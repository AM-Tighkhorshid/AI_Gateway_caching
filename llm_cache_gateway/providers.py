"""LLM providers behind the gateway.

`OpenRouterClient` speaks the OpenAI-compatible OpenRouter API:
    POST {base}/chat/completions
    POST {base}/embeddings
    GET  {base}/models
    GET  {base}/embeddings/models

Free model IDs on OpenRouter rotate often (models are added and delisted week
to week), so the client discovers a currently-free model at runtime instead of
hard-coding one. Pin `chat_model` in the config for reproducible behaviour.

`MockProvider` returns deterministic canned answers and is used by the test
suite and the offline demo, so the whole cache stack can be exercised without
network access or an API key.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class ProviderError(RuntimeError):
    """Raised when the upstream provider cannot serve the request."""


class BaseProvider(ABC):
    name = "base"

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None,
             **params) -> Dict[str, Any]:
        """Return {'text', 'model', 'usage', 'cost_usd', 'raw'}."""


# --------------------------------------------------------------------------- #
class OpenRouterClient(BaseProvider):
    name = "openrouter"

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.request_timeout
        self.max_retries = config.max_retries
        self._models_cache: Optional[List[Dict[str, Any]]] = None
        self._embedding_models_cache: Optional[List[Dict[str, Any]]] = None
        self._chat_model: Optional[str] = config.chat_model
        self._lock = threading.RLock()

    # -- HTTP ----------------------------------------------------------- #
    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Export it, or run the gateway "
                "with MockProvider for offline testing."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers used by OpenRouter's dashboards.
            "HTTP-Referer": self.config.referer,
            "X-Title": self.config.app_title,
        }

    def _request(self, method: str, path: str,
                 payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                url, data=body, headers=self._headers(), method=method
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                # 429 rate limit and 5xx / 529 provider overload are retryable.
                if exc.code in {408, 429, 500, 502, 503, 504, 529}:
                    last_error = ProviderError(f"HTTP {exc.code}: {detail}")
                    self._sleep_backoff(attempt, exc.headers.get("Retry-After"))
                    continue
                raise ProviderError(f"HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = ProviderError(f"network error: {exc}")
                self._sleep_backoff(attempt, None)

        raise last_error or ProviderError("request failed")

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: Optional[str]) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        # Exponential backoff with jitter.
        time.sleep(min(2 ** attempt, 16) * (0.5 + random.random() * 0.5))

    # -- model discovery ------------------------------------------------ #
    def list_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            if self._models_cache is None or refresh:
                self._models_cache = self._request("GET", "/models").get("data", [])
            return self._models_cache

    def list_embedding_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            if self._embedding_models_cache is None or refresh:
                try:
                    data = self._request("GET", "/embeddings/models").get("data", [])
                except ProviderError:
                    data = []
                self._embedding_models_cache = data
            return self._embedding_models_cache

    @staticmethod
    def _is_free(model: Dict[str, Any]) -> bool:
        pricing = model.get("pricing") or {}
        try:
            return (float(pricing.get("prompt", 1)) == 0.0
                    and float(pricing.get("completion", 1)) == 0.0)
        except (TypeError, ValueError):
            return False

    def free_models(self) -> List[str]:
        """Model IDs that currently cost nothing, longest context first."""
        models = [m for m in self.list_models() if self._is_free(m)]
        models.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
        return [m["id"] for m in models]

    def pick_chat_model(self) -> str:
        """Resolve the chat model: pinned config value, else a free model."""
        with self._lock:
            if self._chat_model:
                return self._chat_model
            candidates = self.free_models() if self.config.prefer_free_models else []
            if not candidates:
                raise ProviderError(
                    "No free chat model found. Set config.chat_model explicitly."
                )
            # Prefer instruct-style general models over specialised endpoints.
            preferred = [m for m in candidates
                         if not any(tag in m for tag in ("embed", "vision", "audio"))]
            self._chat_model = (preferred or candidates)[0]
            return self._chat_model

    def pick_embedding_model(self) -> str:
        if self.config.embedding_model:
            return self.config.embedding_model
        models = self.list_embedding_models()
        free = [m["id"] for m in models if self._is_free(m)]
        if free:
            return free[0]
        if models:
            return models[0]["id"]
        # Reasonable default if the discovery endpoint is unavailable.
        return "openai/text-embedding-3-small"

    def model_pricing(self, model_id: str) -> Dict[str, float]:
        for model in self.list_models():
            if model.get("id") == model_id:
                pricing = model.get("pricing") or {}
                try:
                    return {"prompt": float(pricing.get("prompt", 0)),
                            "completion": float(pricing.get("completion", 0))}
                except (TypeError, ValueError):
                    break
        return {"prompt": 0.0, "completion": 0.0}

    # -- inference ------------------------------------------------------ #
    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None,
             **params) -> Dict[str, Any]:
        model_id = model or self.pick_chat_model()
        payload: Dict[str, Any] = {"model": model_id, "messages": messages}
        for key, value in params.items():
            if value is not None:
                payload[key] = value

        data = self._request("POST", "/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"empty response from provider: {str(data)[:300]}")

        text = (choices[0].get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        pricing = self.model_pricing(model_id)
        cost = (prompt_tokens * pricing["prompt"]
                + completion_tokens * pricing["completion"])

        return {
            "text": text,
            "model": data.get("model", model_id),
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens},
            "cost_usd": cost,
            "raw": data,
        }

    def embeddings(self, texts: Sequence[str],
                   model: Optional[str] = None) -> List[List[float]]:
        model_id = model or self.pick_embedding_model()
        data = self._request("POST", "/embeddings",
                             {"model": model_id, "input": list(texts)})
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        if not items:
            raise ProviderError(f"no embeddings returned: {str(data)[:300]}")
        return [item["embedding"] for item in items]


# --------------------------------------------------------------------------- #
class MockProvider(BaseProvider):
    """Deterministic offline provider for tests, demos, and benchmarking."""

    name = "mock"

    def __init__(self, latency: float = 0.05, canned: Optional[Dict[str, str]] = None):
        self.latency = latency
        self.canned = canned or {}
        self.call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None,
             **params) -> Dict[str, Any]:
        self.call_count += 1
        self.calls.append({"messages": messages, "model": model, "params": params})
        time.sleep(self.latency)  # stand-in for real inference latency

        last_user = next(
            (m.get("content", "") for m in reversed(messages)
             if m.get("role") == "user"), ""
        )
        if isinstance(last_user, list):
            last_user = " ".join(str(b) for b in last_user)

        for needle, answer in self.canned.items():
            if needle.lower() in str(last_user).lower():
                text = answer
                break
        else:
            digest = hashlib.sha1(str(last_user).encode("utf-8")).hexdigest()[:8]
            text = f"[mock answer #{self.call_count} · {digest}] {str(last_user)[:120]}"

        prompt_tokens = max(1, len(str(messages)) // 4)
        completion_tokens = max(1, len(text) // 4)
        return {
            "text": text,
            "model": model or "mock/echo",
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens},
            "cost_usd": 0.0,
            "raw": {"mock": True},
        }
