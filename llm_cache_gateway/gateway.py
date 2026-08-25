"""Section 2.5 / 5 - the hierarchical gateway.

Pipeline, in the order proposed by the design document:

    request
      -> conversation compaction        (2.8)
      -> normalization                  (2.2)
      -> exact cache                    (2.1 / 2.6)
      -> keyword cache                  (2.3)
      -> semantic cache                 (2.4)
      -> few-shot retrieval             (2.7)
      -> LLM call
      -> adaptive store decision        (2.11)

Each stage is cheaper than the one after it, so the expensive stages only run
for the requests that actually need them: no embedding is computed when the
exact key hits, and the model is only called when nothing was reusable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from .config import CacheConfig
from .context_cache import ConversationCache, FewShotCache
from .embedding import build_embedder
from .exact_cache import ExactCache
from .keyword_cache import KeywordCache
from .normalize import PromptNormalizer
from .policy import AdaptivePolicy
from .providers import BaseProvider, MockProvider, OpenRouterClient
from .repository import EntryRepository
from .semantic_cache import SemanticCache
from .stats import GatewayStats
from .store import build_store
from .types import (CacheRequest, GatewayResponse, LayerResult, Message,
                    messages_to_text)

logger = logging.getLogger("llm_cache_gateway")


class CachingGateway:
    """An AI gateway with a full hierarchical cache in front of the provider."""

    def __init__(self, config: Optional[CacheConfig] = None,
                 provider: Optional[BaseProvider] = None,
                 store=None, embedder=None):
        self.config = config or CacheConfig.from_env()
        logging.basicConfig(level=getattr(logging, self.config.log_level, 20))

        self.store = store or build_store(self.config)
        self.provider = provider or OpenRouterClient(self.config)

        client = self.provider if isinstance(self.provider, OpenRouterClient) else None
        self.embedder = embedder or build_embedder(self.config, client, self.store)

        self.repo = EntryRepository(self.store, self.config)
        self.normalizer = PromptNormalizer(self.config)
        self.exact = ExactCache(self.repo, self.config)
        self.keyword = KeywordCache(self.repo, self.config, self.embedder)
        self.semantic = SemanticCache(self.repo, self.config, self.embedder)
        self.fewshot = FewShotCache(self.semantic, self.repo, self.config)
        self.conversation = ConversationCache(self.store, self.config, self.provider)
        self.policy = AdaptivePolicy(self.config, self.store)
        self.stats = GatewayStats()
        self._calibrate_threshold()

        logger.info("gateway ready: backend=%s embedder=%s threshold=%.3f",
                    self.config.backend, self.embedder.name,
                    self.policy.threshold)

    def _calibrate_threshold(self) -> None:
        """The hashing fallback scores on a different scale than real models.

        Only applied when the user left `similarity_threshold` at its default,
        so an explicit setting is never silently overridden.
        """
        if "hashing" not in self.embedder.name:
            return
        default = CacheConfig.__dataclass_fields__["similarity_threshold"].default
        if self.config.similarity_threshold != default:
            return
        adjusted = self.config.hashing_similarity_threshold
        self.config.similarity_threshold = adjusted
        self.semantic.threshold = adjusted
        self.policy._threshold = adjusted  # noqa: SLF001 - same package
        logger.warning(
            "Using the offline hashing embedder: similarity threshold set to "
            "%.2f. This backend is lexical, not semantic - install "
            "sentence-transformers or set OPENROUTER_API_KEY for real "
            "semantic caching.", adjusted,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete(self, messages: List[Message] | str,
                 model: Optional[str] = None,
                 user_id: Optional[str] = None,
                 conversation_id: Optional[str] = None,
                 no_cache: bool = False,
                 no_store: bool = False,
                 ttl: Optional[float] = None,
                 **params) -> GatewayResponse:
        """Main entry point. Mirrors an OpenAI-style chat completion call."""
        started = time.perf_counter()
        self.stats.record_request()

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        model_id = model or self._default_model()
        request = CacheRequest(
            messages=list(messages), model=model_id, params=dict(params),
            user_id=user_id, conversation_id=conversation_id,
            namespace=self.config.namespace, no_cache=no_cache,
            no_store=no_store, ttl=ttl,
        )

        trace: List[Dict[str, Any]] = []

        # -- 2.8 conversation context caching --------------------------- #
        working_messages, compacted = self.conversation.compact(
            request.messages, conversation_id, model_id
        )
        if compacted:
            self.stats.conversations_compacted += 1
            trace.append({"layer": "conversation", "hit": True,
                          "detail": "history compacted into rolling summary"})

        # -- 2.2 normalization ------------------------------------------ #
        norm_started = time.perf_counter()
        request.raw_prompt = messages_to_text(working_messages, include_system=False)
        canonical = messages_to_text(working_messages, include_system=True)
        request.normalized_prompt = self.normalizer.normalize(canonical)
        trace.append({"layer": "normalize", "hit": True,
                      "latency_ms": round((time.perf_counter() - norm_started) * 1000, 3),
                      "detail": f"{len(canonical)} -> "
                                f"{len(request.normalized_prompt)} chars"})

        # -- cache read path -------------------------------------------- #
        query_vector: Optional[List[float]] = None
        hit: Optional[LayerResult] = None
        rejected_keys: List[str] = []

        if not no_cache:
            hit, query_vector, rejected_keys = self._read_path(request, trace)

        if hit is not None and hit.entry is not None:
            entry = hit.entry
            entry.touch()
            self.repo.update(entry)
            saved = self.policy.estimate_saving(entry)
            latency = (time.perf_counter() - started) * 1000
            self.stats.record_result(True, latency, saved=saved)
            return GatewayResponse(
                text=entry.response_text, cached=True, source=hit.layer,
                model=entry.model, request_id=request.request_id,
                latency_ms=latency, similarity=hit.score, entry_key=entry.key,
                conversation_compacted=compacted,
                prompt_tokens=entry.prompt_tokens,
                completion_tokens=entry.completion_tokens,
                cost_saved_usd=saved,
                trace=trace if self.config.collect_traces else [],
            )

        # -- 2.7 few-shot augmentation ---------------------------------- #
        outbound = working_messages
        examples: Sequence = ()
        if self.config.fewshot_enabled and not no_cache:
            fs_started = time.perf_counter()
            examples = self.fewshot.retrieve(request, vector=query_vector,
                                             exclude=rejected_keys)
            if examples:
                outbound = self.fewshot.build_messages(working_messages, examples)
                self.stats.fewshot_augmented += 1
            trace.append({
                "layer": "fewshot", "hit": bool(examples),
                "latency_ms": round((time.perf_counter() - fs_started) * 1000, 3),
                "detail": f"{len(examples)} example(s) injected",
            })

        # -- provider call ---------------------------------------------- #
        try:
            result = self.provider.chat(outbound, model=model_id, **params)
        except Exception as exc:  # noqa: BLE001
            self.stats.errors += 1
            logger.error("provider call failed: %s", exc)
            raise

        # -- 2.11 store decision ---------------------------------------- #
        entry_key = self._write_path(request, result, query_vector, trace)

        latency = (time.perf_counter() - started) * 1000
        usage = result.get("usage") or {}
        self.stats.record_result(False, latency, cost=result.get("cost_usd", 0.0))

        return GatewayResponse(
            text=result.get("text", ""), cached=False, source="llm",
            model=result.get("model", model_id), request_id=request.request_id,
            latency_ms=latency, entry_key=entry_key,
            fewshot_used=len(examples), conversation_compacted=compacted,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=float(result.get("cost_usd", 0.0)),
            trace=trace if self.config.collect_traces else [],
            raw=result.get("raw", {}),
        )

    # ------------------------------------------------------------------ #
    # Read path: exact -> keyword -> semantic
    # ------------------------------------------------------------------ #
    def _read_path(self, request: CacheRequest, trace: List[Dict[str, Any]]):
        cfg = self.config
        query_vector: Optional[List[float]] = None
        rejected: List[str] = []

        # 2.1 exact
        if cfg.exact_enabled:
            result = self.exact.lookup(request)
            self.stats.record_layer("exact", result.hit)
            if result.hit and result.entry is not None:
                decision = self.policy.should_serve(request, result.entry,
                                                    result.score, "exact")
                if decision.allowed:
                    trace.append(result.to_dict())
                    return result, query_vector, rejected
                self.stats.rejected_by_policy += 1
                rejected.append(result.entry.key)
                result = LayerResult("exact", False, result.latency_ms,
                                     result.score, detail=decision.reason)
            trace.append(result.to_dict())

        # 2.3 keyword
        if cfg.keyword_enabled:
            result = self.keyword.lookup(request)
            if result.hit and result.entry is not None:
                confirmed = True
                if cfg.keyword_requires_semantic_confirm and cfg.semantic_enabled:
                    query_vector = self.semantic.embed(request.normalized_prompt)
                    neighbours = dict(self.semantic.index.search(query_vector, 10))
                    similarity = neighbours.get(result.entry.key, 0.0)
                    confirmed = similarity >= cfg.keyword_confirm_threshold
                    result.detail += f"; semantic confirm={similarity:.4f}"
                    result.score = similarity if confirmed else result.score
                if confirmed:
                    decision = self.policy.should_serve(request, result.entry,
                                                        result.score, "keyword")
                    if decision.allowed:
                        self.stats.record_layer("keyword", True)
                        trace.append(result.to_dict())
                        return result, query_vector, rejected
                    self.stats.rejected_by_policy += 1
                    rejected.append(result.entry.key)
                    result = LayerResult("keyword", False, result.latency_ms,
                                         result.score, detail=decision.reason)
                else:
                    result = LayerResult("keyword", False, result.latency_ms,
                                         result.score,
                                         detail="rejected by semantic confirmation")
            self.stats.record_layer("keyword", result.hit)
            trace.append(result.to_dict())

        # 2.4 semantic
        if cfg.semantic_enabled:
            if query_vector is None:
                query_vector = self.semantic.embed(request.normalized_prompt)
            result = self.semantic.lookup(request, vector=query_vector,
                                          threshold=self.policy.threshold)
            if result.hit and result.entry is not None:
                decision = self.policy.should_serve(request, result.entry,
                                                    result.score, "semantic")
                if decision.allowed:
                    self.stats.record_layer("semantic", True)
                    trace.append(result.to_dict())
                    return result, query_vector, rejected
                self.stats.rejected_by_policy += 1
                rejected.append(result.entry.key)
                result = LayerResult("semantic", False, result.latency_ms,
                                     result.score, detail=decision.reason)
            self.stats.record_layer("semantic", result.hit)
            trace.append(result.to_dict())

        return None, query_vector, rejected

    # ------------------------------------------------------------------ #
    # Write path
    # ------------------------------------------------------------------ #
    def _write_path(self, request: CacheRequest, result: Dict[str, Any],
                    query_vector: Optional[List[float]],
                    trace: List[Dict[str, Any]]) -> Optional[str]:
        decision = self.policy.should_store(request, result)
        if not decision.allowed:
            self.stats.rejected_by_policy += 1
            trace.append({"layer": "store", "hit": False,
                          "detail": decision.reason})
            return None

        keywords = (self.keyword.extract(request.normalized_prompt)
                    if self.config.keyword_enabled else [])

        entry = self.exact.store(
            request, result, ttl=decision.ttl, keywords=keywords,
            volatile=bool(decision.metadata.get("volatile")),
        )

        if self.config.keyword_enabled:
            self.keyword.index_entry(entry.key, request.normalized_prompt, keywords)

        if self.config.semantic_enabled or self.config.fewshot_enabled:
            self.semantic.index_entry(entry.key, request.normalized_prompt,
                                      vector=query_vector)
            entry.embedding_id = entry.key
            self.repo.update(entry)

        self.stats.stored += 1
        trace.append({"layer": "store", "hit": True,
                      "detail": f"cached with ttl={decision.ttl}",
                      "entry_key": entry.key})
        return entry.key

    # ------------------------------------------------------------------ #
    # Feedback, maintenance, introspection
    # ------------------------------------------------------------------ #
    def feedback(self, entry_key: str, accepted: bool) -> float:
        """Report whether a served cache hit was actually correct.

        This is the signal that drives the adaptive similarity threshold and
        the quality term in the eviction score. Wire it to a thumbs-up/down
        control, or to an automatic verifier.
        """
        entry = self.repo.get(entry_key)
        if entry is not None:
            if accepted:
                entry.accepted += 1
            else:
                entry.rejected += 1
            self.repo.update(entry)
            if not accepted and entry.rejected >= entry.accepted + 2:
                self.invalidate(entry_key)
        return self.policy.record_feedback(accepted)

    def set_similarity_threshold(self, value: float) -> float:
        """Override the semantic reuse threshold at runtime."""
        value = max(0.0, min(1.0, float(value)))
        self.config.similarity_threshold = value
        self.semantic.threshold = value
        self.policy._threshold = value  # noqa: SLF001 - same package
        return value

    def invalidate(self, entry_key: str) -> None:
        self.repo.delete(entry_key)
        self.keyword.remove_entry(entry_key)
        self.semantic.remove_entry(entry_key)

    def invalidate_by_tag(self, tag: str) -> int:
        count = 0
        for entry in list(self.repo.iter_entries()):
            if tag in entry.tags:
                self.invalidate(entry.key)
                count += 1
        return count

    def warm(self, pairs: Sequence, model: Optional[str] = None) -> int:
        """Pre-populate the cache with known question/answer pairs."""
        model_id = model or self._default_model()
        count = 0
        for question, answer in pairs:
            request = CacheRequest(
                messages=[{"role": "user", "content": question}],
                model=model_id, namespace=self.config.namespace,
            )
            request.raw_prompt = question
            request.normalized_prompt = self.normalizer.normalize(
                f"user: {question}"
            )
            fake_response = {
                "text": answer, "model": model_id,
                "usage": {"prompt_tokens": len(question) // 4,
                          "completion_tokens": len(answer) // 4},
                "cost_usd": 0.0,
            }
            self._write_path(request, fake_response, None, [])
            count += 1
        return count

    def purge_expired(self) -> int:
        removed = self.repo.purge_expired()
        if removed:
            self.keyword.rebuild()
            self.semantic.rebuild()
        return removed

    def clear(self) -> None:
        self.repo.clear()
        self.keyword.rebuild()
        self.semantic.index.clear()

    def report(self) -> Dict[str, Any]:
        snapshot = self.stats.snapshot()
        snapshot.update({
            "entries": self.repo.count(),
            "evicted": self.repo.evicted,
            "vectors": len(self.semantic.index),
            "similarity_threshold": round(self.policy.threshold, 4),
            "embedder": self.embedder.name,
            "policy_skips": dict(self.policy.stats),
        })
        return snapshot

    # ------------------------------------------------------------------ #
    def _default_model(self) -> str:
        if self.config.chat_model:
            return self.config.chat_model
        if isinstance(self.provider, OpenRouterClient):
            return self.provider.pick_chat_model()
        return getattr(self.provider, "name", "mock") + "/default"


# --------------------------------------------------------------------------- #
def build_offline_gateway(**overrides) -> CachingGateway:
    """A fully local gateway (mock provider + hashing embedder) for tests."""
    config = CacheConfig(
        backend="memory", embedding_backend="hashing",
        chat_model="mock/echo", **overrides,
    )
    return CachingGateway(config=config, provider=MockProvider(latency=0.0))
