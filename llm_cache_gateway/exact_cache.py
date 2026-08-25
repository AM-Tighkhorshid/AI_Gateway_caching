"""Sections 2.1 + 2.6 - Exact prompt matching and response caching.

The cache key is a hash over:
    normalized prompt  +  model id  +  the parameters that affect the output
                       +  (optionally) the user id

Parameters not in `key_param_fields` are treated as output-irrelevant, so a
different `stream` flag or a different `user` label still hits the cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Optional

from .types import CacheEntry, CacheRequest, LayerResult


class ExactCache:
    def __init__(self, repository, config):
        self.repo = repository
        self.config = config

    # ------------------------------------------------------------------ #
    def params_signature(self, request: CacheRequest) -> str:
        if not self.config.key_includes_params:
            return "-"
        selected = {
            field: request.params.get(field)
            for field in self.config.key_param_fields
            if request.params.get(field) is not None
        }
        if not selected:
            return "-"
        blob = json.dumps(selected, sort_keys=True, ensure_ascii=False,
                          default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def build_key(self, request: CacheRequest) -> str:
        parts = [request.normalized_prompt]
        if self.config.key_includes_model:
            parts.append(f"model={request.model}")
        parts.append(f"params={self.params_signature(request)}")
        if self.config.key_includes_user and request.user_id:
            parts.append(f"user={request.user_id}")
        digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return digest[:40]

    # ------------------------------------------------------------------ #
    def lookup(self, request: CacheRequest) -> LayerResult:
        started = time.perf_counter()
        key = self.build_key(request)
        entry = self.repo.get(key)
        elapsed = (time.perf_counter() - started) * 1000

        if entry is None:
            return LayerResult("exact", False, elapsed, detail="miss")
        return LayerResult("exact", True, elapsed, score=1.0, entry=entry,
                           detail="exact key match")

    def store(self, request: CacheRequest, response: dict,
              ttl: Optional[float] = None, keywords=None,
              embedding_id: Optional[str] = None,
              volatile: bool = False) -> CacheEntry:
        key = self.build_key(request)
        usage = response.get("usage") or {}
        entry = CacheEntry(
            key=key,
            namespace=request.namespace,
            model=request.model,
            params_hash=self.params_signature(request),
            raw_prompt=request.raw_prompt,
            normalized_prompt=request.normalized_prompt,
            response_text=response.get("text", ""),
            response_raw={"model": response.get("model")},
            keywords=list(keywords or []),
            embedding_id=embedding_id,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=float(response.get("cost_usd", 0.0)),
            user_id=request.user_id,
            volatile=volatile,
            expires_at=time.time() + ttl if ttl else None,
        )
        self.repo.put(entry, ttl=ttl)
        return entry
