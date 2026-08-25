"""Central configuration for the gateway caching stack.

Every mechanism described in the design document is switchable here, so the
gateway can be run with any subset of the design space enabled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass
class CacheConfig:
    # ------------------------------------------------------------------ #
    # Provider (OpenRouter)                                              #
    # ------------------------------------------------------------------ #
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    # None => the client discovers a currently-free model at runtime.
    # Free model IDs on OpenRouter rotate frequently, so hard-coding one is
    # a maintenance hazard; discovery is the safer default.
    chat_model: Optional[str] = None
    embedding_model: Optional[str] = None
    prefer_free_models: bool = True
    request_timeout: float = 120.0
    max_retries: int = 4
    referer: str = "https://localhost"
    app_title: str = "llm-cache-gateway"

    # ------------------------------------------------------------------ #
    # Storage backend                                                    #
    # ------------------------------------------------------------------ #
    backend: str = "memory"                 # memory | disk | redis
    disk_path: str = "./.llm_cache/cache.sqlite3"
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "default"              # logical cache partition
    max_entries: int = 20_000
    default_ttl: float = 24 * 3600.0        # seconds

    # ------------------------------------------------------------------ #
    # 2.2 Prompt normalization                                           #
    # ------------------------------------------------------------------ #
    normalization_enabled: bool = True
    norm_unicode_form: str = "NFKC"
    norm_lowercase: bool = True
    norm_collapse_whitespace: bool = True
    norm_standardize_punctuation: bool = True
    norm_strip_metadata: bool = True        # timestamps, request ids, uuids
    norm_strip_politeness: bool = False     # "please", "could you", ...
    norm_preserve_code_blocks: bool = True  # never touch ``` fenced regions
    norm_sort_json_keys: bool = True

    # ------------------------------------------------------------------ #
    # 2.1 / 2.6 Exact prompt + response cache                            #
    # ------------------------------------------------------------------ #
    exact_enabled: bool = True
    key_includes_model: bool = True
    key_includes_params: bool = True
    # Parameters that participate in the cache key. Anything not listed is
    # considered irrelevant for response equivalence.
    key_param_fields: List[str] = field(
        default_factory=lambda: ["temperature", "top_p", "max_tokens", "seed",
                                 "stop", "response_format", "tools"]
    )
    key_includes_user: bool = False         # True => strict per-user isolation

    # ------------------------------------------------------------------ #
    # 2.3 Keyword cache                                                  #
    # ------------------------------------------------------------------ #
    keyword_enabled: bool = True
    keyword_extractor: str = "tfidf"        # tfidf | rake | yake | keybert
    keyword_top_k: int = 10
    keyword_candidates: int = 20            # BM25 candidates pulled per query
    keyword_min_score: float = 0.55         # normalized lexical score in [0,1]
    # A lexical match alone is a weak signal, so by default the winner is
    # confirmed by the embedding model before it is served.
    keyword_requires_semantic_confirm: bool = True
    keyword_confirm_threshold: float = 0.88

    # ------------------------------------------------------------------ #
    # 2.4 Semantic cache                                                 #
    # ------------------------------------------------------------------ #
    semantic_enabled: bool = True
    embedding_backend: str = "auto"         # auto | openrouter | sbert | hashing
    hashing_dim: int = 768                  # only for the offline fallback
    similarity_threshold: float = 0.90
    # The hashing fallback is lexical, so its cosine values live on a
    # different scale than a real embedding model. Applied automatically only
    # when `similarity_threshold` was left at its default.
    hashing_similarity_threshold: float = 0.62
    semantic_top_k: int = 5
    # Minimum gap between the best and second-best neighbour. Guards against
    # serving a response when two different cached prompts are equally close.
    semantic_min_margin: float = 0.0
    embedding_cache_enabled: bool = True

    # ------------------------------------------------------------------ #
    # 2.7 Few-shot example retrieval                                     #
    # ------------------------------------------------------------------ #
    fewshot_enabled: bool = True
    fewshot_k: int = 3
    fewshot_min_similarity: float = 0.55    # floor for "related enough"
    fewshot_max_chars: int = 4000           # budget for injected examples

    # ------------------------------------------------------------------ #
    # 2.8 Conversation context cache                                     #
    # ------------------------------------------------------------------ #
    conversation_enabled: bool = True
    conversation_keep_recent_turns: int = 6
    conversation_summary_trigger_chars: int = 6000
    conversation_summary_max_chars: int = 1200
    conversation_ttl: float = 7 * 24 * 3600.0

    # ------------------------------------------------------------------ #
    # 2.11 Adaptive cache policies                                       #
    # ------------------------------------------------------------------ #
    adaptive_enabled: bool = True
    store_after_n_requests: int = 1         # popularity gate (1 = always)
    max_temperature_for_reuse: float = 0.5  # high-temp answers are not reused
    min_response_chars: int = 1             # skip caching trivial/empty output
    volatile_ttl: float = 120.0             # TTL for time-sensitive queries
    volatile_serve_enabled: bool = False    # serve volatile hits at all?
    eviction_policy: str = "cost_aware"     # lru | lfu | cost_aware
    # Adaptive similarity threshold: raised on reported false hits, lowered
    # slowly while hits are accepted.
    adaptive_threshold_enabled: bool = True
    threshold_min: float = 0.80
    threshold_max: float = 0.985
    threshold_step_up: float = 0.02         # on a rejected (bad) hit
    threshold_step_down: float = 0.002      # on an accepted hit
    # Reference price used to report savings even when the model is free.
    reference_cost_per_1k_prompt: float = 0.0005
    reference_cost_per_1k_completion: float = 0.0015

    # ------------------------------------------------------------------ #
    # Misc                                                               #
    # ------------------------------------------------------------------ #
    collect_traces: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, **overrides) -> "CacheConfig":
        """Build a config from environment variables, then apply overrides."""
        cfg = cls(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            chat_model=os.getenv("OPENROUTER_CHAT_MODEL") or None,
            embedding_model=os.getenv("OPENROUTER_EMBEDDING_MODEL") or None,
            backend=os.getenv("CACHE_BACKEND", "memory"),
            disk_path=os.getenv("CACHE_DISK_PATH", "./.llm_cache/cache.sqlite3"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            namespace=os.getenv("CACHE_NAMESPACE", "default"),
            default_ttl=_env_float("CACHE_TTL", 24 * 3600.0),
            similarity_threshold=_env_float("CACHE_SIMILARITY_THRESHOLD", 0.90),
            semantic_enabled=_env_bool("CACHE_SEMANTIC", True),
            keyword_enabled=_env_bool("CACHE_KEYWORD", True),
            fewshot_enabled=_env_bool("CACHE_FEWSHOT", True),
            max_entries=_env_int("CACHE_MAX_ENTRIES", 20_000),
            log_level=os.getenv("CACHE_LOG_LEVEL", "INFO"),
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"Unknown config field: {key}")
            setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***"
        return data
