"""Section 2.11 - Adaptive cache policies.

Three decisions are made here, and each is what separates a cache that helps
from one that quietly serves wrong answers:

  should_serve  - is it safe to return this cached entry *now*?
  should_store  - is this response worth keeping at all?
  ttl_for       - how long should it stay valid?

Plus a feedback-driven similarity threshold: reported bad hits raise the bar
sharply, accepted hits lower it slowly (additive-increase / additive-decrease).
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .types import CacheEntry, CacheRequest

# Queries whose correct answer changes over time. Serving these from cache is
# how a gateway ends up reporting yesterday's price or last week's news.
_VOLATILE_PATTERNS = [
    r"\b(today|tonight|right now|currently|current|latest|newest|recent|"
    r"this (week|month|year|morning)|as of|up[- ]to[- ]date|breaking)\b",
    r"\b(price|stock|weather|forecast|score|news|headline|exchange rate|"
    r"traffic|availability|in stock|deadline)\b",
    r"\b(اکنون|امروز|الان|جدیدترین|آخرین|قیمت|اخبار|هم اکنون)\b",
    r"\b(20[2-9]\d)\b",                       # explicit recent years
]
_VOLATILE_RE = re.compile("|".join(_VOLATILE_PATTERNS), re.IGNORECASE)

# Requests that must never be answered from a shared cache.
_PERSONAL_PATTERNS = [
    r"\b(my|mine|our) (account|balance|order|password|token|api key|address)\b",
    r"\b(who am i|what did i (just )?say|my name is)\b",
]
_PERSONAL_RE = re.compile("|".join(_PERSONAL_PATTERNS), re.IGNORECASE)

_NONDETERMINISTIC_RE = re.compile(
    r"\b(random|randomly|surprise me|be creative|creative|brainstorm|"
    r"generate \d+ different|another (one|version)|different answer|"
    r"variation|joke|poem|story)\b",
    re.IGNORECASE,
)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptivePolicy:
    def __init__(self, config, store=None):
        self.config = config
        self.store = store
        self._lock = threading.RLock()
        self._threshold = config.similarity_threshold
        self.stats = {"volatile_skips": 0, "popularity_skips": 0,
                      "temperature_skips": 0, "personal_skips": 0,
                      "nondeterministic_skips": 0, "stale_skips": 0}

    # ------------------------------------------------------------------ #
    # Classification helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_volatile(text: str) -> bool:
        return bool(_VOLATILE_RE.search(text or ""))

    @staticmethod
    def is_personal(text: str) -> bool:
        return bool(_PERSONAL_RE.search(text or ""))

    @staticmethod
    def is_nondeterministic(text: str) -> bool:
        return bool(_NONDETERMINISTIC_RE.search(text or ""))

    def classify(self, request: CacheRequest) -> Dict[str, bool]:
        text = request.normalized_prompt or request.raw_prompt
        temperature = float(request.params.get("temperature", 0.0) or 0.0)
        return {
            "volatile": self.is_volatile(text),
            "personal": self.is_personal(text),
            "nondeterministic": self.is_nondeterministic(text),
            "high_temperature": temperature > self.config.max_temperature_for_reuse,
            "has_tools": bool(request.params.get("tools")),
        }

    # ------------------------------------------------------------------ #
    # Read-side decision
    # ------------------------------------------------------------------ #
    def should_serve(self, request: CacheRequest, entry: CacheEntry,
                     score: Optional[float], layer: str) -> PolicyDecision:
        if not self.config.adaptive_enabled:
            return PolicyDecision(True)

        if request.no_cache:
            return PolicyDecision(False, "request set no_cache")

        if entry.is_expired():
            self.stats["stale_skips"] += 1
            return PolicyDecision(False, "entry expired")

        flags = self.classify(request)

        if flags["personal"] and entry.user_id != request.user_id:
            self.stats["personal_skips"] += 1
            return PolicyDecision(False, "personal query, different user")

        if flags["volatile"] and not self.config.volatile_serve_enabled:
            self.stats["volatile_skips"] += 1
            return PolicyDecision(False, "time-sensitive query")

        if flags["volatile"]:
            age = time.time() - entry.created_at
            if age > self.config.volatile_ttl:
                self.stats["volatile_skips"] += 1
                return PolicyDecision(
                    False, f"volatile entry too old ({age:.0f}s)"
                )

        if flags["nondeterministic"]:
            self.stats["nondeterministic_skips"] += 1
            return PolicyDecision(False, "user asked for fresh/varied output")

        if flags["high_temperature"]:
            self.stats["temperature_skips"] += 1
            return PolicyDecision(
                False,
                f"temperature above reuse limit "
                f"({request.params.get('temperature')})",
            )

        # Entries with a bad feedback record are retired from the read path.
        if entry.rejected > 0 and entry.rejected >= entry.accepted + 2:
            return PolicyDecision(False, "entry has negative feedback")

        return PolicyDecision(True, f"{layer} hit accepted")

    # ------------------------------------------------------------------ #
    # Write-side decision
    # ------------------------------------------------------------------ #
    def should_store(self, request: CacheRequest,
                     response: Dict[str, Any]) -> PolicyDecision:
        cfg = self.config
        if request.no_store:
            return PolicyDecision(False, "request set no_store")

        text = response.get("text") or ""
        if len(text.strip()) < cfg.min_response_chars:
            return PolicyDecision(False, "response too short")

        if not cfg.adaptive_enabled:
            return PolicyDecision(True, ttl=request.ttl or cfg.default_ttl)

        flags = self.classify(request)
        if flags["personal"] and not cfg.key_includes_user:
            self.stats["personal_skips"] += 1
            return PolicyDecision(False, "personal query, shared cache")

        if flags["nondeterministic"]:
            return PolicyDecision(False, "non-deterministic request")

        if flags["high_temperature"]:
            return PolicyDecision(False, "sampling temperature too high")

        # Popularity gate: only promote a prompt into the cache once it has
        # been seen N times, which keeps one-off prompts out of the index.
        if cfg.store_after_n_requests > 1 and self.store is not None:
            seen = self.store.incr(
                f"{cfg.namespace}:pop:{_short_hash(request.normalized_prompt)}"
            )
            if seen < cfg.store_after_n_requests:
                self.stats["popularity_skips"] += 1
                return PolicyDecision(
                    False, f"below popularity threshold ({seen}/"
                           f"{cfg.store_after_n_requests})"
                )

        return PolicyDecision(True, "stored",
                              ttl=self.ttl_for(request, flags),
                              metadata={"volatile": flags["volatile"]})

    def ttl_for(self, request: CacheRequest,
                flags: Optional[Dict[str, bool]] = None) -> float:
        if request.ttl:
            return request.ttl
        flags = flags or self.classify(request)
        if flags["volatile"]:
            return self.config.volatile_ttl
        return self.config.default_ttl

    # ------------------------------------------------------------------ #
    # Adaptive similarity threshold
    # ------------------------------------------------------------------ #
    @property
    def threshold(self) -> float:
        with self._lock:
            return self._threshold

    def record_feedback(self, accepted: bool) -> float:
        """Move the similarity threshold based on hit quality feedback."""
        cfg = self.config
        if not cfg.adaptive_threshold_enabled:
            return self._threshold
        with self._lock:
            if accepted:
                self._threshold = max(cfg.threshold_min,
                                      self._threshold - cfg.threshold_step_down)
            else:
                self._threshold = min(cfg.threshold_max,
                                      self._threshold + cfg.threshold_step_up)
            return self._threshold

    # ------------------------------------------------------------------ #
    def estimate_saving(self, entry: CacheEntry) -> float:
        """Dollar value of a cache hit, using real cost when known."""
        if entry.cost_usd:
            return entry.cost_usd
        cfg = self.config
        return (entry.prompt_tokens / 1000.0 * cfg.reference_cost_per_1k_prompt
                + entry.completion_tokens / 1000.0
                * cfg.reference_cost_per_1k_completion)


def _short_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:24]
