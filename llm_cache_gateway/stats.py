"""Runtime metrics for the caching gateway."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, List


class GatewayStats:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.requests = 0
        self.hits = Counter()          # per-layer hit counts
        self.misses = Counter()        # per-layer miss counts
        self.llm_calls = 0
        self.stored = 0
        self.rejected_by_policy = 0
        self.fewshot_augmented = 0
        self.conversations_compacted = 0
        self.errors = 0
        self.cost_usd = 0.0
        self.cost_saved_usd = 0.0
        self._latencies: List[float] = []
        self._hit_latencies: List[float] = []
        self._miss_latencies: List[float] = []

    # ------------------------------------------------------------------ #
    def record_request(self) -> None:
        with self._lock:
            self.requests += 1

    def record_layer(self, layer: str, hit: bool) -> None:
        with self._lock:
            (self.hits if hit else self.misses)[layer] += 1

    def record_result(self, cached: bool, latency_ms: float,
                      cost: float = 0.0, saved: float = 0.0) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            (self._hit_latencies if cached else self._miss_latencies).append(latency_ms)
            self.cost_usd += cost
            self.cost_saved_usd += saved
            if not cached:
                self.llm_calls += 1

    @property
    def total_hits(self) -> int:
        return sum(self.hits.values())

    @property
    def hit_rate(self) -> float:
        return (self.total_hits / self.requests) if self.requests else 0.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1))))
        return ordered[idx]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            hit_lat = list(self._hit_latencies)
            miss_lat = list(self._miss_latencies)
            return {
                "requests": self.requests,
                "llm_calls": self.llm_calls,
                "hit_rate": round(self.hit_rate, 4),
                "hits_by_layer": dict(self.hits),
                "misses_by_layer": dict(self.misses),
                "stored": self.stored,
                "rejected_by_policy": self.rejected_by_policy,
                "fewshot_augmented": self.fewshot_augmented,
                "conversations_compacted": self.conversations_compacted,
                "errors": self.errors,
                "cost_usd": round(self.cost_usd, 6),
                "cost_saved_usd": round(self.cost_saved_usd, 6),
                "latency_ms": {
                    "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                    "p50": round(self._percentile(latencies, 50), 2),
                    "p95": round(self._percentile(latencies, 95), 2),
                    "mean_on_hit": round(sum(hit_lat) / len(hit_lat), 2) if hit_lat else 0.0,
                    "mean_on_miss": round(sum(miss_lat) / len(miss_lat), 2) if miss_lat else 0.0,
                },
            }

    def reset(self) -> None:
        with self._lock:
            self.__init__()  # noqa: PLC2801 - deliberate full reset
