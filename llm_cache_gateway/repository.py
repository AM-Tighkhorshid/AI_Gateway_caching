"""Shared storage of cache entries plus eviction.

All layers (exact, keyword, semantic, few-shot) point at the *same* entry
objects; only the index used to find them differs. Keeping one repository
means a response is stored once and can be reached three different ways.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, List, Optional

from .types import CacheEntry


class EntryRepository:
    """CRUD over CacheEntry objects with pluggable eviction."""

    def __init__(self, store, config):
        self.store = store
        self.config = config
        self.namespace = config.namespace
        self._lock = threading.RLock()
        self._evicted = 0

    # ------------------------------------------------------------------ #
    def _entry_key(self, key: str) -> str:
        return f"{self.namespace}:entry:{key}"

    def get(self, key: str) -> Optional[CacheEntry]:
        record = self.store.get(self._entry_key(key))
        if record is None:
            return None
        entry = CacheEntry.from_dict(record)
        if entry.is_expired():
            self.delete(key)
            return None
        return entry

    def put(self, entry: CacheEntry, ttl: Optional[float] = None) -> None:
        with self._lock:
            if ttl and not entry.expires_at:
                entry.expires_at = time.time() + ttl
            remaining = None
            if entry.expires_at:
                remaining = max(1.0, entry.expires_at - time.time())
            self.store.set(self._entry_key(entry.key), entry.to_dict(), ttl=remaining)
            self._maybe_evict()

    def update(self, entry: CacheEntry) -> None:
        """Persist mutations (hit counters, feedback) without touching TTL."""
        remaining = None
        if entry.expires_at:
            remaining = max(1.0, entry.expires_at - time.time())
        self.store.set(self._entry_key(entry.key), entry.to_dict(), ttl=remaining)

    def delete(self, key: str) -> None:
        self.store.delete(self._entry_key(key))

    def iter_entries(self) -> Iterator[CacheEntry]:
        prefix = f"{self.namespace}:entry:"
        for _, record in self.store.scan(prefix):
            entry = CacheEntry.from_dict(record)
            if not entry.is_expired():
                yield entry

    def all_keys(self) -> List[str]:
        prefix = f"{self.namespace}:entry:"
        return [k[len(prefix):] for k in self.store.keys(prefix)]

    def count(self) -> int:
        return len(self.all_keys())

    @property
    def evicted(self) -> int:
        return self._evicted

    # ------------------------------------------------------------------ #
    def _maybe_evict(self) -> None:
        """Enforce max_entries using the configured replacement policy."""
        limit = self.config.max_entries
        if limit <= 0:
            return
        entries = list(self.iter_entries())
        overflow = len(entries) - limit
        if overflow <= 0:
            return

        policy = self.config.eviction_policy
        now = time.time()
        if policy == "lru":
            entries.sort(key=lambda e: e.last_access)
        elif policy == "lfu":
            entries.sort(key=lambda e: (e.hit_count, e.last_access))
        else:  # cost_aware (section 2.11): keep what is expensive to recompute
            entries.sort(key=lambda e: self._utility(e, now))

        for entry in entries[:overflow]:
            self.delete(entry.key)
            self._evicted += 1

    def _utility(self, entry: CacheEntry, now: float) -> float:
        """Higher = more worth keeping.

        Combines recency, frequency, and the cost of regenerating the entry,
        which is the whole point of a cost-aware policy: a rarely-hit but very
        expensive answer can be worth more than a cheap popular one.
        """
        age_hours = max(1e-6, (now - entry.last_access) / 3600.0)
        recency = 1.0 / (1.0 + age_hours)
        frequency = 1.0 + entry.hit_count
        cost = entry.cost_usd or self._reference_cost(entry)
        quality = 1.0 + entry.accepted - 2.0 * entry.rejected
        return recency * frequency * (1.0 + 1000.0 * cost) * max(0.1, quality)

    def _reference_cost(self, entry: CacheEntry) -> float:
        cfg = self.config
        return (entry.prompt_tokens / 1000.0 * cfg.reference_cost_per_1k_prompt
                + entry.completion_tokens / 1000.0
                * cfg.reference_cost_per_1k_completion)

    # ------------------------------------------------------------------ #
    def purge_expired(self) -> int:
        removed = 0
        prefix = f"{self.namespace}:entry:"
        for key, record in list(self.store.scan(prefix)):
            entry = CacheEntry.from_dict(record)
            if entry.is_expired():
                self.store.delete(key)
                removed += 1
        return removed

    def clear(self) -> int:
        return self.store.clear(f"{self.namespace}:")
