"""Storage backends for the cache.

Every layer persists through the same tiny KV interface, so switching between
in-process memory, an on-disk SQLite file, and Redis is a config change.
Values are JSON-serializable dicts; TTL is enforced by the backend.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple


class KVStore(ABC):
    """Minimal key/value contract shared by all backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def set(self, key: str, value: Dict[str, Any],
            ttl: Optional[float] = None) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def scan(self, prefix: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]: ...

    @abstractmethod
    def keys(self, prefix: str = "") -> List[str]: ...

    @abstractmethod
    def incr(self, key: str, amount: int = 1) -> int: ...

    @abstractmethod
    def clear(self, prefix: str = "") -> int: ...

    def size(self, prefix: str = "") -> int:
        return len(self.keys(prefix))

    def close(self) -> None:  # pragma: no cover - backend specific
        pass


# --------------------------------------------------------------------------- #
class MemoryStore(KVStore):
    """Thread-safe in-process store. Fastest, lost on restart."""

    def __init__(self) -> None:
        self._data: Dict[str, Tuple[Optional[float], Dict[str, Any]]] = {}
        self._counters: Dict[str, int] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at is not None and time.time() >= expires_at:
                self._data.pop(key, None)
                return None
            return json.loads(json.dumps(value))  # defensive copy

    def set(self, key: str, value: Dict[str, Any],
            ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        with self._lock:
            self._data[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._counters.pop(key, None)

    def scan(self, prefix: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            items = [(k, v) for k, v in self._data.items() if k.startswith(prefix)]
        now = time.time()
        for key, (expires_at, value) in items:
            if expires_at is not None and now >= expires_at:
                self.delete(key)
                continue
            yield key, value

    def keys(self, prefix: str = "") -> List[str]:
        return [k for k, _ in self.scan(prefix)]

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount
            return self._counters[key]

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            targets = [k for k in self._data if k.startswith(prefix)]
            for key in targets:
                self._data.pop(key, None)
            for key in [k for k in self._counters if k.startswith(prefix)]:
                self._counters.pop(key, None)
            return len(targets)


# --------------------------------------------------------------------------- #
class DiskStore(KVStore):
    """SQLite-backed store: survives restarts, no external service needed."""

    def __init__(self, path: str):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._lock = threading.RLock()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0,
                                   check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    expires_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv(expires_at);
                CREATE TABLE IF NOT EXISTS counters (
                    key   TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and time.time() >= expires_at:
                self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
                self._conn.commit()
                return None
            return json.loads(value)

    def set(self, key: str, value: Dict[str, Any],
            ttl: Optional[float] = None) -> None:
        expires_at = time.time() + ttl if ttl else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(key, value, expires_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "expires_at=excluded.expires_at",
                (key, json.dumps(value, ensure_ascii=False), expires_at),
            )
            self._conn.commit()

    def delete(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.execute("DELETE FROM counters WHERE key = ?", (key,))
            self._conn.commit()

    def scan(self, prefix: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value, expires_at FROM kv WHERE key LIKE ? ESCAPE '\\'",
                (_like_prefix(prefix),),
            ).fetchall()
        now = time.time()
        for key, value, expires_at in rows:
            if expires_at is not None and now >= expires_at:
                self.delete(key)
                continue
            yield key, json.loads(value)

    def keys(self, prefix: str = "") -> List[str]:
        return [k for k, _ in self.scan(prefix)]

    def incr(self, key: str, amount: int = 1) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO counters(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = value + ?",
                (key, amount, amount),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT value FROM counters WHERE key = ?", (key,)
            ).fetchone()
            return int(row[0]) if row else 0

    def clear(self, prefix: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM kv WHERE key LIKE ? ESCAPE '\\'", (_like_prefix(prefix),)
            )
            self._conn.execute(
                "DELETE FROM counters WHERE key LIKE ? ESCAPE '\\'",
                (_like_prefix(prefix),),
            )
            self._conn.commit()
            return cur.rowcount or 0

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def _like_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


# --------------------------------------------------------------------------- #
class RedisStore(KVStore):
    """Redis backend for multi-process / multi-node gateway deployments."""

    def __init__(self, url: str, key_prefix: str = "llmcache:"):
        try:
            import redis  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The redis backend requires `pip install redis`."
            ) from exc
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self.key_prefix = key_prefix

    def _k(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self._redis.get(self._k(key))
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Dict[str, Any],
            ttl: Optional[float] = None) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        if ttl:
            self._redis.setex(self._k(key), int(ttl), payload)
        else:
            self._redis.set(self._k(key), payload)

    def delete(self, key: str) -> None:
        self._redis.delete(self._k(key))

    def scan(self, prefix: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]:
        pattern = f"{self.key_prefix}{prefix}*"
        for raw_key in self._redis.scan_iter(match=pattern, count=500):
            value = self._redis.get(raw_key)
            if value:
                yield raw_key[len(self.key_prefix):], json.loads(value)

    def keys(self, prefix: str = "") -> List[str]:
        pattern = f"{self.key_prefix}{prefix}*"
        return [k[len(self.key_prefix):]
                for k in self._redis.scan_iter(match=pattern, count=500)]

    def incr(self, key: str, amount: int = 1) -> int:
        return int(self._redis.incrby(f"{self.key_prefix}ctr:{key}", amount))

    def clear(self, prefix: str = "") -> int:
        count = 0
        pattern = f"{self.key_prefix}{prefix}*"
        for raw_key in list(self._redis.scan_iter(match=pattern, count=500)):
            self._redis.delete(raw_key)
            count += 1
        return count


# --------------------------------------------------------------------------- #
def build_store(config) -> KVStore:
    """Factory used by the gateway."""
    backend = (config.backend or "memory").lower()
    if backend == "memory":
        return MemoryStore()
    if backend in {"disk", "sqlite"}:
        return DiskStore(config.disk_path)
    if backend == "redis":
        return RedisStore(config.redis_url)
    raise ValueError(f"Unknown cache backend: {backend}")
