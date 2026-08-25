"""Section 2.4 - Semantic prompt caching.

    prompt -> embedding -> vector search -> similarity -> cached response

The vector index here is an exact brute-force cosine search (numpy-accelerated
when available). That is the right choice up to roughly 10^5 vectors, which
covers most single-tenant gateways; past that, swap `VectorIndex` for FAISS,
Qdrant, Milvus, or Redis vector search behind the same three methods
(`add`, `search`, `remove`).

Two safeguards beyond the plain threshold check:

  margin  - if the top two neighbours are equally close but belong to
            different prompts, the retrieval is ambiguous and is rejected.
  scope   - model and parameter signature must match, because the same
            question asked of a different model is a different question.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .types import CacheRequest, LayerResult

try:  # numpy is optional; the pure-Python path is correct, just slower
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


class VectorIndex:
    """Flat cosine index over L2-normalized vectors, persisted in the store."""

    def __init__(self, store, namespace: str):
        self.store = store
        self.namespace = namespace
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._matrix = None                     # numpy cache, invalidated on write
        self._lock = threading.RLock()
        self._load()

    @property
    def _key(self) -> str:
        return f"{self.namespace}:vecindex"

    def _load(self) -> None:
        record = self.store.get(self._key)
        if record:
            self._ids = list(record.get("ids", []))
            self._vectors = [list(v) for v in record.get("vectors", [])]

    def _save(self) -> None:
        self.store.set(self._key, {"ids": self._ids, "vectors": self._vectors})

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, entry_id: str, vector: Sequence[float]) -> None:
        with self._lock:
            if entry_id in self._ids:
                self._vectors[self._ids.index(entry_id)] = list(vector)
            else:
                self._ids.append(entry_id)
                self._vectors.append(list(vector))
            self._matrix = None
            self._save()

    def remove(self, entry_id: str) -> None:
        with self._lock:
            if entry_id not in self._ids:
                return
            idx = self._ids.index(entry_id)
            self._ids.pop(idx)
            self._vectors.pop(idx)
            self._matrix = None
            self._save()

    def search(self, vector: Sequence[float],
               top_k: int = 5) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            ids = list(self._ids)
            if _np is not None:
                if self._matrix is None:
                    self._matrix = _np.asarray(self._vectors, dtype="float32")
                matrix = self._matrix
                query = _np.asarray(vector, dtype="float32")
                if matrix.shape[1] != query.shape[0]:
                    # Dimension change (embedder switched): index is unusable.
                    return []
                scores = matrix @ query
                order = _np.argsort(-scores)[:top_k]
                return [(ids[int(i)], float(scores[int(i)])) for i in order]

            scored = [(ids[i], _dot(vector, vec))
                      for i, vec in enumerate(self._vectors)
                      if len(vec) == len(vector)]
            scored.sort(key=lambda kv: -kv[1])
            return scored[:top_k]

    def clear(self) -> None:
        with self._lock:
            self._ids, self._vectors, self._matrix = [], [], None
            self._save()


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity that does not assume normalized inputs."""
    dot = _dot(a, b)
    na = math.sqrt(_dot(a, a)) or 1e-12
    nb = math.sqrt(_dot(b, b)) or 1e-12
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
class SemanticCache:
    def __init__(self, repository, config, embedder):
        self.repo = repository
        self.config = config
        self.embedder = embedder
        self.index = VectorIndex(repository.store, config.namespace)
        # Runtime threshold, moved by the adaptive controller (section 2.11).
        self.threshold = config.similarity_threshold

    # ------------------------------------------------------------------ #
    def embed(self, text: str) -> List[float]:
        return self.embedder.embed_one(text)

    def index_entry(self, entry_key: str, text: str,
                    vector: Optional[Sequence[float]] = None) -> str:
        vec = list(vector) if vector is not None else self.embed(text)
        self.index.add(entry_key, vec)
        return entry_key

    def remove_entry(self, entry_key: str) -> None:
        self.index.remove(entry_key)

    # ------------------------------------------------------------------ #
    def search(self, text: str, top_k: Optional[int] = None,
               vector: Optional[Sequence[float]] = None) -> List[Tuple[str, float]]:
        vec = list(vector) if vector is not None else self.embed(text)
        return self.index.search(vec, top_k or self.config.semantic_top_k)

    def lookup(self, request: CacheRequest,
               vector: Optional[Sequence[float]] = None,
               threshold: Optional[float] = None) -> LayerResult:
        started = time.perf_counter()
        if len(self.index) == 0:
            return LayerResult("semantic", False,
                               (time.perf_counter() - started) * 1000,
                               detail="empty index")

        active_threshold = threshold if threshold is not None else self.threshold
        neighbours = self.search(request.normalized_prompt, vector=vector)
        elapsed = (time.perf_counter() - started) * 1000

        if not neighbours:
            return LayerResult("semantic", False, elapsed, detail="no neighbours")

        # Walk the ranked list: the closest neighbour may be out of scope
        # (different model or expired) while the next one is valid.
        for rank, (entry_key, score) in enumerate(neighbours):
            if score < active_threshold:
                return LayerResult(
                    "semantic", False, elapsed, score=neighbours[0][1],
                    detail=f"below threshold ({neighbours[0][1]:.4f} < "
                           f"{active_threshold:.4f})",
                )
            entry = self.repo.get(entry_key)
            if entry is None:
                self.remove_entry(entry_key)
                continue
            if entry.model != request.model:
                continue
            if (self.config.key_includes_user and request.user_id
                    and entry.user_id != request.user_id):
                continue

            # Ambiguity guard: two distinct prompts nearly tied at the top.
            margin_ok = True
            if self.config.semantic_min_margin > 0 and len(neighbours) > rank + 1:
                runner_up = neighbours[rank + 1][1]
                margin_ok = (score - runner_up) >= self.config.semantic_min_margin
            if not margin_ok:
                return LayerResult("semantic", False, elapsed, score=score,
                                   detail="ambiguous neighbourhood")

            return LayerResult("semantic", True, elapsed, score=score,
                               entry=entry, detail=f"cosine={score:.4f}")

        return LayerResult("semantic", False, elapsed,
                           score=neighbours[0][1], detail="no in-scope neighbour")

    # ------------------------------------------------------------------ #
    def rebuild(self) -> int:
        """Re-embed every stored entry. Needed after switching embedder."""
        self.index.clear()
        entries = list(self.repo.iter_entries())
        if not entries:
            return 0
        vectors = self.embedder.embed([e.normalized_prompt for e in entries])
        for entry, vector in zip(entries, vectors):
            self.index.add(entry.key, vector)
        return len(entries)
