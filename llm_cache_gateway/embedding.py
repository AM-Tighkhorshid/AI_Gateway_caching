"""Embedding backends for the semantic layer (design doc section 2.4).

Three implementations, selected by config or auto-detected:

  openrouter - POST /api/v1/embeddings (OpenAI-compatible). Best quality.
               Free embedding models exist but rotate, so the model is
               discovered at runtime when not pinned.
  sbert      - local sentence-transformers, if installed. No network.
  hashing    - dependency-free fallback: hashed word n-grams + character
               n-grams. Catches paraphrases that share vocabulary, but it is
               NOT a semantic model. Use it for tests and offline demos only.

All backends return L2-normalized vectors, so cosine similarity is a dot
product and stays in [-1, 1].
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Sequence

_WORD_RE = re.compile(r"[\w\u0600-\u06ff\u4e00-\u9fff]+", re.UNICODE)


# --------------------------------------------------------------------------- #
class BaseEmbedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch(list(texts))

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# --------------------------------------------------------------------------- #
class HashingEmbedder(BaseEmbedder):
    """Offline fallback. Deterministic, zero dependencies, zero network.

    Word unigrams/bigrams and character 4-grams are hashed into a fixed-size
    vector with sublinear term weighting. Character n-grams give partial
    robustness to typos and morphology; they do not give real semantics.
    """

    name = "hashing"

    def __init__(self, dim: int = 768, char_ngram: int = 4):
        self.dim = dim
        self.char_ngram = char_ngram

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        text = (text or "").lower()
        words = _WORD_RE.findall(text)

        features: List[str] = []
        features.extend(f"w:{w}" for w in words)
        features.extend(f"b:{a}_{b}" for a, b in zip(words, words[1:]))
        joined = " ".join(words)
        n = self.char_ngram
        features.extend(
            f"c:{joined[i:i + n]}" for i in range(max(0, len(joined) - n + 1))
        )
        if not features:
            return vec

        # Sublinear weighting keeps long prompts from dominating.
        weights = {"w": 1.0, "b": 1.4, "c": 0.35}
        for feature in features:
            bucket, sign = self._hash(feature)
            vec[bucket] += sign * weights.get(feature[0], 1.0)

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def _hash(self, feature: str):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0


# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder(BaseEmbedder):
    """Local sentence-transformers model. Good quality, no API cost."""

    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sbert backend requires `pip install sentence-transformers`."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


# --------------------------------------------------------------------------- #
class OpenRouterEmbedder(BaseEmbedder):
    """Embeddings through OpenRouter's OpenAI-compatible /embeddings endpoint."""

    name = "openrouter"

    def __init__(self, client, model: Optional[str] = None):
        self.client = client
        self.model = model or client.pick_embedding_model()
        self.dim = 0  # learned from the first response

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self.client.embeddings(list(texts), model=self.model)
        normalized = [_l2_normalize(v) for v in vectors]
        if normalized and not self.dim:
            self.dim = len(normalized[0])
        return normalized


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec)) or 1.0
    return [float(x) / norm for x in vec]


# --------------------------------------------------------------------------- #
class CachingEmbedder(BaseEmbedder):
    """Memoizes vectors by content hash.

    Embeddings are deterministic, so this removes the single largest source of
    redundant API calls in the semantic layer: re-embedding the same prompt on
    every lookup and again on store.
    """

    def __init__(self, inner: BaseEmbedder, max_items: int = 5000,
                 store=None, namespace: str = "default"):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.max_items = max_items
        self._mem: "OrderedDict[str, List[float]]" = OrderedDict()
        self._lock = threading.RLock()
        self.store = store
        self.namespace = namespace
        self.hits = 0
        self.misses = 0

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self.inner.dim

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing: List[int] = []

        for idx, text in enumerate(texts):
            key = self._key(text)
            with self._lock:
                cached = self._mem.get(key)
                if cached is not None:
                    self._mem.move_to_end(key)
            if cached is None and self.store is not None:
                record = self.store.get(f"{self.namespace}:emb:{key}")
                cached = record.get("vector") if record else None
                if cached is not None:
                    self._put(key, cached)
            if cached is not None:
                results[idx] = cached
                self.hits += 1
            else:
                missing.append(idx)

        if missing:
            self.misses += len(missing)
            fresh = self.inner.embed([texts[i] for i in missing])
            for idx, vector in zip(missing, fresh):
                key = self._key(texts[idx])
                self._put(key, vector)
                if self.store is not None:
                    self.store.set(f"{self.namespace}:emb:{key}",
                                   {"vector": vector})
                results[idx] = vector

        return [r for r in results if r is not None]

    def _put(self, key: str, vector: List[float]) -> None:
        with self._lock:
            self._mem[key] = vector
            self._mem.move_to_end(key)
            while len(self._mem) > self.max_items:
                self._mem.popitem(last=False)


# --------------------------------------------------------------------------- #
def build_embedder(config, client=None, store=None) -> BaseEmbedder:
    """Select an embedding backend according to config, with graceful fallback."""
    backend = (config.embedding_backend or "auto").lower()

    def _wrap(inner: BaseEmbedder) -> BaseEmbedder:
        if not config.embedding_cache_enabled:
            return inner
        return CachingEmbedder(inner, store=store, namespace=config.namespace)

    if backend == "hashing":
        return _wrap(HashingEmbedder(config.hashing_dim))

    if backend == "sbert":
        return _wrap(SentenceTransformerEmbedder())

    if backend == "openrouter":
        if client is None:
            raise ValueError("openrouter embedding backend needs a provider client")
        return _wrap(OpenRouterEmbedder(client, config.embedding_model))

    # auto: OpenRouter -> sentence-transformers -> hashing
    if client is not None and getattr(client, "api_key", None):
        try:
            embedder = OpenRouterEmbedder(client, config.embedding_model)
            embedder.embed(["warmup"])  # fail fast if the model is unavailable
            return _wrap(embedder)
        except Exception:  # noqa: BLE001 - fall through to a local backend
            pass
    try:
        return _wrap(SentenceTransformerEmbedder())
    except Exception:  # noqa: BLE001
        return _wrap(HashingEmbedder(config.hashing_dim))
