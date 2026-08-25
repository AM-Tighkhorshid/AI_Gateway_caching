"""Section 2.3 - Keyword-based caching.

Extracted keywords go into an inverted index (term -> entry keys). At lookup
time the query's keywords pull a small candidate set, which is ranked with
BM25 and a Jaccard tie-breaker. The winner is accepted only if its normalized
score clears `keyword_min_score`.

This layer costs no embedding call, so it sits between the exact and semantic
layers in the hierarchy and absorbs the "same intent, different wording" cases
that share vocabulary.

Because lexical overlap is a weak proxy for meaning ("how do I *start* a
container" vs "how do I *stop* a container" share most keywords), the default
config asks the semantic layer to confirm the candidate before it is served.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .keywords import BM25Scorer, CorpusStats, KeywordExtractor, jaccard, tokenize
from .types import CacheRequest, LayerResult


class KeywordCache:
    def __init__(self, repository, config, embedder=None):
        self.repo = repository
        self.config = config
        self.store = repository.store
        self.namespace = config.namespace
        self._lock = threading.RLock()

        self.stats = self._load_stats()
        self.extractor = KeywordExtractor(
            method=config.keyword_extractor,
            top_k=config.keyword_top_k,
            stats=self.stats,
            embedder=embedder,
        )
        self.bm25 = BM25Scorer(self.stats)
        self._index: Dict[str, List[str]] = defaultdict(list)
        self._doc_terms: Dict[str, List[str]] = {}
        self._load_index()

    # -- persistence ---------------------------------------------------- #
    @property
    def _stats_key(self) -> str:
        return f"{self.namespace}:kwstats"

    @property
    def _index_key(self) -> str:
        return f"{self.namespace}:kwindex"

    def _load_stats(self) -> CorpusStats:
        record = self.store.get(self._stats_key)
        return CorpusStats.from_dict(record) if record else CorpusStats()

    def _save_stats(self) -> None:
        self.store.set(self._stats_key, self.stats.to_dict())

    def _load_index(self) -> None:
        record = self.store.get(self._index_key)
        if not record:
            return
        self._index = defaultdict(list, {k: list(v) for k, v
                                         in record.get("index", {}).items()})
        self._doc_terms = {k: list(v) for k, v
                           in record.get("docs", {}).items()}

    def _save_index(self) -> None:
        self.store.set(self._index_key,
                       {"index": {k: v for k, v in self._index.items()},
                        "docs": self._doc_terms})

    # -- write path ----------------------------------------------------- #
    def extract(self, text: str) -> List[str]:
        return self.extractor.extract(text)

    def index_entry(self, entry_key: str, text: str,
                    keywords: Optional[Sequence[str]] = None) -> List[str]:
        """Add one entry to the inverted index and update corpus statistics."""
        terms = list(keywords) if keywords else self.extract(text)
        # Multi-word phrases (RAKE/KeyBERT) are also indexed by their tokens so
        # that single-word queries can still reach them.
        flat: List[str] = []
        for term in terms:
            flat.append(term)
            if " " in term:
                flat.extend(tokenize(term))

        with self._lock:
            self.stats.add(tokenize(text))
            self._doc_terms[entry_key] = flat
            for term in set(flat):
                bucket = self._index[term]
                if entry_key not in bucket:
                    bucket.append(entry_key)
            self._save_stats()
            self._save_index()
        return terms

    def remove_entry(self, entry_key: str) -> None:
        with self._lock:
            terms = self._doc_terms.pop(entry_key, [])
            for term in set(terms):
                bucket = self._index.get(term)
                if bucket and entry_key in bucket:
                    bucket.remove(entry_key)
                    if not bucket:
                        self._index.pop(term, None)
            self._save_index()

    # -- read path ------------------------------------------------------ #
    def lookup(self, request: CacheRequest) -> LayerResult:
        started = time.perf_counter()
        query_terms = self.extract(request.normalized_prompt)
        flat_query: List[str] = []
        for term in query_terms:
            flat_query.append(term)
            if " " in term:
                flat_query.extend(tokenize(term))

        if not flat_query:
            return LayerResult("keyword", False,
                               (time.perf_counter() - started) * 1000,
                               detail="no keywords extracted")

        # Candidate generation from the inverted index.
        candidate_hits: Dict[str, int] = defaultdict(int)
        with self._lock:
            for term in set(flat_query):
                for entry_key in self._index.get(term, []):
                    candidate_hits[entry_key] += 1

        if not candidate_hits:
            return LayerResult("keyword", False,
                               (time.perf_counter() - started) * 1000,
                               detail="no candidates")

        candidates = sorted(candidate_hits.items(), key=lambda kv: -kv[1])
        candidates = candidates[: self.config.keyword_candidates]

        # Ranking.
        self.bm25.stats = self.stats
        normalizer = self.bm25.self_score(flat_query)
        scored: List[Tuple[str, float]] = []
        for entry_key, _ in candidates:
            doc_terms = self._doc_terms.get(entry_key, [])
            bm25 = self.bm25.score(flat_query, doc_terms) / normalizer
            lexical = 0.7 * min(1.0, bm25) + 0.3 * jaccard(flat_query, doc_terms)
            scored.append((entry_key, lexical))
        scored.sort(key=lambda kv: -kv[1])

        best_key, best_score = scored[0]
        entry = self.repo.get(best_key)
        elapsed = (time.perf_counter() - started) * 1000

        if entry is None:
            self.remove_entry(best_key)
            return LayerResult("keyword", False, elapsed, score=best_score,
                               detail="stale index reference")

        # A cached answer produced for a different model/params is not
        # interchangeable, so those must still match.
        if entry.model != request.model:
            return LayerResult("keyword", False, elapsed, score=best_score,
                               detail="model mismatch")

        if best_score < self.config.keyword_min_score:
            return LayerResult("keyword", False, elapsed, score=best_score,
                               detail=f"below min_score "
                                      f"({best_score:.3f} < "
                                      f"{self.config.keyword_min_score})")

        return LayerResult("keyword", True, elapsed, score=best_score,
                           entry=entry, detail="lexical match")

    def rebuild(self) -> int:
        """Recreate the index from the entry repository."""
        with self._lock:
            self._index = defaultdict(list)
            self._doc_terms = {}
            self.stats = CorpusStats()
            self.extractor.stats = self.stats
        count = 0
        for entry in self.repo.iter_entries():
            self.index_entry(entry.key, entry.normalized_prompt, entry.keywords)
            count += 1
        return count
