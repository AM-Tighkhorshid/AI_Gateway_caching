"""Section 2.3 - Keyword extraction and lexical scoring.

Four extractors are provided, all dependency-free re-implementations:

  tfidf    - term frequency weighted by an online-updated inverse doc freq
  rake     - Rapid Automatic Keyword Extraction (degree / frequency)
  yake     - simplified YAKE: casing, position, frequency, dispersion
  keybert  - embedding-based: candidate n-grams ranked by cosine similarity
             against the document embedding (needs an embedder)

BM25 is used for retrieval over the inverted index built from the extracted
keywords, which is what actually answers "have I seen a prompt like this".
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Unicode-aware tokenizer: keeps Latin, Arabic/Persian, CJK, digits.
_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff\u4e00-\u9fff]+", re.UNICODE)

ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again", "further",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "what", "which", "who",
    "whom", "how", "why", "where", "there", "here", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "please",
    "tell", "give", "explain", "want", "need", "like", "get", "make", "know",
}

# Small Persian stopword list - useful because gateway traffic is often mixed.
PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "را", "با", "این", "است", "برای", "آن", "یک",
    "می", "هم", "تا", "کن", "کنید", "بر", "شود", "شد", "های", "ها", "هست",
    "بود", "کرد", "چه", "چی", "چرا", "چگونه", "کدام", "یا", "اگر", "ولی",
    "اما", "خود", "بی", "من", "تو", "او", "ما", "شما", "آنها", "باید", "لطفا",
}

STOPWORDS = ENGLISH_STOPWORDS | PERSIAN_STOPWORDS

_SENTENCE_SPLIT_RE = re.compile(r"[.!?;:\n\u061f\u06d4]+")


def tokenize(text: str, keep_stopwords: bool = False,
             min_len: int = 2) -> List[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if keep_stopwords:
        return tokens
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Corpus statistics (shared IDF table, updated online)
# --------------------------------------------------------------------------- #
class CorpusStats:
    """Online document-frequency table backing TF-IDF and BM25."""

    def __init__(self) -> None:
        self.doc_count: int = 0
        self.doc_freq: Counter = Counter()
        self.total_len: int = 0

    def add(self, tokens: Sequence[str]) -> None:
        self.doc_count += 1
        self.total_len += len(tokens)
        for term in set(tokens):
            self.doc_freq[term] += 1

    @property
    def avg_doc_len(self) -> float:
        return (self.total_len / self.doc_count) if self.doc_count else 1.0

    def idf(self, term: str) -> float:
        """Smoothed IDF; falls back to a high value for unseen rare terms."""
        n = self.doc_count or 1
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def to_dict(self) -> dict:
        return {"doc_count": self.doc_count, "total_len": self.total_len,
                "doc_freq": dict(self.doc_freq)}

    @classmethod
    def from_dict(cls, data: dict) -> "CorpusStats":
        stats = cls()
        stats.doc_count = int(data.get("doc_count", 0))
        stats.total_len = int(data.get("total_len", 0))
        stats.doc_freq = Counter(data.get("doc_freq", {}))
        return stats


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #
class KeywordExtractor:
    """Dispatches to one of the four supported extraction algorithms."""

    def __init__(self, method: str = "tfidf", top_k: int = 10,
                 stats: Optional[CorpusStats] = None, embedder=None):
        self.method = method
        self.top_k = top_k
        self.stats = stats or CorpusStats()
        self.embedder = embedder

    def extract(self, text: str, top_k: Optional[int] = None) -> List[str]:
        k = top_k or self.top_k
        method = self.method
        if method == "rake":
            scored = self._rake(text)
        elif method == "yake":
            scored = self._yake(text)
        elif method == "keybert" and self.embedder is not None:
            scored = self._keybert(text)
        else:
            scored = self._tfidf(text)
        return [term for term, _ in scored[:k]]

    def extract_scored(self, text: str,
                       top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        k = top_k or self.top_k
        if self.method == "rake":
            return self._rake(text)[:k]
        if self.method == "yake":
            return self._yake(text)[:k]
        if self.method == "keybert" and self.embedder is not None:
            return self._keybert(text)[:k]
        return self._tfidf(text)[:k]

    # -- tfidf ---------------------------------------------------------- #
    def _tfidf(self, text: str) -> List[Tuple[str, float]]:
        tokens = tokenize(text)
        if not tokens:
            return []
        tf = Counter(tokens)
        max_tf = max(tf.values())
        scored = [
            (term, (0.5 + 0.5 * count / max_tf) * self.stats.idf(term))
            for term, count in tf.items()
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

    # -- rake ----------------------------------------------------------- #
    def _rake(self, text: str) -> List[Tuple[str, float]]:
        """Phrases are maximal runs of non-stopwords; score = degree/frequency."""
        phrases: List[List[str]] = []
        for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
            current: List[str] = []
            for token in _TOKEN_RE.findall(sentence.lower()):
                if token in STOPWORDS or len(token) < 2:
                    if current:
                        phrases.append(current)
                        current = []
                else:
                    current.append(token)
            if current:
                phrases.append(current)

        freq: Counter = Counter()
        degree: Counter = Counter()
        for phrase in phrases:
            deg = len(phrase) - 1
            for word in phrase:
                freq[word] += 1
                degree[word] += deg
        word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}

        scored = [(" ".join(p), sum(word_score[w] for w in p)) for p in phrases]
        # Deduplicate, keeping the best score per phrase.
        best: Dict[str, float] = {}
        for phrase, score in scored:
            if score > best.get(phrase, -1.0):
                best[phrase] = score
        out = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return out

    # -- yake ----------------------------------------------------------- #
    def _yake(self, text: str) -> List[Tuple[str, float]]:
        """Simplified YAKE: lower score = more important, so we invert it."""
        raw_tokens = _TOKEN_RE.findall(text or "")
        if not raw_tokens:
            return []
        lowered = [t.lower() for t in raw_tokens]
        n = len(lowered)
        tf = Counter(t for t in lowered if t not in STOPWORDS and len(t) > 1)
        if not tf:
            return []
        mean_tf = sum(tf.values()) / len(tf)
        std_tf = math.sqrt(
            sum((v - mean_tf) ** 2 for v in tf.values()) / len(tf)
        ) or 1.0

        positions: Dict[str, List[int]] = defaultdict(list)
        casing: Counter = Counter()
        for idx, (raw, low) in enumerate(zip(raw_tokens, lowered)):
            if low in tf:
                positions[low].append(idx)
                if raw[0].isupper() or raw.isupper():
                    casing[low] += 1

        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]
        sent_count = len(sentences) or 1
        sent_of_term: Dict[str, set] = defaultdict(set)
        for s_idx, sentence in enumerate(sentences):
            for token in set(t.lower() for t in _TOKEN_RE.findall(sentence)):
                if token in tf:
                    sent_of_term[token].add(s_idx)

        scored: List[Tuple[str, float]] = []
        for term, count in tf.items():
            t_case = casing[term] / count
            t_pos = math.log(3 + (sum(positions[term]) / len(positions[term])) / max(n, 1) * 10)
            t_freq = count / (mean_tf + std_tf)
            t_disp = len(sent_of_term[term]) / sent_count
            # YAKE's composite: smaller is better.
            score = t_pos / (1.0 + t_case + t_freq + t_disp)
            scored.append((term, 1.0 / (1e-6 + score)))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

    # -- keybert -------------------------------------------------------- #
    def _keybert(self, text: str) -> List[Tuple[str, float]]:
        """Rank candidate n-grams by cosine similarity to the doc embedding."""
        candidates = self._candidate_ngrams(text)
        if not candidates:
            return []
        vectors = self.embedder.embed([text] + candidates)
        doc_vec, cand_vecs = vectors[0], vectors[1:]
        scored = [
            (cand, _cosine(doc_vec, vec))
            for cand, vec in zip(candidates, cand_vecs)
        ]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored

    @staticmethod
    def _candidate_ngrams(text: str, max_n: int = 3,
                          limit: int = 40) -> List[str]:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        seen, out = set(), []
        for size in range(1, max_n + 1):
            for i in range(len(tokens) - size + 1):
                gram = tokens[i:i + size]
                if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:
                    continue
                if any(len(t) < 2 for t in gram):
                    continue
                phrase = " ".join(gram)
                if phrase not in seen:
                    seen.add(phrase)
                    out.append(phrase)
        return out[:limit]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# BM25 over an inverted index
# --------------------------------------------------------------------------- #
class BM25Scorer:
    """Okapi BM25 with the standard k1/b parameters."""

    def __init__(self, stats: CorpusStats, k1: float = 1.5, b: float = 0.75):
        self.stats = stats
        self.k1 = k1
        self.b = b

    def score(self, query_terms: Sequence[str],
              doc_terms: Sequence[str]) -> float:
        if not query_terms or not doc_terms:
            return 0.0
        doc_tf = Counter(doc_terms)
        doc_len = len(doc_terms)
        avg_len = self.stats.avg_doc_len or 1.0
        total = 0.0
        for term in set(query_terms):
            freq = doc_tf.get(term, 0)
            if not freq:
                continue
            idf = self.stats.idf(term)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            total += idf * freq * (self.k1 + 1) / denom
        return total

    def self_score(self, query_terms: Sequence[str]) -> float:
        """Score of the query against itself: used to normalize into [0, 1]."""
        return self.score(query_terms, query_terms) or 1.0


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
