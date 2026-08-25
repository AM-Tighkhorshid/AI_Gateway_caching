"""Section 2.2 - Prompt normalization.

Normalization turns a prompt into a canonical form so that cosmetic
differences (casing, spacing, smart quotes, injected timestamps) do not cause
a cache miss. It runs before every other layer and costs microseconds.

Design note: code blocks are protected by default. Lowercasing a Python
snippet would change its meaning, and two different snippets must never
collapse onto the same cache key.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import List, Tuple

from .config import CacheConfig

# --------------------------------------------------------------------------- #
# Regex table
# --------------------------------------------------------------------------- #
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_WS_RE = re.compile(r"[ \t\u00a0\u200c]+")          # incl. NBSP + ZWNJ runs
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I
)
_ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?Z?\b")
_EPOCH_RE = re.compile(r"\b1[6-9]\d{8}(\d{3})?\b")   # 10/13-digit unix time
_REQ_ID_RE = re.compile(r"\b(req|request|trace|session|correlation)[-_ ]?id\s*[:=]\s*\S+", re.I)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

_POLITENESS_RE = re.compile(
    r"\b(please|kindly|could you( please)?|can you( please)?|would you( please)?|"
    r"i would like you to|i want you to|thanks in advance|thank you)\b",
    re.I,
)

# Punctuation variants collapsed onto ASCII equivalents.
_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u2026": "...",
    "\u00ab": '"', "\u00bb": '"',
    "\u060c": ",",   # Arabic comma
    "\u061b": ";",   # Arabic semicolon
    "\u061f": "?",   # Arabic question mark
    "\u06d4": ".",   # Arabic full stop
    "\u064a": "\u06cc",  # Arabic yeh   -> Persian yeh
    "\u0643": "\u06a9",  # Arabic kaf   -> Persian kaf
}
_PUNCT_TABLE = {ord(k): v for k, v in _PUNCT_MAP.items()}

# Arabic/Persian diacritics that carry no retrieval signal.
_DIACRITIC_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")

# Digits only: the placeholder must survive lowercasing and punctuation
# rewriting unchanged, otherwise the protected block can never be restored.
_PLACEHOLDER = "\x00{}\x00"


class PromptNormalizer:
    """Configurable, order-stable normalization pipeline."""

    def __init__(self, config: CacheConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    def normalize(self, text: str) -> str:
        cfg = self.config
        if not cfg.normalization_enabled or not text:
            return text or ""

        protected: List[str] = []
        if cfg.norm_preserve_code_blocks:
            text, protected = self._protect_code(text)

        if cfg.norm_unicode_form:
            text = unicodedata.normalize(cfg.norm_unicode_form, text)

        if cfg.norm_strip_metadata:
            text = self._strip_metadata(text)

        if cfg.norm_standardize_punctuation:
            text = text.translate(_PUNCT_TABLE)
            text = _DIACRITIC_RE.sub("", text)
            # Collapse repeated terminal punctuation: "what???" -> "what?"
            text = re.sub(r"([!?.,;:])\1{1,}", r"\1", text)
            # Drop space before punctuation, ensure one space after.
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)

        if cfg.norm_strip_politeness:
            text = _POLITENESS_RE.sub(" ", text)

        if cfg.norm_lowercase:
            text = text.lower()

        if cfg.norm_sort_json_keys:
            text = self._canonicalize_json(text)

        if cfg.norm_collapse_whitespace:
            text = _WS_RE.sub(" ", text)
            text = _MULTI_NEWLINE_RE.sub("\n\n", text)
            text = "\n".join(line.strip() for line in text.split("\n"))
            text = text.strip()

        if protected:
            text = self._restore_code(text, protected)
        return text

    # ------------------------------------------------------------------ #
    @staticmethod
    def _protect_code(text: str) -> Tuple[str, List[str]]:
        """Replace code regions with placeholders so they survive untouched."""
        blocks: List[str] = []

        def _swap(match: re.Match) -> str:
            blocks.append(match.group(0))
            return _PLACEHOLDER.format(len(blocks) - 1)

        text = _CODE_FENCE_RE.sub(_swap, text)
        text = _INLINE_CODE_RE.sub(_swap, text)
        return text, blocks

    @staticmethod
    def _restore_code(text: str, blocks: List[str]) -> str:
        for idx, block in enumerate(blocks):
            text = text.replace(_PLACEHOLDER.format(idx), block)
        return text

    @staticmethod
    def _strip_metadata(text: str) -> str:
        """Remove volatile identifiers that would otherwise defeat the cache."""
        text = _HTML_COMMENT_RE.sub(" ", text)
        text = _REQ_ID_RE.sub(" ", text)
        text = _UUID_RE.sub("<uuid>", text)
        text = _ISO_TS_RE.sub("<ts>", text)
        text = _EPOCH_RE.sub("<ts>", text)
        return text

    @staticmethod
    def _canonicalize_json(text: str) -> str:
        """If the prompt is a JSON payload, re-serialize with sorted keys."""
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return text
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return text
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
