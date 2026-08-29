"""Data structures shared by every layer of the gateway."""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]  # {"role": "user", "content": "..."}


@dataclass
class CacheRequest:
    """A normalized view of one incoming gateway request."""

    messages: List[Message]
    model: str
    params: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    namespace: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Per-request cache overrides (mirrors HTTP cache-control semantics).
    no_cache: bool = False   # do not read from cache
    no_store: bool = False   # do not write to cache
    ttl: Optional[float] = None

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    # Filled in by the normalization stage.
    raw_prompt: str = ""
    normalized_prompt: str = ""
    system_prompt: str = ""

    @property
    def last_user_message(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return _content_to_text(msg.get("content", ""))
        return ""


@dataclass
class CacheEntry:
    """One cached prompt/response pair plus everything the policies need."""

    key: str
    namespace: str
    model: str
    params_hash: str
    raw_prompt: str
    normalized_prompt: str
    response_text: str
    response_raw: Dict[str, Any] = field(default_factory=dict)

    keywords: List[str] = field(default_factory=list)
    embedding_id: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hit_count: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    user_id: Optional[str] = None
    volatile: bool = False
    tags: List[str] = field(default_factory=list)

    # Quality feedback used by the adaptive threshold controller.
    accepted: int = 0
    rejected: int = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or time.time()) >= self.expires_at

    def touch(self) -> None:
        self.last_access = time.time()
        self.hit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class LayerResult:
    """Outcome of a single cache layer, used for tracing and metrics."""

    layer: str
    hit: bool
    latency_ms: float = 0.0
    score: Optional[float] = None
    entry: Optional[CacheEntry] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "hit": self.hit,
            "latency_ms": round(self.latency_ms, 3),
            "score": None if self.score is None else round(self.score, 4),
            "detail": self.detail,
            "entry_key": self.entry.key if self.entry else None,
        }


@dataclass
class GatewayResponse:
    """What the gateway returns to the caller."""

    text: str
    cached: bool
    source: str                       # exact | keyword | semantic | llm
    model: str
    request_id: str
    latency_ms: float = 0.0
    similarity: Optional[float] = None
    entry_key: Optional[str] = None
    fewshot_used: int = 0
    conversation_compacted: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI-style content field (string or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    url = (block.get("image_url") or {}).get("url", "")
                    # Only the identity of the image matters for cache keys.
                    parts.append(f"[image:{hash(url) & 0xffffffff:08x}]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def messages_to_text(messages: List[Message], include_system: bool = True) -> str:
    """Render a message list into the canonical text used for cache keys."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system" and not include_system:
            continue
        lines.append(f"{role}: {_content_to_text(msg.get('content', ''))}")
    return "\n".join(lines)
