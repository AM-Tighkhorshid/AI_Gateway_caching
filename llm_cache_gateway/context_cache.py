"""Sections 2.7 and 2.8 - retrieval memory and conversation compaction.

FewShotCache
    When no layer is confident enough to *replace* the model call, previously
    answered neighbours are still useful: they are injected as demonstrations.
    The model is still called, so the answer stays fresh, but it is grounded
    in prior accepted answers. This is the safe way to exploit similarity
    scores that sit below the reuse threshold.

ConversationCache
    Long chats are compacted into a rolling summary plus the last N turns.
    The summary is produced by the same free model and is itself cached, so
    it is generated once per growth step rather than on every request.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import CacheRequest, Message, _content_to_text


class FewShotCache:
    """Retrieves related cached Q/A pairs and formats them as demonstrations."""

    def __init__(self, semantic_cache, repository, config):
        self.semantic = semantic_cache
        self.repo = repository
        self.config = config

    def retrieve(self, request: CacheRequest,
                 vector: Optional[Sequence[float]] = None,
                 exclude: Optional[Sequence[str]] = None) -> List[Tuple[Any, float]]:
        cfg = self.config
        if not cfg.fewshot_enabled:
            return []

        excluded = set(exclude or ())
        neighbours = self.semantic.search(
            request.normalized_prompt,
            top_k=cfg.fewshot_k + len(excluded) + 3,
            vector=vector,
        )

        selected: List[Tuple[Any, float]] = []
        budget = cfg.fewshot_max_chars
        for entry_key, score in neighbours:
            if len(selected) >= cfg.fewshot_k:
                break
            if entry_key in excluded or score < cfg.fewshot_min_similarity:
                continue
            entry = self.repo.get(entry_key)
            if entry is None or not entry.response_text:
                continue
            # A rejected answer must not be promoted as a demonstration.
            if entry.rejected > entry.accepted:
                continue
            cost = len(entry.raw_prompt) + len(entry.response_text)
            if cost > budget:
                continue
            budget -= cost
            selected.append((entry, score))
        return selected

    @staticmethod
    def build_messages(messages: List[Message],
                       examples: Sequence[Tuple[Any, float]]) -> List[Message]:
        """Insert demonstrations after the system prompt, before the live turn."""
        if not examples:
            return messages

        demo_block = ["Here are previously answered, related questions. "
                      "Use them for style and consistency; answer the new "
                      "question on its own merits."]
        for idx, (entry, score) in enumerate(examples, start=1):
            demo_block.append(
                f"\n### Example {idx} (similarity {score:.2f})\n"
                f"Q: {entry.raw_prompt.strip()}\n"
                f"A: {entry.response_text.strip()}"
            )
        demo_message: Message = {"role": "system",
                                 "content": "\n".join(demo_block)}

        out: List[Message] = []
        inserted = False
        for msg in messages:
            if msg.get("role") == "system":
                out.append(msg)
                continue
            if not inserted:
                out.append(demo_message)
                inserted = True
            out.append(msg)
        if not inserted:
            out.append(demo_message)
        return out


# --------------------------------------------------------------------------- #
class ConversationCache:
    """Rolling summary + recent turns, so prompts stop growing without bound."""

    def __init__(self, store, config, provider=None):
        self.store = store
        self.config = config
        self.provider = provider
        self.namespace = config.namespace

    def _key(self, conversation_id: str) -> str:
        return f"{self.namespace}:conv:{conversation_id}"

    # ------------------------------------------------------------------ #
    def get_state(self, conversation_id: str) -> Dict[str, Any]:
        return self.store.get(self._key(conversation_id)) or {
            "summary": "", "summarized_upto": 0, "updated_at": 0.0,
            "turns": 0, "digest": "",
        }

    def save_state(self, conversation_id: str, state: Dict[str, Any]) -> None:
        state["updated_at"] = time.time()
        self.store.set(self._key(conversation_id), state,
                       ttl=self.config.conversation_ttl)

    def clear(self, conversation_id: str) -> None:
        self.store.delete(self._key(conversation_id))

    # ------------------------------------------------------------------ #
    def compact(self, messages: List[Message],
                conversation_id: Optional[str],
                model: Optional[str] = None) -> Tuple[List[Message], bool]:
        """Return (possibly shortened messages, whether compaction happened)."""
        cfg = self.config
        if not cfg.conversation_enabled or not conversation_id:
            return messages, False

        total_chars = sum(len(_content_to_text(m.get("content", "")))
                          for m in messages)
        if total_chars < cfg.conversation_summary_trigger_chars:
            return messages, False

        system_msgs = [m for m in messages if m.get("role") == "system"]
        dialogue = [m for m in messages if m.get("role") != "system"]
        keep = cfg.conversation_keep_recent_turns
        if len(dialogue) <= keep:
            return messages, False

        older, recent = dialogue[:-keep], dialogue[-keep:]
        state = self.get_state(conversation_id)
        digest = hashlib.sha256(
            "".join(_content_to_text(m.get("content", "")) for m in older)
            .encode("utf-8")
        ).hexdigest()[:16]

        # Reuse the cached summary unless the older segment actually changed.
        if state.get("digest") == digest and state.get("summary"):
            summary = state["summary"]
        else:
            summary = self._summarize(older, model)
            state.update({"summary": summary, "digest": digest,
                          "turns": len(older)})
            self.save_state(conversation_id, state)

        summary_msg: Message = {
            "role": "system",
            "content": f"Summary of the earlier conversation:\n{summary}",
        }
        return system_msgs + [summary_msg] + recent, True

    # ------------------------------------------------------------------ #
    def _summarize(self, messages: List[Message],
                   model: Optional[str]) -> str:
        transcript = "\n".join(
            f"{m.get('role')}: {_content_to_text(m.get('content', ''))}"
            for m in messages
        )
        limit = self.config.conversation_summary_max_chars

        if self.provider is None:
            return self._truncate_fallback(transcript, limit)

        prompt = [
            {"role": "system",
             "content": "You compress conversations for reuse as context. "
                        "Keep facts, decisions, names, numbers, and open "
                        "questions. Drop pleasantries. Be terse."},
            {"role": "user",
             "content": f"Summarize this conversation in under {limit} "
                        f"characters:\n\n{transcript}"},
        ]
        try:
            result = self.provider.chat(prompt, model=model, temperature=0.0,
                                        max_tokens=max(256, limit // 3))
            return (result.get("text") or "").strip()[:limit]
        except Exception:  # noqa: BLE001 - never fail a request over a summary
            return self._truncate_fallback(transcript, limit)

    @staticmethod
    def _truncate_fallback(transcript: str, limit: int) -> str:
        """Deterministic fallback: keep the head and tail of the transcript."""
        if len(transcript) <= limit:
            return transcript
        half = limit // 2 - 20
        return f"{transcript[:half]}\n...[trimmed]...\n{transcript[-half:]}"
