from __future__ import annotations

import json
from typing import List

from rag_service.domain.models import ConversationAttachment, ConversationMessage


class RedisConversationMemory:
    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "chat:conv:",
        max_items: int = 8,
        pending_ttl_seconds: int = 7200,
    ) -> None:
        from redis import Redis

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix
        self._max_items = max(1, int(max_items))
        self._pending_ttl_seconds = max(60, int(pending_ttl_seconds))

    def load_messages(self, conversation_id: str) -> List[ConversationMessage]:
        if not conversation_id:
            return []

        try:
            payloads = self._client.lrange(
                f"{self._key_prefix}{conversation_id}:messages",
                0,
                self._max_items - 1,
            )
        except Exception as exc:
            print(f"[memory] failed to load history for conversation_id={conversation_id}: {exc}")
            return []

        messages: List[ConversationMessage] = []
        for raw in reversed(payloads):
            try:
                parsed = json.loads(raw)
            except Exception:
                continue

            role = str(parsed.get("role") or "").strip()
            text = str(parsed.get("text") or "")
            if role not in {"user", "assistant"}:
                continue

            try:
                ts = int(parsed.get("ts") or 0)
            except Exception:
                ts = 0

            attachments = []
            for item in parsed.get("attachments") or []:
                if not isinstance(item, dict):
                    continue

                source = str(item.get("source") or "").strip()
                name = str(item.get("name") or "").strip()
                kind = str(item.get("kind") or "").strip()
                if not name and not source:
                    continue

                attachments.append(
                    ConversationAttachment(
                        name=name,
                        kind=kind or "document",
                        source=source,
                    )
                )

            messages.append(
                ConversationMessage(
                    role=role,
                    text=text,
                    ts=ts,
                    attachments=attachments,
                )
            )

        return messages

    def load_pending_attachment(self, conversation_id: str) -> str:
        if not conversation_id:
            return ""
        try:
            return str(self._client.get(self._pending_key(conversation_id)) or "").strip()
        except Exception as exc:
            print(f"[memory] failed to load pending attachment for conversation_id={conversation_id}: {exc}")
            return ""

    def remember_pending_attachment(self, conversation_id: str, source: str) -> bool:
        normalized_source = str(source or "").strip()
        if not conversation_id or not normalized_source:
            return False
        try:
            self._client.setex(
                self._pending_key(conversation_id),
                self._pending_ttl_seconds,
                normalized_source,
            )
            return True
        except Exception as exc:
            print(f"[memory] failed to remember pending attachment for conversation_id={conversation_id}: {exc}")
            return False

    def clear_pending_attachment(self, conversation_id: str) -> None:
        if not conversation_id:
            return
        try:
            self._client.delete(self._pending_key(conversation_id))
        except Exception as exc:
            print(f"[memory] failed to clear pending attachment for conversation_id={conversation_id}: {exc}")

    def _pending_key(self, conversation_id: str) -> str:
        return f"{self._key_prefix}{conversation_id}:pending_attachment"
