from __future__ import annotations

import json
from typing import List

from rag_service.domain.models import ConversationMessage


class RedisConversationMemory:
    def __init__(self, redis_url: str, key_prefix: str = "chat:conv:", max_items: int = 8) -> None:
        from redis import Redis

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._key_prefix = key_prefix
        self._max_items = max(1, int(max_items))

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

            messages.append(ConversationMessage(role=role, text=text, ts=ts))

        return messages
