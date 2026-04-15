from __future__ import annotations

import unittest

from rag_service.infrastructure.conversation_memory import RedisConversationMemory


class FakeRedisClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []
        self.values = {}

    def lrange(self, key: str, start: int, stop: int):
        self.calls.append((key, start, stop))
        return self.payloads

    def get(self, key: str):
        self.calls.append(("get", key))
        return self.values.get(key, "")

    def setex(self, key: str, ttl: int, value: str):
        self.calls.append(("setex", key, ttl, value))
        self.values[key] = value

    def delete(self, key: str):
        self.calls.append(("delete", key))
        self.values.pop(key, None)


class RedisConversationMemoryTests(unittest.TestCase):
    def test_load_messages_parses_attachment_metadata(self) -> None:
        memory = RedisConversationMemory.__new__(RedisConversationMemory)
        memory._client = FakeRedisClient(
            [
                '{"role":"assistant","text":"Here is the file.","ts":2,"attachments":[{"name":"sample.pdf","kind":"document","source":"docs/sample.pdf"}]}',
                '{"role":"user","text":"Send the sample again","ts":1}',
            ]
        )
        memory._key_prefix = "chat:conv:"
        memory._max_items = 8

        history = memory.load_messages("conv-1")

        self.assertEqual(memory._client.calls, [("chat:conv:conv-1:messages", 0, 7)])
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(len(history[1].attachments), 1)
        self.assertEqual(history[1].attachments[0].name, "sample.pdf")
        self.assertEqual(history[1].attachments[0].kind, "document")
        self.assertEqual(history[1].attachments[0].source, "docs/sample.pdf")

    def test_pending_attachment_round_trip(self) -> None:
        memory = RedisConversationMemory.__new__(RedisConversationMemory)
        memory._client = FakeRedisClient([])
        memory._key_prefix = "chat:conv:"
        memory._max_items = 8
        memory._pending_ttl_seconds = 7200

        saved = memory.remember_pending_attachment("conv-1", "italy/Visa_en.pdf")
        pending = memory.load_pending_attachment("conv-1")
        memory.clear_pending_attachment("conv-1")
        cleared = memory.load_pending_attachment("conv-1")

        self.assertTrue(saved)
        self.assertEqual(pending, "italy/Visa_en.pdf")
        self.assertEqual(cleared, "")
        self.assertEqual(
            memory._client.calls,
            [
                ("setex", "chat:conv:conv-1:pending_attachment", 7200, "italy/Visa_en.pdf"),
                ("get", "chat:conv:conv-1:pending_attachment"),
                ("delete", "chat:conv:conv-1:pending_attachment"),
                ("get", "chat:conv:conv-1:pending_attachment"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
