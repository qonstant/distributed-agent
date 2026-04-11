from __future__ import annotations

import unittest

from rag_service.infrastructure.conversation_memory import RedisConversationMemory


class FakeRedisClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def lrange(self, key: str, start: int, stop: int):
        self.calls.append((key, start, stop))
        return self.payloads


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


if __name__ == "__main__":
    unittest.main()
