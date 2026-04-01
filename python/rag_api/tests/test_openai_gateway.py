from __future__ import annotations

import sys
import unittest
from types import ModuleType, SimpleNamespace

fake_openai_module = ModuleType("openai")


class FakeOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        pass


fake_openai_module.OpenAI = FakeOpenAI
sys.modules.setdefault("openai", fake_openai_module)

from rag_service.infrastructure.openai_gateway import OpenAIGateway


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text='{"intent":"GREETING","explain":"short greeting"}')


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


class OpenAIGatewayTests(unittest.TestCase):
    def test_classify_query_uses_local_language_detection(self) -> None:
        settings = SimpleNamespace(
            openai_api_key="test-key",
            embed_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            class_model="gpt-4o-mini",
        )
        gateway = OpenAIGateway(settings)
        fake_client = FakeClient()
        gateway._client = fake_client

        classification = gateway.classify_query("Сәлем")

        self.assertEqual(classification.intent, "GREETING")
        self.assertEqual(classification.language, "kk")
        self.assertEqual(classification.model, "gpt-4o-mini")
        self.assertEqual(len(fake_client.responses.calls), 1)
        self.assertNotIn('"language"', fake_client.responses.calls[0]["input"])


if __name__ == "__main__":
    unittest.main()
