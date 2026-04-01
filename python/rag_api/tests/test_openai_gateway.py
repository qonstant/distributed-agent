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
    def __init__(self, output_text: str, usage=None) -> None:
        self.output_text = output_text
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text, usage=self.usage)


class FakeClient:
    def __init__(self, output_text: str, usage=None) -> None:
        self.responses = FakeResponses(output_text, usage=usage)


class OpenAIGatewayTests(unittest.TestCase):
    def _gateway_with_output(self, output_text: str, usage=None) -> tuple[OpenAIGateway, FakeClient]:
        settings = SimpleNamespace(
            openai_api_key="test-key",
            embed_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            class_model="gpt-4o-mini",
        )
        gateway = OpenAIGateway(settings)
        fake_client = FakeClient(output_text, usage=usage)
        gateway._client = fake_client
        return gateway, fake_client

    def test_classify_query_normalizes_language_to_supported_codes(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"intent":"GREETING","explain":"short greeting","language":"Russian"}',
            usage=SimpleNamespace(input_tokens=11, output_tokens=5, total_tokens=16),
        )

        classification, usage = gateway.classify_query("hello")

        self.assertEqual(classification.intent, "GREETING")
        self.assertEqual(classification.language, "ru")
        self.assertIsNotNone(usage)
        self.assertEqual(usage.input_tokens, 11)
        self.assertEqual(len(fake_client.responses.calls), 1)

    def test_classify_query_maps_unknown_language_to_other(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"intent":"OTHER","explain":"unsupported language","language":"German"}'
        )

        classification, usage = gateway.classify_query("hallo")

        self.assertEqual(classification.language, "other")
        self.assertIsNone(usage)

    def test_answer_factual_includes_preferred_name_in_prompt(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            "The code is ALPHA-123.",
            usage=SimpleNamespace(input_tokens=15, output_tokens=6, total_tokens=21),
        )

        answer, usage = gateway.answer_factual("What is the test code?", "en", preferred_name="Test User")

        self.assertEqual(answer, "The code is ALPHA-123.")
        self.assertIsNotNone(usage)
        self.assertEqual(len(fake_client.responses.calls), 1)
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Preferred user name: Test User", prompt)


if __name__ == "__main__":
    unittest.main()
