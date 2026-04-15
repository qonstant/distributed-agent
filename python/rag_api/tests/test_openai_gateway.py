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
from rag_service.domain.models import RetrievedHit


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
        self.assertEqual(classification.profile_action, "")
        self.assertEqual(classification.preferred_name, "")
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

    def test_classify_query_extracts_preferred_name_action(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"intent":"CHIT_CHAT","explain":"user sets a preferred name","language":"ru","profile_action":"set_preferred_name","preferred_name":"Heisenberg"}'
        )

        classification, _ = gateway.classify_query("зовут меня теперь Heisenberg")

        self.assertEqual(classification.intent, "CHIT_CHAT")
        self.assertEqual(classification.profile_action, "set_preferred_name")
        self.assertEqual(classification.preferred_name, "Heisenberg")

    def test_classify_query_prompt_guides_russian_vs_kazakh_cyrillic(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"intent":"FACTUAL_QUESTION","explain":"user asks what their name is in Russian","language":"ru","profile_action":"","preferred_name":""}'
        )

        classification, _ = gateway.classify_query("Как меня зовут?")

        self.assertEqual(classification.language, "ru")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn('Do NOT choose Kazakh just because the text is written in Cyrillic.', prompt)
        self.assertIn('Prefer "ru" for standard Russian wording such as "Как меня зовут?"', prompt)
        self.assertIn('Choose "kk" only when there are clear Kazakh signals', prompt)

    def test_classify_query_prompt_limits_scope_to_education_abroad(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"intent":"OTHER","explain":"tourist visa is outside education-abroad scope","language":"en","profile_action":"","preferred_name":""}'
        )

        classification, _ = gateway.classify_query("How do I get an Italian tourist visa?")

        self.assertEqual(classification.intent, "OTHER")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("only for education/study-abroad support", prompt)
        self.assertIn("tourist visas", prompt)
        self.assertIn("For out-of-scope requests, choose OTHER", prompt)

    def test_classify_attachment_follow_up_extracts_resend_action(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"attachment_action":"resend_last_attachment","explain":"user asks to resend the previous file"}'
        )

        action, _ = gateway.classify_attachment_follow_up(
            "Еще раз",
            history=[],
        )

        self.assertEqual(action, "")

        action, _ = gateway.classify_attachment_follow_up(
            "Еще раз",
            history=[SimpleNamespace(role="assistant", text="I sent the file", attachments=[], ts=1)],
        )

        self.assertEqual(action, "resend_last_attachment")

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

    def test_clarify_or_rewrite_query_returns_standalone_query_when_clear(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"What documents are needed for an Italian student visa?","clarifying_question":"","reason":"visa topic is clear"}',
            usage=SimpleNamespace(input_tokens=18, output_tokens=7, total_tokens=25),
        )

        clarity, usage = gateway.clarify_or_rewrite_query("visa docs", "en", "GUIDANCE")

        self.assertTrue(clarity.is_clear)
        self.assertTrue(clarity.is_retrieval_related)
        self.assertEqual(clarity.standalone_query, "What documents are needed for an Italian student visa?")
        self.assertEqual(clarity.clarifying_question, "")
        self.assertIsNotNone(usage)
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Country defaults to Italy", prompt)
        self.assertIn("Treat short/lazy queries as clear", prompt)
        self.assertIn("Tourist visas, travel visas, work visas", prompt)
        self.assertIn("For a generic visa query without enough context, ask whether the user means the student visa", prompt)
        self.assertIn("Short follow-up handling", prompt)
        self.assertIn("все вообще", prompt)
        self.assertIn('"is_retrieval_related": boolean', prompt)

    def test_clarify_or_rewrite_query_returns_question_when_unclear(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_retrieval_related":true,"is_clear":false,"standalone_query":"","clarifying_question":"Which topic do you mean: student visa, CV, scholarship, motivation letter, or recommendation letter?","reason":"topic missing"}'
        )

        clarity, _ = gateway.clarify_or_rewrite_query("what documents do I need?", "en", "GUIDANCE")

        self.assertFalse(clarity.is_clear)
        self.assertTrue(clarity.is_retrieval_related)
        self.assertEqual(
            clarity.clarifying_question,
            "Which topic do you mean: student visa, CV, scholarship, motivation letter, or recommendation letter?",
        )

    def test_clarify_or_rewrite_query_can_mark_other_as_not_retrieval_related(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_retrieval_related":false,"is_clear":false,"standalone_query":"","clarifying_question":"","reason":"not a document question"}'
        )

        clarity, _ = gateway.clarify_or_rewrite_query("random unrelated reply", "en", "OTHER")

        self.assertFalse(clarity.is_retrieval_related)
        self.assertFalse(clarity.is_clear)
        self.assertEqual(clarity.clarifying_question, "")

    def test_assess_retrieval_sufficiency_returns_sufficient_when_context_matches(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"is_sufficient":true,"clarifying_question":"","reason":"The excerpts directly cover the visa topic."}',
            usage=SimpleNamespace(input_tokens=21, output_tokens=6, total_tokens=27),
        )
        chunks = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={
                    "source_file": "italy/Visa_en.pdf",
                    "page": 1,
                    "text": "Italian student visa document requirements.",
                },
            )
        ]

        sufficiency, usage = gateway.assess_retrieval_sufficiency(
            "What documents are needed for an Italian student visa?",
            "en",
            "GUIDANCE",
            chunks,
        )

        self.assertTrue(sufficiency.is_sufficient)
        self.assertEqual(sufficiency.clarifying_question, "")
        self.assertIsNotNone(usage)
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("retrieval sufficiency judge", prompt)
        self.assertIn("Retrieved excerpts", prompt)
        self.assertIn("italy/Visa_en.pdf", prompt)
        self.assertIn("Do NOT ask the user to choose between documents and process", prompt)
        self.assertIn('"is_sufficient": boolean', prompt)

    def test_assess_retrieval_sufficiency_returns_clarifying_question_when_context_is_weak(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_sufficient":false,"clarifying_question":"Which education-abroad topic do you mean?","reason":"The excerpts are empty."}'
        )

        sufficiency, _ = gateway.assess_retrieval_sufficiency("how does it work?", "en", "GUIDANCE", [])

        self.assertFalse(sufficiency.is_sufficient)
        self.assertEqual(sufficiency.clarifying_question, "Which education-abroad topic do you mean?")


if __name__ == "__main__":
    unittest.main()
