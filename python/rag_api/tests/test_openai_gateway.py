from __future__ import annotations

import sys
import unittest
from typing import Any
from types import ModuleType, SimpleNamespace

fake_openai_module = ModuleType("openai")


class FakeOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        pass


fake_openai_module.OpenAI = FakeOpenAI
sys.modules.setdefault("openai", fake_openai_module)

from rag_service.infrastructure.openai_gateway import OpenAIGateway
from rag_service.domain.models import ConversationMessage, RetrievedHit


class FakeResponses:
    def __init__(self, output_text: Any, usage=None) -> None:
        self.output_texts = output_text if isinstance(output_text, list) else [output_text]
        self.usage = usage
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.output_texts) - 1)
        return SimpleNamespace(output_text=self.output_texts[index], usage=self.usage)


class FakeClient:
    def __init__(self, output_text: Any, usage=None) -> None:
        self.responses = FakeResponses(output_text, usage=usage)


class OpenAIGatewayTests(unittest.TestCase):
    def _gateway_with_output(self, output_text: Any, usage=None) -> tuple[OpenAIGateway, FakeClient]:
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
            '{"intent":"CHITCHAT","confidence":0.93,"needs_rag":false,"route":"SMALL_MODEL_RESPONSE","reason":"user sets a preferred name","rewritten_query":"","language":"ru","profile_action":"set_preferred_name","preferred_name":"Heisenberg"}'
        )

        classification, _ = gateway.classify_query("зовут меня теперь Heisenberg")

        self.assertEqual(classification.intent, "CHITCHAT")
        self.assertEqual(classification.profile_action, "set_preferred_name")
        self.assertEqual(classification.preferred_name, "Heisenberg")
        self.assertEqual(classification.route, "SMALL_MODEL_RESPONSE")

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

    def test_guard_query_allows_in_scope_study_abroad_queries(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"allowed":true,"needs_context":false,"violation":"","reason":"DSU defaults to the Italian student scholarship topic.","language":"English"}',
            usage=SimpleNamespace(input_tokens=12, output_tokens=6, total_tokens=18),
        )

        guardrail, usage = gateway.guard_query("How to get dsu")

        self.assertTrue(guardrail.allowed)
        self.assertEqual(guardrail.violation, "")
        self.assertEqual(guardrail.language, "en")
        self.assertIsNotNone(usage)
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("You are a guardrail", prompt)
        self.assertIn("called in two passes", prompt)
        self.assertIn("Recent conversation context: not provided in this pass.", prompt)
        self.assertIn("set needs_context=true", prompt)
        self.assertIn("DSU scholarship/student financial aid in Italy", prompt)
        self.assertIn("If the user mentions DSU without another explicit meaning", prompt)
        self.assertIn("Short confirmations or refusals are allowed", prompt)
        self.assertIn("assistant just asked an in-scope clarification question", prompt)
        self.assertIn("MUST make a final safety/scope decision", prompt)
        self.assertIn("immediate previous meaningful topic", prompt)
        self.assertIn("Do not treat short replies as unsafe merely because", prompt)
        self.assertIn("Profanity, impatience, or rude wording is not unsafe by itself", prompt)
        self.assertIn("Fucking yes, send me already", prompt)
        self.assertIn("What do you mean?", prompt)
        self.assertIn("Fuck u mean", prompt)
        self.assertIn("Hey", prompt)
        self.assertIn("What about student", prompt)
        self.assertIn("So am I", prompt)
        self.assertIn("Docs required", prompt)
        self.assertIn("A previous assistant refusal is not evidence of unsafe user intent", prompt)
        self.assertIn('"visa docs"', prompt)
        self.assertIn("How do I get residence permit", prompt)
        self.assertIn("residence permit documents", prompt)
        self.assertIn("Do not require the words Italy or student", prompt)
        self.assertIn("Как получить ВНЖ", prompt)
        self.assertIn("тұруға рұқсатты қалай алуға болады", prompt)
        self.assertIn("What is the capital of France?", prompt)
        self.assertIn("Италияда оқып жүріп саяхаттай аламын ба?", prompt)
        self.assertIn("How do I move to Italy permanently?", prompt)
        self.assertIn("previous in-scope visa-photo file offer", prompt)
        self.assertIn("previous explicit document-fraud request", prompt)
        self.assertNotIn("Recent conversation context (oldest to newest):", prompt)
        self.assertIn("Block as out_of_scope", prompt)
        self.assertIn("Block as unsafe", prompt)
        self.assertIn("fake a bank statement", prompt)

    def test_guard_query_can_request_context_for_short_confirmation(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"allowed":false,"needs_context":true,"violation":"","reason":"The latest message is only a confirmation and needs recent conversation context.","language":"ru"}'
        )

        guardrail, _ = gateway.guard_query("Да да да")

        self.assertFalse(guardrail.allowed)
        self.assertTrue(guardrail.needs_context)
        self.assertEqual(guardrail.violation, "")
        self.assertEqual(guardrail.language, "ru")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Recent conversation context: not provided in this pass.", prompt)
        self.assertIn("Latest user input: \"Да да да\"", prompt)

    def test_guard_query_uses_history_for_short_confirmation(self) -> None:
        history = [
            ConversationMessage(role="user", text="Как получить внж", ts=1),
            ConversationMessage(
                role="assistant",
                text="Вы имеете в виду разрешение на проживание для студентов или что-то другое?",
                ts=2,
            ),
        ]
        gateway, fake_client = self._gateway_with_output(
            '{"allowed":true,"needs_context":false,"violation":"","reason":"The user confirms the previous in-scope student residence permit clarification.","language":"ru"}'
        )

        guardrail, _ = gateway.guard_query("Да да да", history=history)

        self.assertTrue(guardrail.allowed)
        self.assertFalse(guardrail.needs_context)
        self.assertEqual(guardrail.violation, "")
        self.assertEqual(guardrail.language, "ru")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Recent conversation context (oldest to newest):", prompt)
        self.assertIn("user: Как получить внж", prompt)
        self.assertIn("assistant: Вы имеете в виду разрешение на проживание для студентов", prompt)
        self.assertIn("Latest user input: \"Да да да\"", prompt)

    def test_guard_query_repairs_needs_context_when_history_was_provided(self) -> None:
        history = [
            ConversationMessage(role="user", text="How to get dsu", ts=1),
            ConversationMessage(role="assistant", text="DSU is the Italian student scholarship.", ts=2),
        ]
        gateway, fake_client = self._gateway_with_output(
            [
                '{"allowed":false,"needs_context":true,"violation":"","reason":"The latest message is too short and needs recent conversation context.","language":"en"}',
                '{"allowed":true,"needs_context":false,"violation":"","reason":"The user asks how to continue with the in-scope DSU topic.","language":"en"}',
            ]
        )

        guardrail, _ = gateway.guard_query("How", history=history)

        self.assertTrue(guardrail.allowed)
        self.assertFalse(guardrail.needs_context)
        self.assertEqual(guardrail.violation, "")
        self.assertEqual(len(fake_client.responses.calls), 2)
        repair_prompt = fake_client.responses.calls[1]["input"]
        self.assertIn("context was already provided", repair_prompt)
        self.assertIn("Set needs_context=false", repair_prompt)
        self.assertIn("Previous guardrail JSON", repair_prompt)

    def test_guard_query_blocks_out_of_scope_or_unsafe_queries(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"allowed":false,"needs_context":false,"violation":"unsafe","reason":"The user asks for help falsifying documents.","language":"en"}'
        )

        guardrail, _ = gateway.guard_query("How can I fake a bank statement for visa?")

        self.assertFalse(guardrail.allowed)
        self.assertEqual(guardrail.violation, "unsafe")
        self.assertEqual(guardrail.language, "en")

    def test_classify_query_prompt_runs_after_guardrail_scope_check(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"intent":"PROCEDURE","confidence":0.88,"needs_rag":true,"route":"RAG_SEARCH","reason":"user asks for university admission procedure","rewritten_query":"Italian university admission application steps","language":"en","profile_action":"","preferred_name":""}'
        )

        classification, _ = gateway.classify_query("How to apply to uni")

        self.assertEqual(classification.intent, "PROCEDURE")
        self.assertTrue(classification.needs_rag)
        self.assertEqual(classification.route, "RAG_SEARCH")
        self.assertEqual(classification.rewritten_query, "Italian university admission application steps")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("compact routing classifier", prompt)
        self.assertIn('"intent": one of ["GREETING","CHITCHAT","FACTUAL_QUESTION","PROCEDURE","COMPARISON","OUT_OF_DOMAIN"]', prompt)
        self.assertIn('"confidence": number from 0.0 to 1.0', prompt)
        self.assertIn('"route": one of ["CANNED_RESPONSE","SMALL_MODEL_RESPONSE","RAG_SEARCH","CLARIFY","REFUSE_OR_REDIRECT"]', prompt)
        self.assertIn("short replies like \"yes\", \"how\", \"how to apply\"", prompt)
        self.assertIn("Bare residence-permit queries in English, Russian, or Kazakh default", prompt)
        self.assertIn("How do I get residence permit", prompt)
        self.assertIn("How to apply for residence permit", prompt)
        self.assertIn("residence permit documents", prompt)
        self.assertIn("тұруға рұқсатты қалай алуға болады", prompt)
        self.assertIn("Profanity or impatience does not change the route", prompt)
        self.assertIn("greeting-only or social-only message", prompt)
        self.assertIn("Fuck u mean", prompt)
        self.assertIn("Good", prompt)
        self.assertIn("General", prompt)
        self.assertIn("What about student", prompt)
        self.assertIn("So am I", prompt)
        self.assertIn("Docs required", prompt)
        self.assertIn("do not ask whether they mean process or documents", prompt)
        self.assertIn("Compare Italy and Germany", prompt)
        self.assertNotIn("For out-of-scope requests, choose OTHER", prompt)

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

    def test_classify_attachment_follow_up_extracts_pending_file_action(self) -> None:
        gateway, fake_client = self._gateway_with_output(
            '{"attachment_action":"send_pending_attachment","explain":"user accepts the offered file"}'
        )

        action, _ = gateway.classify_attachment_follow_up(
            "yes please",
            history=[],
            pending_file="italy/Visa_en.pdf",
        )

        self.assertEqual(action, "send_pending_attachment")
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Pending offered attachment source", prompt)
        self.assertIn("send_pending_attachment", prompt)

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
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"What documents are needed for an Italian student visa?","clarifying_question":"","target_language":"","reason":"visa topic is clear"}',
            usage=SimpleNamespace(input_tokens=18, output_tokens=7, total_tokens=25),
        )

        clarity, usage = gateway.clarify_or_rewrite_query("visa docs", "en", "GUIDANCE")

        self.assertTrue(clarity.is_clear)
        self.assertTrue(clarity.is_retrieval_related)
        self.assertEqual(clarity.standalone_query, "What documents are needed for an Italian student visa?")
        self.assertEqual(clarity.clarifying_question, "")
        self.assertEqual(clarity.target_language, "")
        self.assertIsNotNone(usage)
        prompt = fake_client.responses.calls[0]["input"]
        self.assertIn("Country defaults to Italy", prompt)
        self.assertIn("Italian university admission/application guidance", prompt)
        self.assertIn("DSU defaults to the Italian student scholarship/financial-aid topic", prompt)
        self.assertIn("Treat short/lazy queries as clear", prompt)
        self.assertIn("Default no-clarification rules", prompt)
        self.assertIn('"visa docs", "visa documents", and "visa requirements"', prompt)
        self.assertIn('"How do I get residence permit"', prompt)
        self.assertIn('"How to apply for residence permit', prompt)
        self.assertIn('"residence permit documents"', prompt)
        self.assertIn("тұруға рұқсатты қалай алуға болады", prompt)
        self.assertIn("Docs required", prompt)
        self.assertIn("Fucking yes, send me already", prompt)
        self.assertIn("NEVER ask whether they want process or documents", prompt)
        self.assertIn("If two comparison targets are present but criteria are missing", prompt)
        self.assertIn('"how to apply to uni"', prompt)
        self.assertIn('"how to get dsu"', prompt)
        self.assertIn("Do NOT ask sub-aspect clarifications inside an already identified topic", prompt)
        self.assertIn("what photo format is needed for the visa", prompt)
        self.assertIn("Tourist visas, travel visas unrelated to study, work visas", prompt)
        self.assertIn("For a generic visa query without enough context, ask whether the user means the student visa", prompt)
        self.assertIn("For a generic residence permit query, assume student residence permit", prompt)
        self.assertIn("can I travel while studying", prompt)
        self.assertIn("Short follow-up handling", prompt)
        self.assertIn("Clarification-loop prevention", prompt)
        self.assertIn("process-vs-requirements", prompt)
        self.assertIn('If the latest user asks a short continuation like "how"', prompt)
        self.assertIn("Language-switch follow-up handling", prompt)
        self.assertIn('"target_language": string', prompt)
        self.assertIn("все вообще", prompt)
        self.assertIn('"is_retrieval_related": boolean', prompt)

    def test_clarify_or_rewrite_query_returns_target_language_for_language_switch(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_retrieval_related":true,"is_clear":true,"standalone_query":"How to apply for an Italian student visa?","clarifying_question":"","target_language":"English","reason":"user asks for the previous answer in English"}'
        )

        clarity, _ = gateway.clarify_or_rewrite_query("А можно на английском?", "ru", "OTHER")

        self.assertTrue(clarity.is_clear)
        self.assertTrue(clarity.is_retrieval_related)
        self.assertEqual(clarity.standalone_query, "How to apply for an Italian student visa?")
        self.assertEqual(clarity.target_language, "en")

    def test_clarify_or_rewrite_query_returns_question_when_unclear(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_retrieval_related":true,"is_clear":false,"standalone_query":"","clarifying_question":"Which topic do you mean: university admission, student visa, student residence permit, DSU scholarship, CV, motivation letter, or recommendation letter?","reason":"topic missing"}'
        )

        clarity, _ = gateway.clarify_or_rewrite_query("what documents do I need?", "en", "GUIDANCE")

        self.assertFalse(clarity.is_clear)
        self.assertTrue(clarity.is_retrieval_related)
        self.assertEqual(
            clarity.clarifying_question,
            "Which topic do you mean: university admission, student visa, student residence permit, DSU scholarship, CV, motivation letter, or recommendation letter?",
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
        self.assertIn("university admission/application guidance", prompt)
        self.assertIn("DSU is a supported Italian student scholarship/financial-aid topic by default", prompt)
        self.assertIn("Treat short topic queries", prompt)
        self.assertIn("Do NOT ask the user to choose between documents and process", prompt)
        self.assertIn("Do NOT ask the user to choose sub-aspects inside an already identified document topic", prompt)
        self.assertIn("already passed a clarity gate", prompt)
        self.assertIn("Do NOT ask process-vs-documents", prompt)
        self.assertIn("not a question", prompt)
        self.assertIn("enough reliable information was not found in the available documents", prompt)
        self.assertIn("NEVER ask whether the user means student residence permit specifically or general residence permit information", prompt)
        self.assertIn("enough reliable student residence permit information was not found", prompt)
        self.assertIn("тұруға рұқсат", prompt)
        self.assertIn("visa photo format", prompt)
        self.assertIn('"is_sufficient": boolean', prompt)

    def test_assess_retrieval_sufficiency_fails_closed_on_gateway_error(self) -> None:
        gateway, fake_client = self._gateway_with_output("{}")

        def raise_connection_error(**_kwargs):
            raise RuntimeError("connection boom")

        fake_client.responses.create = raise_connection_error

        sufficiency, usage = gateway.assess_retrieval_sufficiency(
            "How do I get residence permit",
            "en",
            "PROCEDURE",
            [],
        )

        self.assertFalse(sufficiency.is_sufficient)
        self.assertEqual(sufficiency.clarifying_question, "")
        self.assertIn("sufficiency check failed", sufficiency.reason)
        self.assertIsNone(usage)

    def test_assess_retrieval_sufficiency_returns_clarifying_question_when_context_is_weak(self) -> None:
        gateway, _ = self._gateway_with_output(
            '{"is_sufficient":false,"clarifying_question":"Which education-abroad topic do you mean?","reason":"The excerpts are empty."}'
        )

        sufficiency, _ = gateway.assess_retrieval_sufficiency("how does it work?", "en", "GUIDANCE", [])

        self.assertFalse(sufficiency.is_sufficient)
        self.assertEqual(sufficiency.clarifying_question, "Which education-abroad topic do you mean?")


if __name__ == "__main__":
    unittest.main()
