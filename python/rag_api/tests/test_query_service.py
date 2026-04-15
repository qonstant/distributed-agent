from __future__ import annotations

import unittest

import numpy as np

from rag_service.application.query_service import QueryService
from rag_service.application.usage_estimation import (
    usage_event_from_model_usage,
)
from rag_service.domain.models import (
    Classification,
    ConversationAttachment,
    ConversationMessage,
    ModelUsage,
    QueryResult,
    RetrievedHit,
)
from rag_service.infrastructure.prompts import prepare_document_request_prompt


class FakeGateway:
    def __init__(self, classification: Classification, attachment_action: str = "") -> None:
        self.classification = classification
        self.attachment_action = attachment_action
        self.classify_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.attachment_follow_up_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.answer_factual_calls: list[tuple[str, str, str, list[ConversationMessage]]] = []
        self.rewrite_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.embedded_queries: list[str] = []
        self.generated_prompts: list[str] = []
        self.classification_usage = ModelUsage(model="gpt-4o-mini", input_tokens=12, output_tokens=6, total_tokens=18)
        self.attachment_action_usage = ModelUsage(model="gpt-4o-mini", input_tokens=8, output_tokens=4, total_tokens=12)
        self.greeting_usage = ModelUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=3, total_tokens=13)
        self.factual_usage = ModelUsage(model="gpt-4o-mini", input_tokens=16, output_tokens=7, total_tokens=23)
        self.rewrite_usage = ModelUsage(model="gpt-4o-mini", input_tokens=20, output_tokens=4, total_tokens=24)
        self.embedding_usage = ModelUsage(model="text-embedding-3-small", input_tokens=9, output_tokens=0, total_tokens=9)
        self.json_usage = ModelUsage(model="gpt-4o-mini", input_tokens=30, output_tokens=8, total_tokens=38)

    def classify_query(self, query: str, history=None):
        self.classify_calls.append((query, history or []))
        return self.classification, self.classification_usage

    def classify_attachment_follow_up(self, query: str, history=None):
        self.attachment_follow_up_calls.append((query, history or []))
        return self.attachment_action, self.attachment_action_usage

    def generate_greeting_reply(self, user_text: str, language_hint: str, preferred_name: str = "", history=None):
        return "hello", self.greeting_usage

    def answer_factual(self, query: str, language_hint: str, preferred_name: str = "", history=None):
        self.answer_factual_calls.append((query, language_hint, preferred_name, history or []))
        return "The test code is ALPHA-123.", self.factual_usage

    def rewrite_query_with_history(self, query: str, history=None):
        self.rewrite_calls.append((query, history or []))
        return "sample onboarding guide pdf", self.rewrite_usage

    def embed_text(self, text: str):
        self.embedded_queries.append(text)
        return np.array([1.0], dtype=np.float32), self.embedding_usage

    def generate_json_response(self, prompt: str, max_tokens: int = 512):
        self.generated_prompts.append(prompt)
        return {"answer": "Use this sample.", "file": "docs/test-guide.pdf"}, self.json_usage


class FakeStore:
    def __init__(self, results=None) -> None:
        self.results = results or []
        self.search_calls = []

    def search(self, query_embedding, k: int = 64, **filters):
        self.search_calls.append({"query_embedding": query_embedding, "k": k, **filters})
        return self.results[:k]


class FakeConversationMemory:
    def __init__(self, messages):
        self.messages = messages
        self.requested_ids: list[str] = []

    def load_messages(self, conversation_id: str):
        self.requested_ids.append(conversation_id)
        return self.messages


class QueryServiceTests(unittest.TestCase):
    def test_factual_query_uses_history_loaded_from_conversation_id(self) -> None:
        history = [
            ConversationMessage(role="user", text="The test code is ALPHA-123", ts=1),
            ConversationMessage(role="assistant", text="Acknowledged. I will remember ALPHA-123 for this test.", ts=2),
        ]
        gateway = FakeGateway(Classification(intent="FACTUAL_QUESTION", explain="needs memory", language="en"))
        memory = FakeConversationMemory(history)
        service = QueryService(gateway, FakeStore(), conversation_memory=memory)

        result = service.handle_query("What is the test code?", conversation_id="conv-1", preferred_name="Test User")

        self.assertEqual(
            result,
            QueryResult(
                answer="The test code is ALPHA-123.",
                file=None,
                classification=Classification(intent="FACTUAL_QUESTION", explain="needs memory", language="en"),
                usage_events=[
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("chat_completion", gateway.factual_usage),
                ],
            ),
        )
        self.assertEqual(memory.requested_ids, ["conv-1"])
        self.assertEqual(gateway.classify_calls, [("What is the test code?", history)])
        self.assertEqual(gateway.attachment_follow_up_calls, [])
        self.assertEqual(gateway.answer_factual_calls, [("What is the test code?", "en", "Test User", history)])

    def test_document_request_rewrites_retrieval_query_and_includes_history_in_prompt(self) -> None:
        history = [
            ConversationMessage(role="user", text="Send me the sample onboarding guide", ts=1),
            ConversationMessage(
                role="assistant",
                text="I can help with that.",
                ts=2,
                attachments=[ConversationAttachment(name="test-guide.pdf", kind="document", source="docs/test-guide.pdf")],
            ),
        ]
        gateway = FakeGateway(Classification(intent="DOCUMENT_REQUEST", explain="follow-up request", language="en"))
        memory = FakeConversationMemory(history)
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "docs/test-guide.pdf", "page": 1, "text": "sample document excerpt"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=memory)

        result = service.handle_query("Which sample guide was that?", conversation_id="conv-1", preferred_name="Test User")

        self.assertEqual(
            result,
            QueryResult(
                answer=(
                    "Use this sample.\n\n"
                    "I already sent this file earlier in the conversation: test-guide.pdf. "
                    "You can find it above in the chat. If you want, I can resend it."
                ),
                file=None,
                classification=Classification(intent="DOCUMENT_REQUEST", explain="follow-up request", language="en"),
                usage_events=[
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                    usage_event_from_model_usage("other", gateway.rewrite_usage),
                    usage_event_from_model_usage("embedding", gateway.embedding_usage),
                    usage_event_from_model_usage("chat_completion", gateway.json_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Which sample guide was that?", history)])
        self.assertEqual(gateway.rewrite_calls, [("Which sample guide was that?", history)])
        self.assertEqual(gateway.embedded_queries, ["sample onboarding guide pdf"])
        self.assertEqual(store.search_calls[0]["language"], "en")
        self.assertEqual(store.search_calls[0]["query_text"], "sample onboarding guide pdf")
        self.assertEqual(len(gateway.generated_prompts), 1)
        self.assertIn("Preferred user name: Test User", gateway.generated_prompts[0])
        self.assertIn("Recent conversation context", gateway.generated_prompts[0])
        self.assertIn("user: Send me the sample onboarding guide", gateway.generated_prompts[0])

    def test_document_request_resends_same_file_when_user_explicitly_asks(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="I already sent the sample.",
                ts=2,
                attachments=[ConversationAttachment(name="test-guide.pdf", kind="document", source="docs/test-guide.pdf")],
            )
        ]
        gateway = FakeGateway(
            Classification(intent="DOCUMENT_REQUEST", explain="resend request", language="en"),
            attachment_action="resend_last_attachment",
        )
        memory = FakeConversationMemory(history)
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "docs/test-guide.pdf", "page": 1, "text": "sample document excerpt"},
            )
        ]
        service = QueryService(gateway, FakeStore(results), conversation_memory=memory)

        result = service.handle_query("Please resend the file", conversation_id="conv-1", preferred_name="Test User")

        self.assertEqual(
            result,
            QueryResult(
                answer="Here is the file again: test-guide.pdf.",
                file="docs/test-guide.pdf",
                classification=Classification(intent="DOCUMENT_REQUEST", explain="resend request", language="en"),
                usage_events=[
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Please resend the file", history)])
        self.assertEqual(gateway.rewrite_calls, [])
        self.assertEqual(gateway.generated_prompts, [])

    def test_short_resend_follow_up_resends_latest_attachment(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="I already sent the sample.",
                ts=2,
                attachments=[ConversationAttachment(name="CV_ru.pdf", kind="document", source="italy/CV_ru.pdf")],
            )
        ]
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="short follow-up", language="ru"),
            attachment_action="resend_last_attachment",
        )
        memory = FakeConversationMemory(history)
        service = QueryService(gateway, FakeStore(), conversation_memory=memory)

        result = service.handle_query("Еще раз", conversation_id="conv-1", preferred_name="Test User")

        self.assertEqual(
            result,
            QueryResult(
                answer="Вот файл еще раз: CV_ru.pdf.",
                file="italy/CV_ru.pdf",
                classification=Classification(intent="CHIT_CHAT", explain="short follow-up", language="ru"),
                usage_events=[
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Еще раз", history)])
        self.assertEqual(gateway.answer_factual_calls, [])
        self.assertEqual(gateway.generated_prompts, [])

    def test_document_prompt_helper_includes_history(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="I sent the sample.",
                ts=1,
                attachments=[ConversationAttachment(name="test-guide.pdf", kind="document", source="docs/test-guide.pdf")],
            )
        ]
        prompt = prepare_document_request_prompt(
            "send that one again",
            [RetrievedHit(score=1.0, nid=1, meta={"source_file": "doc.pdf", "page": 1, "text": "excerpt"})],
            history=history,
            preferred_name="Test User",
        )

        self.assertIn("Preferred user name: Test User", prompt)
        self.assertIn("Recent conversation context", prompt)
        self.assertIn("assistant: I sent the sample.", prompt)
        self.assertIn("attachments sent: document(docs/test-guide.pdf)", prompt)

    def test_profile_update_short_circuits_after_classification(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="CHIT_CHAT",
                explain="user sets a preferred name",
                language="ru",
                profile_action="set_preferred_name",
                preferred_name="Heisenberg",
            )
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Неа, зовут меня теперь Heisenberg", conversation_id="conv-1")

        self.assertEqual(
            result,
            QueryResult(
                answer="",
                file=None,
                classification=Classification(
                    intent="CHIT_CHAT",
                    explain="user sets a preferred name",
                    language="ru",
                    profile_action="set_preferred_name",
                    preferred_name="Heisenberg",
                ),
                usage_events=[
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                ],
            ),
        )
        self.assertEqual(gateway.answer_factual_calls, [])
        self.assertEqual(gateway.generated_prompts, [])
        self.assertEqual(gateway.attachment_follow_up_calls, [])


if __name__ == "__main__":
    unittest.main()
