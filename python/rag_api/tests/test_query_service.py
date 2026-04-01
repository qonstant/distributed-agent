from __future__ import annotations

import unittest

import numpy as np

from rag_service.application.query_service import QueryService
from rag_service.domain.models import (
    Classification,
    ConversationAttachment,
    ConversationMessage,
    QueryResult,
    RetrievedHit,
    UsageEventRecord,
)
from rag_service.infrastructure.prompts import prepare_document_request_prompt


class FakeGateway:
    def __init__(self, classification: Classification) -> None:
        self.classification = classification
        self.classify_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.answer_factual_calls: list[tuple[str, str, list[ConversationMessage]]] = []
        self.rewrite_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.embedded_queries: list[str] = []
        self.generated_prompts: list[str] = []

    def classify_query(self, query: str, history=None) -> Classification:
        self.classify_calls.append((query, history or []))
        return self.classification

    def generate_greeting_reply(self, user_text: str, language_hint: str, history=None) -> str:
        return "hello"

    def answer_factual(self, query: str, language_hint: str, history=None) -> str:
        self.answer_factual_calls.append((query, language_hint, history or []))
        return "The test code is ALPHA-123."

    def rewrite_query_with_history(self, query: str, history=None) -> str:
        self.rewrite_calls.append((query, history or []))
        return "sample onboarding guide pdf"

    def embed_text(self, text: str) -> np.ndarray:
        self.embedded_queries.append(text)
        return np.array([1.0], dtype=np.float32)

    def generate_json_response(self, prompt: str, max_tokens: int = 512):
        self.generated_prompts.append(prompt)
        return {"answer": "Use this sample.", "file": "docs/test-guide.pdf"}


class FakeStore:
    def __init__(self, results=None) -> None:
        self.results = results or []

    def search(self, query_embedding, k: int = 64):
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

        result = service.handle_query("What is the test code?", conversation_id="conv-1")

        self.assertEqual(
            result,
            QueryResult(
                answer="The test code is ALPHA-123.",
                file=None,
                classification=Classification(intent="FACTUAL_QUESTION", explain="needs memory", language="en"),
                usage_events=[
                    UsageEventRecord(event_type="classification"),
                    UsageEventRecord(event_type="chat_completion"),
                ],
            ),
        )
        self.assertEqual(memory.requested_ids, ["conv-1"])
        self.assertEqual(gateway.classify_calls, [("What is the test code?", history)])
        self.assertEqual(gateway.answer_factual_calls, [("What is the test code?", "en", history)])

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
        service = QueryService(gateway, FakeStore(results), conversation_memory=memory)

        result = service.handle_query("Which sample guide was that?", conversation_id="conv-1")

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
                    UsageEventRecord(event_type="classification"),
                    UsageEventRecord(event_type="embedding"),
                    UsageEventRecord(event_type="rag_query"),
                    UsageEventRecord(event_type="chat_completion"),
                ],
            ),
        )
        self.assertEqual(gateway.rewrite_calls, [("Which sample guide was that?", history)])
        self.assertEqual(gateway.embedded_queries, ["sample onboarding guide pdf"])
        self.assertEqual(len(gateway.generated_prompts), 1)
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
        gateway = FakeGateway(Classification(intent="DOCUMENT_REQUEST", explain="resend request", language="en"))
        memory = FakeConversationMemory(history)
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "docs/test-guide.pdf", "page": 1, "text": "sample document excerpt"},
            )
        ]
        service = QueryService(gateway, FakeStore(results), conversation_memory=memory)

        result = service.handle_query("Please resend the file", conversation_id="conv-1")

        self.assertEqual(
            result,
            QueryResult(
                answer="Use this sample.",
                file="docs/test-guide.pdf",
                classification=Classification(intent="DOCUMENT_REQUEST", explain="resend request", language="en"),
                usage_events=[
                    UsageEventRecord(event_type="classification"),
                    UsageEventRecord(event_type="embedding"),
                    UsageEventRecord(event_type="rag_query"),
                    UsageEventRecord(event_type="chat_completion"),
                ],
            ),
        )

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
        )

        self.assertIn("Recent conversation context", prompt)
        self.assertIn("assistant: I sent the sample.", prompt)
        self.assertIn("attachments sent: document(docs/test-guide.pdf)", prompt)


if __name__ == "__main__":
    unittest.main()
