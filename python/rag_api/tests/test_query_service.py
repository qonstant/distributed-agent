from __future__ import annotations

import unittest

import numpy as np

from rag_service.application.query_service import (
    QueryService,
    _choose_retrieval_query,
    _detect_language_switch_follow_up,
    _extract_preferred_name_update,
    _infer_text_language,
    _looks_like_obvious_in_scope_education_query,
    _looks_like_preferred_name_update,
    _query_explicitly_references_attachment,
)
from rag_service.application.usage_estimation import (
    usage_event_from_model_usage,
)
from rag_service.domain.models import (
    Classification,
    ConversationAttachment,
    ConversationMessage,
    GuardrailResult,
    ModelUsage,
    QueryResult,
    RetrievalClarity,
    RetrievalSufficiency,
    RetrievedHit,
)
from rag_service.infrastructure.prompts import prepare_document_request_prompt


EN_PAGE_1_REFERENCE = "Especially check page 1 in the attached file; it has the most relevant details for this answer."
RU_PAGE_1_REFERENCE = "Особенно проверьте страницу 1 в приложенном файле: там самые релевантные детали по этому ответу."
RU_PAGE_3_REFERENCE = "Особенно проверьте страницу 3 в приложенном файле: там самые релевантные детали по этому ответу."
EN_FILE_OFFER = "Should I send you the file with this information?"
EN_TOPIC_CLARIFICATION = "Which topic do you mean: admission, student visa, residence permit, DSU scholarship, documents, deadlines, tuition, housing, exchange, CV, or letters?"
EN_RETRIEVAL_FOLLOW_UP = "To find the right answer in the documents, which topic do you mean: admission, student visa, residence permit, DSU, documents, deadlines, tuition, housing, exchange, CV, or letters?"
EN_RETRIEVAL_DISAMBIGUATION = "I found a couple of relevant document directions for this topic."
EN_SCOPE_ANSWER = "I can help only with education-abroad questions: admission, documents, scholarships, deadlines, tuition, exchange programs, student visas, residence permits, housing, CVs, and letters."
RU_SCOPE_ANSWER = "Я могу помогать только с вопросами про обучение за рубежом: поступление, документы, стипендии, дедлайны, стоимость обучения, exchange, студенческую визу, ВНЖ, жилье, CV и письма."


class FakeGateway:
    def __init__(
        self,
        classification: Classification,
        attachment_action: str = "",
        guardrail: GuardrailResult | list[GuardrailResult] | None = None,
        clarity: RetrievalClarity | None = None,
        sufficiency: RetrievalSufficiency | None = None,
        json_response=None,
    ) -> None:
        self.classification = classification
        self.attachment_action = attachment_action
        if isinstance(guardrail, list):
            self.guardrails = guardrail
        else:
            self.guardrails = [guardrail or GuardrailResult(allowed=True, reason="allowed", language=classification.language)]
        self.clarity = clarity
        self.sufficiency = sufficiency
        self.json_response = json_response
        self.guard_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.classify_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.attachment_follow_up_calls: list[tuple[str, list[ConversationMessage], str]] = []
        self.clarity_calls: list[tuple[str, str, str, list[ConversationMessage]]] = []
        self.sufficiency_calls: list[tuple[str, str, str, list[RetrievedHit], list[ConversationMessage]]] = []
        self.answer_factual_calls: list[tuple[str, str, str, list[ConversationMessage]]] = []
        self.rewrite_calls: list[tuple[str, list[ConversationMessage]]] = []
        self.embedded_queries: list[str] = []
        self.generated_prompts: list[str] = []
        self.guardrail_usage = ModelUsage(model="gpt-4o-mini", input_tokens=9, output_tokens=4, total_tokens=13)
        self.classification_usage = ModelUsage(model="gpt-4o-mini", input_tokens=12, output_tokens=6, total_tokens=18)
        self.attachment_action_usage = ModelUsage(model="gpt-4o-mini", input_tokens=8, output_tokens=4, total_tokens=12)
        self.greeting_usage = ModelUsage(model="gpt-4o-mini", input_tokens=10, output_tokens=3, total_tokens=13)
        self.factual_usage = ModelUsage(model="gpt-4o-mini", input_tokens=16, output_tokens=7, total_tokens=23)
        self.clarity_usage = ModelUsage(model="gpt-4o-mini", input_tokens=18, output_tokens=5, total_tokens=23)
        self.sufficiency_usage = ModelUsage(model="gpt-4o-mini", input_tokens=22, output_tokens=6, total_tokens=28)
        self.rewrite_usage = ModelUsage(model="gpt-4o-mini", input_tokens=20, output_tokens=4, total_tokens=24)
        self.embedding_usage = ModelUsage(model="text-embedding-3-small", input_tokens=9, output_tokens=0, total_tokens=9)
        self.json_usage = ModelUsage(model="gpt-4o-mini", input_tokens=30, output_tokens=8, total_tokens=38)

    def guard_query(self, query: str, history=None):
        self.guard_calls.append((query, history or []))
        index = min(len(self.guard_calls) - 1, len(self.guardrails) - 1)
        return self.guardrails[index], self.guardrail_usage

    def classify_query(self, query: str, history=None):
        self.classify_calls.append((query, history or []))
        return self.classification, self.classification_usage

    def classify_attachment_follow_up(self, query: str, history=None, pending_file: str = ""):
        self.attachment_follow_up_calls.append((query, history or [], pending_file))
        return self.attachment_action, self.attachment_action_usage

    def generate_greeting_reply(self, user_text: str, language_hint: str, preferred_name: str = "", history=None):
        return "hello", self.greeting_usage

    def answer_factual(self, query: str, language_hint: str, preferred_name: str = "", history=None):
        self.answer_factual_calls.append((query, language_hint, preferred_name, history or []))
        return "The test code is ALPHA-123.", self.factual_usage

    def rewrite_query_with_history(self, query: str, history=None):
        self.rewrite_calls.append((query, history or []))
        return "sample onboarding guide pdf", self.rewrite_usage

    def clarify_or_rewrite_query(self, query: str, language_hint: str, intent: str, history=None):
        self.clarity_calls.append((query, language_hint, intent, history or []))
        if self.clarity is not None:
            return self.clarity, self.clarity_usage
        if history:
            return RetrievalClarity(is_clear=True, standalone_query="sample onboarding guide pdf"), self.clarity_usage
        return RetrievalClarity(is_clear=True, standalone_query=query), self.clarity_usage

    def assess_retrieval_sufficiency(self, query: str, language_hint: str, intent: str, top_chunks, history=None):
        self.sufficiency_calls.append((query, language_hint, intent, top_chunks, history or []))
        if self.sufficiency is not None:
            return self.sufficiency, self.sufficiency_usage
        return RetrievalSufficiency(is_sufficient=True, reason="retrieved excerpts are enough"), self.sufficiency_usage

    def embed_text(self, text: str):
        self.embedded_queries.append(text)
        return np.array([1.0], dtype=np.float32), self.embedding_usage

    def generate_json_response(self, prompt: str, max_tokens: int = 512):
        self.generated_prompts.append(prompt)
        if self.json_response is not None:
            return self.json_response, self.json_usage
        return {"answer": "Use this sample.", "file": "docs/test-guide.pdf"}, self.json_usage


class FakeStore:
    def __init__(self, results=None, refined_results=None, refine_meta=None) -> None:
        self.results = results or []
        self.refined_results = refined_results
        self.refine_meta = refine_meta or {}
        self.search_calls = []
        self.refine_calls = []

    def search(self, query_embedding, k: int = 64, **filters):
        self.search_calls.append({"query_embedding": query_embedding, "k": k, **filters})
        return self.results[:k]

    def refine_results(self, *, query_text: str, initial_hits, k: int, language: str, preferred_source: str | None = None):
        self.refine_calls.append(
            {
                "query_text": query_text,
                "initial_hits": initial_hits,
                "k": k,
                "language": language,
                "preferred_source": preferred_source,
            }
        )
        if self.refined_results is None:
            return initial_hits, self.refine_meta
        return self.refined_results[:k], self.refine_meta


class FakeConversationMemory:
    def __init__(self, messages, pending_attachment: str = ""):
        self.messages = messages
        self.pending_attachment = pending_attachment
        self.remembered_pending: list[tuple[str, str]] = []
        self.cleared_pending: list[str] = []
        self.requested_ids: list[str] = []

    def load_messages(self, conversation_id: str):
        self.requested_ids.append(conversation_id)
        return self.messages

    def load_pending_attachment(self, conversation_id: str):
        return self.pending_attachment

    def remember_pending_attachment(self, conversation_id: str, source: str):
        self.pending_attachment = source
        self.remembered_pending.append((conversation_id, source))
        return True

    def clear_pending_attachment(self, conversation_id: str):
        self.pending_attachment = ""
        self.cleared_pending.append(conversation_id)


class QueryServiceTests(unittest.TestCase):
    def test_attachment_reference_detection_uses_plain_phrase_and_page_checks(self) -> None:
        self.assertTrue(
            _query_explicitly_references_attachment(
                "What does this file say about photos?",
                "italy/Residence_Permit_eng.pdf",
            )
        )
        self.assertTrue(
            _query_explicitly_references_attachment(
                "Что написано на странице 7?",
                "italy/Residence_Permit_ru.pdf",
            )
        )
        self.assertTrue(
            _query_explicitly_references_attachment(
                "Осы құжатта photo туралы не айтылған?",
                "italy/Residence_Permit_kz.pdf",
            )
        )
        self.assertFalse(
            _query_explicitly_references_attachment(
                "How to apply for a visa?",
                "italy/Residence_Permit_eng.pdf",
            )
        )

    def test_detect_language_switch_follow_up_for_supported_languages(self) -> None:
        self.assertEqual(_detect_language_switch_follow_up("На русском можно"), "ru")
        self.assertEqual(_detect_language_switch_follow_up("А можно на Английском?"), "en")
        self.assertEqual(_detect_language_switch_follow_up("qazaqsha bola ma"), "kk")
        self.assertEqual(_detect_language_switch_follow_up("scholarship in russian language"), "")

    def test_language_aware_retrieval_query_prefers_search_language(self) -> None:
        self.assertEqual(_infer_text_language("Как подать на ВНЖ в Италии?"), "ru")
        self.assertEqual(_infer_text_language("How to apply for residence permit?"), "en")
        self.assertEqual(_infer_text_language("Италияда тұруға рұқсатты қалай аламын?"), "kk")
        self.assertEqual(
            _choose_retrieval_query(
                "ru",
                "How to apply for an Italian student residence permit?",
                "required documents for Italian student residence permit",
                "Как подать на студенческий ВНЖ в Италии?",
            ),
            "Как подать на студенческий ВНЖ в Италии?",
        )

    def test_query_service_uses_same_language_query_for_retrieval_when_rewrites_are_cross_language(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="FACTUAL_QUESTION",
                explain="supported factual question",
                language="ru",
                rewritten_query="required documents for Italian student residence permit",
                needs_rag=True,
                route="RAG_SEARCH",
            ),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query="How to apply for an Italian student residence permit?",
                reason="clear",
            ),
        )
        store = FakeStore(
            results=[
                RetrievedHit(
                    score=0.9,
                    nid=1,
                    meta={
                        "source_file": "italy/Residence_Permit_ru.pdf",
                        "filename": "italy/Residence_Permit_ru.pdf",
                        "page": "1",
                        "text": "Подача на ВНЖ в Италии.",
                    },
                )
            ]
        )
        service = QueryService(gateway, store)

        result = service.handle_query("Как подать на внж")

        self.assertEqual(store.search_calls[0]["query_text"], "Как подать на внж")
        self.assertEqual(gateway.embedded_queries[0], "Как подать на внж")
        self.assertIn("страницу 1", result.answer)

    def test_query_service_uses_store_refined_results_before_sufficiency(self) -> None:
        gateway = FakeGateway(
            Classification(intent="PROCEDURE", explain="supported guidance", language="ru"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query="Как подать на студенческий ВНЖ в Италии: процесс и документы.",
                reason="clear process question",
            ),
            sufficiency=RetrievalSufficiency(is_sufficient=True, reason="enough"),
            json_response={"answer": "Следуйте шагам подачи на ВНЖ.", "file": "italy/Residence_Permit_ru.pdf"},
        )
        raw_results = [
            RetrievedHit(
                score=1.75,
                nid=1,
                meta={"source_file": "italy/Visa_ru.pdf", "page": "3", "text": "Виза"},
            )
        ]
        refined_results = [
            RetrievedHit(
                score=2.1,
                nid=2,
                meta={"source_file": "italy/Residence_Permit_ru.pdf", "page": "1", "text": "Подача на ВНЖ в течение 8 дней."},
            )
        ]
        store = FakeStore(
            results=raw_results,
            refined_results=refined_results,
            refine_meta={"selected_files": ["italy/Residence_Permit_ru.pdf"], "focused": True},
        )
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Как подать на внж", conversation_id="conv-1")

        self.assertEqual(store.refine_calls[0]["query_text"], "Как подать на студенческий ВНЖ в Италии: процесс и документы.")
        self.assertEqual(gateway.sufficiency_calls[0][3], refined_results)
        self.assertEqual(result.file, "italy/Residence_Permit_ru.pdf")
        self.assertIn("страницу 1", result.answer)

    def test_ambiguous_education_query_retrieves_first_then_asks_file_level_disambiguation(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="FACTUAL_QUESTION",
                explain="ambiguous but education-related",
                language="en",
                confidence=0.79,
                needs_rag=True,
                route="RAG_SEARCH",
            ),
            clarity=RetrievalClarity(
                is_clear=False,
                is_retrieval_related=True,
                standalone_query="photo format for Italy study documents",
                clarifying_question="Do you mean visa or residence permit?",
                reason="Photo requirements may depend on the document path.",
            ),
        )
        raw_results = [
            RetrievedHit(
                score=1.2,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": "3", "text": "Visa photo must follow ICAO format."},
            ),
            RetrievedHit(
                score=1.05,
                nid=2,
                meta={"source_file": "italy/Residence_Permit_eng.pdf", "page": "7", "text": "Residence permit photos are brought to Questura."},
            ),
        ]
        refine_meta = {
            "selected_files": ["italy/Visa_en.pdf", "italy/Residence_Permit_eng.pdf"],
            "focus_files": ["italy/Visa_en.pdf", "italy/Residence_Permit_eng.pdf"],
            "focused": True,
            "candidates": [
                {
                    "source_file": "italy/Visa_en.pdf",
                    "title": "Visa_en.pdf",
                    "doc_type": "visa",
                    "summary": "Student visa application steps, required documents, and ICAO photo requirements for the consulate.",
                },
                {
                    "source_file": "italy/Residence_Permit_eng.pdf",
                    "title": "Residence_Permit_eng.pdf",
                    "doc_type": "residence_permit",
                    "summary": "Residence permit process after arrival in Italy, including the postal kit, fingerprints, and supporting documents.",
                },
            ],
        }
        store = FakeStore(results=raw_results, refined_results=raw_results, refine_meta=refine_meta)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("What photo format do I need?", conversation_id="conv-1")

        self.assertTrue(result.answer.startswith(EN_RETRIEVAL_DISAMBIGUATION))
        self.assertIn("Visa", result.answer)
        self.assertIn("Residence Permit", result.answer)
        self.assertIsNone(result.file)
        self.assertEqual(store.search_calls[0]["query_text"], "photo format for Italy study documents")
        self.assertEqual(gateway.sufficiency_calls, [])

    def test_short_reply_after_retrieval_disambiguation_expands_previous_query_before_rag(self) -> None:
        history = [
            ConversationMessage(role="user", text="What photo format do I need?", ts=1),
            ConversationMessage(
                role="assistant",
                text=(
                    "I found a couple of relevant document directions for this topic.\n"
                    "Which one do you mean?\n"
                    "1. student visa: Student visa application steps and ICAO photo requirements.\n"
                    "2. residence permit: Residence permit process after arrival in Italy.\n"
                    "Reply with the option you mean, and I will narrow the search to that document."
                ),
                ts=2,
            ),
        ]
        gateway = FakeGateway(
            Classification(
                intent="FACTUAL_QUESTION",
                explain="now specific enough after follow-up answer",
                language="en",
                needs_rag=True,
                route="RAG_SEARCH",
            ),
            clarity=RetrievalClarity(
                is_clear=True,
                is_retrieval_related=True,
                standalone_query="What photo format do I need? student visa",
                reason="The follow-up picks the visa branch.",
            ),
            json_response={
                "answer": "The visa photo must follow ICAO standards.",
                "file": "italy/Visa_en.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": "3", "text": "Photo must comply with ICAO standards."},
            )
        ]
        memory = FakeConversationMemory(history)
        service = QueryService(gateway, FakeStore(results), conversation_memory=memory)

        result = service.handle_query("student visa", conversation_id="conv-1")

        self.assertEqual(gateway.classify_calls[0][0], "What photo format do I need? student visa")
        self.assertEqual(service._store.search_calls[0]["query_text"], "What photo format do I need? student visa")
        self.assertIn("ICAO", result.answer)

    def test_pretty_trace_formats_final_response_without_raw_json_noise(self) -> None:
        service = QueryService(
            FakeGateway(Classification(intent="GREETING", language="en")),
            FakeStore(),
            trace_log_format="pretty",
        )

        line = service._format_trace_payload(
            {
                "event": "rag_trace",
                "trace_id": "abc123",
                "stage": "response.final",
                "outcome": "rag_answer",
                "classification_intent": "PROCEDURE",
                "file": "italy/Visa_en.pdf",
                "answer_chars": 873,
                "usage_events": [
                    {"event_type": "guardrail", "total_tokens": 13, "estimated_cost": 0.1},
                    {"event_type": "classification", "total_tokens": 18, "estimated_cost": 0.2},
                ],
                "elapsed_ms": 18164.38,
            }
        )

        self.assertIn("[rag][abc123] response.final", line)
        self.assertIn("outcome=rag_answer", line)
        self.assertIn("usage=classificationx1,guardrailx1", line)
        self.assertIn("tokens=31", line)
        self.assertIn("cost=$0.300000", line)
        self.assertNotIn("usage_events", line)
        self.assertNotIn("{", line)

    def test_pretty_trace_formats_retrieval_hits_compactly(self) -> None:
        service = QueryService(
            FakeGateway(Classification(intent="PROCEDURE", language="en")),
            FakeStore(),
            trace_log_format="pretty",
        )

        line = service._format_trace_payload(
            {
                "event": "rag_trace",
                "trace_id": "abc123",
                "stage": "retrieval.results",
                "count": 64,
                "top_chunks": 8,
                "top_hits": [
                    {"score": 0.714209, "source_file": "italy/Visa_en.pdf", "page": 1},
                    {"score": 0.602951, "source_file": "italy/DSU_Scholarship_en.pdf", "page": 4},
                ],
            }
        )

        self.assertIn("hits=64", line)
        self.assertIn('files="#1 0.714 Visa_en.pdf:p1; #2 0.603 DSU_Scholarship_en.pdf:p4"', line)
        self.assertNotIn("top_hits", line)

    def test_pretty_trace_formats_retrieval_raw_and_focus_with_file_names(self) -> None:
        service = QueryService(
            FakeGateway(Classification(intent="PROCEDURE", language="ru")),
            FakeStore(),
            trace_log_format="pretty",
        )

        raw_line = service._format_trace_payload(
            {
                "event": "rag_trace",
                "trace_id": "abc123",
                "stage": "retrieval.raw",
                "count": 12,
                "top_hits": [
                    {"score": 1.75, "source_file": "italy/Visa_ru.pdf", "page": 3},
                    {"score": 0.917, "source_file": "italy/DSU_Scholarship_ru.pdf", "page": 4},
                ],
            }
        )
        self.assertIn("hits=12", raw_line)
        self.assertIn('files="#1 1.750 Visa_ru.pdf:p3; #2 0.917 DSU_Scholarship_ru.pdf:p4"', raw_line)

        focus_line = service._format_trace_payload(
            {
                "event": "rag_trace",
                "trace_id": "abc123",
                "stage": "retrieval.focus",
                "selected_files": [
                    "italy/Visa_ru.pdf",
                    "italy/Application_ru.pdf",
                    "italy/Residence_Permit_ru.pdf",
                ],
                "focus_files": [
                    "italy/Residence_Permit_ru.pdf",
                    "italy/Visa_ru.pdf",
                ],
                "focused": True,
                "count": 9,
                "top_hits": [
                    {"score": 1.569, "source_file": "italy/Residence_Permit_ru.pdf", "page": 10},
                    {"score": 1.469, "source_file": "italy/Residence_Permit_ru.pdf", "page": 1},
                ],
            }
        )
        self.assertIn('selected="Visa_ru.pdf, Application_ru.pdf, Residence_Permit_ru.pdf"', focus_line)
        self.assertIn('focus="Residence_Permit_ru.pdf, Visa_ru.pdf"', focus_line)
        self.assertIn("focused=yes", focus_line)
        self.assertIn('files="#1 1.569 Residence_Permit_ru.pdf:p10; #2 1.469 Residence_Permit_ru.pdf:p1"', focus_line)

    def test_factual_query_uses_history_loaded_from_conversation_id(self) -> None:
        history = [
            ConversationMessage(role="user", text="The test code is ALPHA-123", ts=1),
            ConversationMessage(role="assistant", text="Acknowledged. I will remember ALPHA-123 for this test.", ts=2),
        ]
        gateway = FakeGateway(
            Classification(intent="FACTUAL_QUESTION", explain="needs memory", language="en"),
            clarity=RetrievalClarity(
                is_clear=False,
                is_retrieval_related=False,
                reason="The user asks about recent conversation memory, not documents.",
            ),
        )
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
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.clarity_usage),
                    usage_event_from_model_usage("chat_completion", gateway.factual_usage),
                ],
            ),
        )
        self.assertEqual(memory.requested_ids, ["conv-1"])
        self.assertEqual(gateway.guard_calls, [("What is the test code?", [])])
        self.assertEqual(gateway.classify_calls, [("What is the test code?", history)])
        self.assertEqual(gateway.clarity_calls, [("What is the test code?", "en", "FACTUAL_QUESTION", history)])
        self.assertEqual(gateway.attachment_follow_up_calls, [])
        self.assertEqual(gateway.answer_factual_calls, [("What is the test code?", "en", "Test User", history)])

    def test_guardrail_retries_with_history_only_when_needed(self) -> None:
        history = [
            ConversationMessage(role="user", text="Как получить внж", ts=1),
            ConversationMessage(
                role="assistant",
                text="Вы имеете в виду разрешение на проживание для студентов или что-то другое?",
                ts=2,
            ),
        ]
        gateway = FakeGateway(
            Classification(intent="GREETING", explain="short confirmation after context guard", language="ru"),
            guardrail=[
                GuardrailResult(
                    allowed=False,
                    reason="The latest message is only a confirmation and needs context.",
                    language="ru",
                    needs_context=True,
                ),
                GuardrailResult(
                    allowed=True,
                    reason="The user confirms the previous in-scope student residence permit clarification.",
                    language="ru",
                ),
            ],
        )
        memory = FakeConversationMemory(history)
        service = QueryService(gateway, FakeStore(), conversation_memory=memory)

        result = service.handle_query("Да да да", conversation_id="conv-1")

        self.assertEqual(result.answer, "hello")
        self.assertEqual(memory.requested_ids, ["conv-1"])
        self.assertEqual(gateway.guard_calls, [("Да да да", []), ("Да да да", history)])
        self.assertEqual(gateway.classify_calls, [("Да да да", history)])

    def test_guardrail_blocked_first_pass_retries_with_history_before_refusing(self) -> None:
        history = [
            ConversationMessage(role="user", text="How to apply for residence permit", ts=1),
            ConversationMessage(
                role="assistant",
                text=(
                    "Are you looking for the general application steps for the student "
                    "residence permit, or do you need specific details about required documents?"
                ),
                ts=2,
            ),
        ]
        standalone_query = "Italian student residence permit application process and required documents."
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="frustrated confirmation", language="en"),
            guardrail=[
                GuardrailResult(
                    allowed=False,
                    reason="The latest message contains profanity.",
                    language="en",
                    violation="unsafe",
                ),
                GuardrailResult(
                    allowed=True,
                    reason="The user impatiently confirms the previous in-scope residence permit topic.",
                    language="en",
                ),
            ],
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="The frustrated confirmation continues the previous student residence permit topic.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student residence permit process"},
            )
        ]
        service = QueryService(gateway, FakeStore(results), conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("Fucking yes, send me already", conversation_id="conv-1")

        self.assertEqual(result.answer, f"Use this sample.\n\n{EN_PAGE_1_REFERENCE}")
        self.assertEqual(result.file, "italy/Visa_en.pdf")
        self.assertEqual(gateway.guard_calls, [("Fucking yes, send me already", []), ("Fucking yes, send me already", history)])
        self.assertEqual(gateway.clarity_calls, [("Fucking yes, send me already", "en", "CHIT_CHAT", history)])
        self.assertEqual(gateway.embedded_queries, [standalone_query])

    def test_guardrail_out_of_scope_follow_up_can_return_to_student_residence_permit_topic(self) -> None:
        history = [
            ConversationMessage(role="user", text="How to apply for residence permit", ts=1),
            ConversationMessage(
                role="assistant",
                text="Are you looking for the application process for the student residence permit specifically, or general information about residence permits in Italy?",
                ts=2,
            ),
            ConversationMessage(role="user", text="General", ts=3),
            ConversationMessage(role="assistant", text=EN_SCOPE_ANSWER, ts=4),
        ]
        standalone_query = "Italian student residence permit application process and required documents."
        gateway = FakeGateway(
            Classification(intent="PROCEDURE", explain="student residence permit follow-up", language="en"),
            guardrail=[
                GuardrailResult(
                    allowed=False,
                    reason="The latest message needs context.",
                    language="en",
                    needs_context=True,
                ),
                GuardrailResult(
                    allowed=True,
                    reason="The user returns to the student residence permit option.",
                    language="en",
                ),
            ],
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="The user returns to the student residence permit option.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student residence permit process"},
            )
        ]
        service = QueryService(gateway, FakeStore(results), conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("What about student", conversation_id="conv-1")

        self.assertNotEqual(result.answer, EN_SCOPE_ANSWER)
        self.assertEqual(result.answer, f"Use this sample.\n\n{EN_PAGE_1_REFERENCE}")
        self.assertEqual(gateway.guard_calls, [("What about student", []), ("What about student", history)])
        self.assertEqual(gateway.classify_calls, [("What about student", history)])
        self.assertEqual(gateway.clarity_calls, [("What about student", "en", "PROCEDURE", history)])
        self.assertEqual(gateway.embedded_queries, [standalone_query])

    def test_clarify_classification_with_history_uses_clarity_before_asking_again(self) -> None:
        history = [
            ConversationMessage(role="user", text="How to apply for residence permit", ts=1),
            ConversationMessage(
                role="assistant",
                text="Are you looking for the application process details, required documents, or both for the student residence permit?",
                ts=2,
            ),
        ]
        standalone_query = "Required documents for an Italian student residence permit / permesso di soggiorno."
        gateway = FakeGateway(
            Classification(
                intent="FACTUAL_QUESTION",
                explain="short follow-up could be ambiguous without history",
                language="en",
                confidence=0.42,
                needs_rag=False,
                route="CLARIFY",
            ),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="Docs required continues the previous student residence permit topic.",
            ),
            json_response={
                "answer": "For a student residence permit, prepare the required documents listed in the retrieved guide.",
                "file": "italy/Visa_en.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student residence permit required documents"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("Docs required", conversation_id="conv-1")

        self.assertNotEqual(result.answer, EN_TOPIC_CLARIFICATION)
        self.assertEqual(result.file, None)
        self.assertIn("student residence permit", result.answer)
        self.assertEqual(gateway.clarity_calls, [("Docs required", "en", "FACTUAL_QUESTION", history)])
        self.assertEqual(gateway.embedded_queries, [standalone_query])
        self.assertEqual(store.search_calls[0]["query_text"], standalone_query)

    def test_low_confidence_classification_asks_clarification_without_rag(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="PROCEDURE",
                explain="too ambiguous",
                language="en",
                confidence=0.42,
                needs_rag=True,
                route="RAG_SEARCH",
            )
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("How do I apply?", conversation_id="conv-1")

        self.assertEqual(result.answer, EN_TOPIC_CLARIFICATION)
        self.assertEqual(gateway.embedded_queries, [])
        self.assertEqual(gateway.clarity_calls, [])

    def test_factual_education_question_uses_rag_and_offers_file_without_sending(self) -> None:
        standalone_query = "What photo is required for an Italian student visa?"
        gateway = FakeGateway(
            Classification(intent="FACTUAL_QUESTION", explain="asks a factual visa document detail", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                is_retrieval_related=True,
                standalone_query=standalone_query,
                reason="The question asks for a factual detail from the student visa document.",
            ),
            json_response={
                "answer": "The visa photo must follow ICAO standards.",
                "file": "italy/Visa_en.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 3, "text": "Photo must comply with ICAO standards."},
            )
        ]
        memory = FakeConversationMemory([])
        service = QueryService(gateway, FakeStore(results), conversation_memory=memory)

        result = service.handle_query("What photo do I need for visa?", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            "The visa photo must follow ICAO standards.\n\n"
            "Especially check page 3 in the attached file; it has the most relevant details for this answer.\n\n"
            f"{EN_FILE_OFFER}",
        )
        self.assertIsNone(result.file)
        self.assertEqual(gateway.answer_factual_calls, [])
        self.assertEqual(gateway.clarity_calls, [("What photo do I need for visa?", "en", "FACTUAL_QUESTION", [])])
        self.assertEqual(gateway.embedded_queries, [standalone_query])
        self.assertEqual(gateway.sufficiency_calls, [(standalone_query, "en", "FACTUAL_QUESTION", results, [])])
        self.assertIn("Answer the user's factual question using ONLY the provided excerpts", gateway.generated_prompts[0])
        self.assertIn("answer with the available details", gateway.generated_prompts[0])
        self.assertEqual(memory.remembered_pending, [("conv-1", "italy/Visa_en.pdf")])

    def test_yes_after_factual_file_offer_sends_pending_file(self) -> None:
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="accepts offered file", language="en"),
            attachment_action="send_pending_attachment",
        )
        memory = FakeConversationMemory(
            [
                ConversationMessage(
                    role="assistant",
                    text="The visa photo must follow ICAO standards.\n\nShould I send you the file with this information?",
                    ts=1,
                )
            ],
            pending_attachment="italy/Visa_en.pdf",
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=memory)

        result = service.handle_query("yes please", conversation_id="conv-1")

        self.assertEqual(result.answer, "Sure, here is the file: Visa_en.pdf.")
        self.assertEqual(result.file, "italy/Visa_en.pdf")
        self.assertEqual(gateway.attachment_follow_up_calls, [("yes please", memory.messages, "italy/Visa_en.pdf")])
        self.assertEqual(memory.cleared_pending, ["conv-1"])
        self.assertEqual(gateway.embedded_queries, [])

    def test_document_request_uses_clarity_standalone_query_and_includes_history_in_prompt(self) -> None:
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
                answer=f"Use this sample.\n\n{EN_PAGE_1_REFERENCE}",
                file="docs/test-guide.pdf",
                classification=Classification(intent="DOCUMENT_REQUEST", explain="follow-up request", language="en"),
                usage_events=[
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                    usage_event_from_model_usage("classification", gateway.clarity_usage),
                    usage_event_from_model_usage("embedding", gateway.embedding_usage),
                    usage_event_from_model_usage("classification", gateway.sufficiency_usage),
                    usage_event_from_model_usage("chat_completion", gateway.json_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Which sample guide was that?", history, "")])
        self.assertEqual(gateway.clarity_calls, [("Which sample guide was that?", "en", "DOCUMENT_REQUEST", history)])
        self.assertEqual(gateway.rewrite_calls, [])
        self.assertEqual(gateway.embedded_queries, ["sample onboarding guide pdf"])
        self.assertEqual(store.search_calls[0]["language"], "en")
        self.assertEqual(store.search_calls[0]["query_text"], "sample onboarding guide pdf")
        self.assertEqual(gateway.sufficiency_calls, [("sample onboarding guide pdf", "en", "DOCUMENT_REQUEST", results, history)])
        self.assertEqual(len(gateway.generated_prompts), 1)
        self.assertIn("Preferred user name: Test User", gateway.generated_prompts[0])
        self.assertIn("Recent conversation context", gateway.generated_prompts[0])
        self.assertIn("user: Send me the sample onboarding guide", gateway.generated_prompts[0])

    def test_unrelated_query_does_not_anchor_refinement_to_stale_attachment(self) -> None:
        standalone_query = "How to apply for an Italian student visa?"
        gateway = FakeGateway(
            Classification(intent="PROCEDURE", explain="visa process", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="clear visa question",
            ),
            json_response={"answer": "Use the student visa application steps.", "file": "italy/Visa_en.pdf"},
        )
        store = FakeStore(
            results=[
                RetrievedHit(
                    score=1.75,
                    nid=1,
                    meta={"source_file": "italy/Visa_en.pdf", "page": "3", "text": "Visa application steps."},
                )
            ],
            refined_results=None,
        )
        memory = FakeConversationMemory(
            [
                ConversationMessage(
                    role="assistant",
                    text="Earlier residence permit answer.",
                    ts=1,
                    attachments=[
                        ConversationAttachment(
                            name="Residence_Permit_eng.pdf",
                            kind="document",
                            source="italy/Residence_Permit_eng.pdf",
                        )
                    ],
                )
            ],
            pending_attachment="italy/Residence_Permit_eng.pdf",
        )
        service = QueryService(gateway, store, conversation_memory=memory)

        result = service.handle_query("How to apply for a visa", conversation_id="conv-1")

        self.assertEqual(store.refine_calls[0]["preferred_source"], None)
        self.assertEqual(result.file, "italy/Visa_en.pdf")
        self.assertIn("page 3", result.answer)

    def test_explicit_file_reference_can_anchor_refinement_to_attachment(self) -> None:
        standalone_query = "What does this file say about residence permit photos?"
        gateway = FakeGateway(
            Classification(intent="FACTUAL_QUESTION", explain="attachment follow-up", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="clear file-specific follow-up",
            ),
            json_response={"answer": "It says to bring passport photos.", "file": "italy/Residence_Permit_eng.pdf"},
        )
        store = FakeStore(
            results=[
                RetrievedHit(
                    score=0.8,
                    nid=1,
                    meta={"source_file": "italy/Residence_Permit_eng.pdf", "page": "9", "text": "Bring 4 passport photos."},
                )
            ],
            refined_results=None,
        )
        memory = FakeConversationMemory(
            [
                ConversationMessage(
                    role="assistant",
                    text="Here is the residence permit guide.",
                    ts=1,
                    attachments=[
                        ConversationAttachment(
                            name="Residence_Permit_eng.pdf",
                            kind="document",
                            source="italy/Residence_Permit_eng.pdf",
                        )
                    ],
                )
            ],
            pending_attachment="italy/Residence_Permit_eng.pdf",
        )
        service = QueryService(gateway, store, conversation_memory=memory)

        service.handle_query("What does this file say about photos?", conversation_id="conv-1")

        self.assertEqual(store.refine_calls[0]["preferred_source"], "italy/Residence_Permit_eng.pdf")

    def test_guidance_falls_back_to_best_source_file_when_llm_omits_file(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="en"),
            json_response={"answer": "Use the scholarship application instructions from the retrieved document.", "file": None},
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={
                    "source_file": "italy/DSU_Scholarship_en.pdf",
                    "page": 1,
                    "text": "Scholarship application instructions",
                },
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("how to apply for scholarship", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            f"Use the scholarship application instructions from the retrieved document.\n\n{EN_PAGE_1_REFERENCE}",
        )
        self.assertEqual(result.file, "italy/DSU_Scholarship_en.pdf")

    def test_guidance_uses_retrieved_source_file_instead_of_llm_file_choice(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="en"),
            json_response={"answer": "Use the visa instructions from the retrieved document.", "file": "wrong.pdf"},
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={
                    "source_file": "italy/Visa_en.pdf",
                    "page": 1,
                    "text": "Student visa instructions",
                },
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("how to apply for visa", conversation_id="conv-1")

        self.assertEqual(result.answer, f"Use the visa instructions from the retrieved document.\n\n{EN_PAGE_1_REFERENCE}")
        self.assertEqual(result.file, "italy/Visa_en.pdf")

    def test_guidance_uses_llm_file_choice_when_it_matches_retrieved_source(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="ru"),
            json_response={
                "answer": "Используйте инструкцию по подаче на стипендию из найденного документа.",
                "file": "italy/DSU_Scholarship_ru.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.95,
                nid=1,
                meta={
                    "source_file": "italy/Visa_ru.pdf",
                    "page": 1,
                    "text": "Student visa instructions",
                },
            ),
            RetrievedHit(
                score=0.7,
                nid=2,
                meta={
                    "source_file": "italy/DSU_Scholarship_ru.pdf",
                    "page": 3,
                    "text": "Scholarship application instructions",
                },
            ),
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("как подать на стипендию", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            f"Используйте инструкцию по подаче на стипендию из найденного документа.\n\n{RU_PAGE_3_REFERENCE}",
        )
        self.assertEqual(result.file, "italy/DSU_Scholarship_ru.pdf")

    def test_guidance_matches_llm_file_choice_by_basename_when_retrieved_source_has_prefix(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="ru"),
            json_response={
                "answer": "Используйте инструкцию по подаче на стипендию из найденного документа.",
                "file": "DSU_Scholarship_ru.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={
                    "source_file": "italy/DSU_Scholarship_ru.pdf",
                    "page": 1,
                    "text": "Scholarship application instructions",
                },
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("как подать на стипендию", conversation_id="conv-1")

        self.assertEqual(result.file, "italy/DSU_Scholarship_ru.pdf")

    def test_guidance_asks_follow_up_for_unknown_answer(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="unsupported guidance", language="en"),
            json_response={"answer": "I don't know based on the provided documents.", "file": None},
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/DSU_Scholarship_en.pdf", "page": 1, "text": "Scholarship excerpt"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("unsupported question", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            EN_RETRIEVAL_FOLLOW_UP,
        )
        self.assertIsNone(result.file)

    def test_guidance_returns_not_enough_info_when_clear_query_results_are_insufficient(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="en"),
            sufficiency=RetrievalSufficiency(
                is_sufficient=False,
                clarifying_question="Are you asking about the student visa or scholarship?",
                reason="The retrieved excerpts do not resolve the topic.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student visa excerpt"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("how does it work?", conversation_id="conv-1")

        self.assertIn("could not find enough reliable information", result.answer)
        self.assertNotIn("student visa or scholarship", result.answer)
        self.assertIsNone(result.file)
        self.assertEqual(gateway.sufficiency_calls, [("how does it work?", "en", "GUIDANCE", results, [])])
        self.assertEqual(gateway.generated_prompts, [])

    def test_residence_permit_weak_retrieval_does_not_ask_student_vs_general_loop_question(self) -> None:
        standalone_query = "How to apply for an Italian student residence permit / permesso di soggiorno?"
        gateway = FakeGateway(
            Classification(intent="PROCEDURE", explain="student residence permit process", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="Residence permit defaults to the student residence permit topic.",
            ),
            sufficiency=RetrievalSufficiency(
                is_sufficient=False,
                clarifying_question="Could you specify if you need information on the application process for the student residence permit or details about required documents for it?",
                reason="The excerpts do not specifically address the application process.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.65,
                nid=1,
                meta={
                    "source_file": "italy/Visa_en.pdf",
                    "page": 5,
                    "text": "Residence permit if the visa application is submitted outside the country of citizenship.",
                },
            )
        ]
        service = QueryService(gateway, FakeStore(results), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("How to apply for residence permit", conversation_id="conv-1")

        self.assertIn("could not find enough reliable information", result.answer)
        self.assertNotIn("application process", result.answer)
        self.assertNotIn("required documents", result.answer)
        self.assertIsNone(result.file)
        self.assertEqual(gateway.generated_prompts, [])

    def test_guidance_returns_not_enough_info_when_no_results(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="supported guidance", language="ru"),
            sufficiency=RetrievalSufficiency(
                is_sufficient=False,
                clarifying_question="Уточните, пожалуйста, вы спрашиваете про студенческую визу, CV или стипендию?",
                reason="No retrieved excerpts are available.",
            ),
        )
        store = FakeStore([])
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("что нужно?", conversation_id="conv-1")

        self.assertIn("не нашлось достаточно надежной информации", result.answer)
        self.assertNotIn("студенческую визу, CV или стипендию", result.answer)
        self.assertIsNone(result.file)
        self.assertEqual(gateway.sufficiency_calls, [("что нужно?", "ru", "GUIDANCE", [], [])])
        self.assertEqual(gateway.generated_prompts, [])

    def test_guidance_asks_clarifying_question_without_search_when_unclear(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="ambiguous docs question", language="en"),
            clarity=RetrievalClarity(
                is_clear=False,
                clarifying_question=EN_TOPIC_CLARIFICATION,
                reason="The requested document topic is missing.",
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("what documents do I need?", conversation_id="conv-1")

        self.assertEqual(
            result,
            QueryResult(
                answer=EN_TOPIC_CLARIFICATION,
                file=None,
                classification=Classification(intent="GUIDANCE", explain="ambiguous docs question", language="en"),
                usage_events=[
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.clarity_usage),
                ],
            ),
        )
        self.assertEqual(gateway.embedded_queries, [])
        self.assertEqual(store.search_calls, [])
        self.assertEqual(gateway.generated_prompts, [])

    def test_follow_up_after_clarification_can_run_rag_when_classifier_returns_other(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text=EN_TOPIC_CLARIFICATION,
                ts=1,
            )
        ]
        gateway = FakeGateway(
            Classification(intent="OTHER", explain="fragment answer to previous question", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query="What documents are needed for an Italian student visa?",
                reason="The latest reply resolves the previous clarification question.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student visa document excerpt"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("student visa", conversation_id="conv-1")

        self.assertEqual(result.answer, f"Use this sample.\n\n{EN_PAGE_1_REFERENCE}")
        self.assertEqual(result.file, "italy/Visa_en.pdf")
        self.assertEqual(gateway.embedded_queries, ["What documents are needed for an Italian student visa?"])
        self.assertEqual(store.search_calls[0]["query_text"], "What documents are needed for an Italian student visa?")
        self.assertEqual(gateway.clarity_calls, [("student visa", "en", "OTHER", history)])

    def test_chit_chat_follow_up_after_clarification_can_run_rag(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="Что именно вас интересует по подаче на студенческую визу: документы или процесс подачи?",
                ts=1,
            )
        ]
        standalone_query = "Полный обзор подачи на студенческую визу в Италию: документы и процесс подачи"
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="short broad reply", language="ru"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="The user asked for all visa application details after a clarification question.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_ru.pdf", "page": 1, "text": "student visa process and documents"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("Все все вообще", conversation_id="conv-1")

        self.assertEqual(result.answer, f"Use this sample.\n\n{RU_PAGE_1_REFERENCE}")
        self.assertEqual(result.file, "italy/Visa_ru.pdf")
        self.assertEqual(gateway.clarity_calls, [("Все все вообще", "ru", "CHIT_CHAT", history)])
        self.assertEqual(gateway.embedded_queries, [standalone_query])
        self.assertEqual(store.search_calls[0]["query_text"], standalone_query)

    def test_short_how_follow_up_after_dsu_answer_reuses_dsu_topic(self) -> None:
        history = [
            ConversationMessage(role="user", text="How to get dsu", ts=1),
            ConversationMessage(
                role="assistant",
                text="DSU is the Italian student scholarship/financial aid.",
                ts=2,
            ),
        ]
        standalone_query = "How to apply for the Italian DSU student scholarship?"
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="short follow-up", language="en"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query=standalone_query,
                reason="The user asks how to continue the previous DSU scholarship topic.",
            ),
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={
                    "source_file": "italy/DSU_Scholarship_en.pdf",
                    "page": 1,
                    "text": "DSU scholarship application instructions",
                },
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("How", conversation_id="conv-1")

        self.assertEqual(result.answer, f"Use this sample.\n\n{EN_PAGE_1_REFERENCE}")
        self.assertEqual(result.file, "italy/DSU_Scholarship_en.pdf")
        self.assertEqual(gateway.clarity_calls, [("How", "en", "CHIT_CHAT", history)])
        self.assertEqual(gateway.embedded_queries, [standalone_query])
        self.assertEqual(store.search_calls[0]["query_text"], standalone_query)

    def test_language_switch_follow_up_reruns_rag_in_requested_language(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="Для подачи на студенческую визу в Италию необходимо подготовить документы.",
                ts=1,
                attachments=[ConversationAttachment(name="Visa_ru.pdf", kind="document", source="italy/Visa_ru.pdf")],
            )
        ]
        gateway = FakeGateway(
            Classification(intent="OTHER", explain="language switch follow-up", language="ru"),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query="How to apply for an Italian student visa?",
                reason="The user asks to continue the previous visa topic in English.",
                target_language="en",
            ),
            json_response={
                "answer": "To apply for an Italian student visa, prepare the required documents and book an appointment.",
                "file": "italy/Visa_en.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/Visa_en.pdf", "page": 1, "text": "student visa application process"},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("А можно на Английском?", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            "To apply for an Italian student visa, prepare the required documents and book an appointment."
            f"\n\n{EN_PAGE_1_REFERENCE}",
        )
        self.assertEqual(result.file, "italy/Visa_en.pdf")
        self.assertEqual(store.search_calls[0]["language"], "en")
        self.assertEqual(store.search_calls[0]["query_text"], "How to apply for an Italian student visa?")
        self.assertIn("Answer in the same language as detected/requested: English", gateway.generated_prompts[0])

    def test_language_switch_follow_up_can_bypass_guardrail_block_and_continue_topic(self) -> None:
        history = [
            ConversationMessage(
                role="assistant",
                text="To apply for the DSU scholarship, prepare the required family income documents.",
                ts=1,
                attachments=[ConversationAttachment(name="DSU_Scholarship_en.pdf", kind="document", source="italy/DSU_Scholarship_en.pdf")],
            )
        ]
        gateway = FakeGateway(
            Classification(intent="OTHER", explain="language switch follow-up", language="ru"),
            guardrail=GuardrailResult(
                allowed=False,
                reason="The latest message alone is not a standalone study-abroad request.",
                language="ru",
            ),
            clarity=RetrievalClarity(
                is_clear=True,
                standalone_query="How to apply for the Italian DSU student scholarship?",
                reason="The user asks to continue the previous DSU topic in Russian.",
                target_language="ru",
            ),
            json_response={
                "answer": "Для подачи на стипендию DSU подготовьте документы о доходах семьи.",
                "file": "italy/DSU_Scholarship_ru.pdf",
            },
        )
        results = [
            RetrievedHit(
                score=0.9,
                nid=1,
                meta={"source_file": "italy/DSU_Scholarship_ru.pdf", "page": 1, "text": "Документы для подачи на стипендию DSU."},
            )
        ]
        store = FakeStore(results)
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("На русском можно", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            "Для подачи на стипендию DSU подготовьте документы о доходах семьи."
            f"\n\n{RU_PAGE_1_REFERENCE}",
        )
        self.assertEqual(result.file, "italy/DSU_Scholarship_ru.pdf")
        self.assertEqual(store.search_calls[0]["language"], "ru")

    def test_chit_chat_with_history_still_greets_when_not_retrieval_related(self) -> None:
        history = [
            ConversationMessage(role="assistant", text="I can help with education-abroad documents.", ts=1),
        ]
        gateway = FakeGateway(
            Classification(intent="CHIT_CHAT", explain="thanks", language="en"),
            clarity=RetrievalClarity(
                is_clear=False,
                standalone_query="",
                clarifying_question="",
                reason="The user is thanking the assistant.",
                is_retrieval_related=False,
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("thanks", conversation_id="conv-1")

        self.assertEqual(result.answer, "hello")
        self.assertIsNone(result.file)
        self.assertEqual(gateway.clarity_calls, [("thanks", "en", "CHIT_CHAT", history)])
        self.assertEqual(gateway.embedded_queries, [])
        self.assertEqual(store.search_calls, [])

    def test_other_with_history_returns_scope_message_when_not_retrieval_related(self) -> None:
        history = [
            ConversationMessage(role="assistant", text="I can help with admissions documents.", ts=1),
        ]
        gateway = FakeGateway(
            Classification(intent="OTHER", explain="not a retrieval question", language="en"),
            clarity=RetrievalClarity(
                is_clear=False,
                standalone_query="",
                clarifying_question="",
                reason="The user is not asking a document-grounded question.",
                is_retrieval_related=False,
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory(history))

        result = service.handle_query("never mind", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            EN_SCOPE_ANSWER,
        )
        self.assertEqual(gateway.clarity_calls, [("never mind", "en", "OTHER", history)])
        self.assertEqual(gateway.answer_factual_calls, [])
        self.assertEqual(store.search_calls, [])

    def test_guidance_out_of_scope_clarity_blocks_retrieval_even_for_retrieval_intent(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="user asks visa guidance", language="en"),
            clarity=RetrievalClarity(
                is_clear=False,
                standalone_query="",
                clarifying_question="",
                reason="The user asks about a tourist visa, not education abroad.",
                is_retrieval_related=False,
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("How do I get a tourist visa for Italy?", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            EN_SCOPE_ANSWER,
        )
        self.assertIsNone(result.file)
        self.assertEqual(gateway.clarity_calls, [("How do I get a tourist visa for Italy?", "en", "GUIDANCE", [])])
        self.assertEqual(gateway.embedded_queries, [])
        self.assertEqual(gateway.generated_prompts, [])
        self.assertEqual(store.search_calls, [])

    def test_other_without_history_returns_scope_message_without_factual_answer(self) -> None:
        gateway = FakeGateway(Classification(intent="OTHER", explain="tourism request", language="ru"))
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Как получить туристическую визу в Италию?", conversation_id="conv-1")

        self.assertEqual(
            result.answer,
            RU_SCOPE_ANSWER,
        )
        self.assertIsNone(result.file)
        self.assertEqual(gateway.clarity_calls, [])
        self.assertEqual(gateway.answer_factual_calls, [])
        self.assertEqual(store.search_calls, [])

    def test_guardrail_blocks_out_of_scope_before_classification(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="would not be used", language="en"),
            guardrail=GuardrailResult(
                allowed=False,
                reason="Tourist visa is outside the study-abroad scope.",
                language="en",
                violation="out_of_scope",
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("How do I get a tourist visa for Italy?", conversation_id="conv-1")

        self.assertEqual(
            result,
            QueryResult(
                answer=EN_SCOPE_ANSWER,
                file=None,
                classification=None,
                usage_events=[
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                ],
            ),
        )
        self.assertEqual(gateway.guard_calls, [("How do I get a tourist visa for Italy?", [])])
        self.assertEqual(gateway.classify_calls, [])
        self.assertEqual(gateway.clarity_calls, [])
        self.assertEqual(gateway.embedded_queries, [])
        self.assertEqual(store.search_calls, [])

    def test_guardrail_blocks_unsafe_request_before_classification(self) -> None:
        gateway = FakeGateway(
            Classification(intent="GUIDANCE", explain="would not be used", language="en"),
            guardrail=GuardrailResult(
                allowed=False,
                reason="The user asks for help falsifying documents.",
                language="en",
                violation="unsafe",
            ),
        )
        store = FakeStore()
        service = QueryService(gateway, store, conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("How can I fake a bank statement for visa?", conversation_id="conv-1")

        self.assertIn("I can't help with illegal or unsafe requests.", result.answer)
        self.assertIsNone(result.file)
        self.assertIsNone(result.classification)
        self.assertEqual(gateway.classify_calls, [])
        self.assertEqual(gateway.clarity_calls, [])
        self.assertEqual(store.search_calls, [])

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
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Please resend the file", history, "")])
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
                    usage_event_from_model_usage("guardrail", gateway.guardrail_usage),
                    usage_event_from_model_usage("classification", gateway.classification_usage),
                    usage_event_from_model_usage("classification", gateway.attachment_action_usage),
                ],
            ),
        )
        self.assertEqual(gateway.attachment_follow_up_calls, [("Еще раз", history, "")])
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

    def test_detects_preferred_name_update_phrases(self) -> None:
        self.assertTrue(_looks_like_preferred_name_update("Call me Alex"))
        self.assertTrue(_looks_like_preferred_name_update("My name is Rodrigo"))
        self.assertTrue(_looks_like_preferred_name_update("Зови меня рекстер"))
        self.assertTrue(_looks_like_preferred_name_update("Меня зовут Азамат"))
        self.assertTrue(_looks_like_preferred_name_update("Родриго деп ата"))
        self.assertFalse(_looks_like_preferred_name_update("Как получить визу в Италию?"))

    def test_extracts_preferred_name_from_keyword_path(self) -> None:
        self.assertEqual(_extract_preferred_name_update("Call me Alex"), "Alex")
        self.assertEqual(_extract_preferred_name_update("Зови меня akimshilik"), "akimshilik")
        self.assertEqual(_extract_preferred_name_update("Зови меня 'Bruno"), "Bruno")
        self.assertEqual(_extract_preferred_name_update("Родриго деп ата"), "Родриго")

    def test_profile_update_bypasses_guardrail_keyword_path(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="CHIT_CHAT",
                explain="user sets a preferred name",
                language="ru",
                profile_action="set_preferred_name",
                preferred_name="Рекстер",
            ),
            guardrail=GuardrailResult(
                allowed=False,
                violation="out_of_scope",
                reason="blocked by guardrail",
                language="ru",
            ),
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Зови меня рекстер", conversation_id="conv-1")

        self.assertEqual(result.answer, "")
        self.assertIsNotNone(result.classification)
        self.assertEqual(result.classification.profile_action, "set_preferred_name")
        self.assertEqual(result.classification.preferred_name, "Рекстер")
        self.assertEqual(gateway.guard_calls, [])
        self.assertEqual(len(result.usage_events), 1)
        self.assertEqual(result.usage_events[0].event_type, "classification")

    def test_profile_update_keyword_fallback_overrides_bad_classifier_output(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="OUT_OF_DOMAIN",
                explain="User's input is unrelated to education abroad.",
                language="ru",
                route="REFUSE_OR_REDIRECT",
                confidence=0.9,
                profile_action="",
                preferred_name="",
            )
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Зови меня akimshilik", conversation_id="conv-1")

        self.assertEqual(result.answer, "")
        self.assertIsNotNone(result.classification)
        self.assertEqual(result.classification.profile_action, "set_preferred_name")
        self.assertEqual(result.classification.preferred_name, "akimshilik")
        self.assertEqual(gateway.guard_calls, [])
        self.assertEqual(gateway.clarity_calls, [])
        self.assertEqual(gateway.attachment_follow_up_calls, [])

    def test_detects_obvious_in_scope_education_query_phrases(self) -> None:
        self.assertTrue(_looks_like_obvious_in_scope_education_query("Какой isee нужен для учебы бесплатно"))
        self.assertTrue(_looks_like_obvious_in_scope_education_query("What ISEE is needed to study for free?"))
        self.assertTrue(_looks_like_obvious_in_scope_education_query("How to get DSU scholarship"))
        self.assertFalse(_looks_like_obvious_in_scope_education_query("How can I fake a bank statement for visa?"))
        self.assertTrue(_looks_like_obvious_in_scope_education_query("Where do I buy marca da bollo for residence permit?"))
        self.assertFalse(_looks_like_obvious_in_scope_education_query("How can I buy a visa?"))
        self.assertFalse(_looks_like_obvious_in_scope_education_query("Где купить визу в Италию?"))

    def test_isee_query_bypasses_guardrail_keyword_path(self) -> None:
        gateway = FakeGateway(
            Classification(
                intent="FACTUAL_QUESTION",
                explain="user asks about ISEE needed for scholarship or tuition waiver",
                language="ru",
                route="RAG_SEARCH",
                needs_rag=True,
                confidence=0.88,
                rewritten_query="какой ISEE нужен для бесплатной учебы и стипендии в Италии",
            ),
            guardrail=GuardrailResult(
                allowed=False,
                violation="out_of_scope",
                reason="blocked by guardrail",
                language="ru",
            ),
        )
        service = QueryService(gateway, FakeStore(), conversation_memory=FakeConversationMemory([]))

        result = service.handle_query("Какой isee нужен для учебы бесплатно", conversation_id="conv-1")

        self.assertIsNotNone(result.classification)
        self.assertEqual(result.classification.intent, "FACTUAL_QUESTION")
        self.assertEqual(gateway.guard_calls, [])
        self.assertEqual(result.usage_events[0].event_type, "classification")


if __name__ == "__main__":
    unittest.main()
