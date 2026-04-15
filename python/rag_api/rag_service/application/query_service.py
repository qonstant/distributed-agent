from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from rag_service.application.usage_estimation import (
    estimate_classification_event,
    estimate_embedding_event,
    estimate_factual_completion_event,
    estimate_greeting_completion_event,
    estimate_prompt_completion_event,
    usage_event_from_model_usage,
)
from rag_service.domain.models import (
    Classification,
    ConversationAttachment,
    ConversationMessage,
    QueryResult,
    RetrievalClarity,
    RetrievedHit,
    UsageEventRecord,
)
from rag_service.infrastructure.prompts import (
    prepare_document_request_prompt,
    prepare_guidance_prompt,
)

if TYPE_CHECKING:
    from rag_service.infrastructure.openai_gateway import OpenAIGateway


RETRIEVAL_INTENTS = {"GUIDANCE", "DOCUMENT_REQUEST"}


def _aggregate_by_file(results: List[RetrievedHit]) -> Tuple[Optional[str], Optional[RetrievedHit]]:
    file_sum: dict[str, float] = {}
    best_chunk_for_file: dict[str, RetrievedHit] = {}
    for hit in results:
        meta = hit.meta
        source_file = meta.get("source_file") or meta.get("filename") or "unknown"
        file_sum[source_file] = file_sum.get(source_file, 0.0) + hit.score
        if source_file not in best_chunk_for_file or hit.score > best_chunk_for_file[source_file].score:
            best_chunk_for_file[source_file] = hit
    if not file_sum:
        return None, None
    best_file = max(file_sum.items(), key=lambda item: item[1])[0]
    return best_file, best_chunk_for_file[best_file]


def _basename(value: str) -> str:
    normalized = (value or "").strip().rstrip("/")
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def _find_latest_assistant_attachment(history: List[ConversationMessage]) -> Optional[ConversationAttachment]:
    for message in reversed(history):
        if message.role != "assistant":
            continue
        for attachment in message.attachments:
            if (attachment.source or "").strip() or (attachment.name or "").strip():
                return attachment
    return None


def _resend_attachment_source(attachment: ConversationAttachment) -> Optional[str]:
    source = (attachment.source or "").strip()
    if source:
        return source
    name = (attachment.name or "").strip()
    if name:
        return name
    return None


def _resend_attachment_answer(attachment: ConversationAttachment, language: str) -> str:
    file_label = _attachment_display_label(attachment, attachment.source or attachment.name)
    normalized_language = (language or "").strip().lower()

    if normalized_language in {"kk", "kazakh", "қазақ", "қазақша"}:
        return f"Міне, файлды қайта жібердім: {file_label}."
    if normalized_language in {"ru", "russian", "русский"}:
        return f"Вот файл еще раз: {file_label}."
    return f"Here is the file again: {file_label}."


def _find_previously_sent_attachment(
    history: List[ConversationMessage],
    file_chosen: Optional[str],
) -> Optional[ConversationAttachment]:
    target_source = (file_chosen or "").strip()
    if not target_source:
        return None

    target_name = _basename(target_source)
    for message in reversed(history):
        if message.role != "assistant":
            continue
        for attachment in message.attachments:
            source = (attachment.source or "").strip()
            name = (attachment.name or "").strip()
            if source and source == target_source:
                return attachment
            if not source and name and name == target_name:
                return attachment
    return None


def _attachment_display_label(
    attachment: Optional[ConversationAttachment],
    file_chosen: Optional[str],
) -> str:
    if attachment is not None:
        name = (attachment.name or "").strip()
        if name:
            return name

        source = (attachment.source or "").strip()
        if source:
            return _basename(source)

    return _basename(file_chosen or "")


def _normalize_file_choice(value) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null", "nil"}:
        return None
    return normalized


def _source_file_from_hit(hit: Optional[RetrievedHit]) -> Optional[str]:
    if hit is None:
        return None
    return _normalize_file_choice(hit.meta.get("source_file") or hit.meta.get("filename"))


def _should_attach_supporting_file(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    if not normalized:
        return False
    return normalized != "i don't know based on the provided documents."


def _language_label(language: str) -> str:
    normalized = (language or "").strip().lower()
    mapping = {
        "en": "English",
        "ru": "Russian",
        "kk": "Kazakh",
        "other": "",
    }
    return mapping.get(normalized, (language or "").strip())


def _should_run_retrieval_clarity(intent: str, history: List[ConversationMessage]) -> bool:
    if intent in RETRIEVAL_INTENTS:
        return True
    if intent == "OTHER" and history:
        return True
    return False


def _fallback_clarifying_question(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return "Қай тақырып бойынша сұрап тұрсыз: студенттік виза, CV, DSU шәкіртақысы, мотивациялық хат немесе ұсыныс хат?"
    if normalized_language == "ru":
        return "По какой теме вы спрашиваете: студенческая виза, CV, стипендия DSU, мотивационное письмо или рекомендательное письмо?"
    return "Which topic do you mean: student visa, CV, DSU scholarship, motivation letter, or recommendation letter?"


def _out_of_scope_answer(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return (
            "Мен тек шетелде оқу бойынша сұрақтарға көмектесе аламын: оқуға түсу, "
            "студенттік виза, шәкіртақы, CV, мотивациялық және ұсыныс хаттар."
        )
    if normalized_language == "ru":
        return (
            "Я могу помогать только с вопросами про обучение за рубежом: поступление, "
            "студенческую визу, стипендия, CV, мотивационное и рекомендательное письма."
        )
    return (
        "I can help only with education-abroad questions: admission, student visas, "
        "DSU scholarships, scholarship, CVs, motivation letters, and recommendation letters."
    )


class QueryService:
    def __init__(self, gateway: "OpenAIGateway", store, conversation_memory=None) -> None:
        self._gateway = gateway
        self._store = store
        self._conversation_memory = conversation_memory

    def handle_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        preferred_name: Optional[str] = None,
        raw_k: Optional[int] = None,
        top_for_llm: Optional[int] = None,
    ) -> QueryResult:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("query is empty")
        normalized_preferred_name = (preferred_name or "").strip()

        history = self._load_history(conversation_id)
        classification, classification_usage = self._gateway.classify_query(normalized_query, history=history)
        usage_events = [
            self._classification_usage_event(
                normalized_query,
                history,
                classification,
                classification_usage,
            )
        ]
        intent = classification.intent
        language = classification.language or ""
        language_label = _language_label(language)
        print(
            f"[query] classifier -> intent={intent} lang={language} "
            f"explain={classification.explain}"
        )

        if classification.profile_action == "set_preferred_name" and (classification.preferred_name or "").strip():
            return QueryResult(
                answer="",
                file=None,
                classification=classification,
                usage_events=usage_events,
            )

        latest_attachment = _find_latest_assistant_attachment(history)
        if latest_attachment is not None:
            attachment_action, attachment_action_usage = self._gateway.classify_attachment_follow_up(
                normalized_query,
                history=history,
            )
            if attachment_action_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", attachment_action_usage))

            resend_source = _resend_attachment_source(latest_attachment)
            if attachment_action == "resend_last_attachment" and resend_source is not None:
                return QueryResult(
                    answer=_resend_attachment_answer(latest_attachment, language),
                    file=resend_source,
                    classification=classification,
                    usage_events=usage_events,
                )

        if intent in ("GREETING", "CHIT_CHAT"):
            greeting, completion_usage = self._gateway.generate_greeting_reply(
                normalized_query,
                language,
                preferred_name=normalized_preferred_name,
                history=history,
            )
            return QueryResult(
                answer=greeting,
                file=None,
                classification=classification,
                usage_events=usage_events + [
                    self._chat_completion_usage_event(
                        normalized_query,
                        history,
                        greeting,
                        completion_usage,
                        greeting_mode=True,
                    )
                ],
            )

        if intent == "FACTUAL_QUESTION":
            answer, completion_usage = self._gateway.answer_factual(
                normalized_query,
                language,
                preferred_name=normalized_preferred_name,
                history=history,
            )
            return QueryResult(
                answer=answer,
                file=None,
                classification=classification,
                usage_events=usage_events + [
                    self._chat_completion_usage_event(
                        normalized_query,
                        history,
                        answer,
                        completion_usage,
                    )
                ],
            )

        if _should_run_retrieval_clarity(intent, history):
            clarity, clarity_usage = self._retrieval_clarity(
                normalized_query,
                language,
                intent,
                history,
            )
            if clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if not clarity.is_retrieval_related:
                return QueryResult(
                    answer=_out_of_scope_answer(language),
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                )

            if not clarity.is_clear:
                answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(language)
                return QueryResult(
                    answer=answer,
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                )

            retrieval_query = (clarity.standalone_query or "").strip() or normalized_query
            retrieval_intent = intent if intent in RETRIEVAL_INTENTS else "GUIDANCE"
            rewrite_usage = None

            try:
                query_embedding, embedding_usage = self._gateway.embed_text(retrieval_query)
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"embedding failed: {exc}") from exc

            try:
                results = self._store.search(
                    query_embedding,
                    k=max(1, int(raw_k or 64)),
                    language=language,
                    query_text=retrieval_query,
                )
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"search failed: {exc}") from exc

            if not results:
                return QueryResult(
                    answer="I don't know based on the provided documents.",
                    file=None,
                    classification=classification,
                    usage_events=usage_events + self._rag_usage_events(
                        retrieval_query,
                        rewrite_usage,
                        embedding_usage,
                    ),
                )

            best_file_agg, best_chunk = _aggregate_by_file(results)
            supporting_file = _source_file_from_hit(best_chunk) or _normalize_file_choice(best_file_agg)
            top_n = max(1, int(top_for_llm or 8))
            top_chunks = results[:top_n]

            if retrieval_intent == "DOCUMENT_REQUEST":
                prompt = prepare_document_request_prompt(
                    retrieval_query,
                    top_chunks,
                    history=history,
                    preferred_name=normalized_preferred_name,
                )
            else:
                prompt = prepare_guidance_prompt(
                    retrieval_query,
                    top_chunks,
                    history=history,
                    preferred_name=normalized_preferred_name,
                )

            if language_label:
                prompt = f"Answer in the same language as detected: {language_label}\n\n" + prompt
            else:
                prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

            llm_json, completion_usage = self._gateway.generate_json_response(prompt, max_tokens=512)

            if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
                answer = str(llm_json.get("answer", "")).strip()
            else:
                if best_chunk is None:
                    return QueryResult(
                        answer="I don't know based on the provided documents.",
                        file=None,
                    )
                chunk_meta = best_chunk.meta
                answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()

            if not answer:
                answer = "I don't know based on the provided documents."
            if len(answer) > 1600:
                answer = answer[:1600].rstrip() + "..."

            file_chosen = supporting_file if _should_attach_supporting_file(answer) else None
            if file_chosen:
                print(f"[query] selected supporting file={file_chosen}")

            previous_attachment = _find_previously_sent_attachment(history, file_chosen)
            if previous_attachment:
                file_label = _attachment_display_label(previous_attachment, file_chosen)
                print(f"[memory] file was sent before, sending again for current answer: {file_label}")

            return QueryResult(
                answer=answer,
                file=file_chosen,
                classification=classification,
                usage_events=usage_events
                + self._rag_usage_events(retrieval_query, rewrite_usage, embedding_usage)
                + [self._prompt_completion_usage_event(prompt, answer, completion_usage)],
            )

        return QueryResult(
            answer=_out_of_scope_answer(language),
            file=None,
            classification=classification,
            usage_events=usage_events,
        )

    def _retrieval_clarity(
        self,
        query: str,
        language: str,
        intent: str,
        history: List[ConversationMessage],
    ) -> Tuple[RetrievalClarity, object]:
        try:
            return self._gateway.clarify_or_rewrite_query(
                query,
                language,
                intent,
                history=history,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            print("[clarify] clarity gateway failed:", exc)
            return RetrievalClarity(is_clear=True, standalone_query=query), None

    def _load_history(self, conversation_id: Optional[str]) -> List[ConversationMessage]:
        if not conversation_id or self._conversation_memory is None:
            return []

        history = self._conversation_memory.load_messages(conversation_id)
        if history:
            print(
                f"[memory] loaded {len(history)} messages for conversation_id={conversation_id}"
            )
        return history

    @staticmethod
    def _classification_usage_event(
        query: str,
        history: List[ConversationMessage],
        classification: Classification,
        usage,
    ) -> UsageEventRecord:
        if usage is not None:
            return usage_event_from_model_usage("classification", usage)
        return estimate_classification_event(query, history, classification)

    @staticmethod
    def _chat_completion_usage_event(
        query: str,
        history: List[ConversationMessage],
        answer: str,
        usage,
        greeting_mode: bool = False,
    ) -> UsageEventRecord:
        if usage is not None:
            return usage_event_from_model_usage("chat_completion", usage)
        if greeting_mode:
            return estimate_greeting_completion_event(query, history, answer)
        return estimate_factual_completion_event(query, history, answer)

    @staticmethod
    def _prompt_completion_usage_event(prompt: str, answer: str, usage) -> UsageEventRecord:
        if usage is not None:
            return usage_event_from_model_usage("chat_completion", usage)
        return estimate_prompt_completion_event(prompt, answer)

    @staticmethod
    def _rag_usage_events(retrieval_query: str, rewrite_usage, embedding_usage) -> List[UsageEventRecord]:
        events: List[UsageEventRecord] = []
        if rewrite_usage is not None:
            events.append(usage_event_from_model_usage("other", rewrite_usage))
        if embedding_usage is not None:
            events.append(usage_event_from_model_usage("embedding", embedding_usage))
        else:
            events.append(estimate_embedding_event(retrieval_query))
        return events
