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
    RetrievedHit,
    UsageEventRecord,
)
from rag_service.infrastructure.prompts import (
    prepare_document_request_prompt,
    prepare_guidance_prompt,
)

if TYPE_CHECKING:
    from rag_service.infrastructure.openai_gateway import OpenAIGateway


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


def _looks_like_resend_request(query: str) -> bool:
    value = (query or "").strip().lower()
    if not value:
        return False

    phrases = (
        "resend",
        "re-send",
        "send it again",
        "send that again",
        "send the file again",
        "send again",
        "one more time",
        "again please",
        "пришли еще раз",
        "отправь еще раз",
        "перешли еще раз",
        "снова отправь",
        "повтори отправку",
    )
    return any(phrase in value for phrase in phrases)


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


def _duplicate_file_note(file_label: str, language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language in {"kk", "kazakh", "қазақ", "қазақша"}:
        return (
            f"Мен бұл файлды осы диалогта бұрын жібергенмін: {file_label}. "
            "Оны чаттың жоғарғы жағынан таба аласыз. Қаласаңыз, оны қайтадан жіберемін."
        )
    if normalized_language in {"ru", "russian", "русский"}:
        return (
            f"Я уже отправлял этот файл ранее в этом диалоге: {file_label}. "
            "Его можно найти выше в переписке. Если хотите, я могу отправить его еще раз."
        )

    return (
        f"I already sent this file earlier in the conversation: {file_label}. "
        "You can find it above in the chat. If you want, I can resend it."
    )


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


def _merge_answer_with_duplicate_note(answer: str, note: str) -> str:
    base = (answer or "").strip()
    if not base:
        return note
    if note in base:
        return base
    return f"{base}\n\n{note}"


def _language_label(language: str) -> str:
    normalized = (language or "").strip().lower()
    mapping = {
        "en": "English",
        "ru": "Russian",
        "kk": "Kazakh",
        "other": "",
    }
    return mapping.get(normalized, (language or "").strip())


class QueryService:
    def __init__(self, gateway: "OpenAIGateway", store, conversation_memory=None) -> None:
        self._gateway = gateway
        self._store = store
        self._conversation_memory = conversation_memory

    def handle_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        raw_k: Optional[int] = None,
        top_for_llm: Optional[int] = None,
    ) -> QueryResult:
        normalized_query = (query or "").strip()
        if not normalized_query:
            raise ValueError("query is empty")

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

        if intent in ("GREETING", "CHIT_CHAT"):
            greeting, completion_usage = self._gateway.generate_greeting_reply(
                normalized_query,
                language,
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

        if intent in ("GUIDANCE", "DOCUMENT_REQUEST"):
            retrieval_query = normalized_query
            rewrite_usage = None
            if history:
                retrieval_query, rewrite_usage = self._gateway.rewrite_query_with_history(
                    normalized_query,
                    history,
                )

            try:
                query_embedding, embedding_usage = self._gateway.embed_text(retrieval_query)
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"embedding failed: {exc}") from exc

            try:
                results = self._store.search(query_embedding, k=max(1, int(raw_k or 64)))
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
            top_n = max(1, int(top_for_llm or 8))
            top_chunks = results[:top_n]

            if intent == "DOCUMENT_REQUEST":
                prompt = prepare_document_request_prompt(normalized_query, top_chunks, history=history)
            else:
                prompt = prepare_guidance_prompt(normalized_query, top_chunks, history=history)

            if language_label:
                prompt = f"Answer in the same language as detected: {language_label}\n\n" + prompt
            else:
                prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

            llm_json, completion_usage = self._gateway.generate_json_response(prompt, max_tokens=512)

            if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
                answer = str(llm_json.get("answer", "")).strip()
                file_chosen = llm_json.get("file")
                if file_chosen is not None:
                    file_chosen = str(file_chosen)
            else:
                if best_chunk is None:
                    return QueryResult(
                        answer="I don't know based on the provided documents.",
                        file=None,
                    )
                chunk_meta = best_chunk.meta
                answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()
                file_chosen = (
                    chunk_meta.get("source_file")
                    or chunk_meta.get("filename")
                    or best_file_agg
                )

            if not answer:
                answer = "I don't know based on the provided documents."
            if len(answer) > 1600:
                answer = answer[:1600].rstrip() + "..."

            previous_attachment = _find_previously_sent_attachment(history, file_chosen)
            if previous_attachment and not _looks_like_resend_request(normalized_query):
                file_label = _attachment_display_label(previous_attachment, file_chosen)
                answer = _merge_answer_with_duplicate_note(
                    answer,
                    _duplicate_file_note(file_label, language),
                )
                print(f"[memory] skipping duplicate attachment resend for file={file_label}")
                file_chosen = None

            return QueryResult(
                answer=answer,
                file=file_chosen,
                classification=classification,
                usage_events=usage_events
                + self._rag_usage_events(retrieval_query, rewrite_usage, embedding_usage)
                + [self._prompt_completion_usage_event(prompt, answer, completion_usage)],
            )

        answer, completion_usage = self._gateway.answer_factual(
            normalized_query,
            language,
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
