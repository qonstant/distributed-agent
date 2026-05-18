from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from rag_service.application.usage_estimation import (
    estimate_classification_event,
    estimate_embedding_event,
    estimate_factual_completion_event,
    estimate_greeting_completion_event,
    estimate_guardrail_event,
    estimate_prompt_completion_event,
    usage_event_from_model_usage,
)
from rag_service.domain.models import (
    Classification,
    ConversationAttachment,
    ConversationMessage,
    GuardrailResult,
    QueryResult,
    RetrievalClarity,
    RetrievalSufficiency,
    RetrievedHit,
    UsageEventRecord,
    normalize_intent,
    normalize_language,
)
from rag_service.infrastructure.prompts import (
    prepare_comparison_prompt,
    prepare_document_request_prompt,
    prepare_factual_rag_prompt,
    prepare_guidance_prompt,
)

if TYPE_CHECKING:
    from rag_service.infrastructure.openai_gateway import OpenAIGateway


RETRIEVAL_INTENTS = {
    "FACTUAL_QUESTION",
    "PROCEDURE",
    "COMPARISON",
    # Legacy labels accepted while older tests/data are migrated.
    "GUIDANCE",
    "DOCUMENT_REQUEST",
}

INTENT_CONFIDENCE_THRESHOLDS = {
    "GREETING": 0.85,
    "CHITCHAT": 0.75,
    "FACTUAL_QUESTION": 0.70,
    "PROCEDURE": 0.70,
    "COMPARISON": 0.75,
    "OUT_OF_DOMAIN": 0.80,
}


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


def _send_pending_attachment_answer(file_source: str, language: str) -> str:
    file_label = _basename(file_source)
    normalized_language = (language or "").strip().lower()

    if normalized_language in {"kk", "kazakh", "қазақ", "қазақша"}:
        return f"Әрине, файлды жібердім: {file_label}."
    if normalized_language in {"ru", "russian", "русский"}:
        return f"Конечно, отправляю файл: {file_label}."
    return f"Sure, here is the file: {file_label}."


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


def _source_files_from_hits(results: List[RetrievedHit]) -> set[str]:
    files: set[str] = set()
    for hit in results:
        source_file = _source_file_from_hit(hit)
        if source_file:
            files.add(source_file)
    return files


def _validated_retrieved_file_choice(value, retrieved_files: set[str]) -> Optional[str]:
    file_choice = _normalize_file_choice(value)
    if not file_choice:
        return None
    if file_choice in retrieved_files:
        return file_choice

    chosen_name = _basename(file_choice)
    for retrieved_file in retrieved_files:
        if _basename(retrieved_file) == chosen_name:
            return retrieved_file
    return None


def _best_hit_for_file(results: List[RetrievedHit], file_chosen: Optional[str]) -> Optional[RetrievedHit]:
    normalized_file = _normalize_file_choice(file_chosen)
    if not normalized_file:
        return None

    chosen_name = _basename(normalized_file)
    matching_hits: List[RetrievedHit] = []
    for hit in results:
        source_file = _source_file_from_hit(hit)
        if not source_file:
            continue
        if source_file == normalized_file or _basename(source_file) == chosen_name:
            matching_hits.append(hit)

    if not matching_hits:
        return None
    return max(matching_hits, key=lambda hit: hit.score)


def _page_reference_from_hit(hit: Optional[RetrievedHit], language: str) -> str:
    if hit is None:
        return ""

    page = str(hit.meta.get("page") or "").strip()
    if not page or page.lower() in {"none", "null", "nil"}:
        return ""

    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return f"Әсіресе қосылған файлдағы {page}-бетті қараңыз: осы жауапқа қатысты ең маңызды ақпарат сол жерде."
    if normalized_language == "ru":
        return f"Особенно проверьте страницу {page} в приложенном файле: там самые релевантные детали по этому ответу."
    return f"Especially check page {page} in the attached file; it has the most relevant details for this answer."


def _append_page_reference(answer: str, hit: Optional[RetrievedHit], language: str) -> str:
    page_reference = _page_reference_from_hit(hit, language)
    if not page_reference:
        return answer
    return f"{answer.rstrip()}\n\n{page_reference}"


def _file_offer_question(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return "Осы ақпарат бар файлды жіберейін бе?"
    if normalized_language == "ru":
        return "Отправить вам файл с этой информацией?"
    return "Should I send you the file with this information?"


def _append_file_offer(answer: str, language: str) -> str:
    return f"{answer.rstrip()}\n\n{_file_offer_question(language)}"


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


def _effective_language(language: str, target_language: str = "") -> str:
    target = (target_language or "").strip()
    if not target:
        return language
    normalized_target = normalize_language(target)
    if normalized_target in {"en", "ru", "kk"}:
        return normalized_target
    return language


def _should_run_retrieval_clarity(intent: str, history: List[ConversationMessage]) -> bool:
    normalized_intent = normalize_intent(intent)
    if normalized_intent in RETRIEVAL_INTENTS or intent in RETRIEVAL_INTENTS:
        return True
    if normalized_intent == "OUT_OF_DOMAIN" and history:
        return True
    return False


def _below_confidence_threshold(intent: str, confidence: float) -> bool:
    if confidence <= 0:
        return False
    return confidence < INTENT_CONFIDENCE_THRESHOLDS.get(normalize_intent(intent), 0.0)


def _fallback_clarifying_question(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return "Қай тақырып бойынша сұрап тұрсыз: университетке түсу, студенттік виза, тұруға рұқсат, DSU шәкіртақысы, құжаттар, дедлайн, оқу ақысы, жатақхана, exchange, CV немесе хаттар?"
    if normalized_language == "ru":
        return "По какой теме вы спрашиваете: поступление, студенческая виза, студенческий ВНЖ, DSU, документы, дедлайны, стоимость обучения, жилье, exchange, CV или письма?"
    return "Which topic do you mean: admission, student visa, residence permit, DSU scholarship, documents, deadlines, tuition, housing, exchange, CV, or letters?"


def _out_of_scope_answer(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return (
            "Мен тек шетелде оқу бойынша сұрақтарға көмектесе аламын: оқуға түсу, "
            "құжаттар, шәкіртақы, дедлайн, оқу ақысы, exchange, студенттік виза, тұруға рұқсат, жатақхана, CV және хаттар."
        )
    if normalized_language == "ru":
        return (
            "Я могу помогать только с вопросами про обучение за рубежом: поступление, "
            "документы, стипендии, дедлайны, стоимость обучения, exchange, студенческую визу, ВНЖ, жилье, CV и письма."
        )
    return (
        "I can help only with education-abroad questions: admission, documents, scholarships, deadlines, tuition, "
        "exchange programs, student visas, residence permits, housing, CVs, and letters."
    )


def _unsafe_request_answer(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return (
            "Мен заңсыз немесе қауіпті әрекеттерге көмектесе алмаймын. "
            "Бірақ оқу, құжаттарды заңды түрде дайындау, студенттік виза, тұруға рұқсат, DSU шәкіртақысы, CV және хаттар бойынша көмектесе аламын."
        )
    if normalized_language == "ru":
        return (
            "Я не могу помогать с незаконными или небезопасными действиями. "
            "Но могу помочь с поступлением, легальной подготовкой документов, студенческой визой, ВНЖ, стипендией DSU, CV и письмами."
        )
    return (
        "I can't help with illegal or unsafe requests. "
        "I can help with admission, legal document preparation, student visas, residence permits, DSU scholarships, CVs, and letters."
    )


def _guardrail_blocked_answer(language: str, violation: str) -> str:
    if (violation or "").strip().lower() == "unsafe":
        return _unsafe_request_answer(language)
    return _out_of_scope_answer(language)


def _fallback_retrieval_follow_up_question(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return "Құжаттардан нақты жауап табу үшін тақырыпты нақтылай аласыз ба: түсу, студенттік виза, тұруға рұқсат, DSU, құжаттар, дедлайн, оқу ақысы, жатақхана, exchange, CV немесе хаттар?"
    if normalized_language == "ru":
        return "Чтобы найти точный ответ в документах, уточните тему: поступление, студенческая виза, ВНЖ, DSU, документы, дедлайны, стоимость обучения, жилье, exchange, CV или письма?"
    return "To find the right answer in the documents, which topic do you mean: admission, student visa, residence permit, DSU, documents, deadlines, tuition, housing, exchange, CV, or letters?"


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

        history: List[ConversationMessage] = []
        history_loaded = False

        def ensure_history_loaded() -> List[ConversationMessage]:
            nonlocal history, history_loaded
            if not history_loaded:
                history = self._load_history(conversation_id)
                history_loaded = True
            return history

        guardrail, guardrail_usage = self._gateway.guard_query(normalized_query, history=None)
        usage_events = [
            self._guardrail_usage_event(
                normalized_query,
                [],
                guardrail,
                guardrail_usage,
            )
        ]

        if guardrail.needs_context:
            contextual_history = ensure_history_loaded()
            if contextual_history:
                guardrail, guardrail_usage = self._gateway.guard_query(
                    normalized_query,
                    history=contextual_history,
                )
                usage_events.append(
                    self._guardrail_usage_event(
                        normalized_query,
                        contextual_history,
                        guardrail,
                        guardrail_usage,
                    )
                )

        guardrail_language = guardrail.language or ""
        print(
            f"[query] guardrail -> allowed={guardrail.allowed} "
            f"needs_context={guardrail.needs_context} violation={guardrail.violation} "
            f"lang={guardrail_language} reason={guardrail.reason}"
        )

        if not guardrail.allowed:
            return QueryResult(
                answer=_guardrail_blocked_answer(guardrail_language, guardrail.violation),
                file=None,
                classification=None,
                usage_events=usage_events,
            )

        history = ensure_history_loaded()
        classification, classification_usage = self._gateway.classify_query(normalized_query, history=history)
        usage_events.append(
            self._classification_usage_event(
                normalized_query,
                history,
                classification,
                classification_usage,
            )
        )
        raw_intent = (classification.intent or "").strip().upper()
        intent = normalize_intent(raw_intent)
        clarity_intent = raw_intent if raw_intent in {"CHIT_CHAT", "GUIDANCE", "DOCUMENT_REQUEST", "OTHER"} else intent
        language = classification.language or guardrail_language
        print(
            f"[query] classifier -> intent={intent} lang={language} "
            f"route={classification.route} confidence={classification.confidence:.2f} "
            f"explain={classification.explain}"
        )

        if classification.profile_action == "set_preferred_name" and (classification.preferred_name or "").strip():
            return QueryResult(
                answer="",
                file=None,
                classification=classification,
                usage_events=usage_events,
            )

        if classification.route == "CLARIFY" or _below_confidence_threshold(intent, classification.confidence):
            return QueryResult(
                answer=_fallback_clarifying_question(language),
                file=None,
                classification=classification,
                usage_events=usage_events,
            )

        pending_file = self._pending_attachment_source(conversation_id)
        latest_attachment = _find_latest_assistant_attachment(history)
        if latest_attachment is not None or pending_file:
            attachment_action, attachment_action_usage = self._gateway.classify_attachment_follow_up(
                normalized_query,
                history=history,
                pending_file=pending_file,
            )
            if attachment_action_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", attachment_action_usage))

            if attachment_action == "send_pending_attachment" and pending_file:
                self._clear_pending_attachment(conversation_id)
                return QueryResult(
                    answer=_send_pending_attachment_answer(pending_file, language),
                    file=pending_file,
                    classification=classification,
                    usage_events=usage_events,
                )

            resend_source = _resend_attachment_source(latest_attachment) if latest_attachment is not None else None
            if attachment_action == "resend_last_attachment" and resend_source is not None:
                return QueryResult(
                    answer=_resend_attachment_answer(latest_attachment, language),
                    file=resend_source,
                    classification=classification,
                    usage_events=usage_events,
                )

        forced_clarity: Optional[RetrievalClarity] = None
        if intent == "CHITCHAT" and history:
            clarity, clarity_usage = self._retrieval_clarity(
                normalized_query,
                language,
                clarity_intent,
                history,
            )
            if clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if clarity.is_retrieval_related:
                if not clarity.is_clear:
                    response_language = _effective_language(language, clarity.target_language)
                    answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                    return QueryResult(
                        answer=answer,
                        file=None,
                        classification=classification,
                        usage_events=usage_events,
                    )
                forced_clarity = clarity
                intent = "PROCEDURE"

        if intent in ("GREETING", "CHITCHAT"):
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
            clarity, clarity_usage = self._retrieval_clarity(
                normalized_query,
                language,
                clarity_intent,
                history,
            )
            if clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if clarity.is_retrieval_related:
                if not clarity.is_clear:
                    response_language = _effective_language(language, clarity.target_language)
                    answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                    return QueryResult(
                        answer=answer,
                        file=None,
                        classification=classification,
                        usage_events=usage_events,
                    )
                forced_clarity = clarity
            else:
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

        if forced_clarity is not None or _should_run_retrieval_clarity(intent, history):
            if forced_clarity is not None:
                clarity = forced_clarity
            else:
                clarity, clarity_usage = self._retrieval_clarity(
                    normalized_query,
                    language,
                    clarity_intent,
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
                response_language = _effective_language(language, clarity.target_language)
                answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                return QueryResult(
                    answer=answer,
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                )

            retrieval_query = (
                (clarity.standalone_query or "").strip()
                or (classification.rewritten_query or "").strip()
                or normalized_query
            )
            if raw_intent == "DOCUMENT_REQUEST":
                retrieval_intent = "DOCUMENT_REQUEST"
            elif raw_intent == "GUIDANCE":
                retrieval_intent = "GUIDANCE"
            else:
                retrieval_intent = intent if intent in RETRIEVAL_INTENTS else "PROCEDURE"
            response_language = _effective_language(language, clarity.target_language)
            response_language_label = _language_label(response_language)
            rewrite_usage = None

            try:
                query_embedding, embedding_usage = self._gateway.embed_text(retrieval_query)
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"embedding failed: {exc}") from exc

            try:
                results = self._store.search(
                    query_embedding,
                    k=max(1, int(raw_k or 64)),
                    language=response_language,
                    query_text=retrieval_query,
                )
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                raise RuntimeError(f"search failed: {exc}") from exc

            top_n = max(1, int(top_for_llm or 8))
            top_chunks = results[:top_n]
            rag_usage_events = self._rag_usage_events(retrieval_query, rewrite_usage, embedding_usage)
            sufficiency, sufficiency_usage = self._retrieval_sufficiency(
                retrieval_query,
                response_language,
                retrieval_intent,
                top_chunks,
                history,
            )
            sufficiency_usage_events = (
                [usage_event_from_model_usage("classification", sufficiency_usage)]
                if sufficiency_usage is not None
                else []
            )

            if not results or not sufficiency.is_sufficient:
                answer = (
                    (sufficiency.clarifying_question or "").strip()
                    or _fallback_retrieval_follow_up_question(response_language)
                )
                return QueryResult(
                    answer=answer,
                    file=None,
                    classification=classification,
                    usage_events=usage_events + rag_usage_events + sufficiency_usage_events,
                )

            best_file_agg, best_chunk = _aggregate_by_file(results)
            supporting_file = _source_file_from_hit(best_chunk) or _normalize_file_choice(best_file_agg)
            retrieved_files = _source_files_from_hits(results)
            llm_file_choice = None

            if retrieval_intent == "DOCUMENT_REQUEST":
                prompt = prepare_document_request_prompt(
                    retrieval_query,
                    top_chunks,
                    history=history,
                    preferred_name=normalized_preferred_name,
                )
            elif retrieval_intent == "COMPARISON":
                prompt = prepare_comparison_prompt(
                    retrieval_query,
                    top_chunks,
                    history=history,
                    preferred_name=normalized_preferred_name,
                )
            elif retrieval_intent == "FACTUAL_QUESTION":
                prompt = prepare_factual_rag_prompt(
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

            if response_language_label:
                prompt = f"Answer in the same language as detected/requested: {response_language_label}\n\n" + prompt
            else:
                prompt = "Answer in the same language as the user's query if possible.\n\n" + prompt

            llm_json, completion_usage = self._gateway.generate_json_response(prompt, max_tokens=512)

            if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
                answer = str(llm_json.get("answer", "")).strip()
                llm_file_choice = _validated_retrieved_file_choice(llm_json.get("file"), retrieved_files)
            else:
                if best_chunk is None:
                    return QueryResult(
                        answer=_fallback_retrieval_follow_up_question(response_language),
                        file=None,
                        classification=classification,
                        usage_events=usage_events + rag_usage_events + sufficiency_usage_events,
                    )
                chunk_meta = best_chunk.meta
                answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()

            if not answer or not _should_attach_supporting_file(answer):
                return QueryResult(
                    answer=_fallback_retrieval_follow_up_question(response_language),
                    file=None,
                    classification=classification,
                    usage_events=usage_events
                    + rag_usage_events
                    + sufficiency_usage_events
                    + [self._prompt_completion_usage_event(prompt, answer, completion_usage)],
                )
            if len(answer) > 1600:
                answer = answer[:1600].rstrip() + "..."

            file_chosen = llm_file_choice or supporting_file
            if file_chosen:
                print(f"[query] selected supporting file={file_chosen}")
                page_hit = _best_hit_for_file(results, file_chosen) or best_chunk
                answer = _append_page_reference(answer, page_hit, response_language)
            response_file = file_chosen
            if retrieval_intent == "FACTUAL_QUESTION":
                response_file = None
                if self._remember_pending_attachment(conversation_id, file_chosen):
                    answer = _append_file_offer(answer, response_language)
            elif file_chosen:
                self._clear_pending_attachment(conversation_id)

            previous_attachment = _find_previously_sent_attachment(history, response_file)
            if previous_attachment:
                file_label = _attachment_display_label(previous_attachment, response_file)
                print(f"[memory] file was sent before, sending again for current answer: {file_label}")

            return QueryResult(
                answer=answer,
                file=response_file,
                classification=classification,
                usage_events=usage_events
                + rag_usage_events
                + sufficiency_usage_events
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

    def _retrieval_sufficiency(
        self,
        query: str,
        language: str,
        intent: str,
        top_chunks: List[RetrievedHit],
        history: List[ConversationMessage],
    ) -> Tuple[RetrievalSufficiency, object]:
        try:
            return self._gateway.assess_retrieval_sufficiency(
                query,
                language,
                intent,
                top_chunks,
                history=history,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            print("[sufficiency] retrieval sufficiency gateway failed:", exc)
            return RetrievalSufficiency(is_sufficient=True, reason=f"sufficiency failed: {exc}"), None

    def _load_history(self, conversation_id: Optional[str]) -> List[ConversationMessage]:
        if not conversation_id or self._conversation_memory is None:
            return []

        history = self._conversation_memory.load_messages(conversation_id)
        if history:
            print(
                f"[memory] loaded {len(history)} messages for conversation_id={conversation_id}"
            )
        return history

    def _pending_attachment_source(self, conversation_id: Optional[str]) -> str:
        if not conversation_id or self._conversation_memory is None:
            return ""
        load_pending = getattr(self._conversation_memory, "load_pending_attachment", None)
        if load_pending is None:
            return ""
        try:
            return str(load_pending(conversation_id) or "").strip()
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[memory] failed to load pending attachment: {exc}")
            return ""

    def _remember_pending_attachment(self, conversation_id: Optional[str], source: Optional[str]) -> bool:
        normalized_source = _normalize_file_choice(source)
        if not conversation_id or not normalized_source or self._conversation_memory is None:
            return False
        remember_pending = getattr(self._conversation_memory, "remember_pending_attachment", None)
        if remember_pending is None:
            return False
        try:
            return bool(remember_pending(conversation_id, normalized_source))
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[memory] failed to remember pending attachment: {exc}")
            return False

    def _clear_pending_attachment(self, conversation_id: Optional[str]) -> None:
        if not conversation_id or self._conversation_memory is None:
            return
        clear_pending = getattr(self._conversation_memory, "clear_pending_attachment", None)
        if clear_pending is None:
            return
        try:
            clear_pending(conversation_id)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[memory] failed to clear pending attachment: {exc}")

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
    def _guardrail_usage_event(
        query: str,
        history: List[ConversationMessage],
        guardrail: GuardrailResult,
        usage,
    ) -> UsageEventRecord:
        if usage is not None:
            return usage_event_from_model_usage("guardrail", usage)
        return estimate_guardrail_event(query, history, guardrail)

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
