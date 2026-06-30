from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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

TRACE_LOGGER = logging.getLogger("rag.trace")
TRACE_LOGGER.setLevel(logging.INFO)

INTENT_CONFIDENCE_THRESHOLDS = {
    "GREETING": 0.85,
    "CHITCHAT": 0.75,
    "FACTUAL_QUESTION": 0.70,
    "PROCEDURE": 0.70,
    "COMPARISON": 0.75,
    "OUT_OF_DOMAIN": 0.80,
}

KAZAKH_SPECIFIC_RE = re.compile(r"[әғқңөұүһі]", flags=re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[а-яё]", flags=re.IGNORECASE)
LATIN_RE = re.compile(r"[a-z]", flags=re.IGNORECASE)

_PREFERRED_NAME_UPDATE_PHRASES = (
    "call me ",
    "my name is ",
    "name is now ",
    "зови меня ",
    "назови меня ",
    "называй меня ",
    "меня зовут ",
    "зовут меня ",
    "зовут меня теперь ",
    "мені ",
    "деп ата",
    "деп аташы",
    "деп атай бер",
    "деп атай аласыз",
    "деп атаңыз",
)

_OBVIOUS_IN_SCOPE_EDUCATION_PHRASES = (
    "isee",
    "equivalent isee",
    "dsu",
    "scholarship",
    "financial aid",
    "fee waiver",
    "tuition waiver",
    "study for free",
    "free study",
    "student visa",
    "residence permit",
    "permesso di soggiorno",
    "motivation letter",
    "recommendation letter",
    "universitaly",
    "стипенд",
    "шәкіртақы",
    "грант",
    "виза",
    "внж",
    "вид на жительство",
    "тұруға рұқсат",
    "ықтиярхат",
    "мотивацион",
    "рекомендатель",
)

_OBVIOUS_UNSAFE_PHRASES = (
    "fake",
    "forge",
    "forged",
    "falsify",
    "hack",
    "bypass",
    "evade",
    "lie in",
    "поддел",
    "фальш",
    "взлом",
    "обойти",
    "обман",
)


def _aggregate_by_file(results: List[RetrievedHit]) -> Tuple[Optional[str], Optional[RetrievedHit]]:
    file_sum: dict[str, float] = {}
    best_chunk_for_file: dict[str, RetrievedHit] = {}
    for hit in results:
        source_file = _source_file_from_hit(hit)
        if not source_file:
            continue
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


def _looks_like_preferred_name_update(query: str) -> bool:
    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return False

    for phrase in _PREFERRED_NAME_UPDATE_PHRASES:
        if phrase not in normalized:
            continue
        head, _, tail = normalized.partition(phrase)
        if phrase.startswith("мені ") and not tail:
            continue
        if phrase.startswith("мені ") and " деп ата" not in normalized and " деп аташы" not in normalized:
            continue
        candidate = tail.strip() if tail else head.strip()
        if candidate:
            return True
    return False


def _looks_like_obvious_in_scope_education_query(query: str) -> bool:
    normalized = " ".join((query or "").strip().lower().split())
    if not normalized:
        return False
    if any(marker in normalized for marker in _OBVIOUS_UNSAFE_PHRASES):
        return False
    return any(marker in normalized for marker in _OBVIOUS_IN_SCOPE_EDUCATION_PHRASES)


def _send_pending_attachment_answer(file_source: str, language: str) -> str:
    file_label = _basename(file_source)
    normalized_language = (language or "").strip().lower()

    if normalized_language in {"kk", "kazakh", "қазақ", "қазақша"}:
        return f"Әрине, файлды жібердім: {file_label}."
    if normalized_language in {"ru", "russian", "русский"}:
        return f"Конечно, отправляю файл: {file_label}."
    return f"Sure, here is the file: {file_label}."


_ATTACHMENT_REFERENCE_PHRASES = (
    "this file",
    "that file",
    "the file",
    "this document",
    "that document",
    "the document",
    "this pdf",
    "that pdf",
    "this attachment",
    "that attachment",
    "this guide",
    "that guide",
    "этот файл",
    "этом файле",
    "этого файла",
    "тот файл",
    "том файле",
    "этот документ",
    "этом документе",
    "этого документа",
    "тот документ",
    "том документе",
    "это вложение",
    "этом вложении",
    "мына файл",
    "осы файл",
    "сол файл",
    "мына құжат",
    "осы құжат",
    "сол құжат",
    "мына pdf",
    "осы pdf",
    "сол pdf",
)
_ATTACHMENT_PAGE_WORDS = {
    "page",
    "pages",
    "страница",
    "странице",
    "страницу",
    "страницы",
    "бет",
    "бетте",
    "бетті",
    "беттің",
}
_ATTACHMENT_PUNCT_TRANSLATION = str.maketrans({
    ".": " ",
    ",": " ",
    ":": " ",
    ";": " ",
    "!": " ",
    "?": " ",
    "(": " ",
    ")": " ",
    "[": " ",
    "]": " ",
    "{": " ",
    "}": " ",
    "\"": " ",
    "'": " ",
    "/": " ",
    "\\": " ",
    "-": " ",
    "_": " ",
})
_LANGUAGE_SWITCH_PREFIXES = {
    "en": ("english", "англ", "engl", "eng"),
    "ru": ("russian", "рус", "russ", "rus"),
    "kk": ("kazakh", "qazaq", "қазақ", "қаз", "казах", "каз"),
}
_LANGUAGE_SWITCH_MARKERS = {
    "can",
    "could",
    "please",
    "pls",
    "say",
    "write",
    "answer",
    "reply",
    "version",
    "translate",
    "можно",
    "переведи",
    "перевести",
    "бола",
    "бола ма",
    "жауап",
}


def _query_explicitly_references_attachment(query: str, attachment_source: str = "") -> bool:
    normalized_query = " ".join(
        str(query or "").strip().lower().translate(_ATTACHMENT_PUNCT_TRANSLATION).split()
    )
    if not normalized_query:
        return False

    attachment_name = " ".join(
        _basename(attachment_source).lower().translate(_ATTACHMENT_PUNCT_TRANSLATION).split()
    )
    if attachment_name and attachment_name in normalized_query:
        return True

    if any(phrase in normalized_query for phrase in _ATTACHMENT_REFERENCE_PHRASES):
        return True

    tokens = normalized_query.split()
    for index, token in enumerate(tokens):
        if token not in _ATTACHMENT_PAGE_WORDS:
            continue
        if index + 1 < len(tokens) and tokens[index + 1].isdigit():
            return True
        if index > 0 and tokens[index - 1].isdigit():
            return True

    return False


def _detect_language_switch_follow_up(query: str) -> str:
    normalized_query = " ".join(
        str(query or "").strip().lower().translate(_ATTACHMENT_PUNCT_TRANSLATION).split()
    )
    if not normalized_query:
        return ""

    tokens = normalized_query.split()
    if not tokens or len(tokens) > 8:
        return ""

    detected_target = ""
    for target_language, prefixes in _LANGUAGE_SWITCH_PREFIXES.items():
        if any(token == prefix or token.startswith(prefix) for token in tokens for prefix in prefixes):
            if detected_target and detected_target != target_language:
                return ""
            detected_target = target_language

    if not detected_target:
        return ""

    if len(tokens) <= 3:
        return detected_target

    if any(token in _LANGUAGE_SWITCH_MARKERS for token in tokens):
        return detected_target

    return ""


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
    kind = str(hit.meta.get("kind") or hit.meta.get("type") or "").strip().lower()
    source = str(hit.meta.get("source_file") or hit.meta.get("filename") or "").strip()
    if hit.meta.get("is_faq") or kind == "faq" or source.lower().startswith("faq://"):
        return None
    return _normalize_file_choice(source)


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


def _infer_text_language(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "other"
    if KAZAKH_SPECIFIC_RE.search(normalized):
        return "kk"
    if CYRILLIC_RE.search(normalized):
        return "ru"
    if LATIN_RE.search(normalized):
        return "en"
    return "other"


def _choose_retrieval_query(response_language: str, *candidates: str) -> str:
    non_empty = [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]
    if not non_empty:
        return ""

    target_language = normalize_language(response_language)
    if target_language not in {"en", "ru", "kk"}:
        return non_empty[0]

    for candidate in non_empty:
        if _infer_text_language(candidate) == target_language:
            return candidate
    return non_empty[-1]


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


def _not_enough_reliable_info_answer(language: str) -> str:
    normalized_language = (language or "").strip().lower()
    if normalized_language == "kk":
        return (
            "Қолда бар құжаттардан бұл сұраққа жеткілікті сенімді ақпарат табылмады. "
            "Университеттің, ресми сайттың немесе нақты құжаттың сілтемесін жіберсеңіз, нақтырақ тексеріп беремін."
        )
    if normalized_language == "ru":
        return (
            "В доступных документах не нашлось достаточно надежной информации по этому вопросу. "
            "Пришлите ссылку университета, официальный источник или конкретный документ, и я проверю точнее."
        )
    return (
        "I could not find enough reliable information about this in the available documents. "
        "Please send a university link, official source, or specific document so I can check it more accurately."
    )


class QueryService:
    def __init__(
        self,
        gateway: "OpenAIGateway",
        store,
        conversation_memory=None,
        trace_enabled: bool = True,
        trace_max_chars: int = 240,
        trace_log_format: str = "pretty",
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._conversation_memory = conversation_memory
        self._trace_enabled = trace_enabled
        self._trace_max_chars = max(40, int(trace_max_chars or 240))
        normalized_format = (trace_log_format or "pretty").strip().lower()
        self._trace_log_format = normalized_format if normalized_format in {"pretty", "json"} else "pretty"

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
        trace_id = uuid.uuid4().hex[:12]
        started_at = time.perf_counter()
        language_switch_target = _detect_language_switch_follow_up(normalized_query)

        def trace(stage: str, **fields: Any) -> None:
            self._trace(
                trace_id,
                stage,
                conversation_id=conversation_id,
                **fields,
            )

        def finish(result: QueryResult, outcome: str) -> QueryResult:
            trace(
                "response.final",
                outcome=outcome,
                answer_chars=len(result.answer or ""),
                file=result.file,
                classification_intent=(
                    result.classification.intent if result.classification is not None else ""
                ),
                usage_events=[
                    {
                        "type": item.event_type,
                        "input": item.input_tokens,
                        "output": item.output_tokens,
                        "total": item.total_tokens,
                        "cost": round(item.estimated_cost, 8),
                    }
                    for item in result.usage_events
                ],
                elapsed_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            return result

        trace(
            "request.start",
            query=normalized_query,
            raw_k=raw_k,
            top_for_llm=top_for_llm,
            has_preferred_name=bool(normalized_preferred_name),
        )

        history: List[ConversationMessage] = []
        history_loaded = False

        def ensure_history_loaded() -> List[ConversationMessage]:
            nonlocal history, history_loaded
            if not history_loaded:
                history = self._load_history(conversation_id)
                history_loaded = True
                trace(
                    "history.loaded",
                    count=len(history),
                    messages=self._history_preview(history),
                )
            return history

        if language_switch_target:
            history = ensure_history_loaded()
            trace(
                "language_switch_follow_up",
                target_language=language_switch_target,
                history_count=len(history),
            )

        skip_guardrail_for_profile_update = _looks_like_preferred_name_update(normalized_query)
        skip_guardrail_for_obvious_in_scope = _looks_like_obvious_in_scope_education_query(normalized_query)
        if skip_guardrail_for_profile_update or skip_guardrail_for_obvious_in_scope:
            skip_reason = (
                "explicit preferred-name update phrase"
                if skip_guardrail_for_profile_update
                else "obvious in-scope education-abroad phrase"
            )
            guardrail = GuardrailResult(
                allowed=True,
                reason=skip_reason,
                language=_infer_text_language(normalized_query),
                model="keyword-bypass",
            )
            guardrail_usage = None
            trace("guard.skipped", **self._guardrail_trace(guardrail))
            usage_events = []
        else:
            guardrail, guardrail_usage = self._gateway.guard_query(normalized_query, history=None)
            trace("guard.first", **self._guardrail_trace(guardrail))
            usage_events = [
                self._guardrail_usage_event(
                    normalized_query,
                    [],
                    guardrail,
                    guardrail_usage,
                )
            ]

        if not skip_guardrail_for_profile_update and guardrail.needs_context:
            contextual_history = ensure_history_loaded()
            if contextual_history:
                guardrail, guardrail_usage = self._gateway.guard_query(
                    normalized_query,
                    history=contextual_history,
                )
                trace("guard.context_retry", **self._guardrail_trace(guardrail))
                usage_events.append(
                    self._guardrail_usage_event(
                        normalized_query,
                        contextual_history,
                        guardrail,
                        guardrail_usage,
                    )
                )
        elif not skip_guardrail_for_profile_update and not guardrail.allowed:
            contextual_history = ensure_history_loaded()
            if contextual_history:
                guardrail, guardrail_usage = self._gateway.guard_query(
                    normalized_query,
                    history=contextual_history,
                )
                trace("guard.blocked_retry", **self._guardrail_trace(guardrail))
                usage_events.append(
                    self._guardrail_usage_event(
                        normalized_query,
                        contextual_history,
                        guardrail,
                        guardrail_usage,
                    )
                )

        guardrail_language = guardrail.language or ""
        trace("guard.final", **self._guardrail_trace(guardrail))

        if not guardrail.allowed:
            if language_switch_target and history:
                trace(
                    "guard.language_switch_bypass",
                    target_language=language_switch_target,
                    reason=guardrail.reason,
                )
            else:
                return finish(QueryResult(
                    answer=_guardrail_blocked_answer(guardrail_language, guardrail.violation),
                    file=None,
                    classification=None,
                    usage_events=usage_events,
                ), "guard_blocked")

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
        trace("classifier", **self._classification_trace(classification, normalized_intent=intent))

        forced_clarity: Optional[RetrievalClarity] = None
        skip_classifier_clarify = False

        if language_switch_target and history:
            switch_clarity, switch_clarity_usage = self._retrieval_clarity(
                normalized_query,
                language_switch_target,
                clarity_intent or "OTHER",
                history,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
            trace(
                "language_switch_follow_up.resolved",
                target_language=language_switch_target,
                related=switch_clarity.is_retrieval_related,
                clear=switch_clarity.is_clear,
                query=switch_clarity.standalone_query,
                reason=switch_clarity.reason,
            )
            if switch_clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", switch_clarity_usage))
            if switch_clarity.is_retrieval_related and switch_clarity.is_clear:
                if not switch_clarity.target_language:
                    switch_clarity = RetrievalClarity(
                        is_retrieval_related=switch_clarity.is_retrieval_related,
                        is_clear=switch_clarity.is_clear,
                        standalone_query=switch_clarity.standalone_query,
                        clarifying_question=switch_clarity.clarifying_question,
                        reason=switch_clarity.reason,
                        target_language=language_switch_target,
                    )
                forced_clarity = switch_clarity
                language = _effective_language(language or language_switch_target, switch_clarity.target_language)
                if intent not in RETRIEVAL_INTENTS:
                    intent = "PROCEDURE"
                skip_classifier_clarify = True

        if classification.profile_action == "set_preferred_name" and (classification.preferred_name or "").strip():
            return finish(QueryResult(
                answer="",
                file=None,
                classification=classification,
                usage_events=usage_events,
            ), "profile_update")

        if not skip_classifier_clarify and (classification.route == "CLARIFY" or _below_confidence_threshold(intent, classification.confidence)):
            if history:
                clarity, clarity_usage = self._retrieval_clarity(
                    normalized_query,
                    language,
                    clarity_intent,
                    history,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                )
                if clarity_usage is not None:
                    usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

                if clarity.is_retrieval_related:
                    if not clarity.is_clear:
                        response_language = _effective_language(language, clarity.target_language)
                        answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                        return finish(QueryResult(
                            answer=answer,
                            file=None,
                            classification=classification,
                            usage_events=usage_events,
                        ), "clarity_question_after_classifier")
                    forced_clarity = clarity
                    if intent not in RETRIEVAL_INTENTS:
                        intent = "PROCEDURE"
                else:
                    return finish(QueryResult(
                        answer=_out_of_scope_answer(language),
                        file=None,
                        classification=classification,
                        usage_events=usage_events,
                    ), "clarity_not_retrieval_after_classifier")
            else:
                return finish(QueryResult(
                    answer=_fallback_clarifying_question(language),
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                ), "classifier_clarify_no_history")

        pending_file = self._pending_attachment_source(conversation_id)
        latest_attachment = _find_latest_assistant_attachment(history)
        if latest_attachment is not None or pending_file:
            trace(
                "attachment.context",
                pending_file=pending_file,
                latest_attachment=(
                    (latest_attachment.source or latest_attachment.name)
                    if latest_attachment is not None
                    else ""
                ),
            )
        resend_source = _resend_attachment_source(latest_attachment) if latest_attachment is not None else None
        if latest_attachment is not None or pending_file:
            attachment_action, attachment_action_usage = self._gateway.classify_attachment_follow_up(
                normalized_query,
                history=history,
                pending_file=pending_file,
            )
            trace("attachment.classifier", action=attachment_action)
            if attachment_action_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", attachment_action_usage))

            if attachment_action == "send_pending_attachment" and pending_file:
                self._clear_pending_attachment(conversation_id)
                return finish(QueryResult(
                    answer=_send_pending_attachment_answer(pending_file, language),
                    file=pending_file,
                    classification=classification,
                    usage_events=usage_events,
                ), "send_pending_attachment")

            if attachment_action == "resend_last_attachment" and resend_source is not None:
                return finish(QueryResult(
                    answer=_resend_attachment_answer(latest_attachment, language),
                    file=resend_source,
                    classification=classification,
                    usage_events=usage_events,
                ), "resend_last_attachment")

        if intent == "CHITCHAT" and history:
            clarity, clarity_usage = self._retrieval_clarity(
                normalized_query,
                language,
                clarity_intent,
                history,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
            if clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if clarity.is_retrieval_related:
                if not clarity.is_clear:
                    response_language = _effective_language(language, clarity.target_language)
                    answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                    return finish(QueryResult(
                        answer=answer,
                        file=None,
                        classification=classification,
                        usage_events=usage_events,
                    ), "chitchat_clarity_question")
                forced_clarity = clarity
                intent = "PROCEDURE"

        if intent in ("GREETING", "CHITCHAT"):
            greeting, completion_usage = self._gateway.generate_greeting_reply(
                normalized_query,
                language,
                preferred_name=normalized_preferred_name,
                history=history,
            )
            return finish(QueryResult(
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
            ), "small_reply")

        if intent == "FACTUAL_QUESTION" and forced_clarity is None:
            clarity, clarity_usage = self._retrieval_clarity(
                normalized_query,
                language,
                clarity_intent,
                history,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
            if clarity_usage is not None:
                usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if clarity.is_retrieval_related:
                if not clarity.is_clear:
                    response_language = _effective_language(language, clarity.target_language)
                    answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                    return finish(QueryResult(
                        answer=answer,
                        file=None,
                        classification=classification,
                        usage_events=usage_events,
                    ), "factual_clarity_question")
                forced_clarity = clarity
            else:
                answer, completion_usage = self._gateway.answer_factual(
                    normalized_query,
                    language,
                    preferred_name=normalized_preferred_name,
                    history=history,
                )
                return finish(QueryResult(
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
                ), "non_retrieval_factual")

        if forced_clarity is not None or _should_run_retrieval_clarity(intent, history):
            if forced_clarity is not None:
                clarity = forced_clarity
            else:
                clarity, clarity_usage = self._retrieval_clarity(
                    normalized_query,
                    language,
                    clarity_intent,
                    history,
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                )
                if clarity_usage is not None:
                    usage_events.append(usage_event_from_model_usage("classification", clarity_usage))

            if not clarity.is_retrieval_related:
                return finish(QueryResult(
                    answer=_out_of_scope_answer(language),
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                ), "clarity_not_retrieval")

            if not clarity.is_clear:
                response_language = _effective_language(language, clarity.target_language)
                answer = (clarity.clarifying_question or "").strip() or _fallback_clarifying_question(response_language)
                return finish(QueryResult(
                    answer=answer,
                    file=None,
                    classification=classification,
                    usage_events=usage_events,
                ), "clarity_question")

            if raw_intent == "DOCUMENT_REQUEST":
                retrieval_intent = "DOCUMENT_REQUEST"
            elif raw_intent == "GUIDANCE":
                retrieval_intent = "GUIDANCE"
            else:
                retrieval_intent = intent if intent in RETRIEVAL_INTENTS else "PROCEDURE"
            response_language = _effective_language(language, clarity.target_language)
            retrieval_query = _choose_retrieval_query(
                response_language,
                (clarity.standalone_query or "").strip(),
                (classification.rewritten_query or "").strip(),
                normalized_query,
            )
            response_language_label = _language_label(response_language)
            rewrite_usage = None
            trace(
                "retrieval.start",
                query=retrieval_query,
                retrieval_intent=retrieval_intent,
                response_language=response_language,
                raw_k=raw_k or 64,
                top_for_llm=top_for_llm or 8,
            )

            query_embedding = None
            embedding_usage = None
            if getattr(self._store, "requires_query_embedding", True):
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

            trace(
                "retrieval.raw",
                count=len(results),
                top_hits=self._hits_preview(results[: max(1, min(8, len(results)))]),
            )

            retrieval_anchor_source = None
            anchor_candidate = pending_file or resend_source
            if anchor_candidate and _query_explicitly_references_attachment(normalized_query, anchor_candidate):
                retrieval_anchor_source = anchor_candidate
            if hasattr(self._store, "refine_results"):
                refined_results, refine_meta = self._store.refine_results(
                    query_text=retrieval_query,
                    initial_hits=results,
                    k=max(1, int(raw_k or 64)),
                    language=response_language,
                    preferred_source=retrieval_anchor_source,
                )
                if refined_results:
                    results = refined_results
                    trace(
                        "retrieval.focus",
                        selected_files=(
                            refine_meta.get("selected_files")
                            if isinstance(refine_meta, dict)
                            else []
                        ),
                        focus_files=(
                            refine_meta.get("focus_files")
                            if isinstance(refine_meta, dict)
                            else []
                        ),
                        focused=(
                            refine_meta.get("focused")
                            if isinstance(refine_meta, dict)
                            else None
                        ),
                        count=len(results),
                        top_hits=self._hits_preview(results[: max(1, min(8, len(results)))]),
                    )

            top_n = max(1, int(top_for_llm or 8))
            top_chunks = results[:top_n]
            trace(
                "retrieval.results",
                count=len(results),
                top_chunks=len(top_chunks),
                top_hits=self._hits_preview(top_chunks),
            )
            rag_usage_events = self._rag_usage_events(retrieval_query, rewrite_usage, embedding_usage)
            sufficiency, sufficiency_usage = self._retrieval_sufficiency(
                retrieval_query,
                response_language,
                retrieval_intent,
                top_chunks,
                history,
                trace_id=trace_id,
                conversation_id=conversation_id,
            )
            sufficiency_usage_events = (
                [usage_event_from_model_usage("classification", sufficiency_usage)]
                if sufficiency_usage is not None
                else []
            )

            if not results or not sufficiency.is_sufficient:
                answer = _not_enough_reliable_info_answer(response_language)
                return finish(QueryResult(
                    answer=answer,
                    file=None,
                    classification=classification,
                    usage_events=usage_events + rag_usage_events + sufficiency_usage_events,
                ), "retrieval_weak")

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
            trace(
                "answer.prompt",
                prompt_type=retrieval_intent,
                prompt_chars=len(prompt),
                retrieved_files=sorted(retrieved_files),
                best_supporting_file=supporting_file,
            )

            llm_json, completion_usage = self._gateway.generate_json_response(prompt, max_tokens=512)

            if isinstance(llm_json, dict) and "answer" in llm_json and "file" in llm_json:
                answer = str(llm_json.get("answer", "")).strip()
                llm_file_choice = _validated_retrieved_file_choice(llm_json.get("file"), retrieved_files)
            else:
                if best_chunk is None:
                    return finish(QueryResult(
                        answer=_fallback_retrieval_follow_up_question(response_language),
                        file=None,
                        classification=classification,
                        usage_events=usage_events + rag_usage_events + sufficiency_usage_events,
                    ), "answer_missing_best_chunk")
                chunk_meta = best_chunk.meta
                answer = (chunk_meta.get("text") or chunk_meta.get("md") or "").strip()

            if not answer or not _should_attach_supporting_file(answer):
                return finish(QueryResult(
                    answer=_fallback_retrieval_follow_up_question(response_language),
                    file=None,
                    classification=classification,
                    usage_events=usage_events
                    + rag_usage_events
                    + sufficiency_usage_events
                    + [self._prompt_completion_usage_event(prompt, answer, completion_usage)],
                ), "answer_empty_or_unknown")
            if len(answer) > 1600:
                answer = answer[:1600].rstrip() + "..."

            file_chosen = llm_file_choice or supporting_file
            if file_chosen:
                page_hit = _best_hit_for_file(results, file_chosen) or best_chunk
                trace(
                    "answer.file_selected",
                    file=file_chosen,
                    page=(page_hit.meta.get("page") if page_hit is not None else ""),
                    score=(round(float(page_hit.score), 6) if page_hit is not None else ""),
                    preview=(
                        (page_hit.meta.get("text") or page_hit.meta.get("md") or "")[:220]
                        if page_hit is not None
                        else ""
                    ),
                )
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
                trace("attachment.previously_sent", file=file_label)

            return finish(QueryResult(
                answer=answer,
                file=response_file,
                classification=classification,
                usage_events=usage_events
                + rag_usage_events
                + sufficiency_usage_events
                + [self._prompt_completion_usage_event(prompt, answer, completion_usage)],
            ), "rag_answer")

        return finish(QueryResult(
            answer=_out_of_scope_answer(language),
            file=None,
            classification=classification,
            usage_events=usage_events,
        ), "fallback_out_of_scope")

    def _retrieval_clarity(
        self,
        query: str,
        language: str,
        intent: str,
        history: List[ConversationMessage],
        trace_id: str = "",
        conversation_id: Optional[str] = None,
    ) -> Tuple[RetrievalClarity, object]:
        try:
            clarity, usage = self._gateway.clarify_or_rewrite_query(
                query,
                language,
                intent,
                history=history,
            )
            self._trace(
                trace_id,
                "clarity",
                conversation_id=conversation_id,
                related=clarity.is_retrieval_related,
                clear=clarity.is_clear,
                target_language=clarity.target_language,
                standalone_query=clarity.standalone_query,
                clarifying_question=clarity.clarifying_question,
                reason=clarity.reason,
            )
            return clarity, usage
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._trace(
                trace_id,
                "clarity.error",
                conversation_id=conversation_id,
                error=str(exc),
            )
            return RetrievalClarity(is_clear=True, standalone_query=query), None

    def _retrieval_sufficiency(
        self,
        query: str,
        language: str,
        intent: str,
        top_chunks: List[RetrievedHit],
        history: List[ConversationMessage],
        trace_id: str = "",
        conversation_id: Optional[str] = None,
    ) -> Tuple[RetrievalSufficiency, object]:
        try:
            sufficiency, usage = self._gateway.assess_retrieval_sufficiency(
                query,
                language,
                intent,
                top_chunks,
                history=history,
            )
            self._trace(
                trace_id,
                "sufficiency",
                conversation_id=conversation_id,
                sufficient=sufficiency.is_sufficient,
                clarifying_question=sufficiency.clarifying_question,
                reason=sufficiency.reason,
            )
            return sufficiency, usage
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._trace(
                trace_id,
                "sufficiency.error",
                conversation_id=conversation_id,
                error=str(exc),
            )
            return RetrievalSufficiency(is_sufficient=False, reason=f"sufficiency failed: {exc}"), None

    def _trace(self, trace_id: str, stage: str, **fields: Any) -> None:
        if not self._trace_enabled:
            return
        payload = {
            "event": "rag_trace",
            "trace_id": trace_id or "",
            "stage": stage,
            **fields,
        }
        sanitized = self._sanitize_trace(payload)
        if self._trace_log_format == "json":
            TRACE_LOGGER.info(json.dumps(sanitized, ensure_ascii=False, default=str))
            return
        TRACE_LOGGER.info(self._format_trace_payload(sanitized))

    def _format_trace_payload(self, payload: dict[str, Any]) -> str:
        stage = str(payload.get("stage") or "trace")
        trace_id = str(payload.get("trace_id") or "-")
        prefix = f"[rag][{trace_id}] {stage}"

        if stage == "request.start":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("conv", self._short_conversation_id(payload.get("conversation_id"))),
                    self._trace_field("q", payload.get("query"), quoted=True),
                    self._trace_field("k", payload.get("raw_k")),
                    self._trace_field("llm", payload.get("top_for_llm")),
                    self._trace_field("name", self._yes_no(payload.get("has_preferred_name"))),
                ]
                if part
            )

        if stage.startswith("guard."):
            if payload.get("needs_context"):
                status = "NEEDS_CTX"
            else:
                status = "PASS" if payload.get("allowed") else "BLOCK"
            return " ".join(
                part
                for part in [
                    prefix,
                    status,
                    self._trace_field("lang", payload.get("language")),
                    self._trace_field("violation", payload.get("violation")),
                    self._trace_field("reason", payload.get("reason"), quoted=True),
                ]
                if part
            )

        if stage == "history.loaded":
            messages = self._format_history_messages(payload.get("messages"))
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("count", payload.get("count")),
                    self._trace_field("latest", messages, quoted=True),
                ]
                if part
            )

        if stage == "classifier":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("intent", payload.get("intent")),
                    self._trace_field("route", payload.get("route")),
                    self._trace_field("conf", payload.get("confidence")),
                    self._trace_field("rag", self._yes_no(payload.get("needs_rag"))),
                    self._trace_field("lang", payload.get("language")),
                    self._trace_field("rewrite", payload.get("rewritten_query"), quoted=True),
                    self._trace_field("reason", payload.get("reason"), quoted=True),
                ]
                if part
            )

        if stage == "clarity":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("related", self._yes_no(payload.get("related"))),
                    self._trace_field("clear", self._yes_no(payload.get("clear"))),
                    self._trace_field("lang", payload.get("target_language")),
                    self._trace_field("query", payload.get("standalone_query"), quoted=True),
                    self._trace_field("ask", payload.get("clarifying_question"), quoted=True),
                    self._trace_field("reason", payload.get("reason"), quoted=True),
                ]
                if part
            )

        if stage == "retrieval.start":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("intent", payload.get("retrieval_intent")),
                    self._trace_field("lang", payload.get("response_language")),
                    self._trace_field("q", payload.get("query"), quoted=True),
                    self._trace_field("k", payload.get("raw_k")),
                    self._trace_field("llm", payload.get("top_for_llm")),
                ]
                if part
            )

        if stage == "retrieval.raw":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("hits", payload.get("count")),
                    self._trace_field("files", self._format_top_hits(payload.get("top_hits")), quoted=True),
                ]
                if part
            )

        if stage == "retrieval.focus":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("selected", self._format_file_list(payload.get("selected_files")), quoted=True),
                    self._trace_field("focus", self._format_file_list(payload.get("focus_files")), quoted=True),
                    self._trace_field("focused", self._yes_no(payload.get("focused"))),
                    self._trace_field("count", payload.get("count")),
                    self._trace_field("files", self._format_top_hits(payload.get("top_hits")), quoted=True),
                ]
                if part
            )

        if stage == "retrieval.results":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("hits", payload.get("count")),
                    self._trace_field("top", payload.get("top_chunks")),
                    self._trace_field("files", self._format_top_hits(payload.get("top_hits")), quoted=True),
                ]
                if part
            )

        if stage == "sufficiency":
            status = "ENOUGH" if payload.get("sufficient") else "WEAK"
            return " ".join(
                part
                for part in [
                    prefix,
                    status,
                    self._trace_field("ask", payload.get("clarifying_question"), quoted=True),
                    self._trace_field("reason", payload.get("reason"), quoted=True),
                ]
                if part
            )

        if stage == "answer.prompt":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("type", payload.get("prompt_type")),
                    self._trace_field("chars", payload.get("prompt_chars")),
                    self._trace_field("best", payload.get("best_supporting_file")),
                    self._trace_field("files", self._count_items(payload.get("retrieved_files"))),
                ]
                if part
            )

        if stage == "answer.file_selected":
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("file", payload.get("file")),
                    self._trace_field("page", payload.get("page")),
                    self._trace_field("score", payload.get("score")),
                    self._trace_field("preview", payload.get("preview"), quoted=True),
                ]
                if part
            )

        if stage == "response.final":
            usage = self._format_usage_events(payload.get("usage_events"))
            return " ".join(
                part
                for part in [
                    prefix,
                    self._trace_field("outcome", payload.get("outcome")),
                    self._trace_field("intent", payload.get("classification_intent")),
                    self._trace_field("file", payload.get("file")),
                    self._trace_field("answer_chars", payload.get("answer_chars")),
                    self._trace_field("elapsed", self._format_elapsed(payload.get("elapsed_ms"))),
                    usage,
                ]
                if part
            )

        return self._format_generic_trace(prefix, payload)

    def _format_generic_trace(self, prefix: str, payload: dict[str, Any]) -> str:
        ignored = {"event", "trace_id", "stage", "conversation_id"}
        parts = [prefix]
        for key, value in payload.items():
            if key in ignored or value in ("", None, [], {}):
                continue
            if isinstance(value, (list, dict)):
                parts.append(self._trace_field(key, self._count_items(value)))
            else:
                parts.append(self._trace_field(key, value, quoted=isinstance(value, str)))
        return " ".join(part for part in parts if part)

    def _format_history_messages(self, messages: Any) -> str:
        if not isinstance(messages, list) or not messages:
            return ""
        items = []
        for message in messages[-4:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "?")[:1].lower()
            text = self._compact_trace_text(message.get("text"), limit=90)
            items.append(f"{role}:{text}")
        return " | ".join(items)

    def _format_top_hits(self, hits: Any) -> str:
        if not isinstance(hits, list) or not hits:
            return ""
        items = []
        for index, hit in enumerate(hits[:5], start=1):
            if not isinstance(hit, dict):
                continue
            source = _basename(str(hit.get("source_file") or "unknown"))
            page = hit.get("page")
            score = hit.get("score")
            page_part = f":p{page}" if page not in ("", None) else ""
            score_part = f"{float(score):.3f}" if isinstance(score, (int, float)) else str(score or "")
            items.append(f"#{index} {score_part} {source}{page_part}".strip())
        return "; ".join(items)

    def _format_file_list(self, files: Any) -> str:
        if not isinstance(files, list) or not files:
            return ""
        items = []
        for source_file in files[:6]:
            text = str(source_file or "").strip()
            if not text:
                continue
            items.append(_basename(text))
        return ", ".join(items)

    def _format_usage_events(self, events: Any) -> str:
        if not isinstance(events, list) or not events:
            return ""
        total_tokens = 0
        total_cost = 0.0
        counts: dict[str, int] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or event.get("type") or "usage")
            counts[event_type] = counts.get(event_type, 0) + 1
            total_tokens += int(event.get("total_tokens") or event.get("total") or 0)
            total_cost += float(event.get("estimated_cost") or event.get("cost") or 0.0)
        labels = ",".join(f"{key}x{value}" for key, value in sorted(counts.items()))
        parts = [
            self._trace_field("usage", labels),
            self._trace_field("tokens", total_tokens or None),
            self._trace_field("cost", f"${total_cost:.6f}" if total_cost > 0 else ""),
        ]
        return " ".join(part for part in parts if part)

    def _trace_field(self, key: str, value: Any, *, quoted: bool = False) -> str:
        if value in ("", None, [], {}):
            return ""
        if quoted:
            return f'{key}="{self._compact_trace_text(value)}"'
        return f"{key}={value}"

    @staticmethod
    def _compact_trace_text(value: Any, *, limit: int = 240) -> str:
        text = str(value or "").replace('"', "'").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @staticmethod
    def _yes_no(value: Any) -> str:
        if value is None or value == "":
            return ""
        return "yes" if bool(value) else "no"

    @staticmethod
    def _short_conversation_id(value: Any) -> str:
        text = str(value or "")
        if not text:
            return ""
        return text if len(text) <= 14 else "..." + text[-12:]

    @staticmethod
    def _format_elapsed(value: Any) -> str:
        if value in ("", None):
            return ""
        try:
            return f"{float(value):.0f}ms"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _count_items(value: Any) -> Any:
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        return value

    def _sanitize_trace(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._sanitize_trace(asdict(value))
        if isinstance(value, str):
            normalized = value.replace("\n", "\\n").strip()
            if len(normalized) > self._trace_max_chars:
                return normalized[: self._trace_max_chars].rstrip() + "..."
            return normalized
        if isinstance(value, dict):
            return {
                str(key): self._sanitize_trace(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize_trace(item) for item in value]
        return value

    def _history_preview(self, history: List[ConversationMessage]) -> List[dict[str, Any]]:
        return [
            {
                "role": message.role,
                "text": message.text,
                "attachments": [
                    attachment.source or attachment.name
                    for attachment in message.attachments
                    if (attachment.source or attachment.name)
                ],
            }
            for message in history
        ]

    @staticmethod
    def _guardrail_trace(guardrail: GuardrailResult) -> dict[str, Any]:
        return {
            "allowed": guardrail.allowed,
            "needs_context": guardrail.needs_context,
            "violation": guardrail.violation,
            "language": guardrail.language,
            "reason": guardrail.reason,
        }

    @staticmethod
    def _classification_trace(
        classification: Classification,
        normalized_intent: str,
    ) -> dict[str, Any]:
        return {
            "intent": normalized_intent,
            "raw_intent": classification.intent,
            "route": classification.route,
            "confidence": round(classification.confidence, 4),
            "needs_rag": classification.needs_rag,
            "language": classification.language,
            "rewritten_query": classification.rewritten_query,
            "profile_action": classification.profile_action,
            "reason": classification.explain,
        }

    @staticmethod
    def _hits_preview(hits: List[RetrievedHit]) -> List[dict[str, Any]]:
        preview = []
        for hit in hits:
            meta = hit.meta or {}
            preview.append(
                {
                    "score": round(float(hit.score), 6),
                    "source_file": meta.get("source_file") or meta.get("filename") or "",
                    "page": meta.get("page"),
                    "text": (meta.get("text") or meta.get("md") or "")[:180],
                }
            )
        return preview

    def _load_history(self, conversation_id: Optional[str]) -> List[ConversationMessage]:
        if not conversation_id or self._conversation_memory is None:
            return []

        history = self._conversation_memory.load_messages(conversation_id)
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
