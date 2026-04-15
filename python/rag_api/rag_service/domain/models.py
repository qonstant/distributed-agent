from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SUPPORTED_INTENTS = {
    "GREETING",
    "CHIT_CHAT",
    "FACTUAL_QUESTION",
    "GUIDANCE",
    "DOCUMENT_REQUEST",
    "OTHER",
}

SUPPORTED_LANGUAGES = {
    "kk",
    "ru",
    "en",
    "other",
}

SUPPORTED_PROFILE_ACTIONS = {
    "",
    "set_preferred_name",
}

SUPPORTED_ATTACHMENT_ACTIONS = {
    "",
    "resend_last_attachment",
}


def normalize_intent(intent: str) -> str:
    value = (intent or "").strip().upper()
    if value not in SUPPORTED_INTENTS:
        return "OTHER"
    return value


def normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    mapping = {
        "kk": "kk",
        "kazakh": "kk",
        "қазақ": "kk",
        "қазақша": "kk",
        "kaz": "kk",
        "kz": "kk",
        "ru": "ru",
        "russian": "ru",
        "русский": "ru",
        "рус": "ru",
        "en": "en",
        "english": "en",
        "английский": "en",
        "eng": "en",
        "other": "other",
        "unknown": "other",
        "mixed": "other",
        "und": "other",
        "": "other",
    }
    normalized = mapping.get(value, value)
    if normalized not in SUPPORTED_LANGUAGES:
        return "other"
    return normalized


def normalize_profile_action(action: str) -> str:
    value = (action or "").strip().lower()
    mapping = {
        "": "",
        "none": "",
        "set_preferred_name": "set_preferred_name",
        "set-name": "set_preferred_name",
        "set_name": "set_preferred_name",
        "rename_user": "set_preferred_name",
        "change_name": "set_preferred_name",
    }
    normalized = mapping.get(value, value)
    if normalized not in SUPPORTED_PROFILE_ACTIONS:
        return ""
    return normalized


def normalize_attachment_action(action: str) -> str:
    value = (action or "").strip().lower()
    mapping = {
        "": "",
        "none": "",
        "resend_last_attachment": "resend_last_attachment",
        "resend_attachment": "resend_last_attachment",
        "resend_last_file": "resend_last_attachment",
        "send_again": "resend_last_attachment",
    }
    normalized = mapping.get(value, value)
    if normalized not in SUPPORTED_ATTACHMENT_ACTIONS:
        return ""
    return normalized


@dataclass(frozen=True)
class Classification:
    intent: str
    explain: str = ""
    language: str = ""
    model: str = ""
    version: str = ""
    profile_action: str = ""
    preferred_name: str = ""


@dataclass(frozen=True)
class RetrievalClarity:
    is_clear: bool
    standalone_query: str = ""
    clarifying_question: str = ""
    reason: str = ""
    is_retrieval_related: bool = True


@dataclass(frozen=True)
class RetrievalSufficiency:
    is_sufficient: bool
    clarifying_question: str = ""
    reason: str = ""


@dataclass(frozen=True)
class UsageEventRecord:
    event_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass(frozen=True)
class ModelUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class RetrievedHit:
    score: float
    nid: int
    meta: Dict[str, Any]


@dataclass(frozen=True)
class ConversationAttachment:
    name: str
    kind: str
    source: str = ""


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    text: str
    ts: int = 0
    attachments: List[ConversationAttachment] = field(default_factory=list)


@dataclass(frozen=True)
class QueryResult:
    answer: str
    file: Optional[str]
    classification: Optional[Classification] = None
    usage_events: List[UsageEventRecord] = field(default_factory=list)
