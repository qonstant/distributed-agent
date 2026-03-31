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


def normalize_intent(intent: str) -> str:
    value = (intent or "").strip().upper()
    if value not in SUPPORTED_INTENTS:
        return "OTHER"
    return value


@dataclass(frozen=True)
class Classification:
    intent: str
    explain: str = ""
    language: str = ""


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
