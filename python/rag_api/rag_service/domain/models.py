from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
class ConversationMessage:
    role: str
    text: str
    ts: int = 0


@dataclass(frozen=True)
class QueryResult:
    answer: str
    file: Optional[str]
