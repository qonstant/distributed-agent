from __future__ import annotations

import re
from typing import Iterable, Optional

from rag_service.domain.models import ConversationMessage

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_KAZAKH_SPECIFIC_CHARS = set("әғқңөұүһі")
_KAZAKH_HINT_WORDS = {
    "сәлем",
    "рахмет",
    "қалай",
    "маған",
    "мағана",
    "мағаның",
    "мағаны",
    "мағаның",
    "мағаны",
    "менің",
    "сенің",
    "сіздің",
    "үшін",
    "керек",
    "жоқ",
    "бар",
    "өтініш",
    "жібер",
    "қайта",
    "qalai",
    "salem",
    "rahmet",
    "magan",
    "menin",
    "ushin",
    "kerek",
    "zhok",
    "bar",
    "otinish",
    "jiber",
    "qaita",
}


def detect_language(
    text: str,
    history: Optional[list[ConversationMessage]] = None,
) -> str:
    detected = _detect_from_text(text)
    if detected:
        return detected

    if not history:
        return ""

    history_text = " ".join(_recent_user_texts(history))
    return _detect_from_text(history_text)


def _detect_from_text(text: str) -> str:
    tokens = [token.lower() for token in _WORD_RE.findall(text or "")]
    if not tokens:
        return ""

    flattened = "".join(tokens)
    if any(char in _KAZAKH_SPECIFIC_CHARS for char in flattened):
        return "kk"

    if any(token in _KAZAKH_HINT_WORDS for token in tokens):
        return "kk"

    latin_letters = sum(1 for char in flattened if char.isascii() and char.isalpha())
    cyrillic_letters = sum(1 for char in flattened if "\u0400" <= char <= "\u04FF")

    if cyrillic_letters and not latin_letters:
        return "ru"
    if latin_letters and not cyrillic_letters:
        return "en"
    if cyrillic_letters or latin_letters:
        return "ru" if cyrillic_letters >= latin_letters else "en"

    return ""


def _recent_user_texts(history: list[ConversationMessage]) -> Iterable[str]:
    for message in reversed(history):
        if message.role != "user":
            continue
        text = (message.text or "").strip()
        if text:
            yield text

