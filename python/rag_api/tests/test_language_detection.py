from __future__ import annotations

import unittest

from rag_service.domain.models import ConversationMessage
from rag_service.infrastructure.language_detection import detect_language


class LanguageDetectionTests(unittest.TestCase):
    def test_detects_english_from_latin_text(self) -> None:
        self.assertEqual(detect_language("How do I renew the permit?"), "en")

    def test_detects_russian_from_cyrillic_text(self) -> None:
        self.assertEqual(detect_language("Как подать заявление на визу?"), "ru")

    def test_detects_kazakh_from_specific_cyrillic_letters(self) -> None:
        self.assertEqual(detect_language("Сәлем, маған көмек керек"), "kk")

    def test_detects_kazakh_from_common_latin_transliteration(self) -> None:
        self.assertEqual(detect_language("salem magan komek kerek"), "kk")

    def test_falls_back_to_user_history_when_query_has_no_signal(self) -> None:
        history = [
            ConversationMessage(role="user", text="Сәлем, маған көмек керек", ts=1),
            ConversationMessage(role="assistant", text="Қалай көмектесе аламын?", ts=2),
        ]
        self.assertEqual(detect_language("?", history=history), "kk")


if __name__ == "__main__":
    unittest.main()
