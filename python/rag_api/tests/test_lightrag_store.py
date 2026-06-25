from __future__ import annotations

import unittest
from pathlib import Path

from rag_service.infrastructure.lightrag_store import LightRAGMetadataStore, _language_from_source_file


class LightRAGMetadataStoreTests(unittest.TestCase):
    def test_language_from_source_file_supports_eng_and_kz_aliases(self) -> None:
        self.assertEqual(_language_from_source_file("italy/Residence_Permit_eng.pdf"), "en")
        self.assertEqual(_language_from_source_file("italy/Residence_Permit_kz.pdf"), "kk")

    def test_hits_from_context_keeps_cross_language_chunks_when_no_same_language_match(self) -> None:
        context = """
{"content":"SOURCE_FILE: italy/Residence_Permit_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 2 -->\\nПодача на ВНЖ в Италии через Poste Italiane и Questura."}
{"content":"SOURCE_FILE: italy/Visa_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 3 -->\\nДокументы на студенческую визу."}
"""
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
        )

        hits = store._hits_from_context(context, k=5, language="kk")

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].meta["source_file"], "italy/Residence_Permit_ru.pdf")
        self.assertEqual(hits[0].meta["language"], "ru")

    def test_hits_from_context_still_prefers_same_language_when_available(self) -> None:
        context = """
{"content":"SOURCE_FILE: italy/Residence_Permit_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 2 -->\\nПодача на ВНЖ в Италии."}
{"content":"SOURCE_FILE: italy/Residence_Permit_en.pdf\\nLANGUAGE: en\\n<!-- PAGE 3 -->\\nHow to apply for residence permit in Italy."}
"""
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
        )

        hits = store._hits_from_context(context, k=5, language="en")

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].meta["source_file"], "italy/Residence_Permit_en.pdf")
        self.assertEqual(hits[0].meta["language"], "en")

    def test_hits_from_context_diversifies_by_file_before_trimming(self) -> None:
        context = """
{"content":"SOURCE_FILE: italy/Visa_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 3 -->\\nВиза шаг 1."}
{"content":"SOURCE_FILE: italy/Visa_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 2 -->\\nВиза шаг 2."}
{"content":"SOURCE_FILE: italy/Application_ru.pdf\\nLANGUAGE: ru\\n<!-- PAGE 1 -->\\nApplication summary."}
{"content":"SOURCE_FILE: italy/Residence_Permit_eng.pdf\\nLANGUAGE: en\\n<!-- PAGE 1 -->\\nResidence permit process."}
"""
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
        )

        hits = store._hits_from_context(context, k=3, language="ru")

        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].meta["source_file"], "italy/Visa_ru.pdf")
        self.assertEqual(hits[1].meta["source_file"], "italy/Application_ru.pdf")
        self.assertEqual(hits[2].meta["source_file"], "italy/Residence_Permit_eng.pdf")


if __name__ == "__main__":
    unittest.main()
