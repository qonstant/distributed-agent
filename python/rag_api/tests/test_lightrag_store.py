from __future__ import annotations

import unittest
from pathlib import Path

from rag_service.infrastructure.lightrag_store import LightRAGMetadataStore


class LightRAGMetadataStoreTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
