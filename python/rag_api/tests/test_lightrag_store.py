from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rag_service.domain.models import RetrievedHit
from rag_service.infrastructure.lightrag_store import LightRAGMetadataStore, _language_from_source_file, _load_file_catalog


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

    def test_refine_results_focuses_on_best_matching_file_then_pages(self) -> None:
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
            chunks_by_source_file={
                "italy/Visa_ru.pdf": [
                    {"source_file": "italy/Visa_ru.pdf", "page": "2", "text": "Документы на визу и анкета.", "language": "ru", "doc_type": "visa", "country": "italy"},
                    {"source_file": "italy/Visa_ru.pdf", "page": "3", "text": "Подача на визу и фото.", "language": "ru", "doc_type": "visa", "country": "italy"},
                ],
                "italy/Residence_Permit_ru.pdf": [
                    {"source_file": "italy/Residence_Permit_ru.pdf", "page": "1", "text": "Подача на ВНЖ в Италии. Вы должны подать заявление в течение 8 дней после прибытия.", "language": "ru", "doc_type": "residence_permit", "country": "italy"},
                    {"source_file": "italy/Residence_Permit_ru.pdf", "page": "7", "text": "В конверт для ВНЖ нужно положить копию паспорта, копию визы, страховку и приглашение.", "language": "ru", "doc_type": "residence_permit", "country": "italy"},
                ],
            },
            file_catalog={
                "italy/Visa_ru.pdf": {"source_file": "italy/Visa_ru.pdf", "summary": "Подача на студенческую визу в Италию, анкета и фото.", "language": "ru", "doc_type": "visa"},
                "italy/Residence_Permit_ru.pdf": {"source_file": "italy/Residence_Permit_ru.pdf", "summary": "Подача на студенческий ВНЖ в Италии, сроки, этапы и документы.", "language": "ru", "doc_type": "residence_permit"},
            },
        )
        initial_hits = [
            RetrievedHit(score=1.75, nid=1, meta={"source_file": "italy/Visa_ru.pdf", "filename": "italy/Visa_ru.pdf", "page": "3", "text": "Подача на визу."}),
            RetrievedHit(score=0.81, nid=2, meta={"source_file": "italy/Residence_Permit_ru.pdf", "filename": "italy/Residence_Permit_ru.pdf", "page": "10", "text": "Получение ВНЖ."}),
        ]

        refined_hits, refine_meta = store.refine_results(
            query_text="Как подать на студенческий ВНЖ в Италии: процесс и документы.",
            initial_hits=initial_hits,
            k=5,
            language="ru",
        )

        self.assertTrue(refine_meta["focused"])
        self.assertIn("italy/Residence_Permit_ru.pdf", refine_meta["selected_files"])
        self.assertEqual(refined_hits[0].meta["source_file"], "italy/Residence_Permit_ru.pdf")
        self.assertIn(refined_hits[0].meta["page"], {"1", "7"})

    def test_refine_results_can_pull_best_file_from_catalog_even_if_missing_in_initial_hits(self) -> None:
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
            chunks_by_source_file={
                "italy/Visa_ru.pdf": [
                    {"source_file": "italy/Visa_ru.pdf", "page": "2", "text": "Документы на студенческую визу и анкета.", "language": "ru", "doc_type": "visa", "country": "italy"},
                    {"source_file": "italy/Visa_ru.pdf", "page": "3", "text": "Подача на визу через консульство.", "language": "ru", "doc_type": "visa", "country": "italy"},
                ],
                "italy/Application_ru.pdf": [
                    {"source_file": "italy/Application_ru.pdf", "page": "1", "text": "Подача документов в университет и pre-enrollment.", "language": "ru", "doc_type": "application", "country": "italy"},
                ],
                "italy/Residence_Permit_ru.pdf": [
                    {"source_file": "italy/Residence_Permit_ru.pdf", "page": "1", "text": "Подача на ВНЖ в Италии. Вы должны подать заявление в течение 8 дней после прибытия.", "language": "ru", "doc_type": "residence_permit", "country": "italy"},
                    {"source_file": "italy/Residence_Permit_ru.pdf", "page": "7", "text": "В конверт для ВНЖ нужно положить копию паспорта, визы, страховку и приглашение от университета.", "language": "ru", "doc_type": "residence_permit", "country": "italy"},
                ],
            },
            file_catalog={
                "italy/Visa_ru.pdf": {"source_file": "italy/Visa_ru.pdf", "summary": "Студенческая виза в Италию: анкета, фото, подача в консульство.", "language": "ru", "doc_type": "visa"},
                "italy/Application_ru.pdf": {"source_file": "italy/Application_ru.pdf", "summary": "Поступление в университет Италии: pre-enrollment, admission, документы в вуз.", "language": "ru", "doc_type": "application"},
                "italy/Residence_Permit_ru.pdf": {"source_file": "italy/Residence_Permit_ru.pdf", "summary": "Студенческий ВНЖ в Италии: сроки, этапы подачи, документы, Poste Italiane, Questura.", "language": "ru", "doc_type": "residence_permit"},
            },
        )
        initial_hits = [
            RetrievedHit(score=1.75, nid=1, meta={"source_file": "italy/Visa_ru.pdf", "filename": "italy/Visa_ru.pdf", "page": "3", "text": "Подача на визу."}),
            RetrievedHit(score=1.08, nid=2, meta={"source_file": "italy/Visa_ru.pdf", "filename": "italy/Visa_ru.pdf", "page": "2", "text": "Документы на визу."}),
            RetrievedHit(score=0.87, nid=3, meta={"source_file": "italy/Application_ru.pdf", "filename": "italy/Application_ru.pdf", "page": "1", "text": "Документы для поступления."}),
        ]

        refined_hits, refine_meta = store.refine_results(
            query_text="Как подать на студенческий ВНЖ в Италии: процесс и документы.",
            initial_hits=initial_hits,
            k=5,
            language="ru",
        )

        self.assertTrue(refine_meta["focused"])
        self.assertIn("italy/Residence_Permit_ru.pdf", refine_meta["selected_files"])
        self.assertEqual(refined_hits[0].meta["source_file"], "italy/Residence_Permit_ru.pdf")
        self.assertIn(refined_hits[0].meta["page"], {"1", "7"})

    def test_refine_results_keeps_multiple_focus_files_in_second_pass(self) -> None:
        store = LightRAGMetadataStore(
            rag=None,
            working_dir=Path("."),
            mode="naive",
            source_files=[],
            runner=None,
            chunks_by_source_file={
                "italy/Residence_Permit_eng.pdf": [
                    {"source_file": "italy/Residence_Permit_eng.pdf", "page": "1", "text": "Apply for residence permit within 8 days after arrival.", "language": "en", "doc_type": "residence_permit", "country": "italy"},
                ],
                "italy/Visa_en.pdf": [
                    {"source_file": "italy/Visa_en.pdf", "page": "2", "text": "Student visa application form and required documents.", "language": "en", "doc_type": "visa", "country": "italy"},
                    {"source_file": "italy/Visa_en.pdf", "page": "3", "text": "How to apply for an Italian student visa step by step.", "language": "en", "doc_type": "visa", "country": "italy"},
                ],
            },
            file_catalog={
                "italy/Residence_Permit_eng.pdf": {"source_file": "italy/Residence_Permit_eng.pdf", "summary": "Student residence permit in Italy, Questura, Poste Italiane, fingerprints.", "language": "en", "doc_type": "residence_permit"},
                "italy/Visa_en.pdf": {"source_file": "italy/Visa_en.pdf", "summary": "Italian student visa application process, form, documents, photos.", "language": "en", "doc_type": "visa"},
            },
        )
        initial_hits = [
            RetrievedHit(score=1.75, nid=1, meta={"source_file": "italy/Visa_en.pdf", "filename": "italy/Visa_en.pdf", "page": "3", "text": "How to apply for a visa."}),
            RetrievedHit(score=0.81, nid=2, meta={"source_file": "italy/Residence_Permit_eng.pdf", "filename": "italy/Residence_Permit_eng.pdf", "page": "1", "text": "Residence permit process."}),
        ]

        refined_hits, refine_meta = store.refine_results(
            query_text="How to apply for an Italian student visa?",
            initial_hits=initial_hits,
            k=5,
            language="en",
            preferred_source="italy/Residence_Permit_eng.pdf",
        )

        self.assertTrue(refine_meta["focused"])
        top_files = {hit.meta["source_file"] for hit in refined_hits[:2]}
        self.assertIn("italy/Residence_Permit_eng.pdf", top_files)
        self.assertIn("italy/Visa_en.pdf", top_files)

    def test_load_file_catalog_prefers_generated_file_summaries_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            working_dir = Path(tmpdir)
            (working_dir / "file_summaries.json").write_text(
                json.dumps(
                    {
                        "files": {
                            "italy/Residence_Permit_ru.pdf": {
                                "source_file": "italy/Residence_Permit_ru.pdf",
                                "title": "Residence_Permit_ru.pdf",
                                "language": "ru",
                                "doc_type": "residence_permit",
                                "country": "italy",
                                "summary": "Студенческий ВНЖ в Италии: сроки, этапы подачи, документы, Poste Italiane и Questura.",
                                "page_snippets": [{"page": "1", "excerpt": "Подача на ВНЖ в течение 8 дней."}],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (working_dir / "kv_store_doc_status.json").write_text(
                json.dumps(
                    {
                        "doc-1": {
                            "file_path": "italy/Residence_Permit_ru.pdf",
                            "status": "processed",
                            "content_summary": "SOURCE_FILE: italy/Residence_Permit_ru.pdf\n...\n",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            catalog = _load_file_catalog(
                working_dir,
                {
                    "italy/Residence_Permit_ru.pdf": [
                        {
                            "source_file": "italy/Residence_Permit_ru.pdf",
                            "page": "1",
                            "text": "Подача на ВНЖ в течение 8 дней после приезда.",
                            "language": "ru",
                            "doc_type": "residence_permit",
                            "country": "italy",
                        }
                    ]
                },
            )

            self.assertEqual(
                catalog["italy/Residence_Permit_ru.pdf"]["summary"],
                "Студенческий ВНЖ в Италии: сроки, этапы подачи, документы, Poste Italiane и Questura.",
            )
            self.assertEqual(catalog["italy/Residence_Permit_ru.pdf"]["pages"], ["1"])


if __name__ == "__main__":
    unittest.main()
