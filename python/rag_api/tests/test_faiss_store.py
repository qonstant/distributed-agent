from __future__ import annotations

import unittest

import numpy as np

from rag_service.infrastructure.faiss_store import FaissMetadataStore


class FakeIndex:
    def __init__(self, ids, scores) -> None:
        self.ntotal = len(ids)
        self._ids = ids
        self._scores = scores
        self.search_calls = []

    def search(self, query_embedding, k: int):
        self.search_calls.append(k)
        return (
            np.array([self._scores[:k]], dtype=np.float32),
            np.array([self._ids[:k]], dtype=np.int64),
        )


class FaissMetadataStoreTests(unittest.TestCase):
    def test_search_prioritizes_default_italy_country_and_language(self) -> None:
        meta = {
            "1": {"source_file": "italy/CV_en.pdf", "country": "italy", "language": "en", "page": 1},
            "2": {"source_file": "italy/CV_ru.pdf", "country": "italy", "language": "ru", "page": 1},
            "3": {"source_file": "spain/CV_ru.pdf", "country": "spain", "language": "ru", "page": 1},
            "4": {"source_file": "italy/Visa_en.pdf", "country": "italy", "language": "en", "page": 1},
        }
        index = FakeIndex(ids=[1, 3, 4, 2], scores=[0.99, 0.97, 0.96, 0.60])
        store = FaissMetadataStore(meta=meta, index=index)

        results = store.search(
            np.array([1.0], dtype=np.float32),
            k=2,
            language="ru",
            query_text="cv help",
        )

        self.assertEqual([hit.meta["source_file"] for hit in results], ["italy/CV_ru.pdf", "italy/CV_en.pdf"])
        self.assertEqual(index.search_calls, [4])

    def test_search_uses_single_available_country_when_query_has_no_country(self) -> None:
        meta = {
            "1": {"source_file": "italy/CV_ru.pdf", "country": "italy", "language": "ru", "page": 1},
            "2": {"source_file": "italy/CV_en.pdf", "country": "italy", "language": "en", "page": 1},
        }
        index = FakeIndex(ids=[1, 2], scores=[0.99, 0.50])
        store = FaissMetadataStore(meta=meta, index=index)

        results = store.search(
            np.array([1.0], dtype=np.float32),
            k=1,
            language="en",
            query_text="cv help",
        )

        self.assertEqual(results[0].meta["source_file"], "italy/CV_en.pdf")


if __name__ == "__main__":
    unittest.main()
