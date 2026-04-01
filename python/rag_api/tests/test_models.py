from __future__ import annotations

import unittest

from rag_service.domain.models import normalize_language


class ModelsTests(unittest.TestCase):
    def test_normalize_language_maps_supported_values(self) -> None:
        self.assertEqual(normalize_language("en"), "en")
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(normalize_language("ru"), "ru")
        self.assertEqual(normalize_language("Russian"), "ru")
        self.assertEqual(normalize_language("kk"), "kk")
        self.assertEqual(normalize_language("Kazakh"), "kk")

    def test_normalize_language_maps_unknown_to_other(self) -> None:
        self.assertEqual(normalize_language("de"), "other")
        self.assertEqual(normalize_language(""), "other")
        self.assertEqual(normalize_language("mixed"), "other")


if __name__ == "__main__":
    unittest.main()
