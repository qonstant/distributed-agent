from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "lightrag_s3_pipeline.py"
SPEC = importlib.util.spec_from_file_location("lightrag_s3_pipeline", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LightRAGS3PipelineTests(unittest.TestCase):
    def test_classify_source_signature_change_detects_additive_update(self) -> None:
        base_signature = [
            {"key": "italy/CV_en.pdf", "etag": "a", "size": 100},
            {"key": "italy/Visa_ru.pdf", "etag": "b", "size": 200},
        ]
        current_signature = [
            {"key": "italy/CV_en.pdf", "etag": "a", "size": 100},
            {"key": "italy/Visa_ru.pdf", "etag": "b", "size": 200},
            {"key": "italy/Residence_Permit_ru.pdf", "etag": "c", "size": 300},
        ]

        change = MODULE.classify_source_signature_change(base_signature, current_signature)
        added = MODULE.source_signature_added_keys(base_signature, current_signature)

        self.assertEqual(change, "additive")
        self.assertEqual(added, ["italy/Residence_Permit_ru.pdf"])

    def test_classify_source_signature_change_detects_incompatible_update(self) -> None:
        base_signature = [
            {"key": "italy/CV_en.pdf", "etag": "a", "size": 100},
        ]
        current_signature = [
            {"key": "italy/CV_en.pdf", "etag": "changed", "size": 100},
            {"key": "italy/Residence_Permit_ru.pdf", "etag": "c", "size": 300},
        ]

        change = MODULE.classify_source_signature_change(base_signature, current_signature)

        self.assertEqual(change, "incompatible")


if __name__ == "__main__":
    unittest.main()
