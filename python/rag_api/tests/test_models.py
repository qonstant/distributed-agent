from __future__ import annotations

import unittest

from rag_service.domain.models import (
    normalize_attachment_action,
    normalize_language,
    normalize_profile_action,
)


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

    def test_normalize_profile_action_maps_supported_values(self) -> None:
        self.assertEqual(normalize_profile_action("set_preferred_name"), "set_preferred_name")
        self.assertEqual(normalize_profile_action("change_name"), "set_preferred_name")
        self.assertEqual(normalize_profile_action("none"), "")

    def test_normalize_profile_action_maps_unknown_to_empty(self) -> None:
        self.assertEqual(normalize_profile_action("delete_user"), "")

    def test_normalize_attachment_action_maps_supported_values(self) -> None:
        self.assertEqual(normalize_attachment_action("resend_last_attachment"), "resend_last_attachment")
        self.assertEqual(normalize_attachment_action("send_again"), "resend_last_attachment")
        self.assertEqual(normalize_attachment_action("none"), "")

    def test_normalize_attachment_action_maps_unknown_to_empty(self) -> None:
        self.assertEqual(normalize_attachment_action("delete_attachment"), "")


if __name__ == "__main__":
    unittest.main()
