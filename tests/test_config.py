import os
import unittest
from unittest.mock import patch

from src.config import load_settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_safe(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()
        self.assertEqual(settings.provider_timeout_seconds, 60.0)
        self.assertEqual(settings.max_upload_bytes, 5 * 1024 * 1024)
        self.assertEqual(settings.max_pdf_pages, 6)
        self.assertFalse(settings.supabase_enabled)

    def test_limits_are_clamped(self):
        with patch.dict(
            os.environ,
            {
                "PROVIDER_TIMEOUT_SECONDS": "9999",
                "MAX_UPLOAD_BYTES": "999999999",
                "MAX_PDF_PAGES": "999",
            },
            clear=True,
        ):
            settings = load_settings()
        self.assertEqual(settings.provider_timeout_seconds, 120.0)
        self.assertEqual(settings.max_upload_bytes, 25 * 1024 * 1024)
        self.assertEqual(settings.max_pdf_pages, 20)

    def test_numbered_provider_keys_are_loaded_in_order(self):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY_1": "first",
                "GROQ_API_KEY_3": "third",
                "GEMINI_API_KEY_2": "gemini-second",
            },
            clear=True,
        ):
            settings = load_settings()
        self.assertEqual(settings.groq_api_keys, ("first", "", "third"))
        self.assertEqual(settings.gemini_api_keys[1], "gemini-second")


if __name__ == "__main__":
    unittest.main()
