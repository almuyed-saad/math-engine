import unittest
from unittest.mock import patch

from src.services.ai import _deterministic_fallback, _post_with_retry


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}


class ProviderRetryTests(unittest.TestCase):
    def test_deterministic_fallback_preserves_verified_result(self):
        answer = _deterministic_fallback(
            {"type": "Derivative", "result": "3*x**2", "latex": "3x^2"},
            "provider quota exceeded",
        )
        self.assertIn("SymPy Verified", answer)
        self.assertIn("3*x**2", answer)
        self.assertIn("provider quota exceeded", answer)

    def test_deterministic_fallback_is_empty_without_result(self):
        self.assertEqual(_deterministic_fallback({"type": "general", "result": None}), "")

    @patch("src.services.ai.time.sleep")
    @patch("src.services.ai.requests.post")
    def test_retries_transient_server_failure(self, post, sleep):
        post.side_effect = [FakeResponse(503), FakeResponse(200)]
        response = _post_with_retry(
            "https://provider.example/test",
            headers={},
            json={"prompt": "test"},
            timeout=5,
            attempts=2,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    @patch("src.services.ai.time.sleep")
    @patch("src.services.ai.requests.post")
    def test_retries_timeout_once_then_raises(self, post, sleep):
        import requests

        post.side_effect = requests.exceptions.Timeout("timed out")
        with self.assertRaises(requests.exceptions.Timeout):
            _post_with_retry(
                "https://provider.example/test",
                headers={},
                json={"prompt": "test"},
                timeout=5,
                attempts=2,
            )
        self.assertEqual(post.call_count, 2)
        self.assertEqual(sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
