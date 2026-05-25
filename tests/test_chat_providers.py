import builtins
import os
import types
import unittest
from unittest.mock import patch

from app import (
    CHAT_PROVIDER_FALLBACK,
    CHAT_PROVIDER_GEMINI,
    CHAT_PROVIDER_OPENAI,
    GEMINI_DEFAULT_MODEL,
    GEMINI_QUOTA_FALLBACK_NOTICE,
    call_gemini_for_chat,
    call_llm_for_chat,
    gemini_model_name,
    llm_response_or_fallback,
    selected_chat_provider,
)


class ChatProviderTests(unittest.TestCase):
    def test_gemini_api_key_selects_gemini(self):
        provider = selected_chat_provider({"GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "openai-key"})

        self.assertEqual(provider, CHAT_PROVIDER_GEMINI)

    def test_only_openai_api_key_selects_openai(self):
        provider = selected_chat_provider({"OPENAI_API_KEY": "openai-key"})

        self.assertEqual(provider, CHAT_PROVIDER_OPENAI)

    def test_no_api_keys_selects_fallback(self):
        provider = selected_chat_provider({})

        self.assertEqual(provider, CHAT_PROVIDER_FALLBACK)

    def test_missing_gemini_package_does_not_crash(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "google":
                raise ImportError("google-genai missing")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with patch("builtins.__import__", side_effect=fake_import):
                response = call_llm_for_chat("System prompt", [{"role": "user", "content": "Explain"}])

        self.assertIn("LLM response unavailable", response)
        self.assertIn("google-genai missing", response)

    def test_gemini_model_default_and_override(self):
        self.assertEqual(gemini_model_name({}), GEMINI_DEFAULT_MODEL)
        self.assertEqual(gemini_model_name({"GEMINI_MODEL": "custom-gemini-model"}), "custom-gemini-model")

    def test_gemini_call_uses_model_override(self):
        captured = {}

        class FakeModels:
            def generate_content(self, *, model, contents):
                captured["model"] = model
                captured["contents"] = contents
                return types.SimpleNamespace(text="Gemini answer")

        class FakeClient:
            def __init__(self, api_key):
                captured["api_key"] = api_key
                self.models = FakeModels()

        fake_genai = types.SimpleNamespace(Client=FakeClient)
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key", "GEMINI_MODEL": "override-model"}, clear=True):
            with patch.dict("sys.modules", {"google": fake_google, "google.genai": fake_genai}):
                response = call_gemini_for_chat("System", [{"role": "user", "content": "Question"}])

        self.assertEqual(response, "Gemini answer")
        self.assertEqual(captured["api_key"], "gemini-key")
        self.assertEqual(captured["model"], "override-model")

    def test_gemini_quota_error_returns_clean_notice(self):
        class FakeModels:
            def generate_content(self, *, model, contents):
                raise RuntimeError("429 RESOURCE_EXHAUSTED: raw quota details")

        class FakeClient:
            def __init__(self, api_key):
                self.models = FakeModels()

        fake_genai = types.SimpleNamespace(Client=FakeClient)
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with patch.dict("sys.modules", {"google": fake_google, "google.genai": fake_genai}):
                response = call_gemini_for_chat("System", [{"role": "user", "content": "Question"}])

        self.assertEqual(response, GEMINI_QUOTA_FALLBACK_NOTICE)
        self.assertNotIn("raw quota details", response)

    def test_quota_notice_wraps_deterministic_fallback_without_raw_error(self):
        response = llm_response_or_fallback(GEMINI_QUOTA_FALLBACK_NOTICE, "Deterministic fallback response.")

        self.assertIn(GEMINI_QUOTA_FALLBACK_NOTICE, response)
        self.assertIn("Deterministic fallback response.", response)
        self.assertNotIn("RESOURCE_EXHAUSTED", response)


if __name__ == "__main__":
    unittest.main()
