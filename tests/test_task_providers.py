"""Task-specific providers: isolated routing and legacy migration tests.

All provider calls are mocked. No model download, no API charges.
"""
import unittest
from unittest.mock import patch
from shorts_generator import config
from shorts_generator.local import task_routing


class TaskProviderTests(unittest.TestCase):
    def test_inherit_legacy_ollama_without_explicit_choices(self):
        def get(k, default=""):
            return {"LLM_PROVIDER": "ollama"}.get(k, default)
        with patch("shorts_generator.user_config.get", side_effect=get):
            self.assertEqual(config.current_stage_provider("highlight"), "ollama")
            self.assertEqual(config.current_stage_provider("seo"), "ollama")
            self.assertEqual(config.current_stage_provider("vision"), "off")

    def test_independent_providers_and_models(self):
        values = {"HIGHLIGHT_PROVIDER": "gemini", "VISION_PROVIDER": "groq",
                  "SEO_PROVIDER": "ollama", "SEO_MODEL": "qwen3:4b"}
        with patch("shorts_generator.user_config.get", side_effect=lambda k, d="": values.get(k, d)):
            self.assertEqual(config.current_stage_provider("highlight"), "gemini")
            self.assertEqual(config.current_stage_provider("vision"), "groq")
            self.assertEqual(config.current_stage_provider("seo"), "ollama")
            self.assertEqual(config.current_stage_model("seo"), "qwen3:4b")

    def test_highlight_uses_only_gemini(self):
        with patch.object(task_routing, "current_stage_provider", return_value="gemini"), \
             patch.object(task_routing, "current_stage_model", return_value="gemini-foo"), \
             patch("shorts_generator.user_config.get", return_value="key"), \
             patch("shorts_generator.local.llm.call_gemini_llm", return_value="ranked") as gemini, \
             patch("shorts_generator.local.ollama_api.generate") as ollama:
            self.assertEqual(task_routing.call_highlight_llm("rank"), "ranked")
            gemini.assert_called_once_with("rank", model_override="gemini-foo")
            ollama.assert_not_called()

    def test_seo_uses_only_ollama(self):
        with patch.object(task_routing, "current_stage_provider", return_value="ollama"), \
             patch.object(task_routing, "current_stage_model", return_value="qwen3:1.7b"), \
             patch("shorts_generator.local.ollama_api.generate", return_value="titles") as ollama, \
             patch("shorts_generator.local.llm.call_gemini_llm") as gemini:
            self.assertEqual(task_routing.call_seo_llm("seo"), "titles")
            ollama.assert_called_once_with("seo", model="qwen3:1.7b")
            gemini.assert_not_called()

    def test_disabled_vision_never_calls_cloud(self):
        with patch.object(task_routing, "current_stage_provider", return_value="off"), \
             patch("shorts_generator.local.llm.call_gemini_llm") as gemini:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                task_routing.call_vision_stage(["test"])
            gemini.assert_not_called()

    def test_vision_routes_to_selected_cloud_model(self):
        with patch.object(task_routing, "current_stage_provider", return_value="groq"), \
             patch.object(task_routing, "current_stage_model", return_value="vision-model"), \
             patch("shorts_generator.user_config.get", return_value="key"), \
             patch("shorts_generator.local.llm._openai_compatible_vision", return_value="frames") as fn:
            self.assertEqual(task_routing.call_vision_stage(["image"]), "frames")
            fn.assert_called_once_with(["image"], "groq", model_override="vision-model")

    def test_unconfigured_cloud_key_does_not_fallback(self):
        with patch.object(task_routing, "current_stage_provider", return_value="gemini"), \
             patch.object(task_routing, "current_stage_model", return_value="gemini-test"), \
             patch("shorts_generator.user_config.get", return_value=""), \
             patch("shorts_generator.local.ollama_api.generate") as ollama:
            with self.assertRaisesRegex(RuntimeError, "no API key"):
                task_routing.call_highlight_llm("hello")
            ollama.assert_not_called()


if __name__ == "__main__":
    unittest.main()
