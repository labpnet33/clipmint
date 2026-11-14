"""Local Ollama integration regression tests (never download models)."""
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from shorts_generator.local import ollama_api
from shorts_generator import seo
from shorts_generator.config import current_model


class OllamaIntegrationTests(unittest.TestCase):
    def test_status_ready(self):
        with patch.object(ollama_api, "_request", return_value={"models": [{"name": "qwen3:1.7b"}]}) as http, \
             patch("shorts_generator.config.current_model", return_value="qwen3:1.7b"):
            result = ollama_api.status()
        self.assertTrue(result["running"])
        self.assertTrue(result["model_installed"])
        http.assert_called_once_with("/api/tags", timeout=2.0)

    def test_status_model_missing(self):
        with patch.object(ollama_api, "_request", return_value={"models": [{"name": "qwen3:4b"}]}), \
             patch("shorts_generator.config.current_model", return_value="qwen3:1.7b"):
            result = ollama_api.status()
        self.assertTrue(result["running"])
        self.assertFalse(result["model_installed"])
        self.assertIn("ollama pull", result["error"])

    def test_status_offline_is_nonfatal(self):
        with patch.object(ollama_api, "_request", side_effect=URLError("not listening")):
            result = ollama_api.status()
        self.assertFalse(result["running"])

    def test_model_config(self):
        with patch("shorts_generator.user_config.get", side_effect=lambda key, default="": default):
            self.assertEqual(current_model("ollama"), "qwen3:1.7b")

    def test_compact_seo_prompt_is_short_and_hinglish(self):
        highlight = {"title": "A joke", "hook_sentence": "Kya hua", "first_line": "Kya hua",
                     "transcript_text": "Ek funny baat " * 5000, "scene": {"genre": "standup"}}
        prompt = seo._ollama_seo_prompt(highlight, {"title": "Standup"}, {}, [])
        self.assertLess(len(prompt), 7000)
        self.assertIn("Hinglish", prompt)
        self.assertIn("description_options", prompt)
        self.assertIn("title_options", prompt)

    def test_generate_json_and_no_cloud_key(self):
        with patch.object(ollama_api, "status", return_value={"running": True, "model_installed": True}), \
             patch.object(ollama_api, "_request", return_value={"response": '{"clips": []}'}) as send, \
             patch("shorts_generator.config.current_model", return_value="qwen3:1.7b"):
            output = ollama_api.generate("hello")
        self.assertEqual(json.loads(output)["clips"], [])
        self.assertEqual(send.call_args.args[0], "/api/generate")
        self.assertFalse(send.call_args.args[1]["stream"])
        self.assertFalse(send.call_args.args[1]["think"])


if __name__ == "__main__":
    unittest.main()
