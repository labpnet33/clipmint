"""ClipMint GPU handoff tests; no GPU, network, Ollama, or video needed."""
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError

from shorts_generator.local import gpu_handoff


class GpuHandoffTests(unittest.TestCase):
    @patch("shorts_generator.config.current_model", return_value="qwen3:1.7b")
    @patch("shorts_generator.local.ollama_api._request")
    @patch.object(gpu_handoff, "vram_mb", return_value=None)
    def test_unloads_only_active_selected_model(self, _vram, request, _model):
        request.side_effect = [
            {"models": [{"name": "qwen3:1.7b"}, {"name": "another:latest"}]},
            {"done": True, "done_reason": "unload"},
            {"models": [{"name": "another:latest"}]},
        ]
        self.assertTrue(gpu_handoff.prepare_cuda_transcription())
        self.assertEqual(request.call_args_list[1].args[0], "/api/generate")
        payload = request.call_args_list[1].args[1]
        self.assertEqual(payload["model"], "qwen3:1.7b")
        self.assertEqual(payload["prompt"], "")
        self.assertEqual(payload["keep_alive"], 0)

    @patch("shorts_generator.local.ollama_api._request")
    @patch.object(gpu_handoff, "_clipmint_ollama_models", return_value=["qwen3:1.7b", "qwen3:4b"])
    @patch.object(gpu_handoff, "vram_mb", return_value=None)
    def test_unloads_both_configured_task_models_not_unrelated(self, _vram, _models, request):
        request.side_effect = [
            {"models": [{"name": "qwen3:1.7b"}, {"name": "qwen3:4b"}, {"name": "llama3.2:latest"}]},
            {"done_reason": "unload"}, {"done_reason": "unload"},
            {"models": [{"name": "llama3.2:latest"}]},
        ]
        self.assertTrue(gpu_handoff.prepare_cuda_transcription())
        self.assertEqual([call.args[1]["model"] for call in request.call_args_list
                          if call.args[0] == "/api/generate"], ["qwen3:1.7b", "qwen3:4b"])

    @patch("shorts_generator.config.current_model", return_value="qwen3:1.7b")
    @patch("shorts_generator.local.ollama_api._request")
    @patch.object(gpu_handoff, "vram_mb", return_value=None)
    def test_does_not_unload_another_users_model(self, _vram, request, _model):
        request.return_value = {"models": [{"name": "llama3.2:latest"}]}
        self.assertFalse(gpu_handoff.prepare_cuda_transcription())
        self.assertEqual(request.call_count, 1)

    @patch("shorts_generator.config.current_model", return_value="qwen3:1.7b")
    @patch("shorts_generator.local.ollama_api._request", side_effect=URLError("offline"))
    @patch.object(gpu_handoff, "vram_mb", return_value=None)
    def test_offline_ollama_does_not_block_whisper(self, *_):
        self.assertFalse(gpu_handoff.prepare_cuda_transcription())

    @patch("shorts_generator.config.current_model", return_value="qwen3:1.7b")
    @patch("shorts_generator.local.ollama_api._request")
    @patch.object(gpu_handoff, "vram_mb", return_value=None)
    def test_failed_unload_is_nonfatal(self, _vram, request, _model):
        request.side_effect = [{"models": [{"name": "qwen3:1.7b"}]}, URLError("busy")]
        self.assertFalse(gpu_handoff.prepare_cuda_transcription())

    def test_whisper_cuda_weights_released_safely(self):
        unload = Mock()
        gpu_handoff.release_whisper_model(Mock(model=Mock(unload_model=unload)), "cuda")
        unload.assert_called_once_with()

    def test_cpu_whisper_does_not_touch_cuda_release(self):
        unload = Mock()
        gpu_handoff.release_whisper_model(Mock(model=Mock(unload_model=unload)), "cpu")
        unload.assert_not_called()

    @patch.object(gpu_handoff.shutil, "which", return_value=None)
    def test_nvidia_smi_missing_is_not_a_failure(self, _):
        self.assertIsNone(gpu_handoff.vram_mb())

    @patch.object(gpu_handoff.shutil, "which", return_value="nvidia-smi")
    @patch.object(gpu_handoff.subprocess, "run")
    def test_gpu_memory_diagnostics(self, run, _):
        run.return_value = Mock(returncode=0, stdout="4096, 1700, 2396\n")
        self.assertEqual(gpu_handoff.vram_mb()["total_mb"], 4096)

    def test_model_name_aliases(self):
        self.assertTrue(gpu_handoff._matches("qwen3:latest", "qwen3"))
        self.assertFalse(gpu_handoff._matches("qwen3:4b", "qwen3:1.7b"))

    def test_transcriber_real_cuda_handoff_and_cleanup(self):
        """Exercise transcribe_local without needing CUDA or a real audio file."""
        from shorts_generator.local import transcriber
        import sys
        from types import SimpleNamespace
        from pathlib import Path
        from unittest.mock import patch
        events = []
        class FakeWhisper:
            def __init__(self, *a, **kw):
                events.append("construct")
                self.model = SimpleNamespace(unload_model=lambda: events.append("unload"))
            def transcribe(self, **kw):
                return iter([SimpleNamespace(start=0.0, end=1.0, text="Hello")]), SimpleNamespace(duration=1.0)
        mocked = SimpleNamespace(WhisperModel=FakeWhisper)
        with patch.dict(sys.modules, {"faster_whisper": mocked}), \
             patch.object(transcriber, "_find_cached_transcript", return_value=None), \
             patch.object(transcriber, "_register_cuda_dlls"), \
             patch.object(transcriber, "_resolve_device", return_value="cuda"), \
             patch.object(transcriber, "_write_srt_cache", return_value=Path("fake.srt")), \
             patch.object(transcriber.proc, "wait_if_paused"), \
             patch.object(gpu_handoff, "prepare_cuda_transcription", side_effect=lambda: events.append("handoff")):
            output = transcriber.transcribe_local("fake.mp4")
        self.assertEqual(["handoff", "construct", "unload"], events)
        self.assertEqual(output["segments"][0]["text"], "Hello")


if __name__ == "__main__":
    unittest.main()
