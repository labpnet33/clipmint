"""Best-effort VRAM handoff between Ollama and Faster-Whisper.

Only ClipMint's selected Ollama model is unloaded. No processes are killed,
no CUDA contexts are force-reset, and failure to query Ollama never stops
transcription or changes ClipMint's existing CPU fallback behavior.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional


def vram_mb() -> Optional[dict]:
    """First GPU's memory in MiB; diagnostics only, never a hard dependency."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode != 0:
            return None
        first = (proc.stdout or "").strip().splitlines()[0].split(",")
        total, used, free = (int(s.strip()) for s in first[:3])
        return {"total_mb": total, "used_mb": used, "free_mb": free}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _memory_note() -> str:
    m = vram_mb()
    return (f"GPU VRAM {m['used_mb']}/{m['total_mb']} MiB used"
            if m else "GPU VRAM measurement unavailable")


def _matches(name: str, expected: str) -> bool:
    name = (name or "").strip().lower()
    expected = (expected or "").strip().lower()
    return bool(expected) and (name == expected or name == expected + ":latest"
                                   or expected == name + ":latest")


def _loaded_model_names() -> list[str]:
    from shorts_generator.local import ollama_api
    data = ollama_api._request("/api/ps", timeout=2.0)
    return [str(row.get("name") or row.get("model") or "")
            for row in (data.get("models") or []) if isinstance(row, dict)]


def _clipmint_ollama_models() -> list[str]:
    """Include per-task Ollama overrides when separate AI routing is installed."""
    from shorts_generator.config import current_model
    names = [current_model("ollama")]
    try:
        from shorts_generator.config import current_stage_provider, current_stage_model
        for stage in ("highlight", "seo"):
            if current_stage_provider(stage) == "ollama":
                names.append(current_stage_model(stage))
    except ImportError:
        pass  # Older ClipMint branch only has LLM_PROVIDER/OLLAMA_MODEL.
    return list(dict.fromkeys(n for n in names if n))


def prepare_cuda_transcription() -> bool:
    """Unload the selected *loaded* Ollama model before CUDA transcription.

    Returns True only when an unload is verified; best effort otherwise.
    Ollama is not stopped, and the model remains installed on disk. Later
    /api/generate requests load it automatically on demand.
    """
    from shorts_generator.local import ollama_api

    print(f"[gpu-handoff] Before Whisper: {_memory_note()}", flush=True)
    try:
        running = _loaded_model_names()
    except Exception as exc:
        print(f"[gpu-handoff] Ollama not reachable; continuing GPU transcription "
              f"({str(exc).splitlines()[0][:100]})", flush=True)
        return False
    configured = _clipmint_ollama_models()
    targets = [name for name in running
               if any(_matches(name, chosen) for chosen in configured)]
    if not targets:
        print("[gpu-handoff] No ClipMint Ollama model loaded; Whisper can start", flush=True)
        return False
    for model in targets:
        try:
            ollama_api._request("/api/generate", {
                "model": model, "prompt": "", "stream": False, "keep_alive": 0,
            }, timeout=12.0)
        except Exception as exc:
            print(f"[gpu-handoff] Could not unload {model}; trying Whisper with "
                  f"existing CPU fallback available ({str(exc).splitlines()[0][:100]})", flush=True)
            return False
    # Ollama may release the GPU allocation just after responding.
    for _ in range(8):
        try:
            remaining = _loaded_model_names()
            if not any(any(_matches(m, target) for m in remaining) for target in targets):
                print(f"[gpu-handoff] Unloaded {', '.join(targets)}; {_memory_note()}", flush=True)
                return True
        except Exception:
            break
        time.sleep(0.25)
    print(f"[gpu-handoff] Ollama unload requested, but VRAM release not verified; "
          f"{_memory_note()}", flush=True)
    return False


def release_whisper_model(model: object, device: str) -> None:
    """Release CTranslate2 weights; caller deletes model and runs gc.collect().

    Do not clear or reset the global CUDA device: FFmpeg/NVENC and unrelated
    applications may share it. Keep this best effort and exception safe.
    """
    if device != "cuda":
        return
    try:
        backend = getattr(model, "model", None)
        unload = getattr(backend, "unload_model", None)
        if callable(unload):
            unload()
            print("[gpu-handoff] Faster-Whisper CTranslate2 weights unloaded", flush=True)
    except Exception as exc:
        print(f"[gpu-handoff] CTranslate2 unload unavailable ({str(exc).splitlines()[0][:90]}); "
              "garbage collection will still run", flush=True)


def whisper_cleanup_done(device: str) -> None:
    if device == "cuda":
        print(f"[gpu-handoff] After Whisper cleanup: {_memory_note()}; "
              "Ollama will load on its next AI request", flush=True)
