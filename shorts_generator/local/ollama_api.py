"""Local-only Ollama HTTP client: no API key and no extra dependencies."""
from __future__ import annotations

import json
import os
from urllib import error, parse, request


def _base_url() -> str:
    from shorts_generator import user_config
    url = (user_config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
           or "http://127.0.0.1:11434").rstrip("/")
    parsed = parse.urlsplit(url)
    # Do not let a 'local provider' silently upload transcripts to remote hosts.
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password:
        raise ValueError("OLLAMA_BASE_URL must use http://127.0.0.1 or localhost (local only).")
    if not parsed.port or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_BASE_URL must be a local host and port (e.g. http://127.0.0.1:11434).")
    return url


def _request(endpoint: str, payload: dict | None = None, timeout: int | float = 2) -> dict:
    url = _base_url() + endpoint
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"},
                          method="POST" if payload is not None else "GET")
    # Do not let a custom HTTP proxy intercept sensitive local transcripts.
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as res:
        data = json.loads(res.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Ollama returned an unexpected API response")
    return data


def status(timeout: float = 2.0, model: str | None = None) -> dict:
    """Quick startup check. Verifies server AND installed model, not whether loaded."""
    from shorts_generator.config import current_model
    model = model or current_model("ollama")
    try:
        data = _request("/api/tags", timeout=timeout)
        models = sorted({str(x.get("name") or x.get("model") or "")
                         for x in (data.get("models") or []) if isinstance(x, dict)})
        found = model in models or (":" not in model and model + ":latest" in models)
        return {"running": True, "model_installed": found, "model": model,
                "models": models, "error": "" if found else f"Run: ollama pull {model}"}
    except (OSError, TimeoutError, ValueError, RuntimeError, error.URLError) as exc:
        return {"running": False, "model_installed": False, "model": model,
                "models": [], "error": str(exc).splitlines()[0][:240]}


from . import ai_activity  # ClipMint AI Activity Dashboard v1


@ai_activity.trace_provider("ollama")
def generate(prompt: str, model: str | None = None) -> str:
    """Text generation via Ollama /api/generate using 4K context and JSON mode.

    SEO uses short per-clip prompts. For older/general ClipMint prompts, retain
    beginning and ending rather than sending a >4K-context input unchanged.
    """
    from shorts_generator.config import current_model
    from shorts_generator import user_config
    model = model or current_model("ollama")
    state = status(timeout=2, model=model)
    if not state["running"]:
        raise RuntimeError("Ollama is not running. Open Ollama or run 'ollama serve', then retry.")
    if not state["model_installed"]:
        raise RuntimeError(state["error"])
    # Mixed text scripts can consume more than one token per character.
    # With a 4096-token context, this is an upper bound, not a guarantee.
    if len(prompt) > 9000:
        prompt = prompt[:2000] + "\n[Long middle context shortened for local model]\n" + prompt[-5900:]
        print("[ollama] shortened a long prompt for local context", flush=True)
    try:
        result = _request("/api/generate", {
            "model": model, "prompt": prompt, "stream": False, "think": False,
            "format": "json", "keep_alive": "5m",
            "options": {"num_ctx": 4096, "num_predict": 1000, "temperature": 0.55},
        }, timeout=180)
    except (OSError, TimeoutError, ValueError, RuntimeError, error.URLError) as exc:
        raise RuntimeError(f"Ollama generation failed: {exc}") from exc
    answer = (result.get("response") or "").strip()
    if not answer:
        raise RuntimeError("Ollama returned no text. Try a smaller model or shorter prompt.")
    return answer
