"""Explicit task-specific AI routing. No silent cross-task/cloud failover.

Only selected providers are called. In particular, when a task uses Ollama,
failures never spend Gemini, Groq, OpenRouter or OpenAI credits automatically.
"""
from __future__ import annotations

from shorts_generator import user_config
from shorts_generator.config import current_stage_provider, current_stage_model

PROVIDERS = ("ollama", "gemini", "groq", "openrouter", "openai")
VISION_PROVIDERS = ("off", "gemini", "groq", "openrouter", "openai")
KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}


def readiness(stage: str) -> dict:
    """Local config check; cloud connectivity/remaining quota is not implied."""
    provider = current_stage_provider(stage)
    model = current_stage_model(stage)
    if provider == "off":
        return {"provider": provider, "model": "", "ready": True,
                "message": "Vision intentionally disabled"}
    if provider == "ollama":
        from shorts_generator.local.ollama_api import status
        report = status(timeout=1.5, model=model)
        ready = bool(report["running"] and report["model_installed"])
        message = "Local Ollama model installed" if ready else report["error"]
        return {"provider": provider, "model": model, "ready": ready,
                "message": message}
    key = KEYS.get(provider)
    ready = bool(key and user_config.get(key))
    return {"provider": provider, "model": model, "ready": ready,
            "message": "API key configured; not live-tested" if ready
                       else f"Set {key or 'a supported provider'} in Settings first"}


def call_stage_text(stage: str, prompt: str) -> str:
    if stage not in ("highlight", "seo"):
        raise ValueError("Text AI stage must be highlight or seo")
    provider = current_stage_provider(stage)
    model = current_stage_model(stage)
    if provider == "ollama":
        from shorts_generator.local.ollama_api import generate
        return generate(prompt, model=model)
    if not user_config.get(KEYS.get(provider, "")):
        raise RuntimeError(f"{stage} AI provider {provider} has no API key configured")
    from shorts_generator.local import llm
    if provider == "gemini":
        return llm.call_gemini_llm(prompt, model_override=model)
    if provider == "groq":
        return llm.call_groq_llm(prompt, model_override=model)
    if provider == "openrouter":
        return llm.call_openrouter_llm(prompt, model_override=model)
    if provider == "openai":
        return llm.call_openai_llm(prompt, model_override=model)
    raise ValueError(f"Unsupported {stage} AI provider: {provider}")


def call_highlight_llm(prompt: str) -> str:
    return call_stage_text("highlight", prompt)


def call_seo_llm(prompt: str) -> str:
    return call_stage_text("seo", prompt)


def vision_images_per_request_for_stage() -> int:
    from shorts_generator.local.llm import _IMAGES_PER_REQUEST
    return _IMAGES_PER_REQUEST.get(current_stage_provider("vision"), 5)


def call_vision_stage(parts: list) -> str:
    """Use ONLY the selected vision provider, without automatic paid fallback."""
    provider = current_stage_provider("vision")
    if provider == "off":
        raise RuntimeError("Vision AI is disabled in Settings")
    if provider not in VISION_PROVIDERS:
        raise RuntimeError("Selected vision provider cannot inspect images")
    if not user_config.get(KEYS.get(provider, "")):
        raise RuntimeError(f"Vision provider {provider} has no configured API key")
    from shorts_generator.local import llm
    # Groq's dedicated vision model differs from its default text model.
    model = current_stage_model("vision")
    if provider == "gemini":
        return llm.call_gemini_llm("", contents=llm._as_gemini_parts(parts),
                                   model_override=model)
    return llm._openai_compatible_vision(parts, provider, model_override=model)
