"""ClipMint live AI activity telemetry.

Bound to one job at a time using ContextVars. No keys, prompts, transcripts,
response content or raw provider exceptions leave this module. This is
observability only: exceptions are re-raised and existing fallback stays intact.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import re
import time
import uuid

_CURRENT = ContextVar('clipmint_ai_job', default=None)
_STAGE = ContextVar('clipmint_ai_stage', default='other')
MAX_EVENTS = 120


@contextmanager
def watch(job_id, sink):
    """Route events to this job's thread-safe sink, including child callbacks."""
    tok = _CURRENT.set((str(job_id), sink))
    try:
        yield
    finally:
        _CURRENT.reset(tok)


@contextmanager
def stage(name):
    tok = _STAGE.set(name)
    try:
        yield
    finally:
        _STAGE.reset(tok)


def for_stage(name, fn):
    """Identify which pipeline task issued an actual provider API call."""
    @wraps(fn)
    def inner(*args, **kwargs):
        with stage(name):
            return fn(*args, **kwargs)
    return inner


def _emit(event):
    active = _CURRENT.get()
    if active is not None:
        try:
            active[1](dict(event))
        except Exception:
            pass  # Metrics must never alter AI behavior.


def _model(provider, args, kwargs, kind):
    if provider == 'faster-whisper':
        try:
            from shorts_generator.config import LOCAL_WHISPER_MODEL
            return str(LOCAL_WHISPER_MODEL)
        except Exception:
            return 'Whisper'
    chosen = kwargs.get('model_override') or kwargs.get('model')
    if chosen:
        return str(chosen)[:100]
    try:
        from shorts_generator.config import current_model
        if kind == 'vision' and provider == 'groq':
            from shorts_generator.local.llm import GROQ_VISION_MODEL
            return GROQ_VISION_MODEL
        return str(current_model(provider))[:100]
    except Exception:
        return 'configured model'


def _expected(stage_name):
    try:
        from shorts_generator.config import current_stage_provider
        return str(current_stage_provider(stage_name))
    except (ImportError, AttributeError):
        try:
            from shorts_generator.config import current_provider
            return str(current_provider())
        except Exception:
            return ''


def _safe_code(exc):
    """Only exception class and numeric HTTP code; never token-bearing text."""
    label = re.sub(r'[^A-Za-z0-9_]', '', type(exc).__name__)[:48] or 'Error'
    code = getattr(exc, 'status_code', None)
    if not isinstance(code, int):
        response = getattr(exc, 'response', None)
        code = getattr(response, 'status_code', None)
    if isinstance(code, int) and 100 <= code <= 599:
        return f'{label} (HTTP {code})'
    return label


def trace_provider(provider=None, kind='request'):
    """Wrap the *actual* backend call, capturing retries and actual fallbacks."""
    def decorate(fn):
        @wraps(fn)
        def invoke(*args, **kwargs):
            if _CURRENT.get() is None:
                return fn(*args, **kwargs)
            actual_provider = str(provider(*args, **kwargs) if callable(provider) else provider)
            stage_name = _STAGE.get()
            if stage_name == 'other' and actual_provider == 'faster-whisper':
                stage_name = 'transcribe'
            if stage_name == 'other' and kind == 'vision':
                stage_name = 'vision'
            model = _model(actual_provider, args, kwargs, kind)
            expected = _expected(stage_name) if stage_name in ('highlight', 'vision', 'seo') else ''
            event = {
                'id': uuid.uuid4().hex[:12],
                'stage': stage_name,
                'provider': actual_provider,
                'model': model,
                'kind': kind,
                'status': 'running',
                'started_at': round(time.time(), 3),
                'duration_ms': None,
                'fallback': bool(expected and expected != actual_provider),
                'error': '',
            }
            _emit(event)
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                event['status'] = 'error'
                event['error'] = _safe_code(exc)
                raise
            else:
                event['status'] = 'success'
                return result
            finally:
                event['duration_ms'] = max(0, round((time.monotonic() - start) * 1000))
                _emit(event)
        return invoke
    return decorate


def trace_transcription(fn):
    return trace_provider('faster-whisper', kind='transcription')(fn)


def summarize(events):
    """Only actual completed provider requests count toward usage."""
    requests = [e for e in events if e.get('kind') == 'request']
    return {
        'requests': len(requests),
        'completed': sum(e.get('status') == 'success' for e in requests),
        'failed': sum(e.get('status') == 'error' for e in requests),
        'running': sum(e.get('status') == 'running' for e in requests),
        'fallbacks': sum(bool(e.get('fallback')) for e in requests),
        'cloud_requests': sum(e.get('provider') not in ('ollama', 'faster-whisper') for e in requests),
        'elapsed_ms': sum(int(e.get('duration_ms') or 0) for e in requests),
    }
