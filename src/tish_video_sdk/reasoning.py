"""reasoning: the sole gateway to every Gemini/LLM call outside TTS
synthesis and Imagen image generation (see CONTEXT.md and ADR 0011). No
caller -- SDK-internal (MusicManager, ImageGenerator) or a consuming app's
(Daily_Readings) -- ever receives a raw credential or holds a client;
generate()/embed() look up and inject credentials internally, per call.

REASONING_PACKS_PATH points at a JSON file listing one ReasoningPack entry
per (language, provider) -- see configuration/reasoning_packs.example.json.
Loaded lazily (on first actual lookup, not at import time) and cached, so
importing this module never requires REASONING_PACKS_PATH to be set for
projects that don't use it.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

from .internal.providers.reasoning_packs import (
    ReasoningAPIError,
    ReasoningError,
    ReasoningPack,
    ReasoningRateLimitError,
    load_reasoning_packs,
)

_ALL_REASONING_PACKS: Optional[List[ReasoningPack]] = None

# Retries against the *current* provider before falling through to the next
# one in the chain -- a deliberate addition beyond VoicePack's pattern (see
# ADR 0011): without it, a transient rate limit would be a hard failure
# whenever, as today, only one provider is registered.
_RATE_LIMIT_RETRIES = 3
_RATE_LIMIT_BACKOFF_SECONDS = 5

_RESPONSE_FORMATS = {"text", "json"}


def _get_all_reasoning_packs() -> List[ReasoningPack]:
    global _ALL_REASONING_PACKS
    if _ALL_REASONING_PACKS is None:
        path = os.getenv("REASONING_PACKS_PATH")
        if not path:
            raise RuntimeError(
                "REASONING_PACKS_PATH is not set. Point it at a reasoning_packs.json listing "
                "the per-language, per-provider Gemini credentials available to this process -- see "
                "configuration/reasoning_packs.example.json."
            )
        _ALL_REASONING_PACKS = load_reasoning_packs(path)
    return _ALL_REASONING_PACKS


def _clean_json_fences(text: str) -> str:
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return cleaned.strip()


def _retry_rate_limit(call):
    """Call `call()`, retrying with backoff up to _RATE_LIMIT_RETRIES times
    if it raises ReasoningRateLimitError, before giving up on this provider."""
    last_error: Optional[ReasoningRateLimitError] = None
    for attempt in range(_RATE_LIMIT_RETRIES):
        try:
            return call()
        except ReasoningRateLimitError as e:
            last_error = e
            if attempt < _RATE_LIMIT_RETRIES - 1:
                time.sleep(_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
    raise last_error


def _generate_json_on_pack(pack: ReasoningPack, prompt: str, tier: str) -> Union[dict, list]:
    """generate() against one pack for response_format="json": one shared
    attempt budget covers both a rate-limited call (retried with backoff)
    and a call that succeeded but returned unparseable JSON (retried
    immediately -- a re-roll of the same prompt may just format better)."""
    last_error: Optional[Exception] = None
    for attempt in range(_RATE_LIMIT_RETRIES):
        try:
            raw = pack.generate(prompt, tier)
        except ReasoningRateLimitError as e:
            last_error = e
            if attempt < _RATE_LIMIT_RETRIES - 1:
                time.sleep(_RATE_LIMIT_BACKOFF_SECONDS * (2 ** attempt))
            continue

        try:
            return json.loads(_clean_json_fences(raw))
        except json.JSONDecodeError as e:
            last_error = e
            continue

    if isinstance(last_error, ReasoningError):
        raise last_error
    raise ReasoningAPIError(f"Response was not valid JSON after {_RATE_LIMIT_RETRIES} attempts: {last_error}")


def _walk_chain(language_code: str, provider: Optional[str], attempt):
    """Try attempt(pack) against each ReasoningPack in language_code's
    provider chain, in order, falling back to the next provider on any
    ReasoningError. Raises the last ReasoningError if every provider fails."""
    chain = ReasoningPack.filter_chain(_get_all_reasoning_packs(), language_code, provider)

    last_error: Optional[ReasoningError] = None
    for pack in chain:
        try:
            pack.load_model()
            return attempt(pack)
        except ReasoningError as e:
            last_error = e
            continue

    raise last_error


def is_mock(language_code: str, provider: Optional[str] = None) -> bool:
    """Whether language_code's configured chain (or `provider`, if pinned)
    currently resolves to the mock reasoning provider -- lets a consuming
    app gate its own unrelated mock behavior (e.g. swapping in a fake
    music/image/publishing client for a coherent dummy end-to-end run) on
    the same test-mode signal, without ever inspecting a credential string
    (see CONTEXT.md's reasoning.generate()/embed() entry)."""
    chain = ReasoningPack.filter_chain(_get_all_reasoning_packs(), language_code, provider)
    return chain[0].provider == "mock"


def generate(language_code: str, template: str, values: Optional[Dict[str, Any]] = None, *,
             tier: str = "fast", provider: Optional[str] = None,
             response_format: str = "text") -> Union[str, dict, list]:
    """Generate text from template.format(**values) using language_code's
    reasoning provider chain (see CONTEXT.md's reasoning.generate()/embed()
    entry and ADR 0011).

    Args:
        language_code: Selects the ReasoningPack chain to use.
        template: A str.format()-style prompt template. Pass a plain string
            (no placeholders) with values={} for a prompt that needs no
            substitution.
        values: Substituted into template via .format(**values).
        tier: "fast" | "balanced" | "deep" -- the speed/capability level to
            ask for, not a model name (see CONTEXT.md's "tier" entry).
        provider: Pin to one provider (disables fallback). None walks the
            full chain configured for language_code, falling back to the
            next provider on failure.
        response_format: "text" returns the raw (stripped) response. "json"
            strips markdown fences, json.loads()s the result, and retries
            on parse failure -- returns a dict/list.

    Returns:
        str for response_format="text", dict/list for "json".

    Raises:
        ReasoningError (or a typed subclass) if every provider in the chain
        fails -- never returns None/"" on total failure.
    """
    if response_format not in _RESPONSE_FORMATS:
        raise ValueError(f"Unknown response_format '{response_format}'. Expected one of {sorted(_RESPONSE_FORMATS)}.")

    prompt = template.format(**(values or {}))

    if response_format == "json":
        return _walk_chain(language_code, provider, lambda pack: _generate_json_on_pack(pack, prompt, tier))

    return _walk_chain(language_code, provider, lambda pack: _retry_rate_limit(lambda: pack.generate(prompt, tier)))


def embed(language_code: str, texts: List[str], *, task_type: str = "retrieval_document",
          provider: Optional[str] = None) -> List[List[float]]:
    """Embed texts using language_code's reasoning provider chain (see
    CONTEXT.md's reasoning.generate()/embed() entry and ADR 0011).

    Args:
        language_code: Selects the ReasoningPack chain to use.
        texts: Texts to embed -- pass a single-item list for one query.
        task_type: "retrieval_document" for content being indexed,
            "retrieval_query" for a search query -- Gemini's embedding
            model optimizes differently for each.
        provider: Pin to one provider (disables fallback). None walks the
            full chain configured for language_code, falling back to the
            next provider on failure.

    Returns:
        One vector (list[float]) per input text, same order as texts.

    Raises:
        ReasoningError (or a typed subclass) if every provider in the chain
        fails.
    """
    return _walk_chain(language_code, provider, lambda pack: _retry_rate_limit(lambda: pack.embed(texts, task_type)))
