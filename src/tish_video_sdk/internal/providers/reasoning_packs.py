"""ReasoningPack: a single provider's adapter for one language's generic
LLM/text calls that aren't TTS synthesis or image generation (see
CONTEXT.md). Mirrors VoicePack's ABC+registry+chain shape (see ADR 0011,
which mirrors ADR 0006) -- GeminiReasoningPack is the first concrete
subclass. reasoning.py is the only intended caller; nothing outside this
module and reasoning.py should ever hold a ReasoningPack or its credentials
directly.
"""
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type


class ReasoningError(Exception):
    """A ReasoningPack failed to generate/embed. reasoning.py catches this
    (or a subclass) and falls back to the next ReasoningPack in the chain;
    any other exception is treated as a real bug and propagates instead."""


class ReasoningAuthenticationError(ReasoningError):
    """This ReasoningPack's credentials are missing, invalid, or rejected."""


class ReasoningRateLimitError(ReasoningError):
    """This ReasoningPack's provider rate-limited the request."""


class ReasoningAPIError(ReasoningError):
    """Any other provider-side failure, including a response that still
    isn't valid JSON after response_format="json"'s own retries."""


class ReasoningPack(ABC):
    """Abstract base for a single provider's adapter for one language.

    Only fields every provider genuinely shares live here; provider-only
    fields (e.g. GeminiReasoningPack's per-tier model names) live on the
    concrete subclass that actually uses them.
    """

    provider: str = ""  # set by each concrete subclass; doubles as its registry key
    _registry: Dict[str, Type["ReasoningPack"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.provider:
            ReasoningPack._registry[cls.provider] = cls

    def __init__(self, entry: dict):
        self.language_code = entry["language_code"]
        self.credentials = entry.get("credentials", "")

    @abstractmethod
    def load_model(self) -> None:
        """Set up this provider's client/credentials. Called lazily -- only
        on the ReasoningPack whose turn it currently is in the fallback chain."""

    @abstractmethod
    def generate(self, prompt: str, tier: str) -> str:
        """Generate text for prompt at tier ("fast"/"balanced"/"deep").
        Raises ReasoningError (or a subclass) on provider-level failure."""

    @abstractmethod
    def embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed texts (task_type e.g. "retrieval_document"/"retrieval_query").
        Raises ReasoningError (or a subclass) on provider-level failure."""

    @classmethod
    def from_entries(cls, entries: List[dict]) -> List["ReasoningPack"]:
        """Instantiate every entry in reasoning_packs.json, regardless of
        language -- one process can hold ReasoningPacks for multiple
        languages; filter_chain() narrows this down per generate()/embed() call."""
        packs = []
        for entry in entries:
            provider = entry.get("provider", "gemini")
            pack_cls = cls._registry.get(provider)
            if pack_cls is None:
                known = ", ".join(sorted(cls._registry)) or "(none registered)"
                raise ValueError(f"Unknown reasoning provider '{provider}' in reasoning_packs.json. Known providers: {known}.")
            packs.append(pack_cls(entry))
        return packs

    @classmethod
    def filter_chain(cls, all_packs: List["ReasoningPack"], language_code: str, provider: Optional[str] = None) -> List["ReasoningPack"]:
        """Narrow the full loaded ReasoningPack list to the chain for one
        generate()/embed() call: entries matching `language_code`, in list
        order. If `provider` is given, pins to that single provider --
        fallback is disabled, since there's nothing left to fall back to."""
        matches = [p for p in all_packs if p.language_code == language_code]
        if provider is not None:
            matches = [p for p in matches if p.provider == provider]
        if not matches:
            if provider is not None:
                raise ValueError(f"No ReasoningPack found for language '{language_code}' and provider '{provider}' in reasoning_packs.json.")
            raise ValueError(f"No ReasoningPack entries found for language '{language_code}' in reasoning_packs.json.")
        return matches


def _classify_gemini_error(e: Exception) -> ReasoningError:
    message = str(e)
    if "API key" in message or "PERMISSION_DENIED" in message or "UNAUTHENTICATED" in message:
        return ReasoningAuthenticationError(f"Gemini authentication failed: {e}")
    if "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower() or "429" in message:
        return ReasoningRateLimitError(f"Gemini rate limit hit: {e}")
    return ReasoningAPIError(f"Gemini call failed: {e}")


class GeminiReasoningPack(ReasoningPack):
    provider = "gemini"

    # tier -> model name. Owned here (per provider, in code) rather than per
    # language in reasoning_packs.json, since it's a property of the model
    # family, not of the language -- see CONTEXT.md's "tier" entry.
    _TIER_MODELS = {
        "fast": "gemini-2.5-flash-lite",
        "balanced": "gemini-2.5-flash",
        "deep": "gemini-2.5-pro",
    }

    _EMBEDDING_MODEL = "gemini-embedding-001"

    def __init__(self, entry: dict):
        super().__init__(entry)
        self._client = None

    def load_model(self) -> None:
        if not self.credentials:
            raise ReasoningAuthenticationError("GeminiReasoningPack has no credentials (Gemini API key) configured.")
        from google import genai
        self._client = genai.Client(api_key=self.credentials)

    def _model_for_tier(self, tier: str) -> str:
        try:
            return self._TIER_MODELS[tier]
        except KeyError:
            known = ", ".join(sorted(self._TIER_MODELS))
            raise ValueError(f"Unknown reasoning tier '{tier}'. Known tiers: {known}.")

    def generate(self, prompt: str, tier: str) -> str:
        if self._client is None:
            raise RuntimeError("GeminiReasoningPack.load_model() must be called before generate().")
        model = self._model_for_tier(tier)
        try:
            response = self._client.models.generate_content(model=model, contents=prompt)
        except Exception as e:
            raise _classify_gemini_error(e) from e
        return (response.text or "").strip()

    def embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        if self._client is None:
            raise RuntimeError("GeminiReasoningPack.load_model() must be called before embed().")
        from google.genai import types
        try:
            response = self._client.models.embed_content(
                model=self._EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type.upper()),
            )
        except Exception as e:
            raise _classify_gemini_error(e) from e
        return [list(embedding.values or []) for embedding in (response.embeddings or [])]


class MockReasoningPack(ReasoningPack):
    """A reasoning provider that never calls out to a real API -- for tests.
    Selected the same way any real provider is: a `"provider": "mock"` entry
    in reasoning_packs.json, not by inspecting the credential string (see
    CONTEXT.md's reasoning.generate()/embed() entry)."""

    provider = "mock"

    def load_model(self) -> None:
        pass

    def generate(self, prompt: str, tier: str) -> str:
        return "This is a mock reasoning response."

    def embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        return [[0.0] * 8 for _ in texts]


def load_reasoning_packs(reasoning_packs_path: str) -> List[ReasoningPack]:
    """Load and parse reasoning_packs_path, returning every ReasoningPack it
    describes (all languages, all providers). Use ReasoningPack.filter_chain()
    to narrow this down for a specific generate()/embed() call."""
    if not os.path.exists(reasoning_packs_path):
        raise RuntimeError(
            f"REASONING_PACKS_PATH points at '{reasoning_packs_path}', which does not exist. "
            "See configuration/reasoning_packs.example.json for the expected shape."
        )

    with open(reasoning_packs_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    return ReasoningPack.from_entries(entries)
