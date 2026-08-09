"""VoicePack: a single TTS provider's adapter for one language (see CONTEXT.md).

TTSBuilder tries an ordered list of VoicePacks -- loaded from the JSON file at
TTS_VOICE_PACKS_PATH, filtered to the process's configured TTS_LANGUAGE_CODE --
falling back to the next VoicePack on a VoicePackSynthesisError. Concrete
subclasses (GeminiVoicePack, GoogleCloudVoicePack, ...) auto-register themselves
via __init_subclass__, keyed by their `provider` class attribute, so the JSON's
"provider" field resolves to a class without a separate registry to maintain.
"""
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError


class VoicePackSynthesisError(Exception):
    """A VoicePack failed to synthesize audio. TTSBuilder catches this (or a
    subclass) and falls back to the next VoicePack in the chain; any other
    exception is treated as a real bug and propagates instead."""


class VoicePackAuthenticationError(VoicePackSynthesisError):
    """This VoicePack's credentials are missing, invalid, or rejected."""


class VoicePackRateLimitError(VoicePackSynthesisError):
    """This VoicePack's provider rate-limited the request."""


class VoicePackAPIError(VoicePackSynthesisError):
    """Any other provider-side synthesis failure."""


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    mime_type: str


def _convert_text_to_ssml(text_content: str, api_key: str, model: str, language_code: str) -> str:
    """Convert plain text to SSML suitable for expressive liturgical/biblical
    readings, via a Gemini LLM call. Shared by any SSML-capable VoicePack --
    each one supplies its own api_key/model (its ssml_credentials/ssml_model)."""
    if not api_key:
        raise VoicePackAuthenticationError("No ssml_credentials configured for SSML conversion.")

    os.environ["GOOGLE_API_KEY"] = api_key

    try:
        llm_model = ChatGoogleGenerativeAI(model=model)

        llm_prompt_template = """
        Language: {lang}

        Vous êtes un convertisseur texte-vers-SSML pour une synthèse vocale expressive.
        Générez du SSML dans la langue d'origine pour le texte suivant. Il s'agit d'une lecture liturgique ou biblique destinée à une diffusion orale, votre SSML doit donc :
        - Structurer le discours avec un phrasé naturel en utilisant des balises <break time="Xs"/> (par exemple, à la ponctuation, pour des pauses dramatiques). Exagérez les pauses pour mettre l'accent. X doit être supérieur à 0,5 seconde
        - Mettre l'accent sur les noms importants, les déclarations ou les affirmations théologiques avec <emphasis> (utilisez modéré ou fort
        - Regrouper les pensées cohérentes en utilisant des balises <p>
        - Notez que l'audio sera accompagné d'une musique de fond mélodique à 60 BPM avec une signature rythmique 4/4, faites une pause en conséquence

        Retournez uniquement le SSML valide (sans commentaire ni texte supplémentaire).

        Voici le texte d'entrée : {texte}
        """
        llm_prompt_template_instance = PromptTemplate.from_template(llm_prompt_template)

        ssml_generation_chain = (
            {"texte": RunnablePassthrough(), "lang": lambda _: language_code}
            | llm_prompt_template_instance
            | llm_model
            | StrOutputParser()
        )

        ssml_output = ssml_generation_chain.invoke(text_content)
        ssml_output = re.sub(r'```xml\s*', '', ssml_output)
        ssml_output = re.sub(r'```\s*', '', ssml_output)
        return ssml_output.strip()
    except Exception as e:
        raise VoicePackAPIError(f"SSML conversion failed: {e}") from e


def clean_and_validate_ssml(ssml: str) -> str:
    """Clean up markdown fences/control characters, ensure <speak> wrapping,
    and validate the result is well-formed XML. Raises ValueError if not."""
    cleaned_ssml = ssml.strip()
    if cleaned_ssml.startswith("```xml") and cleaned_ssml.endswith("```"):
        cleaned_ssml = cleaned_ssml[6:-3].strip()

    cleaned_ssml = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned_ssml)
    cleaned_ssml = cleaned_ssml.replace('\xa0', ' ')

    if not cleaned_ssml.startswith('<speak>'):
        cleaned_ssml = f'<speak>{cleaned_ssml}</speak>'
    if not cleaned_ssml.endswith('</speak>'):
        cleaned_ssml = f'{cleaned_ssml}</speak>'

    try:
        ET.fromstring(cleaned_ssml)
        return cleaned_ssml
    except ParseError as e:
        raise ValueError(f"Generated SSML is invalid: {e}") from e


class VoicePack(ABC):
    """Abstract base for a single provider's adapter for one language.

    Only fields every provider genuinely shares live here; provider-only
    fields (e.g. tts_language_code) live on the concrete
    subclass that actually uses them. SSML-capable providers share their own
    fields/behavior via the SSMLVoicePack intermediate base instead.
    """

    provider: str = ""  # set by each concrete subclass; doubles as its registry key
    _registry: Dict[str, Type["VoicePack"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.provider:
            VoicePack._registry[cls.provider] = cls

    def __init__(self, entry: dict):
        self.language_code = entry["language_code"]
        self.voice_name = entry["voice_name"]
        self.credentials = entry["credentials"]
        self.synthesis_character_limit = int(entry.get("synthesis_character_limit", 4800))
        self.max_sentence_character_limit = int(entry.get("max_sentence_character_limit", 800))

    @property
    @abstractmethod
    def supports_ssml(self) -> bool:
        """Whether this VoicePack can synthesize SSML content (not just plain text)."""

    @abstractmethod
    def load_model(self) -> None:
        """Set up this provider's client/credentials. Called lazily -- only
        on the VoicePack whose turn it currently is in the fallback chain."""

    @abstractmethod
    def synthesize(self, chunk: str, is_ssml: bool) -> SynthesisResult:
        """Synthesize one chunk, already sized to this VoicePack's own
        character limits. Raises VoicePackSynthesisError (or a subclass) on
        provider-level failure; TTSBuilder normalizes the returned bytes to
        WAV itself using the returned mime_type."""

    @classmethod
    def from_entries(cls, entries: List[dict]) -> List["VoicePack"]:
        """Instantiate every entry in voice_packs.json, regardless of
        language -- one process can hold VoicePacks for multiple languages;
        filter_chain() narrows this down per TTSBuilder call."""
        packs = []
        for entry in entries:
            provider = entry.get("provider")
            pack_cls = cls._registry.get(provider)
            if pack_cls is None:
                known = ", ".join(sorted(cls._registry)) or "(none registered)"
                raise ValueError(f"Unknown TTS provider '{provider}' in voice_packs.json. Known providers: {known}.")
            packs.append(pack_cls(entry))
        return packs

    @classmethod
    def filter_chain(cls, all_packs: List["VoicePack"], language_code: str, provider: Optional[str] = None) -> List["VoicePack"]:
        """Narrow the full loaded VoicePack list to the chain for one
        TTSBuilder call: entries matching `language_code`, in list order.
        If `provider` is given, pins to that single provider -- fallback is
        disabled, since there's nothing left to fall back to."""
        matches = [p for p in all_packs if p.language_code == language_code]
        if provider is not None:
            matches = [p for p in matches if p.provider == provider]
        if not matches:
            if provider is not None:
                raise ValueError(f"No VoicePack found for language '{language_code}' and provider '{provider}' in voice_packs.json.")
            raise ValueError(f"No VoicePack entries found for language '{language_code}' in voice_packs.json.")
        return matches


class SSMLVoicePack(VoicePack):
    """Intermediate base for VoicePacks that can convert plain text to SSML
    before synthesis. Owns the fields/behavior every SSML-capable provider
    needs (its own ssml_credentials/ssml_model, since SSML conversion is
    always a separate Gemini LLM call regardless of which provider does the
    actual audio synthesis) so a new SSML-capable provider (e.g. Polly,
    Deepgram) only has to subclass this instead of re-implementing
    supports_ssml/convert_to_ssml itself. Providers that don't support SSML
    (e.g. GeminiVoicePack) subclass VoicePack directly instead."""

    def __init__(self, entry: dict):
        super().__init__(entry)
        self.ssml_credentials: Optional[str] = entry.get("ssml_credentials")
        self.ssml_model = entry.get("ssml_model", "gemini-2.5-flash")

    @property
    def supports_ssml(self) -> bool:
        return bool(self.ssml_credentials)

    def convert_to_ssml(self, text: str) -> str:
        """Convert plain text to SSML using this VoicePack's own ssml_credentials/ssml_model."""
        ssml = _convert_text_to_ssml(text, api_key=self.ssml_credentials, model=self.ssml_model, language_code=self.language_code)
        return clean_and_validate_ssml(ssml)


class GeminiVoicePack(VoicePack):
    provider = "gemini"

    def __init__(self, entry: dict):
        super().__init__(entry)
        self.model = entry.get("model", "gemini-2.5-flash-preview-tts")
        self._client = None

    @property
    def supports_ssml(self) -> bool:
        return False

    def load_model(self) -> None:
        if not self.credentials:
            raise VoicePackAuthenticationError("GeminiVoicePack has no credentials (Gemini API key) configured.")
        from google import genai
        self._client = genai.Client(api_key=self.credentials)

    def synthesize(self, chunk: str, is_ssml: bool) -> SynthesisResult:
        from google.genai import types

        if self._client is None:
            raise RuntimeError("GeminiVoicePack.load_model() must be called before synthesize().")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=chunk,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                        )
                    ),
                ),
            )
        except Exception as e:
            message = str(e)
            if "API key" in message or "PERMISSION_DENIED" in message or "UNAUTHENTICATED" in message:
                raise VoicePackAuthenticationError(f"Gemini authentication failed: {e}") from e
            if "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower():
                raise VoicePackRateLimitError(f"Gemini rate limit hit: {e}") from e
            raise VoicePackAPIError(f"Gemini synthesis failed: {e}") from e

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("audio"):
                    return SynthesisResult(audio_bytes=part.inline_data.data, mime_type=part.inline_data.mime_type)

        raise VoicePackAPIError("Gemini response did not contain audio data.")


class GoogleCloudVoicePack(SSMLVoicePack):
    provider = "google_cloud"

    def __init__(self, entry: dict):
        super().__init__(entry)
        self.tts_language_code = entry["tts_language_code"]
        self._client = None
        self._voice_params = None
        self._audio_config = None

    def load_model(self) -> None:
        import google.cloud.texttospeech as tts

        if not self.credentials:
            raise VoicePackAuthenticationError("GoogleCloudVoicePack has no credentials (service account path) configured.")
        if not os.path.exists(self.credentials):
            raise VoicePackAuthenticationError(f"Google Cloud credentials file not found at '{self.credentials}'.")

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials
        try:
            self._client = tts.TextToSpeechClient()
        except Exception as e:
            raise VoicePackAuthenticationError(f"Could not initialize Google Cloud TTS client: {e}") from e

        self._voice_params = tts.VoiceSelectionParams(language_code=self.tts_language_code, name=self.voice_name)
        self._audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16, sample_rate_hertz=24000)

    def synthesize(self, chunk: str, is_ssml: bool) -> SynthesisResult:
        import google.cloud.texttospeech as tts

        if self._client is None:
            raise RuntimeError("GoogleCloudVoicePack.load_model() must be called before synthesize().")

        if is_ssml:
            try:
                ET.fromstring(chunk)
                synthesis_input = tts.SynthesisInput(ssml=chunk)
            except ParseError:
                synthesis_input = tts.SynthesisInput(text=chunk)
        else:
            synthesis_input = tts.SynthesisInput(text=chunk)

        try:
            response = self._client.synthesize_speech(
                input=synthesis_input, voice=self._voice_params, audio_config=self._audio_config
            )
        except Exception as e:
            message = str(e)
            if "PERMISSION_DENIED" in message or "UNAUTHENTICATED" in message or "credential" in message.lower():
                raise VoicePackAuthenticationError(f"Google Cloud authentication failed: {e}") from e
            if "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower() or "Quota" in message:
                raise VoicePackRateLimitError(f"Google Cloud rate limit hit: {e}") from e
            raise VoicePackAPIError(f"Google Cloud TTS synthesis failed: {e}") from e

        return SynthesisResult(audio_bytes=response.audio_content, mime_type="audio/wav")


def load_voice_packs(voice_packs_path: str) -> List[VoicePack]:
    """Load and parse voice_packs_path, returning every VoicePack it
    describes (all languages, all providers). Use VoicePack.filter_chain()
    to narrow this down for a specific TTSBuilder call."""
    import json

    if not os.path.exists(voice_packs_path):
        raise RuntimeError(
            f"TTS_VOICE_PACKS_PATH points at '{voice_packs_path}', which does not exist. "
            "See configuration/voice_packs.example.json for the expected shape."
        )

    with open(voice_packs_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    return VoicePack.from_entries(entries)
