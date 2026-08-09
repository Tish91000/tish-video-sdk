"""Mocked TTSBuilder tests -- no real network/API calls, safe to run anywhere.

Run with: pytest tests/fake
"""
import io
import os
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest
from pydub import AudioSegment

import tish_video_sdk.tts as tts_module
from tish_video_sdk.tts import TTSBuilder
from tish_video_sdk.internal.providers.voice_packs import GeminiVoicePack, GoogleCloudVoicePack, VoicePackAuthenticationError

# No __init__.py in tests/ -- these aren't a package, so read the credentials
# path conftest.py already wrote to disk straight off the loaded chain instead
# of a relative import.
FAKE_CREDENTIALS_PATH = tts_module._ALL_VOICE_PACKS[1].credentials


def _fake_gemini_response(pcm_bytes: bytes, mime_type: str = "audio/l16;rate=24000"):
    """Build a response shaped like genai.Client().models.generate_content()'s
    return value, carrying raw PCM audio in its first part."""
    part = SimpleNamespace(inline_data=SimpleNamespace(mime_type=mime_type, data=pcm_bytes))
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _fake_pcm(num_samples: int = 1000) -> bytes:
    # 16-bit mono silence; only needs to be a valid even-length PCM buffer.
    return b"\x00\x00" * num_samples


def _fake_wav_bytes() -> bytes:
    # A real, minimal, valid WAV file -- so _normalize_to_wav actually
    # exercises the pydub path instead of falling back on a parse error.
    segment = AudioSegment.silent(duration=50, frame_rate=24000)
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


class TestLanguageConfig:
    def test_unconfigured_language_raises(self):
        with pytest.raises(ValueError):
            TTSBuilder("not-a-configured-language")

    def test_configured_language_is_accepted(self):
        builder = TTSBuilder("xx")
        assert builder.language_code == "xx"

    def test_no_language_code_uses_default(self):
        # TTS_LANGUAGE_CODE=xx is set as the default in conftest.py.
        builder = TTSBuilder()
        assert builder.language_code == "xx"

    def test_multiple_languages_in_the_same_process(self):
        # The whole point of the language filter: one process can build for
        # more than one language, unlike the old one-language-per-process design.
        xx_builder = TTSBuilder("xx")
        yy_builder = TTSBuilder("yy")

        assert [p.provider for p in xx_builder._voice_pack_chain] == ["gemini", "google_cloud"]
        assert [p.provider for p in yy_builder._voice_pack_chain] == ["gemini"]


class TestProviderFilter:
    def test_pinning_to_a_provider_restricts_the_chain(self):
        builder = TTSBuilder("xx", provider="google_cloud")
        assert [p.provider for p in builder._voice_pack_chain] == ["google_cloud"]

    def test_pinned_provider_failure_does_not_fall_back(self, tmp_path):
        # Even though a google_cloud VoicePack also exists for "xx", pinning
        # to gemini must disable fallback to it.
        with patch("google.genai.Client") as mock_genai_cls:
            mock_genai_cls.return_value.models.generate_content.side_effect = RuntimeError("gemini is down")

            builder = TTSBuilder.from_text("Hello there", "xx", provider="gemini", use_llm_ssml=False)
            result = builder.build()

            assert result is None

    def test_unknown_provider_for_language_raises(self):
        with pytest.raises(ValueError):
            TTSBuilder("yy", provider="google_cloud")  # yy only has a gemini entry


class TestVoicePackCredentials:
    def test_gemini_voice_pack_requires_credentials(self):
        entry = dict(tts_module._ALL_VOICE_PACKS[0].__dict__)
        pack = GeminiVoicePack({**entry, "credentials": None})
        with pytest.raises(VoicePackAuthenticationError):
            pack.load_model()

    def test_google_cloud_voice_pack_requires_existing_credentials_file(self):
        pack = GoogleCloudVoicePack({
            "language_code": "xx",
            "voice_name": "xx-XX-FakeVoice",
            "tts_language_code": "xx-XX",
            "credentials": "/no/such/file.json",
            "synthesis_character_limit": 4800,
            "max_sentence_character_limit": 800,
        })
        with pytest.raises(VoicePackAuthenticationError):
            pack.load_model()


class TestGeminiSynthesis:
    def test_synthesis_success(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_response(_fake_pcm())

            builder = TTSBuilder.from_text("Hello there", "xx", use_llm_ssml=False)
            result = builder.build()

            assert result is builder
            assert builder.used_provider == "gemini"
            assert builder._audio_chunks

            out_path = builder.save(str(tmp_path / "out.wav"))
            assert out_path is not None
            assert os.path.exists(out_path)
            with open(out_path, "rb") as f:
                assert f.read(4) == b"RIFF"


class TestGoogleCloudSynthesis:
    def test_synthesis_success(self, tmp_path):
        with patch("google.cloud.texttospeech.TextToSpeechClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock(
                synthesize_speech=MagicMock(return_value=SimpleNamespace(audio_content=_fake_wav_bytes()))
            )
            pack = GoogleCloudVoicePack({
                "language_code": "xx",
                "voice_name": "xx-XX-FakeVoice",
                "tts_language_code": "xx-XX",
                "credentials": FAKE_CREDENTIALS_PATH,
                "synthesis_character_limit": 4800,
                "max_sentence_character_limit": 800,
            })
            pack.load_model()
            result = pack.synthesize("Bonjour le monde", is_ssml=False)

            assert result.mime_type == "audio/wav"
            assert result.audio_bytes


class TestFallbackChain:
    def test_gemini_api_failure_falls_back_to_google_cloud(self, tmp_path):
        with patch("google.genai.Client") as mock_genai_cls, \
             patch("google.cloud.texttospeech.TextToSpeechClient") as mock_tts_cls:
            mock_genai_cls.return_value.models.generate_content.side_effect = RuntimeError("gemini is down")
            mock_tts_cls.return_value = MagicMock(
                synthesize_speech=MagicMock(return_value=SimpleNamespace(audio_content=_fake_wav_bytes()))
            )

            builder = TTSBuilder.from_text("Hello there", "xx", use_llm_ssml=False)
            result = builder.build()

            assert result is builder
            assert builder.used_provider == "google_cloud"

    def test_gemini_missing_credentials_falls_back_to_google_cloud(self, monkeypatch):
        monkeypatch.setattr(tts_module._ALL_VOICE_PACKS[0], "credentials", None)
        with patch("google.cloud.texttospeech.TextToSpeechClient") as mock_tts_cls:
            mock_tts_cls.return_value = MagicMock(
                synthesize_speech=MagicMock(return_value=SimpleNamespace(audio_content=_fake_wav_bytes()))
            )

            builder = TTSBuilder.from_text("Hello there", "xx", use_llm_ssml=False)
            result = builder.build()

            assert result is builder
            assert builder.used_provider == "google_cloud"

    def test_all_voice_packs_failing_returns_none(self, monkeypatch):
        monkeypatch.setattr(tts_module._ALL_VOICE_PACKS[0], "credentials", None)
        monkeypatch.setattr(tts_module._ALL_VOICE_PACKS[1], "credentials", "/no/such/file.json")

        builder = TTSBuilder.from_text("Hello there", "xx", use_llm_ssml=False)
        result = builder.build()

        assert result is None

    def test_unexpected_exception_propagates_instead_of_falling_back(self, monkeypatch):
        # A bug inside a VoicePack (not a provider-level failure) must stop the
        # build entirely -- it shouldn't be swallowed and treated the same as
        # "this provider's API is down."
        def _boom():
            raise TypeError("boom - this is a bug, not a provider failure")

        monkeypatch.setattr(tts_module._ALL_VOICE_PACKS[0], "load_model", _boom)

        builder = TTSBuilder.from_text("Hello there", "xx", use_llm_ssml=False)
        with pytest.raises(TypeError):
            builder.build()


class TestBuildGuards:
    def test_save_before_build_returns_none(self, tmp_path):
        builder = TTSBuilder("xx")
        assert builder.save(str(tmp_path / "out.wav")) is None

    def test_build_without_content_returns_none(self):
        builder = TTSBuilder("xx")
        assert builder.build() is None


class TestSubtitleGeneration:
    """MFA isn't installed in this environment, so these force the
    time-based-splitting path (see tests/fake/test_subtitles.py for that
    path's own dedicated tests) -- the point here is TTSBuilder's wiring
    into SubtitleBuilder, not alignment accuracy."""

    def test_with_subtitles_generates_segments_during_build(self, monkeypatch):
        import tish_video_sdk.subtitles as subtitles_module
        monkeypatch.setattr(subtitles_module, "USE_MFA_ALIGNMENT", False)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_response(_fake_pcm())

            builder = TTSBuilder.from_text("Hello there. This is a test.", "xx", use_llm_ssml=False)
            builder.with_subtitles()
            result = builder.build()

            assert result is builder
            segments = builder.get_subtitle_segments()
            assert segments
            assert segments[0]['text'] == 'Hello there.'

    def test_subtitle_failure_fails_the_whole_build(self, monkeypatch):
        import tish_video_sdk.subtitles as subtitles_module
        monkeypatch.setattr(subtitles_module, "USE_MFA_ALIGNMENT", False)
        # No reference text distributable over zero-duration audio -> time-based
        # splitting fails cleanly; simulate by forcing SubtitleBuilder.build() to fail.
        monkeypatch.setattr(subtitles_module.SubtitleBuilder, "build", lambda self: None)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_response(_fake_pcm())

            builder = TTSBuilder.from_text("Hello there.", "xx", use_llm_ssml=False)
            builder.with_subtitles()
            result = builder.build()

            assert result is None

    def test_without_subtitles_get_subtitle_segments_is_empty(self):
        builder = TTSBuilder.from_text("Hello there.", "xx")
        assert builder.get_subtitle_segments() == []

    def test_cache_key_differs_with_and_without_subtitles(self, tmp_path, monkeypatch):
        """Regression test: the hash TTSBuilder uses to key its cache used to
        ignore whether subtitles were requested, so a build without subtitles
        could cache-hit a later call with with_subtitles() (or vice versa),
        silently returning audio with the wrong subtitle state."""
        import tish_video_sdk.subtitles as subtitles_module
        monkeypatch.setattr(subtitles_module, "USE_MFA_ALIGNMENT", False)

        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_response(_fake_pcm())

            plain = TTSBuilder.from_text("Hello there.", "xx", use_llm_ssml=False, cache_dir=str(tmp_path))
            plain.build()

            with_subs = TTSBuilder.from_text("Hello there.", "xx", use_llm_ssml=False, cache_dir=str(tmp_path))
            with_subs.with_subtitles()
            with_subs.build()

            assert with_subs.used_provider != "cache"
            assert with_subs.get_subtitle_segments()
