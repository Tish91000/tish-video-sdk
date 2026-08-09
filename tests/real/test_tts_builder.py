"""TTSBuilder tests that make real Gemini / Google Cloud TTS API calls.

These cost real API quota and require actual credentials, so every test here
is individually skipped (not errored) when its prerequisites aren't available
-- safe to leave in the suite on a machine without them (see conftest.py for
the one case that isn't gracefully skippable).

Uses TTSBuilder(..., provider=...) to pin each test to one provider -- since
TTSBuilder no longer locks a process to a single language, this also proves
the multi-language/provider-filter design against the real APIs, not just the
fake suite.

Run with: pytest tests/real
Exclude with: pytest -m "not integration" (once other suites also opt in)
"""
import os
import wave

import pytest

pytestmark = pytest.mark.integration

LANGUAGE_CONFIGURED_FOR_ENGLISH = os.getenv("TTS_LANGUAGE_CODE") == "en"

_GEMINI_PACK = None
_GOOGLE_CLOUD_PACK = None
if LANGUAGE_CONFIGURED_FOR_ENGLISH:
    import tish_video_sdk.tts as _tts_module
    from tish_video_sdk.internal.providers.voice_packs import VoicePack

    try:
        _en_packs = VoicePack.filter_chain(_tts_module._ALL_VOICE_PACKS, "en")
    except ValueError:
        _en_packs = []

    for _pack in _en_packs:
        if _pack.provider == "gemini" and _GEMINI_PACK is None:
            _GEMINI_PACK = _pack
        elif _pack.provider == "google_cloud" and _GOOGLE_CLOUD_PACK is None:
            _GOOGLE_CLOUD_PACK = _pack

GEMINI_AVAILABLE = _GEMINI_PACK is not None and bool(_GEMINI_PACK.credentials)
GOOGLE_CLOUD_AVAILABLE = (
    _GOOGLE_CLOUD_PACK is not None
    and bool(_GOOGLE_CLOUD_PACK.credentials)
    and os.path.exists(_GOOGLE_CLOUD_PACK.credentials)
)


def _assert_valid_wav(path: str):
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    with wave.open(path, "rb") as wf:
        assert wf.getnframes() > 0


@pytest.mark.skipif(not LANGUAGE_CONFIGURED_FOR_ENGLISH, reason="TTS_LANGUAGE_CODE is not 'en'")
@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="no gemini VoicePack with credentials for 'en' in voice_packs.json (see configuration/voice_packs.example.json)")
def test_real_gemini_synthesis_english(tmp_path):
    from tish_video_sdk.tts import TTSBuilder

    builder = TTSBuilder.from_text("This is a short real test sentence.", "en", provider="gemini", use_llm_ssml=False)
    result = builder.build()

    assert result is not None, "Gemini synthesis failed against the real API"
    assert builder.used_provider == "gemini"

    out_path = builder.save(str(tmp_path / "real_gemini_en.wav"))
    assert out_path is not None
    _assert_valid_wav(out_path)


@pytest.mark.skipif(not LANGUAGE_CONFIGURED_FOR_ENGLISH, reason="TTS_LANGUAGE_CODE is not 'en'")
@pytest.mark.skipif(not GOOGLE_CLOUD_AVAILABLE, reason="no google_cloud VoicePack with an existing credentials file for 'en' in voice_packs.json")
def test_real_google_cloud_synthesis_english(tmp_path):
    from tish_video_sdk.tts import TTSBuilder

    builder = TTSBuilder.from_text("This is a short real test sentence.", "en", provider="google_cloud", use_llm_ssml=False)
    result = builder.build()

    assert result is not None, "Google Cloud TTS synthesis failed against the real API"
    assert builder.used_provider == "google_cloud"

    out_path = builder.save(str(tmp_path / "real_google_cloud_en.wav"))
    assert out_path is not None
    _assert_valid_wav(out_path)


@pytest.mark.skipif(not LANGUAGE_CONFIGURED_FOR_ENGLISH, reason="TTS_LANGUAGE_CODE is not 'en'")
@pytest.mark.skipif(not GEMINI_AVAILABLE, reason="no gemini VoicePack with credentials for 'en' -- needed since it's first in the real chain")
def test_real_full_chain_via_ttsbuilder(tmp_path):
    from tish_video_sdk.tts import TTSBuilder

    builder = TTSBuilder.from_text("This is a short real test sentence.", "en", use_llm_ssml=False)
    result = builder.build()

    assert result is not None, "TTS synthesis failed against the real API (every VoicePack in the chain failed)"

    out_path = builder.save(str(tmp_path / "real_full_chain_en.wav"))
    assert out_path is not None
    _assert_valid_wav(out_path)
