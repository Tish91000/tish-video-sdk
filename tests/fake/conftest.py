"""Points TTS_LANGUAGE_CODE at a synthetic sentinel language ("xx", not a real
language like fr/en -- keeps it visually obvious these are mechanism-test
values) as the default, and TTS_VOICE_PACKS_PATH at a throwaway voice_packs.json
covering two sentinel languages ("xx" with both providers, "yy" with just
gemini) so tests can exercise multi-language filtering -- before any test
module in this directory imports tish_video_sdk.tts (which loads both at
import time). The Google Cloud credentials file only needs to exist on disk --
its content is never read, since the real Google/Gemini clients are always
mocked in this suite.
"""
import json
import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="tish_tts_fake_config_")

FAKE_CREDENTIALS_PATH = os.path.join(_tmp_dir, "xx-credentials.json")
with open(FAKE_CREDENTIALS_PATH, "w", encoding="utf-8") as _f:
    json.dump({"type": "service_account"}, _f)

FAKE_VOICE_PACKS = [
    {
        "provider": "gemini",
        "language_code": "xx",
        "voice_name": "FakeGeminiVoice",
        "credentials": "fake-gemini-api-key",
        "model": "fake-gemini-tts-model",
        "synthesis_character_limit": 4800,
        "max_sentence_character_limit": 800,
    },
    {
        "provider": "google_cloud",
        "language_code": "xx",
        "tts_language_code": "xx-XX",
        "voice_name": "xx-XX-FakeVoice",
        "credentials": FAKE_CREDENTIALS_PATH,
        "ssml_credentials": "fake-ssml-api-key",
        "ssml_model": "fake-ssml-model",
        "synthesis_character_limit": 4800,
        "max_sentence_character_limit": 800,
    },
    {
        "provider": "gemini",
        "language_code": "yy",
        "voice_name": "FakeGeminiVoiceYY",
        "credentials": "fake-gemini-api-key-yy",
        "model": "fake-gemini-tts-model",
        "synthesis_character_limit": 4800,
        "max_sentence_character_limit": 800,
    },
]

_voice_packs_path = os.path.join(_tmp_dir, "voice_packs.json")
with open(_voice_packs_path, "w", encoding="utf-8") as _f:
    json.dump(FAKE_VOICE_PACKS, _f)

os.environ["TTS_LANGUAGE_CODE"] = "xx"
os.environ["TTS_VOICE_PACKS_PATH"] = _voice_packs_path

# "xx" gets two providers (gemini, then mock) so chain-fallback tests can
# exercise a real second entry without inventing a fake third-party
# provider class; "yy" gets just gemini, for the same single-provider
# shape ReasoningPack.filter_chain()/TTSBuilder tests both rely on.
FAKE_REASONING_PACKS = [
    {"provider": "gemini", "language_code": "xx", "credentials": "fake-gemini-api-key"},
    {"provider": "mock", "language_code": "xx", "credentials": ""},
    {"provider": "gemini", "language_code": "yy", "credentials": "fake-gemini-api-key-yy"},
]

_reasoning_packs_path = os.path.join(_tmp_dir, "reasoning_packs.json")
with open(_reasoning_packs_path, "w", encoding="utf-8") as _f:
    json.dump(FAKE_REASONING_PACKS, _f)

os.environ["REASONING_PACKS_PATH"] = _reasoning_packs_path
