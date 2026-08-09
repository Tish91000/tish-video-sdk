"""Mocked SubtitleBuilder tests -- no MFA install or real audio decode needed,
safe to run anywhere. Time-based splitting exercises the real code path;
MFA is only checked for its "not configured" error path (not a real MFA run).

Run with: pytest tests/fake
"""
import inspect
import os
import struct
import wave

import pytest

import tish_video_sdk.subtitles as subtitles_module
from tish_video_sdk.subtitles import SubtitleBuilder


def _make_silent_wav(path: str, duration_seconds: float = 2.0, frame_rate: int = 24000) -> str:
    with wave.open(path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(frame_rate)
        f.writeframes(struct.pack('<h', 0) * int(frame_rate * duration_seconds))
    return path


@pytest.fixture(autouse=True)
def _force_time_based_alignment(monkeypatch):
    # MFA isn't installed in this environment; force the no-external-dependency
    # path so these tests run anywhere. MFA's "not configured" branch is
    # exercised separately in TestMFANotConfigured.
    monkeypatch.setattr(subtitles_module, "USE_MFA_ALIGNMENT", False)


class TestTimeBasedSplitting:
    def test_generates_segments_proportional_to_audio_duration(self, tmp_path):
        wav_path = _make_silent_wav(str(tmp_path / "audio.wav"), duration_seconds=2.0)
        builder = SubtitleBuilder.from_audio_with_reference(wav_path, "Hello world. This is a test.", "en")

        result = builder.build()

        assert result is builder
        segments = builder.get_segments()
        assert len(segments) == 2
        assert segments[0]['text'] == 'Hello world.'
        assert segments[-1]['end'] == pytest.approx(2.0, abs=0.01)

    def test_honors_ssml_break_pauses(self, tmp_path):
        wav_path = _make_silent_wav(str(tmp_path / "audio.wav"), duration_seconds=4.0)
        text = '<speak>First part.<break time="2s"/>Second part.</speak>'
        builder = SubtitleBuilder.from_audio_with_reference(wav_path, text, "en")

        builder.build()
        segments = builder.get_segments()

        assert len(segments) == 2
        assert segments[1]['start'] >= 2.0  # second block starts after the 2s pause

    def test_no_audio_file_returns_none(self):
        builder = SubtitleBuilder.from_audio_with_reference("no/such/file.wav", "Hello", "en")
        assert builder.build() is None

    def test_save_segments_to_json(self, tmp_path):
        wav_path = _make_silent_wav(str(tmp_path / "audio.wav"))
        builder = SubtitleBuilder.from_audio_with_reference(wav_path, "Hello world.", "en")
        builder.build()

        out_path = str(tmp_path / "segments.json")
        result = builder.save_segments_to_json(out_path)

        assert result == out_path
        assert os.path.exists(out_path)


class TestMFANotConfigured:
    def test_mfa_alignment_without_env_var_fails_cleanly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subtitles_module, "USE_MFA_ALIGNMENT", True)
        monkeypatch.delenv("MFA_ENV_PATH", raising=False)

        wav_path = _make_silent_wav(str(tmp_path / "audio.wav"))
        builder = SubtitleBuilder.from_audio_with_reference(wav_path, "Hello world.", "en")

        assert builder.build() is None


class TestNoDeadGeminiParam:
    def test_constructor_has_no_gemini_api_key_param(self):
        """Regression test: the original SubtitleBuilder.__init__ accepted a
        gemini_api_key param and imported google.generativeai, neither of
        which was ever actually used. Both were dropped when porting this
        module -- confirms the dead param doesn't come back by accident."""
        params = inspect.signature(SubtitleBuilder.__init__).parameters
        assert "gemini_api_key" not in params
