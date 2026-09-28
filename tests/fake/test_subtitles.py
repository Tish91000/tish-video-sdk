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

class TestMfaSubsetDictionary:
    """MFA loads and compiles its whole dictionary every run (~34 s of a
    ~62 s run for tamil_cv); aligning against just the entries a transcript
    can use must never change which entries are available to it."""

    @staticmethod
    def _make_dictionary(tmp_path, monkeypatch, entries):
        dict_dir = tmp_path / "pretrained_models" / "dictionary"
        dict_dir.mkdir(parents=True)
        (dict_dir / "xx_test.dict").write_text(
            "".join(f"{word}\t{phones}\n" for word, phones in entries), encoding="utf-8"
        )
        monkeypatch.setenv("MFA_ROOT_DIR", str(tmp_path))

    def test_keeps_only_entries_the_text_can_use(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [
            ("hello", "h e l o"), ("world", "w o r l d"), ("unused", "u n"),
        ])
        out = tmp_path / "subset.dict"

        assert _write_subset_dictionary("xx_test", ["Hello, world!"], str(out)) is True
        assert out.read_text(encoding="utf-8").splitlines() == ["hello\th e l o", "world\tw o r l d"]

    def test_keeps_every_pronunciation_of_a_word(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [("read", "r i d"), ("read", "r e d"), ("x", "x")])
        out = tmp_path / "subset.dict"
        _write_subset_dictionary("xx_test", ["read"], str(out))
        assert len(out.read_text(encoding="utf-8").splitlines()) == 2

    def test_matches_loosely_across_case_punctuation_and_compounds(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [
            ("don't", "d o n t"), ("well", "w e l"), ("known", "n o n"), ("other", "o"),
        ])
        out = tmp_path / "subset.dict"
        _write_subset_dictionary("xx_test", ["DON'T well-known"], str(out))
        kept = [line.split("\t")[0] for line in out.read_text(encoding="utf-8").splitlines()]
        assert kept == ["don't", "well", "known"]

    def test_keeps_tamil_combining_marks(self, tmp_path, monkeypatch):
        # Vowel signs and pulli are combining marks, not punctuation: stripping
        # them would make every Tamil word match the wrong (or no) entry.
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [
            ("\u0b95\u0bbe\u0bb1\u0bcd\u0bb1\u0bc1", "k a t r u"), ("\u0b95\u0bb1\u0bcd\u0bb1\u0bc1", "k a t r u"),
        ])
        out = tmp_path / "subset.dict"
        _write_subset_dictionary("xx_test", ["\u0b95\u0bbe\u0bb1\u0bcd\u0bb1\u0bc1"], str(out))
        assert out.read_text(encoding="utf-8").split("\t")[0] == "\u0b95\u0bbe\u0bb1\u0bcd\u0bb1\u0bc1"

    def test_missing_dictionary_returns_false(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        monkeypatch.setenv("MFA_ROOT_DIR", str(tmp_path))
        assert _write_subset_dictionary("nope", ["hi"], str(tmp_path / "s.dict")) is False

    def test_no_matching_words_returns_false(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [("hello", "h")])
        assert _write_subset_dictionary("xx_test", ["zzz"], str(tmp_path / "s.dict")) is False

    def test_dictionary_arg_falls_back_to_model_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MFA_ROOT_DIR", str(tmp_path))
        assert SubtitleBuilder("xx")._mfa_dictionary_arg("nope", ["hi"], str(tmp_path)) == "nope"

    def test_dictionary_arg_uses_subset_path_when_available(self, tmp_path, monkeypatch):
        self._make_dictionary(tmp_path, monkeypatch, [("hello", "h")])
        arg = SubtitleBuilder("xx")._mfa_dictionary_arg("xx_test", ["hello"], str(tmp_path))
        assert arg == str(tmp_path / "xx_test_subset.dict")

    def test_env_switch_forces_full_dictionary(self, tmp_path, monkeypatch):
        self._make_dictionary(tmp_path, monkeypatch, [("hello", "h")])
        monkeypatch.setenv("TISH_MFA_FULL_DICTIONARY", "1")
        assert SubtitleBuilder("xx")._mfa_dictionary_arg("xx_test", ["hello"], str(tmp_path)) == "xx_test"

    def test_extra_pronunciations_replace_stock_entries(self, tmp_path, monkeypatch):
        from tish_video_sdk.subtitles import _write_subset_dictionary
        self._make_dictionary(tmp_path, monkeypatch, [("kaari", "x"), ("hello", "h")])
        out = tmp_path / "subset.dict"
        assert _write_subset_dictionary(
            "xx_test", ["Kaari, hello randaka"], str(out), {"kaari": "k aː ɾ i", "randaka": "ɾ a"},
        ) is True
        assert out.read_text(encoding="utf-8").splitlines() == [
            "hello\th", "kaari\tk aː ɾ i", "randaka\tɾ a",
        ]

    def test_extra_pronunciations_alone_still_write_a_dictionary(self, tmp_path, monkeypatch):
        self._make_dictionary(tmp_path, monkeypatch, [("hello", "h")])
        builder = SubtitleBuilder.from_audio_segments_with_reference(
            "a.wav", [], "xx", pronunciations={"randaka": "ɾ a"},
        )
        arg = builder._mfa_dictionary_arg("xx_test", ["randaka"], str(tmp_path))
        assert (tmp_path / "xx_test_subset.dict").read_text(encoding="utf-8") == "randaka\tɾ a\n"
        assert arg == str(tmp_path / "xx_test_subset.dict")

    def test_env_switch_keeps_full_dictionary_plus_extras(self, tmp_path, monkeypatch):
        self._make_dictionary(tmp_path, monkeypatch, [("hello", "h"), ("other", "o")])
        monkeypatch.setenv("TISH_MFA_FULL_DICTIONARY", "1")
        builder = SubtitleBuilder.from_audio_with_reference(
            "a.wav", "randaka", "xx", pronunciations={"randaka": "ɾ a"},
        )
        builder._mfa_dictionary_arg("xx_test", ["randaka"], str(tmp_path))
        assert (tmp_path / "xx_test_subset.dict").read_text(encoding="utf-8").splitlines() == [
            "hello\th", "other\to", "randaka\tɾ a",
        ]
