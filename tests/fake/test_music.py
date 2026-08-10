"""Mocked MusicManager tests -- no real network/API calls, safe to run anywhere.

Run with: pytest tests/fake
"""
import json
import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.generators import Sine

from tish_video_sdk.music import MusicManager
from tish_video_sdk.internal.providers.music_packs import JamendoTrack
from tish_video_sdk.internal.providers.reasoning_packs import ReasoningAPIError


def _write_tone(path, duration_ms):
    tone = Sine(440).to_audio_segment(duration=duration_ms)
    tone.export(str(path), format="wav")
    return tone


@pytest.fixture(autouse=True)
def _no_ambient_jamendo_client_id(monkeypatch):
    # tts.py's module-level load_dotenv() can leak a real JAMENDO_CLIENT_ID
    # from .env into os.environ once any other test module in this session
    # imports it -- MusicManager() falls back to that env var, so without
    # this every bare MusicManager() below would risk making a real,
    # unmocked network call instead of the no-op its test expects.
    monkeypatch.delenv("JAMENDO_CLIENT_ID", raising=False)


class TestMoodAnalysis:
    def test_no_language_code_returns_default_mood(self):
        manager = MusicManager(default_mood="calm")
        assert manager.get_mood_from_text("some text") == "calm"

    def test_valid_mood_returned(self):
        with patch("tish_video_sdk.reasoning.generate", return_value="joy"):
            manager = MusicManager(language_code="en")
            assert manager.get_mood_from_text("Everyone cheered and celebrated") == "joy"

    def test_unexpected_mood_falls_back_to_default(self):
        with patch("tish_video_sdk.reasoning.generate", return_value="furious"):
            manager = MusicManager(default_mood="calm", language_code="en")
            assert manager.get_mood_from_text("...") == "calm"

    def test_api_error_falls_back_to_default(self):
        with patch("tish_video_sdk.reasoning.generate", side_effect=ReasoningAPIError("gemini is down")):
            manager = MusicManager(default_mood="calm", language_code="en")
            assert manager.get_mood_from_text("...") == "calm"

    def test_default_moods_are_generic_not_domain_specific(self):
        # Regression check for the biblical-content cleanup: defaults must be
        # plain mood labels, not tied to any one content domain.
        manager = MusicManager()
        assert set(manager.available_moods) == {"calm", "joy", "adventurous", "dark"}
        for description in manager.available_moods.values():
            assert "bibl" not in description.lower()

    def test_prompt_sent_to_gemini_has_no_domain_framing(self):
        with patch("tish_video_sdk.reasoning.generate", return_value="calm") as mock_generate:
            manager = MusicManager(language_code="en")
            manager.get_mood_from_text("some text")

            template, values = mock_generate.call_args.args[1], mock_generate.call_args.args[2]
            assert "bibl" not in template.format(**values).lower()

    def test_mood_packs_path_overrides_moods_and_prompt(self, tmp_path):
        config_path = tmp_path / "custom_moods.json"
        config_path.write_text(
            '{"moods": {"tense": "Suspenseful and anxious."}, '
            '"prompt_template": "Custom prompt. Moods: {mood_list}. {mood_descriptions} Input: {text}"}',
            encoding="utf-8",
        )

        with patch("tish_video_sdk.reasoning.generate", return_value="tense") as mock_generate:
            manager = MusicManager(language_code="en", mood_packs_path=str(config_path))
            assert manager.available_moods == {"tense": "Suspenseful and anxious."}
            assert manager.get_mood_from_text("something ominous") == "tense"

            template, values = mock_generate.call_args.args[1], mock_generate.call_args.args[2]
            assert template.format(**values).startswith("Custom prompt.")

    def test_music_moods_path_env_var_is_used_when_no_explicit_path(self, tmp_path, monkeypatch):
        config_path = tmp_path / "env_moods.json"
        config_path.write_text('{"moods": {"weird": "A very specific mood."}}', encoding="utf-8")
        monkeypatch.setenv("MUSIC_MOODS_PATH", str(config_path))

        manager = MusicManager()
        assert manager.available_moods == {"weird": "A very specific mood."}


class TestSelectAudioMusic:
    def test_missing_mood_directory_returns_default(self, tmp_path):
        default_music = tmp_path / "default.wav"
        _write_tone(default_music, 500)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), default_music_file=str(default_music))
        assert manager.select_audio_music("calm", 5) == str(default_music)

    def test_selects_file_matching_requested_duration(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        short_file = mood_dir / "short.wav"
        long_file = mood_dir / "long.wav"
        _write_tone(short_file, 500)  # 0.5s
        _write_tone(long_file, 3000)  # 3s

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        assert manager.select_audio_music("calm", duration=2) == str(long_file)

    def test_falls_back_to_longest_when_nothing_long_enough(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        short_file = mood_dir / "short.wav"
        medium_file = mood_dir / "medium.wav"
        _write_tone(short_file, 500)
        _write_tone(medium_file, 1000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        assert manager.select_audio_music("calm", duration=10) == str(medium_file)


def _fake_track(id_="123", name="Sample Track", artist="Some Artist", duration=180):
    return JamendoTrack(
        id=id_, name=name, artist_name=artist, duration=duration,
        audio_download_url="https://example.invalid/track.mp3",
        license_ccurl="https://creativecommons.org/licenses/by/4.0/",
    )


def _fake_download_track(track, dest_path, timeout=30.0):
    # A real, decodable mp3 -- so the caller's normal local scan (duration
    # check, loudness measurement) works on it exactly like any other track.
    Sine(440).to_audio_segment(duration=3000).export(dest_path, format="mp3")


class TestJamendoFallback:
    """Covers the opt-in Jamendo fallback: a mood with no usable local track
    downloads one free, Creative-Commons track from Jamendo into
    bgm_directory instead of falling straight back to default_music_file.
    All network access is mocked -- see music_packs.py's own module docstring
    for the (unverified against a live call) API shape assumed here."""

    def test_noop_without_client_id(self, tmp_path):
        default_music = tmp_path / "default.wav"
        _write_tone(default_music, 500)
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), default_music_file=str(default_music))

        with patch("tish_video_sdk.music.music_packs.search_track") as mock_search:
            result = manager.select_audio_music("calm", duration=5)

        mock_search.assert_not_called()
        assert result == str(default_music)

    def test_downloads_track_when_mood_directory_is_missing(self, tmp_path):
        default_music = tmp_path / "default.wav"
        _write_tone(default_music, 500)
        manager = MusicManager(
            bgm_directory=str(tmp_path / "bgm"),
            default_music_file=str(default_music),
            jamendo_client_id="fake-client-id",
        )

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track) as mock_download:
            result = manager.select_audio_music("calm", duration=2)

        # No explicit search_query given -- effective_query defaults to the
        # mood's own (long) description, but with no language_code
        # configured to condense it, _refine_search_query falls back to the
        # bare mood name itself (a working Jamendo tag, unlike raw prose).
        mock_search.assert_called_once_with("calm", 2, "fake-client-id")
        mock_download.assert_called_once()
        assert result != str(default_music)
        assert os.path.exists(result)
        assert "jamendo_123" in os.path.basename(result)

    def test_downloads_track_when_mood_directory_is_empty(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()), \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            result = manager.select_audio_music("calm", duration=2)

        assert os.path.exists(result)
        assert os.path.dirname(result) == str(mood_dir)

    def test_falls_back_to_default_when_jamendo_finds_nothing(self, tmp_path):
        default_music = tmp_path / "default.wav"
        _write_tone(default_music, 500)
        manager = MusicManager(
            bgm_directory=str(tmp_path / "bgm"),
            default_music_file=str(default_music),
            jamendo_client_id="fake-client-id",
        )

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=None):
            result = manager.select_audio_music("calm", duration=5)

        assert result == str(default_music)

    def test_falls_back_to_default_when_download_fails(self, tmp_path):
        default_music = tmp_path / "default.wav"
        _write_tone(default_music, 500)
        manager = MusicManager(
            bgm_directory=str(tmp_path / "bgm"),
            default_music_file=str(default_music),
            jamendo_client_id="fake-client-id",
        )

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()), \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=RuntimeError("network down")):
            result = manager.select_audio_music("calm", duration=5)

        assert result == str(default_music)

    def test_fetch_from_jamendo_skips_redownload_if_file_already_exists(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()), \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track) as mock_download:
            first = manager._fetch_from_jamendo("calm", 2, str(mood_dir), "calm")
            second = manager._fetch_from_jamendo("calm", 2, str(mood_dir), "calm")

        assert first is not None
        assert first == second
        assert mock_download.call_count == 1

    def test_writes_attribution_into_shared_cache_sidecar(self, tmp_path):
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track(name="Sample Track")), \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            result = manager.select_audio_music("calm", duration=2)

        sidecar_path = result + ".cache.json"
        assert os.path.exists(sidecar_path)
        with open(sidecar_path, encoding="utf-8") as f:
            cache = json.load(f)
        attribution = cache["attribution"]
        assert attribution["source"] == "jamendo"
        assert attribution["name"] == "Sample Track"
        assert attribution["license_ccurl"].startswith("https://creativecommons.org")

    def test_attribution_and_loudness_share_the_sidecar_without_clobbering(self, tmp_path):
        # Attribution is written at download time; loudness gets measured
        # later, during mixing -- both must land in the same file rather
        # than one overwriting the other.
        speech_file = tmp_path / "speech.wav"
        _write_tone(speech_file, 3000)
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track(name="Sample Track")), \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            manager.add_background_music(str(speech_file), str(tmp_path / "out.wav"), music_input="calm")

        downloaded = tmp_path / "bgm" / "calm" / "jamendo_123_Sample_Track.mp3"
        assert downloaded.exists()
        with open(str(downloaded) + ".cache.json", encoding="utf-8") as f:
            cache = json.load(f)
        assert cache["attribution"]["name"] == "Sample Track"
        assert isinstance(cache["dbfs"], float)

    def test_long_reference_text_is_condensed_via_gemini_before_reaching_jamendo(self, tmp_path):
        long_text = (
            "On the seashore of endless worlds children meet, the infinite sky is "
            "motionless overhead and the restless water is boisterous."
        )
        assert len(long_text) > 40  # otherwise this wouldn't exercise refinement at all

        with patch("tish_video_sdk.reasoning.generate", return_value="ambient, dreamy, seaside"), \
             patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):

            manager = MusicManager(
                bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id", language_code="en",
            )
            manager.select_audio_music("calm", duration=2, search_query=long_text)

        # Combined -- mood itself, plus the Gemini-condensed keywords.
        mock_search.assert_called_once_with("calm, ambient, dreamy, seaside", 2, "fake-client-id")

    def test_long_query_without_gemini_falls_back_to_bare_mood_name(self, tmp_path):
        long_text = "A " + "very " * 20 + "long piece of reference text."
        assert len(long_text) > 40
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            manager.select_audio_music("dark", duration=2, search_query=long_text)

        mock_search.assert_called_once_with("dark", 2, "fake-client-id")

    def test_long_query_falls_back_to_mood_name_when_gemini_extraction_fails(self, tmp_path):
        long_text = "A " + "very " * 20 + "long piece of reference text."

        with patch("tish_video_sdk.reasoning.generate", side_effect=ReasoningAPIError("gemini is down")), \
             patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):

            manager = MusicManager(
                bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id", language_code="en",
            )
            manager.select_audio_music("dark", duration=2, search_query=long_text)

        mock_search.assert_called_once_with("dark", 2, "fake-client-id")

    def test_short_query_passes_through_unchanged_even_with_gemini_configured(self, tmp_path):
        with patch("tish_video_sdk.reasoning.generate") as mock_generate, \
             patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            manager = MusicManager(
                bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id", language_code="en",
            )
            manager.select_audio_music("dark", duration=2, search_query="stormy seas")

            mock_generate.assert_not_called()

        # Combined -- mood itself, plus the short query as-is (no Gemini needed).
        mock_search.assert_called_once_with("dark, stormy seas", 2, "fake-client-id")

    def test_reference_text_searches_jamendo_but_still_files_under_the_mood(self, tmp_path):
        # get_music_path's reference_text branch should search Jamendo with
        # the richer original text, not just the single classified mood
        # word -- but the download still lands in that mood's own folder,
        # so a later plain mood-name lookup finds it too.
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            # No language_code configured -- mood classification falls back
            # to default_mood ("calm") without needing Gemini mocked too.
            result = manager.get_music_path(reference_text="A wild chase through the mountains", duration=2)

        # Combined -- the classified mood (falls back to default_mood here),
        # plus the original reference text as-is (short enough to need no
        # Gemini condensing).
        mock_search.assert_called_once_with(
            f"{manager.default_mood}, A wild chase through the mountains", 2, "fake-client-id"
        )
        assert os.path.basename(os.path.dirname(result)) == manager.default_mood


class TestFetchFromJamendoExplicit:
    """Covers fetch_from_jamendo(), the explicit -- not fallback-only --
    entry point to Jamendo: always searches/downloads regardless of what's
    already local, so a caller can deliberately grow a mood's local pool
    over time (e.g. once a day for a mostly-single-mood content series)."""

    def test_no_client_id_returns_none_without_searching(self, tmp_path):
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))

        with patch("tish_video_sdk.music.music_packs.search_track") as mock_search:
            result = manager.fetch_from_jamendo("calm")

        mock_search.assert_not_called()
        assert result is None

    def test_always_searches_even_when_a_local_track_already_satisfies_duration(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        _write_tone(mood_dir / "existing.wav", 5000)  # already satisfies any short duration

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            result = manager.fetch_from_jamendo("calm", duration=2)

        mock_search.assert_called_once()
        assert result is not None
        assert os.path.basename(result) == "jamendo_123_Sample_Track.mp3"

    def test_returns_none_when_nothing_matches(self, tmp_path):
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=None):
            assert manager.fetch_from_jamendo("calm") is None

    def test_repeated_calls_with_different_tracks_grow_the_local_pool(self, tmp_path):
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")
        mood_dir = tmp_path / "bgm" / "meditative"

        with patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track(id_="1", name="Track One")):
                manager.fetch_from_jamendo("meditative", search_query="calm piano")
            with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track(id_="2", name="Track Two")):
                manager.fetch_from_jamendo("meditative", search_query="ambient drone")

        mp3_files = sorted(f for f in os.listdir(mood_dir) if f.endswith(".mp3"))
        assert mp3_files == ["jamendo_1_Track_One.mp3", "jamendo_2_Track_Two.mp3"]

    def test_search_query_is_combined_with_mood_like_the_automatic_fallback(self, tmp_path):
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            manager.fetch_from_jamendo("meditative", search_query="rain, piano")

        mock_search.assert_called_once_with("meditative, rain, piano", 60, "fake-client-id")


class TestRandomLocalSelection:
    def test_picks_randomly_among_tracks_satisfying_duration(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        track_a = mood_dir / "a.wav"
        track_b = mood_dir / "b.wav"
        _write_tone(track_a, 5000)
        _write_tone(track_b, 5000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))

        with patch("random.choice") as mock_choice:
            mock_choice.return_value = str(track_b)
            result = manager.select_audio_music("calm", duration=2)

        assert result == str(track_b)
        assert set(mock_choice.call_args.args[0]) == {str(track_a), str(track_b)}

    def test_varies_across_calls_given_multiple_candidates(self, tmp_path):
        # Not mocking random this time -- a real statistical check that both
        # tracks get picked across enough calls, not the same one every time.
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        track_a = mood_dir / "a.wav"
        track_b = mood_dir / "b.wav"
        _write_tone(track_a, 5000)
        _write_tone(track_b, 5000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        results = {manager.select_audio_music("calm", duration=2) for _ in range(30)}

        assert results == {str(track_a), str(track_b)}


class TestGetMusicPath:
    def test_existing_file_path_returned_as_is(self, tmp_path):
        music_file = tmp_path / "track.wav"
        _write_tone(music_file, 500)

        manager = MusicManager()
        assert manager.get_music_path(music_input=str(music_file)) == str(music_file)

    def test_valid_mood_name_selects_from_bgm_directory(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "joy"
        mood_dir.mkdir(parents=True)
        track = mood_dir / "track.wav"
        _write_tone(track, 2000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        assert manager.get_music_path(music_input="joy", duration=1) == str(track)

    def test_unregistered_mood_name_still_selects_from_bgm_directory(self, tmp_path):
        # "meditative" isn't one of MusicManager's available_moods, but
        # music_input isn't limited to that registry -- any name is used as
        # a mood directly.
        mood_dir = tmp_path / "bgm" / "meditative"
        mood_dir.mkdir(parents=True)
        track = mood_dir / "track.wav"
        _write_tone(track, 2000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        assert manager.get_music_path(music_input="meditative", duration=1) == str(track)

    def test_unregistered_mood_name_reaches_jamendo_fallback(self, tmp_path):
        # Previously an unregistered mood name silently fell back to
        # default_mood before ever reaching select_audio_music/Jamendo --
        # it must now actually be looked up (and searched) under its own name.
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), jamendo_client_id="fake-client-id")

        with patch("tish_video_sdk.music.music_packs.search_track", return_value=_fake_track()) as mock_search, \
             patch("tish_video_sdk.music.music_packs.download_track", side_effect=_fake_download_track):
            result = manager.get_music_path(music_input="meditative", duration=2)

        mock_search.assert_called_once_with("meditative", 2, "fake-client-id")
        assert os.path.basename(os.path.dirname(result)) == "meditative"

    def test_reference_text_drives_mood_analysis(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "joy"
        mood_dir.mkdir(parents=True)
        track = mood_dir / "track.wav"
        _write_tone(track, 2000)

        with patch("tish_video_sdk.reasoning.generate", return_value="joy"):
            manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), language_code="en")
            assert manager.get_music_path(reference_text="Everyone cheered!", duration=1) == str(track)

    def test_no_input_falls_back_to_default_mood(self, tmp_path):
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        track = mood_dir / "track.wav"
        _write_tone(track, 2000)

        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"), default_mood="calm")
        assert manager.get_music_path(duration=1) == str(track)


class TestAddBackgroundMusic:
    def test_mixes_and_exports(self, tmp_path):
        speech_file = tmp_path / "speech.wav"
        _write_tone(speech_file, 2000)
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        _write_tone(mood_dir / "track.wav", 5000)

        output_file = tmp_path / "mixed.wav"
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        manager.add_background_music(str(speech_file), str(output_file), music_input="calm")

        assert output_file.exists()
        # Overlay keeps the base (speech) track's own length.
        assert len(AudioSegment.from_wav(str(output_file))) == len(AudioSegment.from_wav(str(speech_file)))

    def test_loops_short_music_to_cover_speech(self, tmp_path):
        speech_file = tmp_path / "speech.wav"
        _write_tone(speech_file, 3000)
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        _write_tone(mood_dir / "track.wav", 500)  # much shorter than speech + 1s buffer

        output_file = tmp_path / "mixed.wav"
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        manager.add_background_music(str(speech_file), str(output_file), music_input="calm")

        assert output_file.exists()

    def test_missing_speech_file_raises(self, tmp_path):
        manager = MusicManager()
        with pytest.raises(Exception):
            manager.add_background_music(str(tmp_path / "no-such-file.wav"), str(tmp_path / "out.wav"))

    def test_silent_speech_falls_back_to_flat_reduction_without_crashing(self, tmp_path):
        # AudioSegment.silent() measures as -inf dBFS -- the loudness-relative
        # gain formula can't target an offset below that, so this exercises
        # the flat-cut fallback instead of dividing by/against -inf.
        speech_file = tmp_path / "silent_speech.wav"
        AudioSegment.silent(duration=2000).export(str(speech_file), format="wav")
        mood_dir = tmp_path / "bgm" / "calm"
        mood_dir.mkdir(parents=True)
        _write_tone(mood_dir / "track.wav", 5000)

        output_file = tmp_path / "mixed.wav"
        manager = MusicManager(bgm_directory=str(tmp_path / "bgm"))
        manager.add_background_music(str(speech_file), str(output_file), music_input="calm")

        assert output_file.exists()


class TestLoudnessBasedMixing:
    """Covers the fix for a flat per-call dB reduction producing wildly
    different relative balance depending on a bgm track's own source
    loudness -- gain is now computed per file from each side's measured
    dBFS instead."""

    def test_gain_targets_measured_offset_below_speech(self):
        manager = MusicManager()
        gain = manager._music_gain_db(speech_dbfs=-15.0, music_dbfs=-30.0, below_speech_db=10.0)
        assert gain == pytest.approx(5.0)  # target -25 dBFS; -30 + 5 = -25

    def test_falls_back_to_flat_cut_when_either_side_is_silent(self):
        manager = MusicManager()
        assert manager._music_gain_db(float("-inf"), -20.0, 10.0) == -10.0
        assert manager._music_gain_db(-20.0, float("-inf"), 10.0) == -10.0

    def test_final_mix_loudness_is_consistent_regardless_of_source_track_volume(self, tmp_path):
        speech_file = tmp_path / "speech.wav"
        _write_tone(speech_file, 3000)

        loud_track = tmp_path / "loud.wav"
        quiet_track = tmp_path / "quiet.wav"
        _write_tone(loud_track, 5000)
        # Same tone, 25 dB quieter -- simulates a bgm library where source
        # tracks vary widely in their own recording level (the real one used
        # for this SDK spans about 15 dB between its quietest and loudest
        # tracks).
        AudioSegment.from_wav(str(loud_track)).apply_gain(-25).export(str(quiet_track), format="wav")

        manager = MusicManager()
        mixed_loud = manager.get_audio_with_bgm_from_file(str(speech_file), str(loud_track))
        mixed_quiet = manager.get_audio_with_bgm_from_file(str(speech_file), str(quiet_track))

        # A flat "-10dB from the track's own volume" would leave the quiet
        # track's contribution far below the loud track's; measuring and
        # normalizing per file makes the two mixes land at essentially the
        # same overall loudness instead.
        assert mixed_loud.dBFS == pytest.approx(mixed_quiet.dBFS, abs=0.5)

    def test_dbfs_is_measured_once_per_file_path_in_memory(self, tmp_path):
        track = tmp_path / "track.wav"
        _write_tone(track, 500)
        segment = AudioSegment.from_wav(str(track))

        manager = MusicManager()
        with patch.object(AudioSegment, "dBFS", new_callable=PropertyMock) as mock_dbfs:
            mock_dbfs.return_value = -20.0
            first = manager._measured_dbfs(str(track), segment)
            second = manager._measured_dbfs(str(track), segment)

        assert first == second == -20.0
        assert mock_dbfs.call_count == 1

    def test_dbfs_persists_across_manager_instances_via_sidecar_file(self, tmp_path):
        track = tmp_path / "track.wav"
        _write_tone(track, 500)
        segment = AudioSegment.from_wav(str(track))

        first_manager = MusicManager()
        with patch.object(AudioSegment, "dBFS", new_callable=PropertyMock) as mock_dbfs:
            mock_dbfs.return_value = -17.5
            measured = first_manager._measured_dbfs(str(track), segment)
        assert measured == -17.5
        assert os.path.exists(str(track) + ".cache.json")

        # A brand-new instance (simulating a fresh process) has an empty
        # in-memory cache, so this only succeeds if it reads the sidecar
        # file _measured_dbfs just wrote rather than re-measuring.
        second_manager = MusicManager()
        with patch.object(AudioSegment, "dBFS", new_callable=PropertyMock) as mock_dbfs_2:
            mock_dbfs_2.return_value = -99.0  # would surface if a re-measure happened
            cached = second_manager._measured_dbfs(str(track), segment)

        assert cached == -17.5
        mock_dbfs_2.assert_not_called()

    def test_sidecar_cache_invalidated_when_source_file_changes(self, tmp_path):
        track = tmp_path / "track.wav"
        _write_tone(track, 500)
        segment_v1 = AudioSegment.from_wav(str(track))

        manager = MusicManager()
        with patch.object(AudioSegment, "dBFS", new_callable=PropertyMock) as mock_dbfs:
            mock_dbfs.return_value = -10.0
            manager._measured_dbfs(str(track), segment_v1)

        # Replace the file with different (longer, so differently-sized)
        # content -- the stale sidecar entry must not be trusted anymore.
        _write_tone(track, 2000)
        segment_v2 = AudioSegment.from_wav(str(track))

        fresh_manager = MusicManager()  # sidestep the in-memory cache
        with patch.object(AudioSegment, "dBFS", new_callable=PropertyMock) as mock_dbfs_2:
            mock_dbfs_2.return_value = -30.0
            result = fresh_manager._measured_dbfs(str(track), segment_v2)

        assert result == -30.0


class TestLoadAudioMoviepyFallback:
    def test_falls_back_to_moviepy_when_pydub_cannot_decode(self, tmp_path):
        bogus_mp3 = tmp_path / "bogus.mp3"
        bogus_mp3.write_bytes(b"not a real mp3")
        expected_wav = tmp_path / "bogus.wav"

        def _fake_write_audiofile(path, logger=None):
            _write_tone(path, 500)

        mock_clip = MagicMock()
        mock_clip.write_audiofile.side_effect = _fake_write_audiofile

        # Only the mp3 load should fail -- the wav read-back after moviepy's
        # conversion must still go through pydub for real.
        real_from_file = AudioSegment.from_file

        def _flaky_from_file(file, *args, **kwargs):
            if str(file).lower().endswith(".mp3"):
                raise CouldntDecodeError("boom")
            return real_from_file(file, *args, **kwargs)

        manager = MusicManager()
        with patch("pydub.AudioSegment.from_file", side_effect=_flaky_from_file), \
             patch("moviepy.AudioFileClip", return_value=mock_clip):
            segment = manager._load_audio(str(bogus_mp3))

        assert isinstance(segment, AudioSegment)
        assert expected_wav.exists()
