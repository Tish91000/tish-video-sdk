"""Mocked VideoBuilder/video_maker tests -- no real moviepy rendering, TTS
calls, or ffmpeg invocations, safe to run anywhere.

Run with: pytest tests/fake
"""
import pytest
from unittest.mock import MagicMock, patch

from tish_video_sdk.video_maker import VideoBuilder, TextStyle


class TestSingleImageVideo:
    @patch('tish_video_sdk.video_maker.os.path.exists')
    @patch('tish_video_sdk.video_maker.AudioFileClip')
    @patch('tish_video_sdk.video_maker.VideoBuilder._load_image_clip_safe')
    def test_create_video_with_audio(self, mock_load_image, mock_audio_clip, mock_exists):
        mock_exists.return_value = True

        mock_audio = MagicMock()
        mock_audio.duration = 15.0
        mock_audio_clip.return_value = mock_audio

        mock_video = MagicMock()
        mock_video.duration = 15.0
        mock_video.with_fps.return_value = mock_video

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_image.with_duration.return_value = mock_video
        mock_video.with_audio.return_value = mock_video

        builder = VideoBuilder.from_single_image_with_audio('path/to/image.jpg', 'path/to/audio.mp3', fps=30)
        video_clip = builder.build()

        assert video_clip is not None
        assert video_clip.duration > 0

    @patch('tish_video_sdk.video_maker.os.path.exists')
    @patch('tish_video_sdk.video_maker.VideoBuilder._load_image_clip_safe')
    def test_create_video_with_duration(self, mock_load_image, mock_exists):
        mock_exists.return_value = True

        mock_video = MagicMock()
        mock_video.duration = 10.0
        mock_video.with_fps.return_value = mock_video

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_image.with_duration.return_value = mock_video

        builder = VideoBuilder.from_single_image('path/to/image.jpg', duration=10.0, fps=30)
        video_clip = builder.build()

        assert video_clip is not None
        assert video_clip.duration == 10.0

    @patch('tish_video_sdk.video_maker.os.path.exists')
    def test_no_audio_and_no_duration_returns_none(self, mock_exists):
        mock_exists.return_value = True
        builder = VideoBuilder.from_single_image('path/to/image.jpg', duration=None)
        assert builder.build() is None


class TestSave:
    def test_save_writes_file(self):
        mock_clip = MagicMock()
        mock_clip.fps = 24

        builder = VideoBuilder()
        builder.video_clip = mock_clip

        result = builder.save('output/video.mp4')

        assert result == 'output/video.mp4'
        mock_clip.write_videofile.assert_called_once()

    def test_save_with_no_clip_returns_none(self):
        assert VideoBuilder().save('output/video.mp4') is None


class TestVideoBuilderTTSWiring:
    @patch('tish_video_sdk.video_maker.TTSBuilder')
    def test_generate_tts_audio_passes_provider_not_credentials(self, mock_tts_builder_cls):
        """Regression test: _generate_tts_audio() used to pass google_api_key
        (a stale pre-VoicePack param) as TTSBuilder.from_text()'s third
        positional argument, which is actually `provider`. Confirms the fixed
        wiring passes the builder's own `provider` filter through correctly."""
        mock_instance = MagicMock()
        mock_instance.build.return_value = MagicMock()
        mock_instance.save.return_value = '/tmp/tts_audio.wav'
        mock_tts_builder_cls.from_text.return_value = mock_instance

        builder = VideoBuilder.from_single_image_with_tts(
            'path/to/image.jpg', 'Hello world', 'xx', provider='gemini'
        )
        result = builder._generate_tts_audio()

        assert result is True
        mock_tts_builder_cls.from_text.assert_called_once_with('Hello world', 'xx', 'gemini')


class TestOverlayChaining:
    def test_with_overlay_image_returns_self(self):
        builder = VideoBuilder()
        result = builder.with_overlay_image('logo.png')
        assert result is builder
        assert builder._overlay_images[0]['path'] == 'logo.png'

    def test_with_overlay_video_returns_self(self):
        builder = VideoBuilder()
        result = builder.with_overlay_video('reactive.mp4')
        assert result is builder
        assert builder._overlay_video_path == 'reactive.mp4'


class TestTextStyle:
    def test_default_values(self):
        style = TextStyle()
        assert style.font_size == 50
        assert style.font_color == 'black'
        assert style.bg_color is None
        assert style.text_position == (0.5, 0.5)

    def test_custom_values(self):
        style = TextStyle(font_size=50, font_color='yellow', bg_color='blue', text_position=('left', 'top'))
        assert style.font_size == 50
        assert style.font_color == 'yellow'
        assert style.bg_color == 'blue'
        assert style.text_position == ('left', 'top')

    def test_invalid_font_size(self):
        with pytest.raises(ValueError):
            TextStyle(font_size=-10)

    def test_invalid_color(self):
        with pytest.raises(ValueError):
            TextStyle(font_color='not_a_color')

    def test_highlight_color_default_and_custom(self):
        style = TextStyle()
        assert style.highlight_color == '#FFD700'

        custom_style = TextStyle(highlight_color='#FF0000')
        assert custom_style.highlight_color == '#FF0000'

    def test_invalid_highlight_color(self):
        with pytest.raises(ValueError):
            TextStyle(highlight_color='not_a_color')


class TestFontResolution:
    """Language-specific font choices must be driven entirely by config
    (SubtitlePack -> TextStyle), and a font name listing several fallback
    variants must resolve to a real font file rather than silently falling
    back to a default."""

    def test_get_font_for_language_returns_configured_font_family(self):
        style = TextStyle(language_code='ta', font_family='Some-Custom-Font')
        assert style.get_font_for_language() == 'Some-Custom-Font'

    @patch('tish_video_sdk.video_maker.os.path.exists')
    def test_get_font_path_resolves_ampersand_joined_fallback_chain(self, mock_exists):
        # Fonts dir exists; only Nirmala.ttc exists on disk (its real-world
        # layout -- Windows ships Nirmala as a single .ttc, not .ttf).
        mock_exists.side_effect = lambda p: p.lower().endswith(('fonts', 'nirmala.ttc'))

        builder = VideoBuilder()
        path = builder._get_font_path(
            'Nirmala-UI-&-Nirmala-UI-Bold-&-Nirmala-UI-Semilight-&-'
            'Nirmala-Text-&-Nirmala-Text-Bold-&-Nirmala-Text-Semilight'
        )
        # The first variant in the chain must resolve to its real font file.
        assert path.lower().endswith('nirmala.ttc')


class TestTextSegments:
    def test_with_text_segments_returns_self(self):
        builder = VideoBuilder()
        segments = [{'text': 'Title', 'start': 0.0, 'end': 3.0}]
        style = TextStyle(font_color='white')

        result = builder.with_text_segments(segments, style=style)

        assert result is builder
        assert builder._text_segments == segments
        assert builder._text_style is style

    @patch('tish_video_sdk.video_maker.os.path.exists')
    @patch('tish_video_sdk.video_maker.VideoBuilder._overlay_text_segments_on_single_clip')
    @patch('tish_video_sdk.video_maker.VideoBuilder._load_image_clip_safe')
    def test_build_applies_text_segments(self, mock_load_image, mock_overlay, mock_exists):
        mock_exists.return_value = True

        mock_video = MagicMock()
        mock_video.duration = 5.0
        mock_video.with_fps.return_value = mock_video

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_image.with_duration.return_value = mock_video

        mock_overlaid = MagicMock()
        mock_overlaid.duration = 5.0
        mock_overlay.return_value = mock_overlaid

        segments = [{'text': 'Title', 'start': 0.0, 'end': 5.0}]
        builder = VideoBuilder.from_single_image('path/to/image.jpg', duration=5.0)
        builder.with_text_segments(segments)
        result = builder.build()

        assert result is mock_overlaid
        mock_overlay.assert_called_once_with(mock_video, segments)

    @patch('tish_video_sdk.video_maker.os.path.exists')
    @patch('tish_video_sdk.video_maker.VideoBuilder._overlay_text_segments_on_single_clip')
    @patch('tish_video_sdk.video_maker.VideoBuilder._load_image_clip_safe')
    def test_build_returns_none_when_overlay_fails(self, mock_load_image, mock_overlay, mock_exists):
        mock_exists.return_value = True

        mock_video = MagicMock()
        mock_video.duration = 5.0
        mock_video.with_fps.return_value = mock_video

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_image.with_duration.return_value = mock_video

        mock_overlay.return_value = None

        builder = VideoBuilder.from_single_image('path/to/image.jpg', duration=5.0)
        builder.with_text_segments([{'text': 'Title', 'start': 0.0, 'end': 5.0}])

        assert builder.build() is None


class TestSubtitlesWiring:
    def test_with_subtitles_is_a_thin_wrapper_over_text_segments(self):
        builder = VideoBuilder()
        segments = [{'text': 'Hello', 'start': 0.0, 'end': 1.0}]
        style = TextStyle(font_color='white')

        result = builder.with_subtitles(segments, style=style)

        assert result is builder
        assert builder._text_segments == segments
        assert builder._text_style is style

    def test_with_tts_subtitles_sets_flag_not_segments(self):
        builder = VideoBuilder()
        style = TextStyle(font_color='yellow')

        result = builder.with_tts_subtitles(style=style)

        assert result is builder
        assert builder._enable_tts_subtitles is True
        assert builder._text_style is style
        assert builder._text_segments is None  # not known until build() runs TTS

    def test_with_tts_subtitles_defaults_to_language_subtitle_pack(self):
        # 'fr' has a built-in SubtitlePack (see DEFAULT_SUBTITLE_PACKS) --
        # no explicit style means resolve one from it rather than a bare TextStyle().
        builder = VideoBuilder.from_single_image_with_tts('path/to/image.jpg', 'Bonjour', 'fr')
        builder.with_tts_subtitles()

        assert builder._text_style is not None
        assert builder._text_style.language_code == 'fr'

    def test_with_subtitles_falls_back_to_bare_text_style_without_a_language(self):
        # No TTS involved, so no language is known -- nothing to resolve a
        # SubtitlePack from, so with_text_segments()'s own TextStyle() default applies.
        builder = VideoBuilder()
        builder.with_subtitles([{'text': 'Hello', 'start': 0.0, 'end': 1.0}])

        assert builder._text_style is None

    @patch('tish_video_sdk.video_maker.TTSBuilder')
    def test_generate_tts_audio_enables_subtitles_on_tts_builder_when_requested(self, mock_tts_builder_cls):
        mock_instance = MagicMock()
        mock_instance.with_subtitles.return_value = mock_instance
        mock_instance.build.return_value = MagicMock()
        mock_instance.save.return_value = '/tmp/tts_audio.wav'
        mock_tts_builder_cls.from_text.return_value = mock_instance

        builder = VideoBuilder.from_single_image_with_tts('path/to/image.jpg', 'Hello world', 'xx')
        builder.with_tts_subtitles()
        builder._generate_tts_audio()

        mock_instance.with_subtitles.assert_called_once()

    @patch('tish_video_sdk.video_maker.os.path.exists')
    @patch('tish_video_sdk.video_maker.VideoBuilder._overlay_text_segments_on_single_clip')
    @patch('tish_video_sdk.video_maker.VideoBuilder._generate_tts_audio')
    @patch('tish_video_sdk.video_maker.VideoBuilder._load_image_clip_safe')
    @patch('tish_video_sdk.video_maker.AudioFileClip')
    def test_build_extracts_and_applies_tts_subtitles(self, mock_audio_clip, mock_load_image, mock_generate_tts, mock_overlay, mock_exists):
        mock_exists.return_value = True

        mock_audio = MagicMock()
        mock_audio.duration = 5.0
        mock_audio_clip.return_value = mock_audio

        def _fake_generate_tts_audio(self=None):
            builder._audio_filepath = 'path/to/tts_audio.wav'
            return True
        mock_generate_tts.side_effect = _fake_generate_tts_audio

        mock_video = MagicMock()
        mock_video.duration = 5.0
        mock_video.with_fps.return_value = mock_video

        mock_image = MagicMock()
        mock_load_image.return_value = mock_image
        mock_image.with_duration.return_value = mock_video
        mock_video.with_audio.return_value = mock_video

        mock_overlaid = MagicMock()
        mock_overlaid.duration = 5.0
        mock_overlay.return_value = mock_overlaid

        segments = [{'text': 'Hello', 'start': 0.0, 'end': 5.0}]
        mock_tts_builder = MagicMock()
        mock_tts_builder.get_subtitle_segments.return_value = segments

        builder = VideoBuilder.from_single_image_with_tts('path/to/image.jpg', 'Hello world', 'xx')
        builder.with_tts_subtitles()
        builder._tts_builder = mock_tts_builder  # what the (mocked) _generate_tts_audio would have set

        result = builder.build()

        assert result is mock_overlaid
        mock_overlay.assert_called_once_with(mock_video, segments)


class TestAdvancedSegments:
    def test_from_multi_segments_advanced_sets_state(self):
        segments = [{'image': 'a.png', 'text': 'Hello', 'repeat': 2}]
        builder = VideoBuilder.from_multi_segments_advanced(segments)
        assert builder._advanced_segments == segments

    def test_expand_segments_skips_missing_fields(self):
        builder = VideoBuilder.from_multi_segments_advanced([{'text': 'no image'}])
        assert builder._expand_segments_with_repetitions() == []

    @patch('tish_video_sdk.video_maker.VideoBuilder._generate_and_cache_audio')
    def test_expand_segments_with_repetitions(self, mock_generate_cache):
        mock_generate_cache.return_value = True

        segments = [
            {'image': 'a.png', 'text': 'Hello', 'language_code': 'xx', 'repeat': 3},
            {'image': 'b.png', 'text': 'World', 'language_code': 'xx'},
        ]
        builder = VideoBuilder.from_multi_segments_advanced(segments)

        expanded = builder._expand_segments_with_repetitions()

        assert len(expanded) == 4  # 3 repeats of "Hello" + 1 "World"
        hello_reps = [s['_repetition'] for s in expanded if s['text'] == 'Hello']
        assert hello_reps == [1, 2, 3]
        # Audio generation runs once per unique text, not once per repetition.
        assert mock_generate_cache.call_count == 2

    @patch('tish_video_sdk.video_maker.VideoBuilder._generate_and_cache_audio')
    def test_expand_segments_skips_segment_when_tts_generation_fails(self, mock_generate_cache):
        mock_generate_cache.return_value = False

        builder = VideoBuilder.from_multi_segments_advanced(
            [{'image': 'a.png', 'text': 'Hello', 'language_code': 'xx'}]
        )

        assert builder._expand_segments_with_repetitions() == []

    def test_create_segment_video_clip_reuses_video_cache(self):
        cached_clip = MagicMock()
        cached_clip.copy.return_value = cached_clip
        cached_clip.duration = 2.0

        builder = VideoBuilder()
        builder._video_cache['key'] = cached_clip

        with patch('tish_video_sdk.video_maker.ImageClip') as mock_image_clip:
            result = builder._create_segment_video_clip(
                {'_video_cache_key': 'key', '_audio_cache_key': 'a', '_repetition': 2}, 0.0
            )
            mock_image_clip.assert_not_called()  # cache hit skips clip creation entirely

        assert result is cached_clip

    def test_cleanup_cache_clears_all_caches_and_closes_clips(self):
        builder = VideoBuilder()
        mock_video_clip = MagicMock()
        mock_bgm_clip = MagicMock()
        builder._audio_cache = {'k': {'audio_path': None}}
        builder._video_cache = {'k': mock_video_clip}
        builder._bgm_cache = {'k': mock_bgm_clip}

        builder.cleanup_cache()

        assert builder._audio_cache == {}
        assert builder._video_cache == {}
        assert builder._bgm_cache == {}
        mock_video_clip.close.assert_called_once()
        mock_bgm_clip.close.assert_called_once()
