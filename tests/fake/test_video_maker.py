"""Mocked VideoBuilder/video_maker tests -- no real moviepy rendering, TTS
calls, or ffmpeg invocations, safe to run anywhere.

Run with: pytest tests/fake

Subtitle support (auto-generating text segments via forced alignment, and the
advanced multi-segment "chapelet" builder) isn't available yet -- see ADR
0003 -- so there's nothing to test for it here; it's built on top of
with_text_segments() once tish_video_sdk.subtitles lands.
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
