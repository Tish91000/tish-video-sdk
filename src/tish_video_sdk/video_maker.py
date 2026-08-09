from __future__ import annotations

import os

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    concatenate_audioclips,
)
from moviepy.video.fx import MaskColor, Loop
import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageColor, ImageDraw, ImageFont
import uuid # For unique temp filenames
import traceback # Debugging

# Simple Dummy Logger to fix MoviePy/proglog 'NoneType' stdout errors
class DummyLogger:
    def __init__(self):
        self.stdout = self # Self-referential to handle logger.stdout.write calls
    def __call__(self, *args, **kwargs):
        pass
    def callback(self, **kwargs):
        pass
    def progress(self, *args, **kwargs):
        pass
    def message(self, *args, **kwargs):
        pass
    def write(self, *args, **kwargs): # Handle stdout.write
        pass
    def flush(self): # Handle stdout.flush
        pass
    def iter_bar(self, **kwargs): # Handle progress bar iteration
        return kwargs.get("chunk", [])

from .tts import TTSBuilder
from .subtitles import get_subtitle_pack
from typing import Optional, List, Dict, Tuple, Union
import tempfile

DEFAULT_FPS = 24 # Frames per second for the output video

# Font mapping for common families, resolved against the Windows Fonts folder.
DEFAULT_FONT_SIZE = 50
DEFAULT_TEXT_POSITION = (0.5, 0.5)  # Centered in the middle of the video
DEFAULT_FONT_COLOR = 'black'
DEFAULT_BG_COLOR = None  # Transparent background
DEFAULT_STROKE_COLOR = 'black'
DEFAULT_STROKE_WIDTH = 3
DEFAULT_FONT_FAMILY = 'Arial-Bold'
DEFAULT_BOX_SIZE = (0.8, None)

# Specific font mapping for certain languages (e.g., Tamil requires a different font)
FONT_FAMILY_BY_LANGUAGE = {
    'ta': 'Nirmala-UI-&-Nirmala-UI-Bold-&-Nirmala-UI-Semilight-&-Nirmala-Text-&-Nirmala-Text-Bold-&-Nirmala-Text-Semilight',
    'default': DEFAULT_FONT_FAMILY,
}


@dataclass
class TextStyle:
    """Styling for a block of text (or word-timed text segments) rendered onto
    a video via :meth:`VideoBuilder.with_text_segments` -- font, color,
    position, and the animated per-word highlight used for karaoke-style
    captions. Not tied to subtitles specifically: a single-segment title/caption
    uses the same style class."""
    font_size: int = DEFAULT_FONT_SIZE
    font_color: str = DEFAULT_FONT_COLOR
    font_family: str = DEFAULT_FONT_FAMILY
    stroke_color: str = DEFAULT_STROKE_COLOR
    stroke_width: int = DEFAULT_STROKE_WIDTH
    bg_color: Optional[str] = DEFAULT_BG_COLOR
    text_position: Tuple = DEFAULT_TEXT_POSITION
    box_size: Tuple[Optional[float], Optional[float]] = DEFAULT_BOX_SIZE
    highlight_color: str = '#FFD700'
    timing_offset: float = 0.0
    language_code: str = 'en'

    def get_font_for_language(self) -> str:
        """Return appropriate font family for the language."""
        return FONT_FAMILY_BY_LANGUAGE.get(self.language_code, self.font_family)

    def __post_init__(self):
        if self.font_size <= 0:
            raise ValueError("font_size must be positive")
        if self.stroke_width < 0:
            raise ValueError("stroke_width must be non-negative")
        if not isinstance(self.text_position, (tuple, list)) or len(self.text_position) != 2:
            raise ValueError("text_position must be a tuple/list of 2 elements")
        if not isinstance(self.box_size, (tuple, list)) or len(self.box_size) != 2:
            raise ValueError("box_size must be a tuple/list of 2 elements")
        if self.font_color:
            try:
                ImageColor.getrgb(self.font_color)
            except ValueError:
                raise ValueError(f"Invalid font_color: {self.font_color}")
        if self.bg_color is not None:
            try:
                ImageColor.getrgb(self.bg_color)
            except ValueError:
                raise ValueError(f"Invalid bg_color: {self.bg_color}")
        if self.highlight_color:
            try:
                ImageColor.getrgb(self.highlight_color)
            except ValueError:
                raise ValueError(f"Invalid highlight_color: {self.highlight_color}")


class VideoBuilder:
    """
    A flexible video builder class that can create videos from various combinations of media.

    The class automatically determines video duration based on the provided inputs:
    - If audio is provided: video duration matches audio duration
    - If TTS text is provided: audio is generated and video duration matches the generated audio
    - If no audio but duration is specified: uses the specified duration
    - For multi-image segments: uses the timestamps from segments
    """

    def __init__(self, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920):
        """
        Initialize the VideoBuilder.

        Args:
            fps (int): Frames per second.
            width (int): Target video width.
            height (int): Target video height.
        """
        self.fps = fps
        self.width = width
        self.height = height

        self.video_clip: Optional[VideoFileClip] = None
        self._active_audio_clip: Optional[AudioFileClip] = None # Prevent premature closing
        self._audio_filepath: Optional[str] = None
        self._duration: Optional[float] = None
        self._image_filepath: Optional[str] = None
        self._image_segments: Optional[List[Dict]] = None
        self._advanced_segments: Optional[List[Dict]] = None
        self._tts_builder: Optional[TTSBuilder] = None
        self._tts_text: Optional[str] = None
        self._language_code: Optional[str] = None
        self._provider: Optional[str] = None

        # Overlay configuration
        self._overlay_images: List[Dict] = []

        # Overlay Video configuration
        self._overlay_video_path: Optional[str] = None
        self._overlay_video_config: Optional[Dict] = None

        # Timed text segments (titles, captions, subtitles, ...) configuration
        self._text_segments: Optional[List[Dict]] = None
        self._text_style: Optional[TextStyle] = None
        self._enable_tts_subtitles: bool = False

        # Two-level caching for from_multi_segments_advanced() (TTS audio +
        # subtitles per segment, complete video clips, background music)
        self._audio_cache: Dict[str, Dict] = {}
        self._video_cache: Dict[str, VideoFileClip] = {}
        self._bgm_cache: Dict[str, AudioFileClip] = {}

    @classmethod
    def from_single_image_with_audio(cls, image_filepath: str, audio_filepath: str, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """Create a VideoBuilder from a single image and an audio file."""
        builder = cls(fps, width, height)
        builder._image_filepath = image_filepath
        builder._audio_filepath = audio_filepath
        return builder

    @classmethod
    def from_single_image_with_tts(cls, image_filepath: str, text: str, language_code: str, provider: Optional[str] = None, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """
        Create a VideoBuilder from a single image and TTS text.
        Audio will be generated from text and video duration will match the generated audio.

        Args:
            image_filepath (str): Path to the image file.
            text (str): Text to be converted to speech.
            language_code (str): Language code for TTS (e.g., 'fr', 'en', 'es', 'it', 'ta').
            provider (str, optional): Pin TTS synthesis to one provider (e.g. "gemini",
                "google_cloud"); omit to try every VoicePack configured for the language
                in order, falling back on failure (see TTSBuilder).
            fps (int): Frames per second for the output video.
            width (int): Target video width.
            height (int): Target video height.

        Returns:
            VideoBuilder: Configured VideoBuilder instance.
        """
        builder = cls(fps, width, height)
        builder._image_filepath = image_filepath
        builder._tts_text = text
        builder._language_code = language_code
        builder._provider = provider
        return builder

    @classmethod
    def from_single_image(cls, image_filepath: str, duration: float, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """Create a VideoBuilder from a single image with specific duration (silent)."""
        builder = cls(fps, width, height)
        builder._image_filepath = image_filepath
        builder._duration = duration
        return builder

    @classmethod
    def from_multi_images_with_audio(cls, image_segments: List[Dict], audio_filepath: str, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """Create a VideoBuilder from multiple images with timestamps and an audio file."""
        builder = cls(fps, width, height)
        builder._image_segments = image_segments
        builder._audio_filepath = audio_filepath
        return builder

    @classmethod
    def from_multi_images_with_tts(cls, image_segments: List[Dict], text: str, language_code: str, provider: Optional[str] = None, fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """
        Create a VideoBuilder from multiple images with timestamps and TTS text.
        Audio will be generated from text.

        Args:
            image_segments (List[Dict]): List of segments with 'path', 'start', and 'end' keys.
            text (str): Text to be converted to speech.
            language_code (str): Language code for TTS.
            provider (str, optional): Pin TTS synthesis to one provider; omit to try every
                VoicePack configured for the language in order, falling back on failure
                (see TTSBuilder).
            fps (int): Frames per second for the output video.
            width (int): Target video width.
            height (int): Target video height.

        Returns:
            VideoBuilder: Configured VideoBuilder instance.
        """
        builder = cls(fps, width, height)
        builder._image_segments = image_segments
        builder._tts_text = text
        builder._language_code = language_code
        builder._provider = provider
        return builder

    @classmethod
    def from_multi_images(cls, image_segments: List[Dict], fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """Create a VideoBuilder from multiple images with timestamps (silent video)."""
        builder = cls(fps, width, height)
        builder._image_segments = image_segments
        return builder

    @classmethod
    def from_multi_segments_advanced(cls, segments: List[Dict], fps: int = DEFAULT_FPS, width: int = 1080, height: int = 1920) -> 'VideoBuilder':
        """
        Create a VideoBuilder from advanced segments with intelligent repetition handling.

        Segment structure::

            {
                "image": "path/to/image.png",
                "text": "Text to be spoken",
                "bgm": "path/to/background_music.wav",  # optional
                "repeat": 10,  # optional, default 1
                "language_code": "fr",  # optional, defaults to 'fr'
                "provider": "gemini",  # optional, pins TTS to one provider for this segment
            }

        Optimizations:

        - TTS audio and subtitles are cached and reused for identical text + language combinations
        - Complete video clips are cached and reused for identical text + language + image combinations
        - Background music is handled per segment

        Args:
            segments (List[Dict]): List of advanced segment configurations.
            fps (int): Frames per second for the output video.

        Returns:
            VideoBuilder: Configured VideoBuilder instance.
        """
        builder = cls(fps, width, height)
        builder._advanced_segments = segments
        return builder

    def with_subtitles(self, segments: List[Dict], style: Optional[TextStyle] = None) -> 'VideoBuilder':
        """
        Add pre-supplied subtitle/caption segments to the video (e.g. from
        tish_video_sdk.subtitles.SubtitleBuilder, or hand-written). Thin
        wrapper over with_text_segments() -- subtitles are just timed text.

        Args:
            segments (List[Dict]): Timed segments, see with_text_segments().
            style (TextStyle, optional): Styling to apply. Defaults to the
                configured SubtitlePack for this builder's language (see
                tish_video_sdk.subtitles), falling back to TextStyle() if
                none is configured or no language is known.

        Returns:
            VideoBuilder: Self for method chaining.
        """
        return self.with_text_segments(segments, style or self._default_subtitle_style())

    def with_tts_subtitles(self, style: Optional[TextStyle] = None) -> 'VideoBuilder':
        """
        Enable automatic subtitle generation from TTS audio via forced
        alignment (see tish_video_sdk.subtitles.SubtitleBuilder). Only works
        when using TTS for audio generation. Segments aren't known until the
        TTS builder has synthesized audio, so build() extracts them from
        self._tts_builder and applies them via with_text_segments() once
        available.

        Args:
            style (TextStyle, optional): Styling to apply. Defaults to the
                configured SubtitlePack for this builder's language (see
                tish_video_sdk.subtitles), falling back to TextStyle() if
                none is configured.

        Returns:
            VideoBuilder: Self for method chaining.
        """
        self._enable_tts_subtitles = True
        self._text_style = style or self._default_subtitle_style()
        return self

    def _default_subtitle_style(self) -> Optional[TextStyle]:
        """Resolve a language-appropriate default style from a configured
        SubtitlePack, when this builder's language is known. Returns None
        (letting with_text_segments() fall back to bare TextStyle()) if no
        language is known yet or no pack is configured for it -- e.g. manual
        subtitles supplied without ever going through TTS."""
        if self._language_code:
            pack = get_subtitle_pack(self._language_code)
            if pack:
                return TextStyle(**pack.style_kwargs())
        return None

    def with_overlay_image(self, image_path: str, position: Union[tuple, str] = ('center', 0.7), width: Optional[int] = None, height: Optional[int] = None, corner_radius: int = 0) -> 'VideoBuilder':
        """
        Add an overlay image (e.g., poster/album art, logo) to the video.

        Args:
            image_path (str): Path to the image file.
            position (tuple or str): Position tuple (x, y) or string (e.g. 'center', 'bottom').
                                     If tuple, values 0-1 are relative.
            width (int): Target width for resizing.
            height (int): Target height for resizing.
            corner_radius (int): Radius for rounding corners.

        Returns:
            VideoBuilder: Self for method chaining.
        """
        self._overlay_images.append({
            'path': image_path,
            'position': position,
            'width': width,
            'height': height,
            'corner_radius': corner_radius
        })
        return self

    def with_overlay_video(self, video_path: str, position: Union[tuple, str] = ('center', 0.8), width: Optional[int] = None, height: Optional[int] = None, mask_color_rgb: Optional[tuple] = None, opacity: float = 1.0, react_to_audio: bool = False, pulse_magnitude: float = 0.2) -> 'VideoBuilder':
        """
        Add an overlay video/GIF to the video.

        Args:
            video_path (str): Path to the video/gif file.
            position (tuple or str): Position tuple (x, y) or string.
            width (int): Target width.
            height (int): Target height.
            mask_color_rgb (tuple): RGB tuple (0-255) to mask out (make transparent). e.g. (0,0,0) for black.
            opacity (float): Opacity of the overlay (0.0 to 1.0).
            react_to_audio (bool): If True, scales the overlay based on audio volume.
            pulse_magnitude (float): How much to scale up during peak volume (e.g. 0.2 = 120% size).

        Returns:
            VideoBuilder: Self for method chaining.
        """
        self._overlay_video_path = video_path
        self._overlay_video_config = {
            'position': position,
            'width': width,
            'height': height,
            'mask_color_rgb': mask_color_rgb,
            'opacity': opacity,
            'react_to_audio': react_to_audio,
            'pulse_magnitude': pulse_magnitude
        }
        return self

    def with_text_segments(self, segments: List[Dict], style: Optional[TextStyle] = None) -> 'VideoBuilder':
        """
        Overlay one or more timed text segments (titles, captions, karaoke-style
        word-highlighted text, ...) onto the video.

        Args:
            segments (List[Dict]): Each segment is a dict with 'text', 'start',
                'end' (seconds), and optionally 'words' -- a list of
                {'word', 'start', 'end'} dicts for per-word highlight timing.
                A single-segment list spanning the whole video is the "title"
                use case; multiple segments with 'words' is how subtitles are
                rendered (see tish_video_sdk.subtitles).
            style (TextStyle, optional): Styling to apply. Defaults to TextStyle().

        Returns:
            VideoBuilder: Self for method chaining.
        """
        self._text_segments = segments
        self._text_style = style
        return self

    def build(self) -> Optional[VideoFileClip]:
        """
        Build the video clip based on the configured parameters.

        Returns:
            Optional[VideoFileClip]: The created video clip, or None if an error occurs.
        """
        print("--- Starting video generation ---")

        # Step 1: Generate TTS audio if needed
        if self._tts_text and self._language_code:
            if not self._generate_tts_audio():
                print("Failed to generate TTS audio.")
                return None

        # Step 2: Create the base video clip
        if self._advanced_segments:
            self.video_clip = self._create_advanced_segments_video_clip()
        elif self._image_segments:
            self.video_clip = self._create_multi_image_video_clip()
        elif self._image_filepath:
            self.video_clip = self._create_single_image_video_clip()
        else:
            print("Error: No image input provided.")
            return None

        if self.video_clip is None:
            print("Failed to create base video clip.")
            return None

        print(f"DEBUG: video_clip type is {type(self.video_clip)}, duration is {self.video_clip.duration} (type: {type(self.video_clip.duration)})")
        try:
            duration_str = f"{self.video_clip.duration:.2f}"
        except Exception as e:
            print(f"DEBUG: Formatting failed: {e}")
            duration_str = str(self.video_clip.duration)
        print(f"Base video clip created. Duration: {duration_str}s")

        # Step 2.5: Apply overlay images if configured
        for overlay_img_conf in self._overlay_images:
            img_path = overlay_img_conf.get('path')
            if img_path and os.path.exists(img_path):
                 print(f"Adding overlay image: {img_path}")
                 try:
                     # Load using PIL first to apply rounded corners if needed
                     from PIL import Image, ImageDraw
                     img = Image.open(img_path).convert("RGBA")

                     target_w = overlay_img_conf.get('width')
                     target_h = overlay_img_conf.get('height')

                     if target_w or target_h:
                         orig_w, orig_h = img.size
                         if target_w and not target_h:
                             target_h = int((target_w / orig_w) * orig_h)
                         elif target_h and not target_w:
                             target_w = int((target_h / orig_h) * orig_w)
                         img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                     else:
                         target_w, target_h = img.size

                     cr = overlay_img_conf.get('corner_radius', 0)
                     if cr > 0:
                         mask = Image.new("L", (target_w, target_h), 0)
                         draw = ImageDraw.Draw(mask)
                         draw.rounded_rectangle((0, 0, target_w, target_h), cr, fill=255)
                         img.putalpha(mask)

                     import numpy as np
                     img_array = np.array(img)
                     overlay_clip = ImageClip(img_array)

                     # Set duration to match video
                     overlay_clip = overlay_clip.with_duration(self.video_clip.duration)

                     pos = overlay_img_conf.get('position', ('center', 0.7))

                     # Check if pos is absolute pixels (both X and Y > 1)
                     is_relative = True
                     if isinstance(pos, tuple) and len(pos) == 2:
                         if isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float)):
                             # If both are > 1, assume absolute pixels. (Relative is usually 0.0-1.0)
                             if pos[0] > 1.0 or pos[1] > 1.0 or pos[0] < 0.0 or pos[1] < 0.0:
                                 is_relative = False

                     overlay_clip = overlay_clip.with_position(pos, relative=is_relative)

                     # Composite on top of base video
                     self.video_clip = CompositeVideoClip([self.video_clip, overlay_clip], size=self.video_clip.size)

                 except Exception as e:
                     print(f"Failed to add overlay image '{img_path}': {e}")
                     traceback.print_exc()

        # Step 2.6: Apply overlay video if configured
        if self._overlay_video_path and os.path.exists(self._overlay_video_path):
             print(f"Adding overlay video: {self._overlay_video_path}")
             try:
                 # Load using VideoFileClip to keep animation
                 # has_mask=True allows transparency if the source has it (e.g. transparent GIF)
                 overlay_clip = VideoFileClip(self._overlay_video_path, has_mask=True)

                 conf = self._overlay_video_config

                 # Resize
                 if conf.get('width') or conf.get('height'):
                      # Use moviepy resize (it maintains aspect ratio if one is None)
                      overlay_clip = overlay_clip.resized(width=conf.get('width'), height=conf.get('height'))

                 # Masking (Chroma Key)
                 if conf.get('mask_color_rgb'):
                     # MaskColor expects color=[r,g,b] and threshold/stiffness parameters.
                     # We use default threshold for now, or could expose it.
                     c = conf.get('mask_color_rgb')
                     overlay_clip = overlay_clip.with_effects([MaskColor(color=c, threshold=10, stiffness=5)])

                 # Opacity
                 if conf.get('opacity') < 1.0:
                     overlay_clip = overlay_clip.with_opacity(conf.get('opacity'))

                 # Loop to match main video duration
                 if overlay_clip.duration < self.video_clip.duration:
                     # Efficient looping
                     overlay_clip = overlay_clip.with_effects([Loop(duration=self.video_clip.duration)])
                 else:
                     overlay_clip = overlay_clip.subclipped(0, self.video_clip.duration)

                 # Audio Reactivity (Pulsing)
                 if conf.get('react_to_audio') and self.video_clip.audio:
                     print("  Applying audio-reactive pulsing...")
                     try:
                         # 1. Analyze Audio
                         # Get sound array sampled at video FPS (one value per frame)
                         audio = self.video_clip.audio
                         # chunk_size should cover one frame duration (1/fps)
                         fps = self.video_clip.fps or DEFAULT_FPS

                         # to_soundarray returns (N, 2) for stereo. take max of channels.
                         # We want RMS or plain max over the frame window?
                         # moviepy's to_soundarray resamples.
                         # Let's get the whole array at fps.
                         sound_array = audio.to_soundarray(fps=fps)
                         # Convert stereo to mono max or rms
                         if sound_array.ndim > 1:
                             vol_array = np.max(np.abs(sound_array), axis=1)
                         else:
                             vol_array = np.abs(sound_array)

                         # Normalize to 0-1 (clip peaks to avoid extreme pulsing)
                         max_vol = np.percentile(vol_array, 95) if len(vol_array) > 0 else 1.0
                         if max_vol == 0: max_vol = 1.0
                         vol_array = np.clip(vol_array / max_vol, 0, 1)

                         pulse_mag = conf.get('pulse_magnitude', 0.2)

                         # 2. Define Transform
                         # We resize frame based on volume
                         def pulse_transform(get_frame, t):
                             frame = get_frame(t)
                             # Find index
                             idx = int(t * fps)
                             if idx >= len(vol_array): idx = len(vol_array) - 1
                             if idx < 0: idx = 0

                             vol = vol_array[idx]
                             scale = 1.0 + (vol * pulse_mag)

                             # Resize using scipy or PIL or moviepy's resize (which uses PIL/OpenCV)
                             # To keep it fast(er), use simple resizing.
                             # But resize changes dimensions. We want to scale from center?
                             # MoviePy resize applies to the clip.
                             # Here we are inside make_frame, returning a numpy array.
                             # If we use clip.resized(), we change the clip definition.
                             # It's better to create a new clip with a custom make_frame.

                             # ACTUALLY, using `transform` or `resize` filter per frame is heavy.
                             # But for a spectrum overlay (small-ish), it might be okay.

                             # Wait, we can't easily resize numpy array inside make_frame without imports.
                             # Simpler approach:
                             # Don't use make_frame raw manipulation unless necessary.
                             # Use MoviePy's `resize` with a function?
                             # clip.resized(lambda t: 1 + ... ) ?
                             # MoviePy's resize(width=...) can take a function t->width.
                             return frame # Fallback for now if complex


                         # Better approach: clip.resized(lambda t: 1 + ...)
                         # But standard resize takes width/height.
                         # We can define a function for width.

                         base_w = overlay_clip.w
                         def width_func(t):
                             idx = int(t * fps)
                             if idx >= len(vol_array): idx = len(vol_array) - 1
                             if idx < 0: idx = 0
                             vol = vol_array[idx]
                             return base_w * (1.0 + vol * pulse_mag)

                         overlay_clip = overlay_clip.resized(width=width_func)
                         # Note: Pulsing width implies position might shift if not centered.
                         # Position is applied AFTER resize.
                         # If position is 'center', it stays centered. Good.

                     except Exception as ex:
                         print(f"  Audio reactivity failed: {ex}")
                         traceback.print_exc()

                 # Position
                 pos = conf.get('position', ('center', 0.8))
                 overlay_clip = overlay_clip.with_position(pos, relative=True)

                 # Composite
                 self.video_clip = CompositeVideoClip([self.video_clip, overlay_clip], size=self.video_clip.size)

             except Exception as e:
                 print(f"Failed to add overlay video: {e}")
                 traceback.print_exc()

        # Step 2.65: Extract TTS subtitle segments if enabled (advanced segments
        # already carry their own per-segment subtitles via _create_segment_video_clip)
        if self._enable_tts_subtitles and self._tts_builder and not self._advanced_segments:
            print("Extracting TTS subtitles...")
            tts_segments = self._tts_builder.get_subtitle_segments()
            if tts_segments:
                self._text_segments = tts_segments
                print(f"Extracted {len(tts_segments)} TTS subtitle segments.")
            else:
                print("No TTS subtitle segments found.")

        # Step 2.7: Add timed text segments (titles, captions, subtitles, ...) if configured
        if self._text_segments and not self._advanced_segments:
            print("Adding text segments to video clip...")
            text_clip = self._overlay_text_segments_on_single_clip(self.video_clip, self._text_segments)

            if text_clip is None:
                print("Failed to add text segments to video clip.")
                self.video_clip.close()
                return None

            self.video_clip = text_clip
            print("Text segments added successfully.")

        print("Video generation completed successfully.")
        try:
            final_duration_str = f"{self.video_clip.duration:.2f}"
        except Exception:
            final_duration_str = str(self.video_clip.duration)
        print(f"Final video clip - Duration: {final_duration_str}s, FPS: {getattr(self.video_clip, 'fps', 'unknown')}")
        return self.video_clip

    def save(self, output_filepath: str) -> Optional[str]:
        """
        Save the video clip to a file.

        Args:
            output_filepath (str): The path where the video will be saved.

        Returns:
            Optional[str]: The path to the saved video file if successful, None otherwise.
        """
        if self.video_clip is None:
            print("Error: No video clip to save. Call build() first.")
            return None

        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        # Helper to generate a unique temp audio path to avoid WinError 32 locking collisions
        import uuid
        def get_temp_audio_path(base_path):
            unique_suffix = str(uuid.uuid4())[:8]
            # Use .m4a for AAC audio to avoid potential FFMPEG/MoviePy issues with mismatches
            return f"{base_path}.temp_audio_{unique_suffix}.m4a"

        # Attempt 1
        temp_audio_1 = get_temp_audio_path(output_filepath)
        try:
            print(f"Writing video to: {output_filepath}")
            self.video_clip.write_videofile(
                output_filepath,
                fps=getattr(self.video_clip, 'fps', DEFAULT_FPS),
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=temp_audio_1,
                remove_temp=True
            )
            print(f"Video saved to: {output_filepath}")
            return output_filepath
        except AttributeError as e:
            if "'NoneType' object has no attribute 'stdout'" in str(e) or "'DummyLogger' object has no attribute" in str(e):
                print("MoviePy Audio Read Error detected. Retrying with FFMPEG fallback (Direct Merge)...")

                if not self._audio_filepath or not os.path.exists(self._audio_filepath):
                    print("Cannot use FFMPEG fallback: No audio filepath available.")
                    raise e

                # Attempt FFMPEG direct merge
                temp_video_silent = output_filepath + ".silent_temp.mp4"
                try:
                    # 1. Write video ONLY (Silent) - This avoids the broken audio reading
                    self.video_clip.write_videofile(
                        temp_video_silent,
                        fps=getattr(self.video_clip, 'fps', DEFAULT_FPS),
                        codec="libx264",
                        audio=False, # Disable audio reading
                        logger=None # Suppress bars
                    )

                    # 2. Merge with original audio file using FFMPEG CLI
                    import subprocess
                    print(f"Merging silent video with audio: {self._audio_filepath}")
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", temp_video_silent,
                        "-i", self._audio_filepath,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest", # Stop when shortest input ends
                        output_filepath
                    ]

                    # Run FFMPEG
                    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    if process.returncode != 0:
                        print(f"FFMPEG Merge Failed: {process.stderr}")
                        raise Exception("FFMPEG Merge Failed")

                    print(f"Video saved (via FFMPEG merge) to: {output_filepath}")

                    # Cleanup
                    if os.path.exists(temp_video_silent):
                        os.remove(temp_video_silent)

                    return output_filepath

                except Exception as e_retry:
                    print(f"Error writing video file (FFMPEG fallback failed): {e_retry}")
                    if os.path.exists(temp_video_silent):
                         try: os.remove(temp_video_silent)
                         except: pass
                    raise e_retry
            else:
                print(f"Error writing video file: {e}")
                print("Check MoviePy/FFmpeg error messages for more details.")
                return None
        except Exception as e:
            print(f"Error writing video file: {e}")
            print("Check MoviePy/FFmpeg error messages for more details.")
            return None
        finally:
            # Clean up temp file 1 manually if needed (in case write_videofile crashed before cleanup)
            if os.path.exists(temp_audio_1):
                try:
                    os.remove(temp_audio_1)
                except:
                    pass

    def save_tts_audio(self, output_filepath: str) -> Optional[str]:
        """
        Save the generated TTS audio to a separate file.
        Only works if TTS was used for audio generation.

        Args:
            output_filepath (str): The path where the audio will be saved.

        Returns:
            Optional[str]: The path to the saved audio file if successful, None otherwise.
        """
        if not self._tts_builder:
            print("Error: No TTS audio available. Use TTS methods to generate audio first.")
            return None

        return self._tts_builder.save(output_filepath)

    def get_tts_processed_content(self) -> Optional[str]:
        """
        Get the processed TTS content (SSML or original text).

        Returns:
            Optional[str]: The processed content if available, None otherwise.
        """
        if self._tts_builder:
            return self._tts_builder.get_processed_content()
        return None

    def _generate_tts_audio(self) -> bool:
        """
        Generate TTS audio from text.

        Returns:
            bool: True if successful, False otherwise.
        """
        print(f"Generating TTS audio for language '{self._language_code}'...")

        try:
            if not self._tts_text or not self._language_code:
                print("Error: TTS text or language code not provided.")
                return False

            # Create TTS builder
            self._tts_builder = TTSBuilder.from_text(
                self._tts_text,
                self._language_code,
                self._provider
            )

            if self._enable_tts_subtitles:
                self._tts_builder = self._tts_builder.with_subtitles()

            # Build TTS
            result = self._tts_builder.build()
            if result is None:
                print("TTS audio generation failed.")
                return False

            # Create a temporary audio file
            import tempfile
            temp_audio_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_audio_dir, f"tts_audio_{id(self)}.wav")

            saved_audio = self._tts_builder.save(temp_audio_path)
            if saved_audio:
                self._audio_filepath = saved_audio
                print(f"TTS audio generated and saved to: {saved_audio}")
                return True
            else:
                print("Failed to save TTS audio.")
                return False

        except Exception as e:
            print(f"Error generating TTS audio: {e}")
            return False

    def _load_image_clip_safe(self, path):
        """Helper to safely load an image clip using PIL to ensure compatibility."""
        try:
             from PIL import Image
             import numpy as np

             img = Image.open(path)
             # Convert to RGB/RGBA to ensure numpy array is standard
             if img.mode not in ['RGB', 'RGBA']:
                 img = img.convert('RGB')

             # Convert to numpy array
             img_array = np.array(img)
             return ImageClip(img_array)
        except Exception as e:
             print(f"Error loading image with PIL '{path}': {e}")
             # Fallback to standard loading
             return ImageClip(path)

    def _create_single_image_video_clip(self) -> Optional[VideoFileClip]:
        """Create a video clip from a single image."""
        if not os.path.exists(self._image_filepath):
            print(f"Error: Image file '{self._image_filepath}' does not exist.")
            return None

        try:
            if self._audio_filepath is not None:
                # Mode 1: Create video with audio (regular or TTS-generated)
                if not os.path.exists(self._audio_filepath):
                    print(f"Error: Audio file '{self._audio_filepath}' does not exist.")
                    return None

                audio_source = "TTS-generated" if self._tts_builder else "provided"
                print(f"Creating video clip: image '{self._image_filepath}' + {audio_source} audio '{self._audio_filepath}'...")
                self._active_audio_clip = AudioFileClip(self._audio_filepath)

                # Use robust loader
                image_clip = self._load_image_clip_safe(self._image_filepath)

                video_clip = image_clip.with_duration(self._active_audio_clip.duration)
                video_clip = video_clip.with_audio(self._active_audio_clip)

                print("Video clip with audio created.")
            else:
                # Mode 2: Create video with specified duration (silent)
                if self._duration is None:
                    print("Error: Duration must be specified when no audio is provided.")
                    return None

                print(f"Creating silent video clip from image '{self._image_filepath}' with duration {self._duration} seconds...")
                image_clip = self._load_image_clip_safe(self._image_filepath)
                video_clip = image_clip.with_duration(self._duration)

                print("Silent video clip created.")

            video_clip = video_clip.with_fps(self.fps)
            return video_clip

        except Exception as e:
            print(f"Error creating video clip: {e}")
            return None

    def _create_multi_image_video_clip(self) -> Optional[VideoFileClip]:
        """Create a video clip from multiple media segments (images or videos) with timestamps."""
        if not self._image_segments:
            print("Error: No media segments provided.")
            return None

        print(f"Creating multi-media video from {len(self._image_segments)} segments...")

        video_clips = []

        for i, segment in enumerate(self._image_segments):
            media_path = segment.get('path', '')
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)

            if not media_path:
                print(f"Warning: Segment {i} has no media path. Skipping.")
                continue

            if start_time >= end_time:
                print(f"Warning: Segment {i} has invalid time range ({start_time}-{end_time}). Skipping.")
                continue

            if not os.path.exists(media_path):
                print(f"Error: Media file '{media_path}' does not exist. Skipping segment {i}.")
                continue

            duration = end_time - start_time

            try:
                # Check if it's a video file based on extension
                lower_path = media_path.lower()
                is_video = lower_path.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))

                if is_video:
                    print(f"Creating video clip for segment {i}: {media_path} ({duration:.2f}s)")
                    clip = VideoFileClip(media_path)

                    # Loop or subclip to match duration
                    if clip.duration < duration:
                        # Loop if shorter
                        clip = concatenate_videoclips([clip] * (int(duration / clip.duration) + 1))
                        clip = clip.subclipped(0, duration)
                    else:
                        # Cut if longer
                        clip = clip.subclipped(0, duration)

                    # Fit within the target box without stretching (resize by whichever
                    # dimension keeps the clip within bounds, matching the image branch below).
                    target_w, target_h = self.width, self.height
                    target_ratio = target_w / target_h

                    w, h = clip.size
                    current_ratio = w / h

                    if current_ratio > target_ratio:
                        # Video is wider than target slot (e.g. 16:9 video in 9:16 slot)
                        # Resize by matching width, height will be smaller (letterbox top/bottom)
                        clip = clip.resized(width=target_w)
                    else:
                        # Video is taller than target slot (e.g. 9:16 video in 1:1 slot, or 9:16 in 9:16)
                        # Resize by matching height, width will be smaller (letterbox left/right)
                        clip = clip.resized(height=target_h)

                    # Now composite over black background
                    # Create black background
                    # clip.duration should be set
                    bg_clip = ColorClip(size=(target_w, target_h), color=(0,0,0), duration=clip.duration)

                    # Center the video clip
                    clip = clip.with_position("center")

                    # Composite
                    clip = CompositeVideoClip([bg_clip, clip], size=(target_w, target_h))

                    clip = clip.with_fps(self.fps)
                    clip = clip.without_audio()

                    video_clips.append(clip)

                else:
                    # Image processing
                    print(f"Creating image clip for segment {i}: {media_path} ({duration:.2f}s)")
                    image_clip = ImageClip(media_path)

                    # Ensure target resolution
                    target_w, target_h = self.width, self.height
                    target_ratio = target_w / target_h

                    w, h = image_clip.size
                    current_ratio = w / h

                    clip = image_clip

                    target_w, target_h = self.width, self.height
                    target_ratio = target_w / target_h

                    w, h = clip.size
                    current_ratio = w / h

                    if current_ratio > target_ratio:
                         # Wider than target
                         clip = clip.resized(width=target_w)
                    else:
                         # Taller than target
                         clip = clip.resized(height=target_h)

                    # Create black background
                    bg_clip = ColorClip(size=(target_w, target_h), color=(0,0,0), duration=duration)

                    # Center
                    clip = clip.with_position("center")

                    # Composite
                    video_clip = CompositeVideoClip([bg_clip, clip], size=(target_w, target_h))

                    video_clip = video_clip.with_duration(duration).with_fps(self.fps)
                    video_clips.append(video_clip)

            except Exception as e:
                print(f"Error creating clip for segment {i}: {e}")
                continue

        if not video_clips:
            print("Error: No valid video clips created from media segments.")
            return None

        try:
            print("Concatenating video clips (mixed media)...")
            final_video = concatenate_videoclips(video_clips, method="compose")

            # Add audio if provided (regular or TTS-generated)
            if self._audio_filepath and os.path.exists(self._audio_filepath):
                audio_source = "TTS-generated" if self._tts_builder else "provided"
                print(f"Adding {audio_source} audio track: {self._audio_filepath}")
                self._active_audio_clip = AudioFileClip(self._audio_filepath)

                if self._active_audio_clip.duration > final_video.duration:
                    self._active_audio_clip = self._active_audio_clip.subclipped(0, final_video.duration)
                elif self._active_audio_clip.duration < final_video.duration:
                    print(f"Warning: Audio duration ({self._active_audio_clip.duration}s) is shorter than video duration ({final_video.duration}s)")

                final_video = final_video.with_audio(self._active_audio_clip)

            print("Multi-media video clip created successfully.")
            return final_video

        except Exception as e:
            print(f"Error concatenating video clips: {e}")
            # Clean up clips in case of error
            for clip in video_clips:
                try:
                    clip.close()
                except:
                    pass
            return None

    def _get_font_path(self, font_name: str) -> str:
        """Find the font file path on Windows."""
        # Clean font name
        cleaned = font_name.split('&')[0].strip() # Handle language specific combinations if any

        # Try standard Windows Fonts folder
        win_dir = os.environ.get('SystemRoot', 'C:\\Windows')
        fonts_dir = os.path.join(win_dir, 'Fonts')

        if not os.path.exists(fonts_dir):
            # Fallback if WINDIR not found or not Windows
            return font_name # return as-is, PIL might fail but we catch it

        # Font mapping dictionary for common families
        font_map = {
            'arial': 'arial.ttf',
            'arial-bold': 'arialbd.ttf',
            'arial_bold': 'arialbd.ttf',
            'helvetica': 'arial.ttf',
            'nirmala-ui': 'Nirmala.ttf',
            'nirmala': 'Nirmala.ttf',
            'nirmala-ui-bold': 'Nirmalab.ttf',
            'nirmala-bold': 'Nirmalab.ttf',
            'courier': 'cour.ttf',
            'courier-new': 'cour.ttf',
            'times-new-roman': 'times.ttf',
            'georgia': 'georgia.ttf',
            'impact': 'impact.ttf',
        }

        name_lower = cleaned.lower()

        # Check direct mapping
        if name_lower in font_map:
            path = os.path.join(fonts_dir, font_map[name_lower])
            if os.path.exists(path):
                return path
            # Fallback to .ttc if direct mapping specifies .ttf but only .ttc exists
            if path.lower().endswith('.ttf'):
                ttc_path = path[:-4] + '.ttc'
                if os.path.exists(ttc_path):
                    return ttc_path

        # Try appending extensions
        for ext in ['.ttf', '.otf', '.ttc', '.TTF', '.OTF', '.TTC']:
            path = os.path.join(fonts_dir, cleaned + ext)
            if os.path.exists(path):
                return path
            path = os.path.join(fonts_dir, name_lower + ext)
            if os.path.exists(path):
                return path

        # Search the directory for containing matches
        try:
            for f in os.listdir(fonts_dir):
                if f.lower().startswith(name_lower) and f.lower().endswith(('.ttf', '.otf', '.ttc')):
                    return os.path.join(fonts_dir, f)
        except Exception:
            pass

        # Default fallback
        return os.path.join(fonts_dir, 'arial.ttf')

    def _render_text_frame_pil(self, width: int, height: int, words: List[Dict], active_word_idx: Optional[int], style: TextStyle, has_word_timestamps: bool = True) -> Image.Image:
        """
        Renders a single frame of text using PIL.
        Supports word-level highlighting, wrapping, outline/stroke, and rounded-corner background box.
        Dynamically adapts font size to fit the available space.
        """
        if not has_word_timestamps:
            active_word_idx = None

        # 1. Create transparent image
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 2. Get font path
        font_path = self._get_font_path(style.get_font_for_language())

        # 3. Load font and calculate layout constraints (potentially scaling down)
        max_w = int(width * (style.box_size[0] or 0.8))
        max_h = int(height * (style.box_size[1] or 0.35))

        current_font_size = style.font_size
        min_font_size = 12
        font = None
        lines = []
        line_height = 0
        total_height = 0

        while current_font_size >= min_font_size:
            try:
                font = ImageFont.truetype(font_path, current_font_size)
                is_default = False
            except Exception as e:
                print(f"Warning: Failed to load font {font_path}, falling back to default. Error: {e}")
                font = ImageFont.load_default()
                is_default = True

            # Helper to get size of a word
            def get_word_size(text):
                if hasattr(draw, 'textbbox'):
                    bbox = draw.textbbox((0, 0), text, font=font)
                    return bbox[2] - bbox[0], bbox[3] - bbox[1]
                elif hasattr(font, 'getbbox'):
                    bbox = font.getbbox(text)
                    return bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    return font.getsize(text)

            # Wrap words into lines to stay within the screen margin
            lines = []
            current_line = []
            current_line_w = 0

            for idx, w_dict in enumerate(words):
                word_text = w_dict.get('word', '').strip()
                if not word_text:
                    continue

                # Measure word with trailing space
                measure_text = word_text + " "
                w_width, w_height = get_word_size(measure_text)

                is_active = (active_word_idx is not None and idx == active_word_idx)

                if current_line_w + w_width > max_w and current_line:
                    # Wrap to next line
                    lines.append(current_line)
                    current_line = [(w_dict, is_active, idx, w_width)]
                    current_line_w = w_width
                else:
                    current_line.append((w_dict, is_active, idx, w_width))
                    current_line_w += w_width

            if current_line:
                lines.append(current_line)

            if lines:
                _, sample_h = get_word_size("Ay")
                line_height = int(sample_h * 1.3)
                total_height = len(lines) * line_height
            else:
                line_height = 0
                total_height = 0

            # If default font or total_height fits within max_h, or we are at min_font_size, we stop
            if is_default or total_height <= max_h or current_font_size <= min_font_size:
                break

            current_font_size -= 2

        # Use final font size for background box padding
        resolved_font_size = current_font_size

        if not lines:
            return img

        # 6. Determine text position on the screen
        pos_x, pos_y = style.text_position
        box_w = int(width * (style.box_size[0] or 0.8))

        # Resolve x coordinate (finding center_x)
        if isinstance(pos_x, str):
            pos_x_lower = pos_x.lower()
            if pos_x_lower == 'left':
                center_x = box_w // 2
            elif pos_x_lower == 'right':
                center_x = width - box_w // 2
            else: # 'center'
                center_x = width // 2
        elif isinstance(pos_x, (int, float)):
            # If relative float
            if 0.0 <= pos_x <= 1.0:
                if pos_x == 0.5:
                    center_x = width // 2
                elif pos_x < 0.4:
                    # e.g., 0.1 is the left margin of an 80% wide box
                    center_x = int(width * (pos_x + (style.box_size[0] or 0.8) / 2))
                else:
                    center_x = int(width * pos_x)
            else:
                # Absolute pixel value
                if pos_x < width * 0.4:
                    center_x = int(pos_x + box_w // 2)
                else:
                    center_x = int(pos_x)
        else:
            center_x = width // 2

        # Resolve y coordinate (finding top of the text block)
        if isinstance(pos_y, str):
            pos_y_lower = pos_y.lower()
            if pos_y_lower == 'top':
                y_start = int(height * 0.05) # 5% top margin
            elif pos_y_lower == 'center':
                y_start = (height - total_height) // 2
            elif pos_y_lower == 'bottom':
                y_start = height - total_height - int(height * 0.05) # 5% bottom margin
            else:
                y_start = height - total_height - int(height * 0.05)
        elif isinstance(pos_y, (int, float)):
            # If relative float
            if 0.0 <= pos_y <= 1.0:
                y_start = int(height * pos_y)
            else:
                y_start = int(pos_y)
        else:
            y_start = height - total_height - int(height * 0.05)

        # Calculate line positions and widths
        line_info = [] # List of (line, line_w, line_x, line_y)
        curr_y = y_start

        for line in lines:
            line_w = sum(item[3] for item in line)
            line_x = center_x - (line_w // 2)
            line_info.append((line, line_w, line_x, curr_y))
            curr_y += line_height

        # 7. Draw background box if style.bg_color is specified
        if style.bg_color:
            try:
                bg_color_rgba = ImageColor.getrgb(style.bg_color)
            except Exception:
                bg_color_rgba = (0, 0, 0, 128) # Default 50% opacity black

            box_padding_x = int(resolved_font_size * 0.5)
            box_padding_y = int(resolved_font_size * 0.25)

            max_line_w = max(info[1] for info in line_info)
            box_w = max_line_w + 2 * box_padding_x
            box_h = total_height + 2 * box_padding_y

            box_left = center_x - (box_w // 2)
            box_top = y_start - box_padding_y
            box_right = box_left + box_w
            box_bottom = y_start + total_height + box_padding_y

            corner_radius = int(resolved_font_size * 0.25)
            draw.rounded_rectangle(
                [(box_left, box_top), (box_right, box_bottom)],
                radius=corner_radius,
                fill=bg_color_rgba
            )

        # 8. Draw each word
        highlight_color = getattr(style, 'highlight_color', '#FFD700')
        font_color = style.font_color or 'white'
        stroke_color = style.stroke_color or 'black'
        stroke_width = style.stroke_width or 0

        for line, line_w, line_x, line_y in line_info:
            curr_x = line_x
            for w_dict, is_active, idx, w_width in line:
                word_text = w_dict.get('word', '')

                # Active word highlight color, else normal font color
                color = highlight_color if is_active else font_color

                # Draw the word
                draw.text(
                    (curr_x, line_y),
                    word_text + " ",
                    fill=color,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color
                )

                curr_x += w_width

        return img

    def _overlay_text_segments_on_single_clip(self, video_clip: VideoFileClip, segments: List[Dict]) -> Optional[VideoFileClip]:
        """Overlay timed text segments on a single video clip using PIL rendering.

        Each segment becomes one or more per-word-interval ImageClips (so a
        segment with word-level timestamps animates a highlighted "active
        word" as it's read); a segment with no 'words' renders as static text
        for its whole start/end span."""
        print(f"DEBUG: _overlay_text_segments_on_single_clip called with {len(segments)} segments")
        if not segments:
            print("DEBUG: No text segments provided, returning original clip")
            return video_clip

        style = self._text_style or TextStyle()

        text_clips = []

        try:
            for i, segment in enumerate(segments):
                text_content = segment.get('text', '').replace('\n', ' ').strip()
                offset = getattr(style, 'timing_offset', 0.0)
                seg_start = segment.get('start', 0.0) + offset
                seg_end = segment.get('end', 0.0) + offset

                if not text_content or seg_start >= seg_end:
                    continue

                words = segment.get('words', [])
                has_word_timestamps = bool(words)
                if not words:
                    # Split plain text into words to allow wrapping, distributing start/end times evenly
                    word_list = text_content.split()
                    duration = seg_end - seg_start
                    if word_list:
                        per_word_duration = duration / len(word_list)
                        words = [
                            {
                                'word': w,
                                'start': seg_start + idx * per_word_duration,
                                'end': seg_start + (idx + 1) * per_word_duration
                            }
                            for idx, w in enumerate(word_list)
                        ]
                    else:
                        words = [{'word': text_content, 'start': seg_start, 'end': seg_end}]
                else:
                    # Shift word timestamps by timing_offset
                    words = [
                        {
                            'word': w.get('word', ''),
                            'start': w.get('start', 0.0) + offset,
                            'end': w.get('end', 0.0) + offset
                        }
                        for w in words
                    ]

                # Ensure words are sorted by start time
                words = sorted(words, key=lambda x: x.get('start', 0.0))

                # Determine time boundaries for sub-intervals within the segment
                boundaries = [seg_start]
                for w in words:
                    w_start = w.get('start', 0.0)
                    w_end = w.get('end', 0.0)
                    # Keep boundaries within segment bounds
                    w_start = max(seg_start, min(w_start, seg_end))
                    w_end = max(seg_start, min(w_end, seg_end))
                    boundaries.append(w_start)
                    boundaries.append(w_end)
                boundaries.append(seg_end)

                # Remove duplicates and sort
                boundaries = sorted(list(set(boundaries)))

                # Generate a clip for each sub-interval
                for idx in range(len(boundaries) - 1):
                    t1 = boundaries[idx]
                    t2 = boundaries[idx + 1]
                    dur = t2 - t1

                    if dur < 0.01:
                        continue

                    # Find which word index is active during this interval
                    mid = (t1 + t2) / 2.0
                    active_idx = None
                    for w_idx, w in enumerate(words):
                        if w.get('start', 0.0) <= mid <= w.get('end', 0.0):
                            active_idx = w_idx
                            break

                    # Render frame using PIL
                    frame_img = self._render_text_frame_pil(video_clip.w, video_clip.h, words, active_idx, style, has_word_timestamps)

                    # Convert to RGBA numpy array
                    frame_arr = np.array(frame_img)

                    rgb_arr = frame_arr[:, :, :3]
                    alpha_arr = frame_arr[:, :, 3] / 255.0  # Normalize alpha to 0.0-1.0

                    # Create the ImageClip and its transparency mask
                    img_clip = ImageClip(rgb_arr)
                    mask_clip = ImageClip(alpha_arr, is_mask=True)
                    img_clip = img_clip.with_mask(mask_clip)

                    # Set time parameters
                    img_clip = img_clip.with_start(t1).with_duration(dur)
                    text_clips.append(img_clip)

        except Exception as e:
            print(f"Error rendering PIL text frames: {e}")
            traceback.print_exc()
            for clip in text_clips:
                try:
                    clip.close()
                except:
                    pass
            return None

        print(f"DEBUG: Created {len(text_clips)} text clips")
        if not text_clips:
            return video_clip

        try:
            print("DEBUG: Compositing video with text clips...")
            final_clip = CompositeVideoClip([video_clip] + text_clips)
            if video_clip.audio:
                final_clip = final_clip.with_audio(video_clip.audio)
            return final_clip

        except Exception as e:
            print(f"Error compositing video with text segments: {e}")
            for clip in text_clips:
                try:
                    clip.close()
                except:
                    pass
            return None

    def _create_advanced_segments_video_clip(self) -> Optional[VideoFileClip]:
        """Create a video clip from advanced segments with intelligent caching."""
        if not self._advanced_segments:
            print("Error: No advanced segments provided.")
            return None

        print(f"Creating advanced multi-segment video from {len(self._advanced_segments)} segments...")

        # STEP 1: Expand segments with repetitions
        expanded_segments = self._expand_segments_with_repetitions()
        if not expanded_segments:
            print("Error: No valid expanded segments.")
            return None

        print(f"Expanded to {len(expanded_segments)} total segments after repetitions.")

        # STEP 2: Process each expanded segment (including repetitions)
        video_clips = []
        current_time = 0.0

        for i, segment in enumerate(expanded_segments):
            print(f"Processing segment {i+1}/{len(expanded_segments)}...")

            # Create video clip for this segment (uses cached TTS)
            clip = self._create_segment_video_clip(segment, current_time)
            if clip:
                video_clips.append(clip)
                current_time += clip.duration
            else:
                print(f"Warning: Failed to create clip for segment {i+1}")

        if not video_clips:
            print("Error: No valid video clips created from advanced segments.")
            return None

        try:
            print("Concatenating video clips...")
            final_video = concatenate_videoclips(video_clips, method="compose")
            print("Advanced segments video clip created successfully.")
            return final_video

        except Exception as e:
            print(f"Error concatenating advanced segment clips: {e}")
            # Clean up clips in case of error
            for clip in video_clips:
                try:
                    clip.close()
                except:
                    pass
            return None

    def _expand_segments_with_repetitions(self) -> List[Dict]:
        """
        Repetition Handling with Two-Level Caching:

        1. AUDIO CACHE: Caches TTS audio + subtitles by text+language
        2. VIDEO CACHE: Caches complete video clips by audio+image combination
        3. EXPANSION: Creates multiple copies of each segment based on repeat count
        4. OPTIMIZATION: Reuses cached audio and video clips when possible

        Cache Keys:
        - Audio Cache Key: "text|language_code"
        - Video Cache Key: "text|language_code|image_path"
        """
        expanded = []

        for segment in self._advanced_segments:
            repeat_count = segment.get('repeat', 1)  # Default to 1 if not specified

            # Validate segment
            if not segment.get('image') or not segment.get('text'):
                print(f"Warning: Segment missing required 'image' or 'text': {segment}")
                continue

            # Create cache keys for two-level caching
            text = segment['text']
            language_code = segment.get('language_code', 'fr')
            image_path = segment['image']

            audio_cache_key = f"{text}|{language_code}"
            video_cache_key = f"{text}|{language_code}|{image_path}"

            # Level 1: Pre-generate TTS audio if not cached
            if audio_cache_key not in self._audio_cache:
                print(f"Generating TTS audio for: '{text[:50]}...' (language: {language_code})")
                tts_result = self._generate_and_cache_audio(text, language_code, segment.get('provider'))
                if not tts_result:
                    print(f"Warning: TTS generation failed for segment with text: '{text[:50]}...'")
                    continue
            else:
                print(f"Using cached TTS audio for: '{text[:50]}...'")

            # Add repeated segments (CORE REPETITION LOGIC)
            for rep in range(repeat_count):
                expanded_segment = segment.copy()
                expanded_segment['_audio_cache_key'] = audio_cache_key
                expanded_segment['_video_cache_key'] = video_cache_key
                expanded_segment['_repetition'] = rep + 1
                expanded.append(expanded_segment)

        return expanded

    def _generate_and_cache_audio(self, text: str, language_code: str, provider: Optional[str] = None) -> bool:
        """Generate TTS audio and subtitles, then cache them in audio cache."""
        audio_cache_key = f"{text}|{language_code}"

        if audio_cache_key in self._audio_cache:
            return True

        try:
            # Create TTS builder
            tts_builder = TTSBuilder.from_text(text, language_code, provider)
            tts_builder = tts_builder.with_subtitles()  # Always enable subtitles for advanced segments

            # Build TTS
            result = tts_builder.build()
            if result is None:
                return False

            # Create temporary audio file
            temp_audio_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_audio_dir, f"audio_cache_{hash(audio_cache_key)}.wav")

            saved_audio = tts_builder.save(temp_audio_path)
            if not saved_audio:
                return False

            # Get subtitle segments
            subtitle_segments = tts_builder.get_subtitle_segments()

            # Cache the results in audio cache
            self._audio_cache[audio_cache_key] = {
                'audio_path': saved_audio,
                'subtitle_segments': subtitle_segments,
                'duration': AudioFileClip(saved_audio).duration if os.path.exists(saved_audio) else 0,
                'tts_builder': tts_builder
            }

            print(f"Cached TTS audio for key: {audio_cache_key[:50]}...")
            return True

        except Exception as e:
            print(f"Error generating TTS for audio cache key {audio_cache_key}: {e}")
            return False

    def _create_segment_video_clip(self, segment: Dict, start_time: float) -> Optional[VideoFileClip]:
        """
        Create a video clip for a single advanced segment with two-level caching.
        Level 1: Check video cache (complete video clip)
        Level 2: Use audio cache + create new video clip
        """
        try:
            video_cache_key = segment['_video_cache_key']
            audio_cache_key = segment['_audio_cache_key']

            # Level 1: Check if complete video clip is cached
            if video_cache_key in self._video_cache:
                print(f"Using cached video clip for: {video_cache_key[:50]}...")
                cached_clip = self._video_cache[video_cache_key]
                # Create a copy of the cached clip for this instance
                video_clip = cached_clip.copy()
                repetition = segment.get('_repetition', 1)
                print(f"Retrieved cached video clip (repetition {repetition}): duration {video_clip.duration:.2f}s")
                return video_clip

            # Level 2: Create new video clip using cached audio
            audio_data = self._audio_cache.get(audio_cache_key)
            if not audio_data:
                print(f"Error: No cached audio data for key: {audio_cache_key}")
                return None

            # Create image clip
            image_path = segment['image']
            if not os.path.exists(image_path):
                print(f"Error: Image file '{image_path}' does not exist.")
                return None

            # Get audio duration from cache
            audio_duration = audio_data['duration']

            # Create base video clip
            image_clip = ImageClip(image_path).with_duration(audio_duration).with_fps(self.fps)

            # Add TTS audio
            audio_clip = AudioFileClip(audio_data['audio_path'])
            video_clip = image_clip.with_audio(audio_clip)

            # Add background music if specified
            if segment.get('bgm'):
                bgm_path = segment['bgm']
                if os.path.exists(bgm_path):
                    # Cache background music
                    if bgm_path not in self._bgm_cache:
                        self._bgm_cache[bgm_path] = AudioFileClip(bgm_path)

                    bgm_clip = self._bgm_cache[bgm_path]

                    # Adjust BGM duration to match video
                    if bgm_clip.duration > audio_duration:
                        bgm_clip = bgm_clip.subclipped(0, audio_duration)
                    elif bgm_clip.duration < audio_duration:
                        # Loop BGM if it's shorter
                        loops_needed = int(audio_duration / bgm_clip.duration) + 1
                        bgm_clip = concatenate_audioclips([bgm_clip] * loops_needed).subclipped(0, audio_duration)

                    # Mix audio (reduce BGM volume)
                    bgm_clip = bgm_clip.with_volume_scaled(0.3)  # 30% volume for BGM
                    mixed_audio = CompositeAudioClip([audio_clip, bgm_clip])
                    video_clip = video_clip.with_audio(mixed_audio)

            # Add subtitles
            subtitle_segments = audio_data['subtitle_segments']
            if subtitle_segments:
                # Create subtitle overlay
                subtitled_clip = self._overlay_text_segments_on_single_clip(video_clip, subtitle_segments)
                if subtitled_clip:
                    video_clip.close()  # Clean up original
                    video_clip = subtitled_clip

            # Cache the complete video clip (without BGM for better reusability)
            if not segment.get('bgm'):  # Only cache if no BGM to keep cache simple
                self._video_cache[video_cache_key] = video_clip.copy()
                print(f"Cached complete video clip for: {video_cache_key[:50]}...")

            repetition = segment.get('_repetition', 1)
            print(f"Created new video clip (repetition {repetition}): duration {audio_duration:.2f}s")
            return video_clip

        except Exception as e:
            print(f"Error creating segment video clip: {e}")
            return None

    def cleanup_cache(self):
        """Clean up cached resources from both cache levels."""
        print("Cleaning up cached resources...")

        # Clean up audio cache
        for cache_key, audio_data in self._audio_cache.items():
            try:
                audio_path = audio_data.get('audio_path')
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"Warning: Could not remove cached audio file: {e}")

        # Clean up video cache
        for cache_key, video_clip in self._video_cache.items():
            try:
                video_clip.close()
            except Exception as e:
                print(f"Warning: Could not close cached video clip: {e}")

        # Clean up BGM cache
        for bgm_path, bgm_clip in self._bgm_cache.items():
            try:
                bgm_clip.close()
            except Exception as e:
                print(f"Warning: Could not close BGM clip: {e}")

        self._audio_cache.clear()
        self._video_cache.clear()
        self._bgm_cache.clear()
        print("Two-level cache cleanup completed.")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_cache()
        except:
            pass
