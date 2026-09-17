from __future__ import annotations

import os

from moviepy import (
    VideoFileClip,
    VideoClip,
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
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont
import uuid # For unique temp filenames
import traceback # Debugging
import bisect # Locating the active boundary interval in _overlay_scrolling_lyrics_on_single_clip

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

# Below this duration, VideoBuilder.save() always uses the single-process
# write_videofile() path -- the fixed cost of spawning worker processes and
# re-running build() once per worker isn't worth it for short clips.
PARALLEL_RENDER_MIN_DURATION_S = 30.0

# Chunked parallel rendering never splits a clip into pieces smaller than
# this -- caps how many workers save() spawns for a given duration.
MIN_CHUNK_DURATION_S = 20.0

# Conservative per-worker memory budget for chunked parallel rendering:
# each worker is a full process re-importing the SDK/moviepy/numpy/PIL,
# holding the base image/audio, and compositing overlay images plus the
# lyrics overlay frame-by-frame. Used to cap worker count against actually
# -available memory, not just CPU count.
PARALLEL_RENDER_MEM_PER_WORKER_BYTES = 2 * 1024 ** 3  # 2 GiB

# Font mapping for common families, resolved against the Windows Fonts folder.
DEFAULT_FONT_SIZE = 50
DEFAULT_TEXT_POSITION = (0.5, 0.5)  # Centered in the middle of the video
DEFAULT_FONT_COLOR = 'black'
DEFAULT_BG_COLOR = None  # Transparent background
DEFAULT_STROKE_COLOR = 'black'
DEFAULT_STROKE_WIDTH = 3
DEFAULT_FONT_FAMILY = 'Arial-Bold'
DEFAULT_BOX_SIZE = (0.8, None)


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
    # 'instant': only the current word is highlighted, one at a time.
    # 'karaoke': already-read words stay highlighted, and the active word
    # fills in progressively rather than popping instantly.
    highlight_style: str = 'instant'
    # >0 shows the current line plus this many lines before/after it,
    # scrolling as it advances; only the current line gets word highlighting.
    context_lines: int = 0
    # Opacity of non-current lines in the scrolling display.
    context_opacity: float = 0.5
    # Font size of non-current lines, relative to the current line's.
    context_font_scale: float = 0.75
    # Seconds before its own start that an upcoming line may appear, so a
    # long instrumental gap doesn't reveal it too early.
    upcoming_lead: float = 3.0
    # Seconds the scrolling display takes to glide to the next current line
    # (position/size/opacity ease smoothly); 0 snaps instantly.
    scroll_duration: float = 0.35
    # Neon/glow look: a blurred halo drawn behind a word's highlighted
    # portion (never behind still-unlit text). 0 glow_radius (default)
    # draws no glow, byte-identical to before this existed; >0 is the
    # Gaussian blur radius in px. glow_color defaults to None, which means
    # "use highlight_color" -- set it to use a different glow color than
    # the highlight color itself.
    glow_radius: int = 0
    glow_color: Optional[str] = None

    def get_font_for_language(self) -> str:
        """Return this style's font family, as resolved from configuration
        (see SubtitlePack) for the current language."""
        return self.font_family

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
        if self.highlight_style not in ('instant', 'karaoke'):
            raise ValueError(f"highlight_style must be 'instant' or 'karaoke', got: {self.highlight_style!r}")
        if self.context_lines < 0:
            raise ValueError("context_lines must be non-negative")
        if not 0.0 <= self.context_opacity <= 1.0:
            raise ValueError("context_opacity must be between 0.0 and 1.0")
        if self.context_font_scale <= 0:
            raise ValueError("context_font_scale must be positive")
        if self.upcoming_lead < 0:
            raise ValueError("upcoming_lead must be non-negative")
        if self.scroll_duration < 0:
            raise ValueError("scroll_duration must be non-negative")
        if self.glow_radius < 0:
            raise ValueError("glow_radius must be non-negative")
        if self.glow_color:
            try:
                ImageColor.getrgb(self.glow_color)
            except ValueError:
                raise ValueError(f"Invalid glow_color: {self.glow_color}")


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

        # A highlighted word's blurred glow halo (TextStyle.glow_radius, in
        # highlight_color) is invariant across the many re-renders karaoke
        # word-highlight animation does of that word's crisp foreground --
        # cached by _paste_word_glow to avoid re-blurring on every one of
        # those calls. Stores only the layer image itself (not its paste
        # position, which is recomputed fresh every call -- see that
        # method's docstring).
        self._glow_cache: Dict[tuple, Image.Image] = {}

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

    def _capture_rebuild_state(self) -> Optional[Dict]:
        """Plain-data snapshot of this builder's *resolved* pre-build inputs,
        picklable across a process boundary so a worker (see
        _render_video_chunk) can reconstruct an equivalent VideoBuilder and
        call build() itself.

        The built self.video_clip can't cross a Windows `spawn` process
        boundary: MoviePy's own VideoClip.__init__ closes over an
        unpicklable lambda, and swapping in a closure-capable pickler
        (dill/`multiprocess`) instead corrupted the ffmpeg audio reader's OS
        handle in testing. Rebuilding from plain data sidesteps both
        failure modes.

        Returns None for construction paths this doesn't support:
        from_multi_segments_advanced's per-segment TTS/caching makes a
        chunked re-build unsafe (it would redo external TTS calls, possibly
        non-deterministically, once per worker), and there's nothing to
        rebuild from if neither a single image nor image segments were ever
        configured.
        """
        if self._advanced_segments:
            return None
        if not self._image_filepath and not self._image_segments:
            return None
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'image_filepath': self._image_filepath,
            'image_segments': self._image_segments,
            'audio_filepath': self._audio_filepath,
            'text_segments': self._text_segments,
            'text_style': self._text_style,
            'overlay_images': self._overlay_images,
            'overlay_video_path': self._overlay_video_path,
            'overlay_video_config': self._overlay_video_config,
        }

    def _save_parallel(self, output_filepath: str) -> Optional[str]:
        """Chunked multiprocess render for long clips: split [0, duration)
        into (cpu_count - 1) pieces, render each piece's frames in its own
        process -- working around the fact that a single process's
        Python-side frame compositing, not FFmpeg encoding, is what
        actually bottlenecks a long karaoke-style render (profiled at ~99%
        of one core in Python vs ~17% of one core in ffmpeg) -- then
        concatenate the silent chunks and mux the original full audio back
        in once.

        Safe only because every per-frame effect VideoBuilder renders (word
        highlight, glow, scroll transition, audio-reactive pulse) is a pure
        function of absolute timestamp t, with no state carried across
        frames: each chunk is rendered independently with no knowledge of
        its neighbors, so the seams are invisible. A future effect that
        accumulates state frame-to-frame would break this.

        Returns None (never partial output) on any failure or when this
        clip doesn't qualify, so callers fall back to the single-process
        path.
        """
        state = self._capture_rebuild_state()
        if state is None:
            return None

        duration = self.video_clip.duration
        if not duration or duration < PARALLEL_RENDER_MIN_DURATION_S:
            return None

        # Cap worker count so chunks don't get smaller than MIN_CHUNK_DURATION_S:
        # each worker still builds its own text-segment overlays (see
        # _render_video_chunk's windowed filtering), so too many workers on a
        # clip that isn't long enough to amortize that per-worker cost wastes
        # memory and CPU rather than saving wall time.
        n_workers = min(
            max(1, (os.cpu_count() or 2) - 1),
            max(1, int(duration // MIN_CHUNK_DURATION_S)),
        )

        # Also cap by actually-available memory: each worker is a full
        # process (its own moviepy/numpy/PIL/SDK imports, base image, audio,
        # and frame-compositing buffers -- overlay images and other
        # per-frame effects add to this), so cpu_count() alone can wildly
        # overcommit on a machine with many cores but modest RAM, or one
        # already under memory pressure from other running applications.
        # Blindly maximizing worker count crashed on real content (multiple
        # concurrent MemoryError under compose_on/compose_mask) even after
        # per-worker memory use was fixed -- N processes at once is still N
        # times one process's peak.
        try:
            import psutil
            available = psutil.virtual_memory().available
            mem_capped_workers = max(1, int(available * 0.8 // PARALLEL_RENDER_MEM_PER_WORKER_BYTES))
            n_workers = min(n_workers, mem_capped_workers)
        except Exception as e:
            print(f"Could not check available memory ({e}); proceeding with CPU/duration-based worker count.")

        if n_workers <= 1:
            return None

        boundaries = [i * duration / n_workers for i in range(n_workers + 1)]
        boundaries[-1] = duration  # exact end, avoid a float-rounding gap

        chunk_dir = tempfile.mkdtemp(prefix="tish_video_chunks_")
        chunk_paths = [os.path.join(chunk_dir, f"chunk_{i:03d}.mp4") for i in range(n_workers)]

        try:
            import multiprocessing
            print(f"Rendering in {n_workers} parallel chunks...")
            with multiprocessing.Pool(n_workers) as pool:
                results = pool.starmap(
                    _render_video_chunk,
                    [
                        (state, boundaries[i], boundaries[i + 1], chunk_paths[i])
                        for i in range(n_workers)
                    ],
                )

            if any(r is None for r in results):
                print("Parallel chunk render failed for at least one chunk; falling back to single-process render.")
                return None

            concat_list_path = os.path.join(chunk_dir, "concat_list.txt")
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for p in chunk_paths:
                    escaped = p.replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            concat_output_path = os.path.join(chunk_dir, "concat_output.mp4")
            import subprocess
            concat_process = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", concat_output_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if concat_process.returncode != 0 or not os.path.exists(concat_output_path):
                print(f"FFMPEG concat failed: {concat_process.stderr}")
                return None

            if self._audio_filepath and os.path.exists(self._audio_filepath):
                mux_process = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", concat_output_path,
                        "-i", self._audio_filepath,
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-shortest",
                        output_filepath,
                    ],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if mux_process.returncode != 0:
                    print(f"FFMPEG audio mux failed: {mux_process.stderr}")
                    return None
            else:
                import shutil
                shutil.move(concat_output_path, output_filepath)

            print(f"Video saved (via {n_workers}-way parallel render) to: {output_filepath}")
            return output_filepath

        except Exception as e:
            print(f"Parallel render failed: {e}")
            traceback.print_exc()
            return None
        finally:
            import shutil
            shutil.rmtree(chunk_dir, ignore_errors=True)

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

        parallel_result = self._save_parallel(output_filepath)
        if parallel_result:
            return parallel_result

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
        # A font name may list several fallback variants joined by '-&-'
        # (some languages need more than one); use the first variant, and
        # trim stray separator hyphens along with whitespace so it still
        # matches the lookup table below.
        cleaned = font_name.split('&')[0].strip().strip('-')

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

    def _paste_word_glow(self, img: Image.Image, text: str, font: ImageFont.FreeTypeFont, x: float, y: float, word_width: float, line_h: int, glow_color, glow_radius: int, crop_w: Optional[float] = None) -> None:
        """Paste a blurred glow_color halo (a 'neon' look) behind one
        word's HIGHLIGHTED portion at (x, y) in img's coordinate space.
        glow_color may be a color string or an RGBA tuple (PIL accepts
        both as a text fill). Glow marks highlighted text specifically --
        callers only reach this for a word (or the already-filled part of
        one, via crop_w) that's actually being painted in highlight_color
        right now, never for text still sitting in its unlit font_color.

        Cached per (text, font identity, glow_color, glow_radius) as a
        small layer in its own LOCAL coordinate space, not this call's
        (x, y) -- karaoke word-highlight animation re-renders a word's
        crisp foreground many times a second (once per active-word
        sub-interval) while its glow's pixel content never changes, so
        re-blurring it on every one of those calls would be pure waste,
        but the paste position is always recomputed fresh from this call's
        (x, y), or a stale position from an earlier frame would paste
        today's glow in yesterday's place (a line's context-row slot
        slides as the song advances; a transition interpolates it).

        crop_w, when given, trims the CACHED layer's width at paste time
        so a karaoke word's glow grows in sync with its own progressive
        color fill without needing a fresh blur per progress value --
        mirrors the crisp fill's own overlay-crop-paste technique. The
        layer's local frame is [pad][word_width][pad] (left margin, glyph,
        right margin, each `pad` wide -- see below), so the crop needs
        `pad` on BOTH sides of crop_w: `pad` to keep the layer's own left
        margin (always present, even at the very first sliver of fill) and
        another `pad` so the blur's softness actually shows past the crisp
        fill edge rather than being clipped flush against it. At
        crop_w == word_width (a finished word) this evaluates to the full
        layer width, so a finished word's glow is exactly as symmetric as
        one pasted with crop_w=None."""
        if glow_radius <= 0:
            return
        pad = glow_radius * 3
        cache_key = (text, getattr(font, 'path', 'default'), font.size, glow_color, glow_radius)
        layer = self._glow_cache.get(cache_key)
        if layer is None:
            w = max(1, int(round(word_width)) + 2 * pad)
            h = max(1, line_h + 2 * pad)
            layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text((pad, pad), text, fill=glow_color, font=font)
            layer = layer.filter(ImageFilter.GaussianBlur(glow_radius))
            self._glow_cache[cache_key] = layer

        paste_layer = layer
        if crop_w is not None:
            crop_px = min(layer.width, max(0, int(round(crop_w)) + 2 * pad))
            if crop_px <= 0:
                return
            paste_layer = layer.crop((0, 0, crop_px, layer.height))

        img.paste(paste_layer, (int(round(x - pad)), int(round(y - pad))), paste_layer)

    def _render_text_frame_pil(self, width: int, height: int, words: List[Dict], active_word_idx: Optional[int], style: TextStyle, has_word_timestamps: bool = True, current_time: Optional[float] = None) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Renders a single frame of text using PIL.
        Supports word-level highlighting, wrapping, outline/stroke, and rounded-corner background box.
        Dynamically adapts font size to fit the available space.

        Returns (image, (x, y)): image is cropped to the actual drawn content's
        bounding box, not the full width x height canvas, and (x, y) is where
        that crop belongs in the full frame. Word-highlight animation calls
        this once per short sub-interval -- potentially thousands of times for
        a whole song under 'karaoke' highlight_style's finer subdivision --
        and each call's caller keeps the result alive until the final
        composite, so a full-canvas RGBA array (+ a separate float mask) per
        call can exhaust memory well before that composite happens. Cropping
        to content keeps each one to a small fraction of the canvas.
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
            return img, (0, 0)

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
        karaoke = has_word_timestamps and style.highlight_style == 'karaoke' and current_time is not None
        glow_radius = style.glow_radius
        glow_color = style.glow_color or highlight_color

        for line, line_w, line_x, line_y in line_info:
            curr_x = line_x
            for w_dict, is_active, idx, w_width in line:
                word_text = w_dict.get('word', '')
                draw_text = word_text + " "

                if karaoke:
                    # Cumulative + progressive: words already finished stay
                    # highlight_color, the word currently being sung fills
                    # from font_color to highlight_color over its own span,
                    # words not yet reached stay font_color.
                    w_start = w_dict.get('start', 0.0)
                    w_end = w_dict.get('end', 0.0)
                    if current_time >= w_end:
                        progress = 1.0
                    elif current_time <= w_start or w_end <= w_start:
                        progress = 0.0
                    else:
                        progress = (current_time - w_start) / (w_end - w_start)

                    fill_w = int(round(w_width * progress))
                    # Glow only the already-highlighted portion, drawn
                    # before the crisp text so it sits behind it -- see
                    # _paste_word_glow's docstring for why this is cheap
                    # even under karaoke's per-sub-interval re-rendering.
                    if glow_radius > 0 and fill_w > 0:
                        self._paste_word_glow(
                            img, draw_text, font, curr_x, line_y, w_width, line_height,
                            glow_color, glow_radius, crop_w=fill_w,
                        )

                    draw.text(
                        (curr_x, line_y), draw_text, fill=font_color,
                        font=font, stroke_width=stroke_width, stroke_fill=stroke_color,
                    )
                    if fill_w > 0:
                        # Word-sized overlay, not img.size -- this runs once
                        # per active-word sub-interval (many per word under
                        # karaoke's fine subdivision), so a full-canvas
                        # allocation here multiplies the same memory problem
                        # documented on this method's docstring.
                        ov_pad = stroke_width + 4
                        ov_left = curr_x
                        ov_top = max(0, line_y - ov_pad)
                        ov_w = w_width + ov_pad
                        ov_h = line_height + 2 * ov_pad
                        overlay = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
                        ImageDraw.Draw(overlay).text(
                            (0, line_y - ov_top), draw_text, fill=highlight_color,
                            font=font, stroke_width=stroke_width, stroke_fill=stroke_color,
                        )
                        region = overlay.crop((0, 0, min(fill_w, ov_w), ov_h))
                        img.paste(region, (ov_left, ov_top), region)
                else:
                    # Active word highlight color, else normal font color
                    color = highlight_color if is_active else font_color
                    if glow_radius > 0 and is_active:
                        self._paste_word_glow(
                            img, draw_text, font, curr_x, line_y, w_width, line_height,
                            glow_color, glow_radius,
                        )
                    draw.text(
                        (curr_x, line_y),
                        draw_text,
                        fill=color,
                        font=font,
                        stroke_width=stroke_width,
                        stroke_fill=stroke_color
                    )

                curr_x += w_width

        # Crop to the actual drawn content's bounding box -- see this
        # method's docstring for why (memory: many small arrays instead of
        # many full-canvas ones).
        pad = stroke_width + max(4, int(resolved_font_size * 0.3))
        if style.glow_radius > 0:
            # Match _paste_word_glow's own layer padding, or the glow's
            # outer edge gets clipped by this crop.
            pad = max(pad, style.glow_radius * 3)
        min_x = min(info[2] for info in line_info)
        max_x = max(info[2] + info[1] for info in line_info)
        min_y = y_start
        max_y = y_start + total_height
        if style.bg_color:
            min_x = min(min_x, box_left)
            max_x = max(max_x, box_right)
            min_y = min(min_y, box_top)
            max_y = max(max_y, box_bottom)

        left = max(0, int(min_x) - pad)
        top = max(0, int(min_y) - pad)
        right = min(width, int(max_x) + pad)
        bottom = min(height, int(max_y) + pad)

        if right <= left or bottom <= top:
            return img, (0, 0)

        return img.crop((left, top, right, bottom)), (left, top)

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

        if style.context_lines > 0:
            return self._overlay_scrolling_lyrics_on_single_clip(video_clip, segments, style)

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
                karaoke = has_word_timestamps and style.highlight_style == 'karaoke'
                max_steps_per_word = 8  # cap so a long held note doesn't blow up clip count
                fps = self.fps or DEFAULT_FPS

                boundaries = [seg_start]
                for w in words:
                    w_start = w.get('start', 0.0)
                    w_end = w.get('end', 0.0)
                    # Keep boundaries within segment bounds
                    w_start = max(seg_start, min(w_start, seg_end))
                    w_end = max(seg_start, min(w_end, seg_end))
                    boundaries.append(w_start)
                    if karaoke and w_end > w_start:
                        step_count = max(1, min(max_steps_per_word, int(round((w_end - w_start) * fps))))
                        step = (w_end - w_start) / step_count
                        boundaries.extend(w_start + i * step for i in range(1, step_count))
                    boundaries.append(w_end)
                boundaries.append(seg_end)

                # Remove duplicates and sort
                boundaries = sorted(list(set(boundaries)))

                # Merge boundaries closer than a frame apart -- otherwise the
                # tiny interval between them gets skipped below, and if that
                # gap lands on an encoded frame it renders blank.
                if len(boundaries) > 1:
                    min_gap = 1.0 / fps
                    merged = [boundaries[0]]
                    for b in boundaries[1:]:
                        if b - merged[-1] >= min_gap:
                            merged.append(b)
                    if merged[-1] != boundaries[-1]:
                        merged[-1] = boundaries[-1]
                    boundaries = merged

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

                    # Render frame using PIL -- cropped to its content's
                    # bounding box, not the full canvas (see
                    # _render_text_frame_pil's docstring)
                    frame_img, (frame_x, frame_y) = self._render_text_frame_pil(
                        video_clip.w, video_clip.h, words, active_idx, style, has_word_timestamps,
                        current_time=mid if karaoke else None,
                    )

                    # Convert to RGBA numpy array
                    frame_arr = np.array(frame_img)

                    rgb_arr = frame_arr[:, :, :3]
                    alpha_arr = (frame_arr[:, :, 3] / 255.0).astype(np.float32)  # Normalize alpha to 0.0-1.0

                    # Create the ImageClip and its transparency mask
                    img_clip = ImageClip(rgb_arr)
                    mask_clip = ImageClip(alpha_arr, is_mask=True)
                    img_clip = img_clip.with_mask(mask_clip)

                    # Set time and position parameters (position is absolute
                    # pixels, matching where _render_text_frame_pil cropped
                    # the content from within the full canvas)
                    img_clip = img_clip.with_start(t1).with_duration(dur).with_position((frame_x, frame_y))
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

    def _overlay_scrolling_lyrics_on_single_clip(self, video_clip: VideoFileClip, segments: List[Dict], style: TextStyle) -> Optional[VideoFileClip]:
        """Scrolling multi-line lyric display (style.context_lines > 0):
        shows the current line plus lines before/after it, stacked around
        the current line's fixed position so earlier/later lines scroll as
        it advances. An upcoming line only joins the window
        style.upcoming_lead seconds before its own start."""
        offset = getattr(style, 'timing_offset', 0.0)
        prepared = []
        for segment in segments:
            text_content = segment.get('text', '').replace('\n', ' ').strip()
            seg_start = segment.get('start', 0.0) + offset
            seg_end = segment.get('end', 0.0) + offset
            if not text_content or seg_start >= seg_end:
                continue

            words = segment.get('words', [])
            if words:
                words = [
                    {
                        'word': w.get('word', ''),
                        'start': w.get('start', 0.0) + offset,
                        'end': w.get('end', 0.0) + offset,
                    }
                    for w in words
                ]
            else:
                word_list = text_content.split()
                duration = seg_end - seg_start
                per_word_duration = duration / len(word_list) if word_list else 0.0
                words = [
                    {
                        'word': w,
                        'start': seg_start + idx * per_word_duration,
                        'end': seg_start + (idx + 1) * per_word_duration,
                    }
                    for idx, w in enumerate(word_list)
                ]
            words = sorted(words, key=lambda x: x.get('start', 0.0))
            prepared.append({'start': seg_start, 'end': seg_end, 'words': words})

        prepared.sort(key=lambda s: s['start'])
        if not prepared:
            return video_clip

        n = len(prepared)
        context_n = max(0, int(style.context_lines))
        upcoming_lead = max(0.0, float(style.upcoming_lead))
        scroll_duration = max(0.0, float(style.scroll_duration))
        clip_duration = video_clip.duration
        fps = self.fps or DEFAULT_FPS
        max_steps_per_word = 8
        karaoke = style.highlight_style == 'karaoke'

        def current_idx_at(t):
            idx = None
            for i, seg in enumerate(prepared):
                if seg['start'] <= t:
                    idx = i
                else:
                    break
            return idx

        def build_window(cur, t):
            """The lines visible at time t if `cur` were the current line."""
            window = []
            if cur is None:
                for j in range(0, min(context_n, n)):
                    seg = prepared[j]
                    if t >= seg['start'] - upcoming_lead:
                        window.append((j, seg, False))
                    else:
                        break
            else:
                for j in range(max(0, cur - context_n), cur):
                    window.append((j, prepared[j], False))
                window.append((cur, prepared[cur], True))
                for j in range(cur + 1, min(n, cur + 1 + context_n)):
                    seg = prepared[j]
                    if t >= seg['start'] - upcoming_lead:
                        window.append((j, seg, False))
                    else:
                        break
            return window

        boundaries = {0.0, clip_duration}
        for seg in prepared:
            boundaries.add(max(0.0, seg['start']))
            boundaries.add(min(clip_duration, seg['end']))
            boundaries.add(max(0.0, seg['start'] - upcoming_lead))
            for w in seg['words']:
                w_start = max(seg['start'], min(w.get('start', 0.0), seg['end']))
                w_end = max(seg['start'], min(w.get('end', 0.0), seg['end']))
                boundaries.add(w_start)
                boundaries.add(w_end)
                if karaoke and w_end > w_start:
                    step_count = max(1, min(max_steps_per_word, int(round((w_end - w_start) * fps))))
                    step = (w_end - w_start) / step_count
                    boundaries.update(w_start + i * step for i in range(1, step_count))

        if scroll_duration > 0:
            scroll_steps = max(1, min(12, int(round(scroll_duration * fps))))
            for i in range(1, n):
                t0 = prepared[i]['start']
                t_end = min(clip_duration, t0 + scroll_duration)
                step = (t_end - t0) / scroll_steps
                boundaries.update(t0 + k * step for k in range(scroll_steps + 1))

        boundaries = sorted(b for b in boundaries if 0.0 <= b <= clip_duration)

        # Merge boundaries closer than a frame apart -- otherwise the tiny
        # interval between them gets skipped below, and if that gap lands
        # on an encoded frame it renders blank.
        if len(boundaries) > 1:
            min_gap = 1.0 / fps
            merged = [boundaries[0]]
            for b in boundaries[1:]:
                if b - merged[-1] >= min_gap:
                    merged.append(b)
            if merged[-1] != boundaries[-1]:
                merged[-1] = boundaries[-1]
            boundaries = merged

        if len(boundaries) < 2:
            return video_clip

        width, height = video_clip.w, video_clip.h

        # Rasterize one boundary interval's frame at a time instead of
        # pre-rendering every interval as its own full-canvas ImageClip and
        # holding all of them in memory simultaneously -- for dense/long
        # lyrics that pre-render approach can need many GB (each interval's
        # rasterized RGBA image is full-canvas-sized before compositing; a
        # few hundred to a few thousand intervals is common for a
        # multi-minute song). A single-slot cache is enough because
        # write_videofile requests frames in increasing time order, so
        # consecutive output frames overwhelmingly land in the same
        # interval and reuse the same rasterized image rather than
        # re-rendering it.
        cache = {'idx': None, 'rgba': None}

        def rasterize(idx: int) -> np.ndarray:
            if cache['idx'] == idx:
                return cache['rgba']

            t1, t2 = boundaries[idx], boundaries[idx + 1]
            mid = (t1 + t2) / 2.0
            cur = current_idx_at(mid)

            in_transition = (
                scroll_duration > 0 and cur is not None and cur > 0
                and (mid - prepared[cur]['start']) < scroll_duration
            )

            frame_img, frame_x, frame_y = None, 0, 0
            try:
                if in_transition:
                    p = max(0.0, min(1.0, (mid - prepared[cur]['start']) / scroll_duration))
                    window_prev = build_window(cur - 1, mid)
                    window_next = build_window(cur, mid)
                    frame_img, (frame_x, frame_y) = self._render_scrolling_lyrics_transition_frame_pil(
                        width, height, window_prev, window_next, p, mid, style,
                    )
                else:
                    window = build_window(cur, mid)
                    if window:
                        frame_img, (frame_x, frame_y) = self._render_scrolling_lyrics_frame_pil(
                            width, height, window, mid, style,
                        )
            except Exception as e:
                print(f"Error rendering scrolling lyrics frame: {e}")
                traceback.print_exc()

            canvas = np.zeros((height, width, 4), dtype=np.uint8)
            if frame_img is not None:
                frame_arr = np.array(frame_img)
                if frame_arr.size:
                    fh, fw = frame_arr.shape[:2]
                    x0, y0 = max(0, frame_x), max(0, frame_y)
                    x1, y1 = min(width, frame_x + fw), min(height, frame_y + fh)
                    if x1 > x0 and y1 > y0:
                        canvas[y0:y1, x0:x1] = frame_arr[y0 - frame_y:y1 - frame_y, x0 - frame_x:x1 - frame_x]

            cache['idx'] = idx
            cache['rgba'] = canvas
            return canvas

        def boundary_index_at(t: float) -> int:
            idx = bisect.bisect_right(boundaries, t) - 1
            return max(0, min(idx, len(boundaries) - 2))

        def make_frame(t):
            return rasterize(boundary_index_at(t))[:, :, :3]

        def make_mask_frame(t):
            return rasterize(boundary_index_at(t))[:, :, 3].astype(np.float32) / 255.0

        try:
            print("DEBUG: Compositing video with lazily-rasterized scrolling lyrics overlay...")
            overlay_mask = VideoClip(frame_function=make_mask_frame, is_mask=True, duration=clip_duration)
            overlay_clip = VideoClip(frame_function=make_frame, duration=clip_duration).with_mask(overlay_mask)
            final_clip = CompositeVideoClip([video_clip, overlay_clip])
            if video_clip.audio:
                final_clip = final_clip.with_audio(video_clip.audio)
            return final_clip

        except Exception as e:
            print(f"Error compositing video with scrolling lyrics: {e}")
            traceback.print_exc()
            return None

    def _layout_lyric_row(self, words: List[Dict], font: ImageFont.FreeTypeFont, max_w: int, get_word_size) -> List[List[Tuple[Dict, int, int]]]:
        """Wrap one lyric line's words into physical lines that fit max_w."""
        lines = []
        current_line = []
        current_line_w = 0
        for idx, w_dict in enumerate(words):
            word_text = w_dict.get('word', '').strip()
            if not word_text:
                continue
            measure_text = word_text + " "
            w_width, _ = get_word_size(measure_text)
            if current_line_w + w_width > max_w and current_line:
                lines.append(current_line)
                current_line = [(w_dict, idx, w_width)]
                current_line_w = w_width
            else:
                current_line.append((w_dict, idx, w_width))
                current_line_w += w_width
        if current_line:
            lines.append(current_line)
        return lines

    def _compute_scroll_geometry(self, window: List[Tuple[int, Dict, bool]], width: int, height: int, style: TextStyle, draw: ImageDraw.ImageDraw):
        """Wrap and stack window's rows vertically around the current row's
        anchor position. Returns (rows, font_path, center_x); each row dict
        is keyed by 'seg_index' so two geometries can be blended row-for-row."""
        font_path = self._get_font_path(style.get_font_for_language())
        max_w = int(width * (style.box_size[0] or 0.8))

        def get_word_size(text, font):
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            elif hasattr(font, 'getbbox'):
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            return font.getsize(text)

        try:
            current_font = ImageFont.truetype(font_path, style.font_size)
        except Exception:
            current_font = ImageFont.load_default()
        context_font_size = max(10, int(style.font_size * style.context_font_scale))
        try:
            context_font = ImageFont.truetype(font_path, context_font_size)
        except Exception:
            context_font = current_font

        _, current_sample_h = get_word_size("Ay", current_font)
        _, context_sample_h = get_word_size("Ay", context_font)
        current_line_h = int(current_sample_h * 1.3)
        context_line_h = int(context_sample_h * 1.3)
        row_gap = int(context_line_h * 0.35)

        rows = []
        for seg_index, seg, is_current in window:
            font = current_font if is_current else context_font
            font_size = style.font_size if is_current else context_font_size
            line_h = current_line_h if is_current else context_line_h
            phys_lines = self._layout_lyric_row(seg['words'], font, max_w, lambda t, f=font: get_word_size(t, f))
            if not phys_lines:
                continue
            rows.append({
                'seg_index': seg_index, 'is_current': is_current, 'font_size': font_size,
                'line_h': line_h, 'phys_lines': phys_lines, 'block_h': len(phys_lines) * line_h,
            })

        if not rows:
            return [], font_path, width // 2

        pos_x, pos_y = style.text_position
        box_w = max_w

        if isinstance(pos_x, str):
            pos_x_lower = pos_x.lower()
            if pos_x_lower == 'left':
                center_x = box_w // 2
            elif pos_x_lower == 'right':
                center_x = width - box_w // 2
            else:
                center_x = width // 2
        elif isinstance(pos_x, (int, float)):
            if 0.0 <= pos_x <= 1.0:
                if pos_x == 0.5:
                    center_x = width // 2
                elif pos_x < 0.4:
                    center_x = int(width * (pos_x + (style.box_size[0] or 0.8) / 2))
                else:
                    center_x = int(width * pos_x)
            else:
                if pos_x < width * 0.4:
                    center_x = int(pos_x + box_w // 2)
                else:
                    center_x = int(pos_x)
        else:
            center_x = width // 2

        current_i = next((i for i, r in enumerate(rows) if r['is_current']), len(rows) // 2)
        current_block_h = rows[current_i]['block_h']

        if isinstance(pos_y, str):
            pos_y_lower = pos_y.lower()
            if pos_y_lower == 'top':
                current_center_y = int(height * 0.05) + current_block_h // 2
            elif pos_y_lower == 'bottom':
                current_center_y = height - int(height * 0.05) - current_block_h // 2
            else:
                current_center_y = height // 2
        elif isinstance(pos_y, (int, float)):
            current_center_y = int(height * pos_y) if 0.0 <= pos_y <= 1.0 else int(pos_y)
        else:
            current_center_y = height // 2

        row_tops = [0] * len(rows)
        row_tops[current_i] = current_center_y - current_block_h // 2

        y = row_tops[current_i]
        for i in range(current_i - 1, -1, -1):
            y -= row_gap + rows[i]['block_h']
            row_tops[i] = y

        y = row_tops[current_i] + current_block_h
        for i in range(current_i + 1, len(rows)):
            row_tops[i] = y + row_gap
            y = row_tops[i] + rows[i]['block_h']

        for r, top in zip(rows, row_tops):
            r['top'] = top

        return rows, font_path, center_x

    def _draw_lyric_row_block(self, img: Image.Image, draw: ImageDraw.ImageDraw, phys_lines, top: float, font: ImageFont.FreeTypeFont, line_h: int, alpha: int, center_x: int, is_current_style: bool, current_time: Optional[float], style: TextStyle, bounds: list) -> None:
        """Draw one lyric row's wrapped lines at the given top/font/alpha,
        with word highlighting only when is_current_style. Word widths are
        measured against `font`, not whatever font the row was wrapped
        with, so an interpolated size during a transition stays centered.
        Mutates bounds ([min_x, min_y, max_x, max_y], None until first write)."""
        if alpha <= 0:
            return

        highlight_color = getattr(style, 'highlight_color', '#FFD700')
        font_color = style.font_color or 'white'
        stroke_color = style.stroke_color or 'black'
        stroke_width = style.stroke_width or 0
        karaoke = style.highlight_style == 'karaoke'
        alpha = max(0, min(255, int(round(alpha))))

        def with_alpha(color_str):
            rgb = ImageColor.getrgb(color_str)[:3]
            return (rgb[0], rgb[1], rgb[2], alpha)

        row_font_color = with_alpha(font_color)
        row_stroke_color = with_alpha(stroke_color)
        row_highlight_color = with_alpha(highlight_color)

        def get_word_size(text):
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            elif hasattr(font, 'getbbox'):
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            return font.getsize(text)

        # Layout pass: resolve every word's (x, y) once, shared by the glow
        # pre-pass below and the crisp draw pass that follows it -- avoids
        # measuring word widths twice.
        positions = []  # (x, y, word_text, w_dict, w_width)
        curr_y = int(round(top))
        for phys_line in phys_lines:
            widths = [get_word_size(w_dict.get('word', '') + " ")[0] for w_dict, _idx, _old_w in phys_line]
            line_w = sum(widths)
            line_x = center_x - line_w // 2
            curr_x = line_x

            for (w_dict, _w_idx, _old_w), w_width in zip(phys_line, widths):
                positions.append((curr_x, curr_y, w_dict.get('word', ''), w_dict, w_width))
                curr_x += w_width

            row_left, row_right = line_x, line_x + line_w
            row_min_y, row_max_y = curr_y, curr_y + line_h
            bounds[0] = row_left if bounds[0] is None else min(bounds[0], row_left)
            bounds[2] = row_right if bounds[2] is None else max(bounds[2], row_right)
            bounds[1] = row_min_y if bounds[1] is None else min(bounds[1], row_min_y)
            bounds[3] = row_max_y if bounds[3] is None else max(bounds[3], row_max_y)

            curr_y += line_h

        glow_radius = style.glow_radius
        row_glow_color = with_alpha(style.glow_color or highlight_color) if glow_radius > 0 else None

        for curr_x, curr_y, word_text, w_dict, w_width in positions:
            draw_text = word_text + " "
            if is_current_style and karaoke and current_time is not None:
                w_start = w_dict.get('start', 0.0)
                w_end = w_dict.get('end', 0.0)
                if current_time >= w_end:
                    progress = 1.0
                elif current_time <= w_start or w_end <= w_start:
                    progress = 0.0
                else:
                    progress = (current_time - w_start) / (w_end - w_start)

                fill_w = int(round(w_width * progress))
                # Glow only the already-highlighted portion, drawn before
                # the crisp text so it sits behind it -- growing in sync
                # with the fill via crop_w, not a fresh blur per frame.
                if row_glow_color is not None and fill_w > 0:
                    self._paste_word_glow(
                        img, draw_text, font, curr_x, curr_y, w_width, line_h,
                        row_glow_color, glow_radius, crop_w=fill_w,
                    )

                draw.text(
                    (curr_x, curr_y), draw_text, fill=row_font_color,
                    font=font, stroke_width=stroke_width, stroke_fill=row_stroke_color,
                )
                if fill_w > 0:
                    ov_pad = stroke_width + 4
                    ov_left = curr_x
                    ov_top = max(0, curr_y - ov_pad)
                    ov_w = w_width + ov_pad
                    ov_h = line_h + 2 * ov_pad
                    overlay = Image.new("RGBA", (ov_w, ov_h), (0, 0, 0, 0))
                    ImageDraw.Draw(overlay).text(
                        (0, curr_y - ov_top), draw_text, fill=row_highlight_color,
                        font=font, stroke_width=stroke_width, stroke_fill=row_stroke_color,
                    )
                    region = overlay.crop((0, 0, min(fill_w, ov_w), ov_h))
                    img.paste(region, (ov_left, ov_top), region)
            elif is_current_style:
                w_start = w_dict.get('start', 0.0)
                w_end = w_dict.get('end', 0.0)
                is_active = current_time is not None and w_start <= current_time <= w_end
                if row_glow_color is not None and is_active:
                    self._paste_word_glow(
                        img, draw_text, font, curr_x, curr_y, w_width, line_h,
                        row_glow_color, glow_radius,
                    )
                draw.text(
                    (curr_x, curr_y), draw_text,
                    fill=row_highlight_color if is_active else row_font_color,
                    font=font, stroke_width=stroke_width, stroke_fill=row_stroke_color,
                )
            else:
                draw.text(
                    (curr_x, curr_y), draw_text, fill=row_font_color,
                    font=font, stroke_width=stroke_width, stroke_fill=row_stroke_color,
                )

    def _crop_to_bounds(self, img: Image.Image, width: int, height: int, bounds: list, style: TextStyle) -> Tuple[Image.Image, Tuple[int, int]]:
        """Shared crop-to-content-bounding-box tail for both scrolling
        renderers -- see _render_text_frame_pil's docstring for why cropping
        (not the full canvas) matters here."""
        if bounds[0] is None:
            return img, (0, 0)
        stroke_width = style.stroke_width or 0
        pad = stroke_width + max(4, int(style.font_size * 0.3))
        if style.glow_radius > 0:
            # Match _paste_word_glow's own layer padding, or the glow's
            # outer edge gets clipped by this crop.
            pad = max(pad, style.glow_radius * 3)
        left = max(0, int(bounds[0]) - pad)
        top = max(0, int(bounds[1]) - pad)
        right = min(width, int(bounds[2]) + pad)
        bottom = min(height, int(bounds[3]) + pad)
        if right <= left or bottom <= top:
            return img, (0, 0)
        return img.crop((left, top, right, bottom)), (left, top)

    def _render_scrolling_lyrics_frame_pil(self, width: int, height: int, window: List[Tuple[int, Dict, bool]], current_time: Optional[float], style: TextStyle) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render one settled (non-transitioning) frame of the scrolling
        display. window: (seg_index, seg, is_current) tuples top-to-bottom,
        exactly one is_current row. Returns (image, (x, y)) cropped to
        content's bounding box."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        rows, font_path, center_x = self._compute_scroll_geometry(window, width, height, style, draw)
        if not rows:
            return img, (0, 0)

        fonts_cache: Dict[int, ImageFont.FreeTypeFont] = {}

        def get_font(size):
            size = max(6, int(round(size)))
            if size not in fonts_cache:
                try:
                    fonts_cache[size] = ImageFont.truetype(font_path, size)
                except Exception:
                    fonts_cache[size] = ImageFont.load_default()
            return fonts_cache[size]

        context_alpha = max(0, min(255, int(round(style.context_opacity * 255))))
        bounds = [None, None, None, None]
        for r in rows:
            self._draw_lyric_row_block(
                img, draw, r['phys_lines'], r['top'], get_font(r['font_size']),
                r['line_h'], 255 if r['is_current'] else context_alpha, center_x,
                r['is_current'], current_time, style, bounds,
            )

        return self._crop_to_bounds(img, width, height, bounds, style)

    def _render_scrolling_lyrics_transition_frame_pil(self, width: int, height: int, window_prev: List[Tuple[int, Dict, bool]], window_next: List[Tuple[int, Dict, bool]], p: float, current_time: Optional[float], style: TextStyle) -> Tuple[Image.Image, Tuple[int, int]]:
        """Render one frame of a line-change transition at progress p in
        [0, 1], blending window_prev's settled geometry into window_next's.
        A line present in both windows interpolates position/size/opacity
        continuously; a line present in only one fades in/out in place."""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        rows_prev, font_path, _center_x_prev = self._compute_scroll_geometry(window_prev, width, height, style, draw)
        rows_next, font_path, center_x = self._compute_scroll_geometry(window_next, width, height, style, draw)

        by_prev = {r['seg_index']: r for r in rows_prev}
        by_next = {r['seg_index']: r for r in rows_next}
        context_alpha = max(0, min(255, int(round(style.context_opacity * 255))))

        fonts_cache: Dict[int, ImageFont.FreeTypeFont] = {}

        def get_font(size):
            size = max(6, int(round(size)))
            if size not in fonts_cache:
                try:
                    fonts_cache[size] = ImageFont.truetype(font_path, size)
                except Exception:
                    fonts_cache[size] = ImageFont.load_default()
            return fonts_cache[size]

        bounds = [None, None, None, None]
        for seg_index in sorted(set(by_prev) | set(by_next)):
            rp = by_prev.get(seg_index)
            rn = by_next.get(seg_index)

            if rp and rn:
                top = rp['top'] + (rn['top'] - rp['top']) * p
                font_size = rp['font_size'] + (rn['font_size'] - rp['font_size']) * p
                alpha_p = 255 if rp['is_current'] else context_alpha
                alpha_n = 255 if rn['is_current'] else context_alpha
                alpha = alpha_p + (alpha_n - alpha_p) * p
                phys_lines = rn['phys_lines']
                is_current_style = rp['is_current'] or rn['is_current']
            elif rn:
                # newly entering the window -- fade in rather than glide from nowhere
                top = rn['top']
                font_size = rn['font_size']
                alpha = (255 if rn['is_current'] else context_alpha) * p
                phys_lines = rn['phys_lines']
                is_current_style = rn['is_current']
            else:
                # leaving the window -- fade out in place
                top = rp['top']
                font_size = rp['font_size']
                alpha = (255 if rp['is_current'] else context_alpha) * (1 - p)
                phys_lines = rp['phys_lines']
                is_current_style = rp['is_current']

            if alpha <= 1:
                continue

            line_h = int(font_size * 1.3)
            self._draw_lyric_row_block(
                img, draw, phys_lines, top, get_font(font_size), line_h,
                alpha, center_x, is_current_style, current_time, style, bounds,
            )

        return self._crop_to_bounds(img, width, height, bounds, style)

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


# Extra time before/after a chunk's own [t0, t1) window that a text segment
# might still need to be included for: TextStyle.upcoming_lead lets a line
# appear before its own start, scroll_duration lets a transition reach past
# a segment's end, plus a fixed safety margin for context_lines neighbors.
_CHUNK_SEGMENT_PAD_S = 10.0


def _segments_overlapping_window(segments: Optional[List[Dict]], window_start: float, window_end: float) -> Optional[List[Dict]]:
    """Subset of `segments` whose own [start, end] overlaps [window_start,
    window_end]. _render_video_chunk uses this so a worker only builds the
    per-segment overlay clips (word-highlight, glow, scrolling-lyrics
    frames) its own chunk can actually show, instead of every segment in
    the whole video -- building every segment in each of N concurrent
    workers is what exhausted memory before this filter existed.
    """
    if not segments:
        return segments
    return [
        seg for seg in segments
        if seg.get('end', seg.get('start', 0)) >= window_start and seg.get('start', 0) <= window_end
    ]


def _render_video_chunk(state: Dict, t0: float, t1: float, chunk_output_path: str) -> Optional[str]:
    """Runs in its own worker process, spawned by VideoBuilder._save_parallel:
    rebuilds an equivalent VideoBuilder from a plain-data state snapshot,
    then writes only the [t0, t1) slice of it, silently -- the caller muxes
    the original full audio back in once, after every chunk is
    concatenated. Must stay a module-level function (not a method or
    closure) so it's importable and picklable across a Windows `spawn`
    process boundary.
    """
    try:
        style = state['text_style']
        pad = _CHUNK_SEGMENT_PAD_S
        if style is not None:
            pad = max(pad, (style.upcoming_lead or 0) + (style.scroll_duration or 0) + 5.0)
        window_start, window_end = t0 - pad, t1 + pad

        builder = VideoBuilder(fps=state['fps'], width=state['width'], height=state['height'])
        builder._image_filepath = state['image_filepath']
        builder._image_segments = state['image_segments']
        builder._audio_filepath = state['audio_filepath']
        builder._text_segments = _segments_overlapping_window(state['text_segments'], window_start, window_end)
        builder._text_style = state['text_style']
        builder._overlay_images = state['overlay_images']
        builder._overlay_video_path = state['overlay_video_path']
        builder._overlay_video_config = state['overlay_video_config']

        clip = builder.build()
        if clip is None:
            return None

        # A worker's own reload of the same audio file can measure a
        # fractionally different duration than the main process did when it
        # first computed chunk boundaries (observed ~0.4s drift in testing,
        # not just float rounding), and the attached audio clip can itself
        # measure shorter than the video clip's own .duration -- subclipped()
        # applies to both, so clamp against whichever is more restrictive.
        clip_duration = clip.duration
        if clip.audio is not None and clip.audio.duration is not None:
            clip_duration = min(clip_duration, clip.audio.duration)
        t0 = max(0.0, min(t0, clip_duration))
        t1 = max(t0, min(t1, clip_duration))
        if t1 <= t0:
            return None

        chunk_clip = clip.subclipped(t0, t1)
        chunk_clip.write_videofile(
            chunk_output_path,
            fps=state['fps'],
            codec="libx264",
            audio=False,
            logger=None,
        )
        return chunk_output_path
    except Exception:
        traceback.print_exc()
        return None
