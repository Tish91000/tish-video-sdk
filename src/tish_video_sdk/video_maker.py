from __future__ import annotations

import os

from moviepy import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx import MaskColor, Loop
import numpy as np
from PIL import Image
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
from typing import TYPE_CHECKING, Optional, List, Dict, Union
import tempfile

if TYPE_CHECKING:
    from .subtitles import SubtitleStyle

DEFAULT_FPS = 24 # Frames per second for the output video


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
        self._tts_builder: Optional[TTSBuilder] = None
        self._tts_text: Optional[str] = None
        self._language_code: Optional[str] = None
        self._provider: Optional[str] = None

        # Overlay configuration
        self._overlay_images: List[Dict] = []

        # Overlay Video configuration
        self._overlay_video_path: Optional[str] = None
        self._overlay_video_config: Optional[Dict] = None

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
        if self._image_segments:
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
