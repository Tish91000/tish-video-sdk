Video Maker
===========

.. currentmodule:: tish_video_sdk.video_maker

Builds a video from a single image or a sequence of timestamped image/video
segments, paired with either a provided audio file or TTS-generated narration
(via :class:`~tish_video_sdk.tts.TTSBuilder`). Supports image and video overlays
(logos, reactive GIFs, ...), and timed text overlays (titles, captions, and
karaoke-style word-highlighted text via :meth:`VideoBuilder.with_text_segments`
and :class:`TextStyle`) composited on top of the base clip.

Subtitle support (auto-generating text segments from narration audio via
forced alignment, and the advanced multi-segment "chapelet" builder that
generates and caches TTS audio + subtitles per segment) is not available yet
— it returns once ``subtitles`` lands, built on top of
:meth:`VideoBuilder.with_text_segments`.

.. automodule:: tish_video_sdk.video_maker
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.video_maker import VideoBuilder, TextStyle

   # Single image + provided audio file:
   builder = VideoBuilder.from_single_image_with_audio("scene.png", "narration.wav")
   builder.build()
   builder.save("output.mp4")

   # Single image + TTS narration, no audio file needed:
   builder = VideoBuilder.from_single_image_with_tts("scene.png", "Bonjour le monde", "fr")
   builder.build()
   builder.save("output.mp4")

   # Single image + a title overlay for the whole clip:
   builder = (
       VideoBuilder.from_single_image("scene.png", duration=5.0)
       .with_text_segments(
           [{"text": "Chapter One", "start": 0.0, "end": 5.0}],
           style=TextStyle(font_size=80, text_position=("center", "top")),
       )
   )
   builder.build()
   builder.save("output.mp4")
