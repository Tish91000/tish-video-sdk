Video Maker
===========

.. currentmodule:: tish_video_sdk.video_maker

Builds a video from a single image, a sequence of timestamped image/video
segments, or advanced per-segment TTS+subtitle "chapelet" segments (see
:meth:`VideoBuilder.from_multi_segments_advanced`), paired with either a
provided audio file or TTS-generated narration (via
:class:`~tish_video_sdk.tts.TTSBuilder`). Supports image and video overlays
(logos, reactive GIFs, ...), and timed text overlays (titles, captions,
karaoke-style word-highlighted text, and subtitles via
:meth:`VideoBuilder.with_text_segments`/:meth:`VideoBuilder.with_subtitles`/
:meth:`VideoBuilder.with_tts_subtitles` and :class:`TextStyle`) composited on
top of the base clip. Subtitle *generation* (forced alignment from audio) is
:mod:`tish_video_sdk.subtitles` — this module only renders whatever timed
segments it's given.

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

   # TTS narration + auto-generated subtitles (see :doc:`subtitles`):
   builder = VideoBuilder.from_single_image_with_tts("scene.png", "Bonjour le monde", "fr")
   builder.with_tts_subtitles(style=TextStyle(font_color="white", bg_color="black"))
   builder.build()
   builder.save("output.mp4")
