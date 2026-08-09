Video Maker
===========

.. currentmodule:: tish_video_sdk.video_maker

Builds a video from a single image or a sequence of timestamped image/video
segments, paired with either a provided audio file or TTS-generated narration
(via :class:`~tish_video_sdk.tts.TTSBuilder`). Supports image and video overlays
(logos, reactive GIFs, ...) composited on top of the base clip.

Subtitle support (:meth:`VideoBuilder.with_subtitles`, TTS auto-subtitles, and
the advanced multi-segment "chapelet" builder that generates and caches TTS
audio + subtitles per segment) is not available yet — it returns once
``subtitles`` lands.

.. automodule:: tish_video_sdk.video_maker
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.video_maker import VideoBuilder

   # Single image + provided audio file:
   builder = VideoBuilder.from_single_image_with_audio("scene.png", "narration.wav")
   builder.build()
   builder.save("output.mp4")

   # Single image + TTS narration, no audio file needed:
   builder = VideoBuilder.from_single_image_with_tts("scene.png", "Bonjour le monde", "fr")
   builder.build()
   builder.save("output.mp4")
