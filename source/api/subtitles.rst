Subtitles
=========

.. currentmodule:: tish_video_sdk.subtitles

Generates timed subtitle segments from narration text and its synthesized
audio, in the shape :meth:`~tish_video_sdk.video_maker.VideoBuilder.with_text_segments`
expects. Two alignment strategies:

- **Montreal Forced Aligner (MFA)** — word-level accuracy, the default
  (``USE_MFA_ALIGNMENT = True``). Requires a separate MFA install (a
  conda/mamba environment, per-language acoustic model + dictionary
  downloads) and the ``MFA_ENV_PATH`` environment variable pointing at it —
  see the error message :class:`SubtitleBuilder` raises if it's missing for
  the exact setup steps.
- **Time-based splitting** — proportionally distributes text over the audio's
  duration, sentence by sentence, honoring SSML ``<break>`` pauses. No
  external dependency, lower accuracy. Set ``USE_MFA_ALIGNMENT = False`` to
  use this instead.

Per-language configuration — which MFA model to align with, and the default
text styling (font, color, position, per-word highlight) to render segments
with — is a SubtitlePack. Every process loads a built-in set covering
``fr``/``en``/``es``/``ta`` at import, optionally extended or overridden via
``SUBTITLE_PACKS_PATH`` (see ``configuration/subtitle_packs.example.json`` and
:doc:`../installation`) — the same JSON-config pattern VoicePack uses for
TTS. :meth:`~tish_video_sdk.video_maker.VideoBuilder.with_tts_subtitles`/
:meth:`~tish_video_sdk.video_maker.VideoBuilder.with_subtitles` resolve a
style from the configured pack automatically when no explicit ``style=`` is
passed and the video's language is known, falling back to
:class:`~tish_video_sdk.video_maker.TextStyle`'s bare defaults otherwise.

.. automodule:: tish_video_sdk.subtitles
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Subtitles are rarely built directly — :class:`~tish_video_sdk.tts.TTSBuilder`
and :class:`~tish_video_sdk.video_maker.VideoBuilder` wire this up for you:

.. code-block:: python

   from tish_video_sdk.video_maker import VideoBuilder, TextStyle

   # Auto-generate subtitles from TTS narration via forced alignment,
   # styled from the configured SubtitlePack for "fr":
   builder = VideoBuilder.from_single_image_with_tts("scene.png", "Bonjour le monde", "fr")
   builder.with_tts_subtitles()
   builder.build()
   builder.save("output.mp4")

   # Or pass an explicit style to override the pack's:
   builder.with_tts_subtitles(style=TextStyle(font_color="white", bg_color="black"))

   # Or supply your own timed segments (no forced alignment needed):
   builder = VideoBuilder.from_single_image_with_audio("scene.png", "narration.wav")
   builder.with_subtitles([
       {"text": "Bonjour le monde", "start": 0.0, "end": 2.0},
   ])
   builder.build()
   builder.save("output.mp4")

To use :class:`SubtitleBuilder` directly (e.g. to inspect segments before
handing them to ``VideoBuilder``):

.. code-block:: python

   from tish_video_sdk.subtitles import SubtitleBuilder

   sb = SubtitleBuilder.from_audio_with_reference("narration.wav", "Bonjour le monde", "fr")
   sb.build()
   segments = sb.get_segments()
