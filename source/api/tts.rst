TTS
===

.. currentmodule:: tish_video_sdk.tts

Text-to-speech synthesis via an ordered chain of VoicePacks (Gemini, Google
Cloud, ...), with automatic SSML generation on providers that support it.
Every VoicePack this process might use is loaded once, at import, from the
JSON file at ``TTS_VOICE_PACKS_PATH`` (see ``.env.template``,
``voice_packs.example.json``, and :doc:`../installation`) — one process can
build :class:`TTSBuilder` instances for multiple languages, each filtering
that pool down to its own chain.

.. automodule:: tish_video_sdk.tts
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.tts import TTSBuilder

   # Uses TTS_LANGUAGE_CODE as the default language, tries every configured
   # provider in order until one succeeds:
   tts = TTSBuilder.from_text("Bonjour le monde", "fr").build()
   tts.save("narration.wav")

   # Or pin to exactly one provider (disables fallback):
   tts = TTSBuilder.from_text("Bonjour le monde", "fr", provider="google_cloud").build()
