Installation
============

Requirements
------------

* Python 3.8+
* A Google Cloud service account with the Text-to-Speech API enabled (for the ``tts`` module)

From source
-----------

.. code-block:: bash

   git clone <repo-url>
   cd tish-video-sdk
   pip install -e ".[dev]"

Configuration
-------------

The ``tts`` module reads its configuration from two places (see ``.env.template``):

* ``TTS_LANGUAGE_CODE`` — an optional *default* language for ``TTSBuilder()`` calls that
  don't pass ``language_code`` explicitly. One process can build ``TTSBuilder`` instances
  for multiple languages; this isn't a hard per-process constraint.
* ``TTS_VOICE_PACKS_PATH`` — path to a JSON file (see ``voice_packs.example.json``) listing
  every VoicePack (provider x language combination, e.g. Gemini or Google Cloud TTS) this
  process might use, including their credentials. Loaded once at import time.

``TTSBuilder(language_code, provider=None)`` filters that list down to the chain for one
call: omit ``provider`` to try every VoicePack configured for that language in order,
falling back to the next on failure, or pass one (e.g. ``provider="google_cloud"``) to pin
to it and disable fallback. ``TTSBuilder.available_providers(language_code)`` lists what's
configured for a language.
