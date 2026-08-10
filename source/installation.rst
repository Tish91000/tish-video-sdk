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

The ``subtitles`` module (subtitle generation via forced alignment, used by
``TTSBuilder.with_subtitles()``/``VideoBuilder.with_tts_subtitles()``) reads:

* ``SUBTITLE_PACKS_PATH`` — optional path to a JSON file (see
  ``subtitle_packs.example.json``) listing SubtitlePacks: per-language MFA acoustic model
  name and default text styling. Built-in defaults already cover ``fr``/``en``/``es``/``ta``;
  this only needs to add languages or override a pack's model/styling.
* ``MFA_ENV_PATH`` — root directory of the conda/mamba environment Montreal Forced Aligner
  was installed into. Only needed if a language's SubtitlePack has an ``mfa_model`` and
  ``USE_MFA_ALIGNMENT`` (the default) is left ``True``; set it to ``False`` to use
  time-based text splitting instead, which needs no external install.

The ``music`` module (background-music selection and mixing, via ``MusicManager``) reads:

* ``MUSIC_MOODS_PATH`` — optional path to a JSON file (see ``music_moods.example.json``)
  overriding the available moods and/or the Gemini mood-analysis prompt template.
  ``MusicManager`` ships generic, domain-neutral moods and a generic prompt by default;
  this only needs to override them to match a specific project's ``bgm_directory`` mood
  subfolders or content domain. Can also be passed directly as ``MusicManager(mood_packs_path=...)``.
* ``JAMENDO_CLIENT_ID`` — optional, free (register at `devportal.jamendo.com
  <https://devportal.jamendo.com/>`_). When set, a mood with no usable local track
  downloads one free, Creative-Commons-licensed track from Jamendo into ``bgm_directory``
  instead of falling back to ``default_music_file`` -- the downloaded file behaves like
  any other local track from then on. Can also be passed directly as
  ``MusicManager(jamendo_client_id=...)``.

The ``media`` module (``ImageFetcher``, ``ImageSearcher``, ``ImageGenerator``) reads:

* ``PEXELS_API_KEY`` — optional, free (register at `pexels.com/api
  <https://www.pexels.com/api/>`_). Used for the ``'pexels'`` image/video source; unset,
  that source is simply skipped (DuckDuckGo and Wikipedia need no credentials). Can also
  be passed directly as ``ImageFetcher(pexels_api_key=...)``/``ImageSearcher(pexels_api_key=...)``.
* ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) — Google GenAI API key, used for Imagen
  image generation and for condensing a search query in ``search_or_generate()``.
  ``generate_image()``/``search_or_generate()`` return ``[]`` without one rather than
  raising. Can also be passed directly as ``ImageGenerator(api_key=...)``.
