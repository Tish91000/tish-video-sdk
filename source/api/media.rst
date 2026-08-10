Media
=====

.. currentmodule:: tish_video_sdk.media

Image/video discovery, local caching, and AI image generation -- one entry
point in the same shape as every other feature module here (``music.py`` ->
:class:`~tish_video_sdk.music.MusicManager`, ``tts.py`` -> ``TTSBuilder``,
``video_maker.py`` -> ``VideoBuilder``). Named "media", not "image", since
Pexels search covers both photos and videos.

:class:`ImageFetcher`
----------------------

Low-level image/video lookup across several web sources -- DuckDuckGo image
search, `Pexels <https://www.pexels.com/api/>`_ (photos and videos), and
Wikipedia article images (ranked by relevance -- see
:mod:`tish_video_sdk.internal.providers.media_packs`). Each ``fetch_*``
method (and the generic ``fetch(provider, query, count)``) returns URLs
plus whatever attribution that source provides; nothing is downloaded or
cached here.

Pexels needs a free API key (``pexels_api_key``/``PEXELS_API_KEY``, see
:doc:`../installation`); without one, Pexels-backed calls simply return
``[]``. DuckDuckGo and Wikipedia need no credentials.

:class:`ImageSearcher`
------------------------

Orchestrates image discovery: combines the sources above behind one
``search()`` call, downloading and caching results locally under
``cache_dir``. Source order/selection is generic and domain-neutral -- a
query is searched as-is against whichever sources are requested (all three
by default); a source with no results, or 'pexels' with no configured API
key, is simply skipped rather than treated as an error.

Each downloaded file gets a ``<file>.meta.json`` sidecar recording its
source, the query that found it, and whatever attribution that source
provided -- several sources' licenses ask for credit where practical, so it
travels with the file rather than only appearing in a log line. Read it
back via :meth:`ImageSearcher.get_metadata`.

:class:`ImageGenerator`
-------------------------

Generates images from text via Google's
`Imagen <https://ai.google.dev/gemini-api/docs/imagen>`_ API
(``api_key``/``GEMINI_API_KEY``/``GOOGLE_API_KEY``). Its prompt template is
plain content, not core SDK behavior -- generic and domain-neutral by
default, overridable via ``prompt_template`` (the same override-a-template
approach :class:`~tish_video_sdk.music.MusicManager`'s mood-analysis prompt
uses).

:meth:`ImageGenerator.search_or_generate` tries a web search first (via a
composed :class:`ImageSearcher` -- pass one explicitly to share a
cache_dir/API key with the rest of a project, or let one be created
automatically), and only falls back to generation when search doesn't find
enough -- generation costs real API usage, search doesn't. The search query
itself is condensed from ``text`` via Gemini when an API key is configured,
falling back to ``text`` itself otherwise.

:class:`ImageConverter`
-------------------------

Independent of image generation -- general-purpose aspect-ratio conversion,
transparent-background cutouts, resizing, and foreground/background
compositing.

.. automodule:: tish_video_sdk.media
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.media import ImageSearcher, ImageGenerator, ImageConverter

   searcher = ImageSearcher(cache_dir="./image_cache", pexels_api_key="...")
   paths = searcher.search("mountain sunrise", max_results=5)
   print(searcher.get_metadata(paths[0]))  # {'source': 'pexels', 'photographer': ..., ...}

   generator = ImageGenerator(output_directory="./images")
   paths = generator.search_or_generate("a quiet mountain sunrise", output_name="sunrise_scene")

   converter = ImageConverter()
   converter.convert_9_to_16_aspect_ratio("portrait.png", "landscape.png")
