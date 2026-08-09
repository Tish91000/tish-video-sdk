Music
=====

.. currentmodule:: tish_video_sdk.music

Background-music selection and mixing for narrated audio. :class:`MusicManager`
picks a track by mood — an explicit mood name, a Gemini-driven text mood
analysis, or a direct file path — from a directory of mood subfolders
(``bgm_directory``), falling back to the longest available track for that mood,
or ``default_music_file``, when nothing matches the requested duration. Mood
analysis is skipped (falling straight to ``default_mood``) when no
``gemini_api_key`` is provided.

The set of moods and the mood-analysis prompt are generic, domain-neutral
defaults, not fixed SDK behavior — override either via a JSON file at
``mood_packs_path``/``MUSIC_MOODS_PATH`` (see ``configuration/music_moods.example.json`` and
:doc:`../installation`) to match a project's own ``bgm_directory`` mood
subfolders or content domain.

When a mood has no usable local track and ``jamendo_client_id``/
``JAMENDO_CLIENT_ID`` is configured (free, see :doc:`../installation`),
:class:`MusicManager` downloads one free, Creative-Commons-licensed track
from `Jamendo <https://www.jamendo.com/>`_ into ``bgm_directory`` instead of
falling back to ``default_music_file`` — from then on it's just a local file,
picked up by the normal scan like any other track. Unset, this is simply off.
Search results are restricted to instrumental tracks (no competing vocals
under narration) and, by default, commercial-use-allowed licenses.
The Jamendo search always includes the mood itself as a tag, plus extra
context when available — ``get_music_path``'s ``reference_text`` as-is when
short, or condensed to one well-chosen tag via Gemini when it's long prose
(occasionally two, only when it genuinely helps — combining too many tags
over-constrains the match and can return nothing) — so different input text
searches differently instead of always resolving to the same single tag.
Whatever gets found is filed under the mood's own local folder regardless of
what query found it. A downloaded track's credit/license info is recorded in
a ``<file>.cache.json`` sidecar next to it (shared with the loudness-analysis
cache).

Jamendo isn't only an automatic fallback: :meth:`MusicManager.fetch_from_jamendo`
exposes it as an explicit action too, searching and downloading regardless of
what's already local — useful for a mostly-single-mood content series (a
daily "meditative" reading, say) that wants its local pool to keep growing
rather than settle on whichever single track the first automatic fallback
happened to find. Once a mood has more than one local track, ``select_audio_music``
picks randomly among the ones that satisfy the requested duration, so repeated
calls for the same mood don't always return the same track.

.. automodule:: tish_video_sdk.music
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.music import MusicManager

   manager = MusicManager(bgm_directory="./music/bgm", gemini_api_key="...")

   # Mix a specific mood under a narration file:
   manager.add_background_music(
       "narration.wav", "narration_with_music.wav", music_input="calm"
   )

   # Or let Gemini pick the mood from the narrated text:
   manager.add_background_music(
       "narration.wav", "narration_with_music.wav",
       reference_text_for_mood="The sun broke through the clouds as the crowd cheered.",
   )

   # Grow a mood's local Jamendo pool explicitly (e.g. a scheduled daily
   # task for a mostly-single-mood content series), independent of playback:
   manager.fetch_from_jamendo("meditative", search_query="today's reading text...")
