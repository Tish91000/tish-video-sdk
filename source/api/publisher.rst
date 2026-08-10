Publisher
=========

.. currentmodule:: tish_video_sdk.publisher

Video publishing to YouTube and Instagram -- one entry point in the same
shape as every other feature module here (``music.py`` ->
:class:`~tish_video_sdk.music.MusicManager`, ``media.py`` -> ``ImageFetcher``/
``ImageSearcher``/``ImageGenerator``). :class:`Publisher` holds one
:class:`~tish_video_sdk.internal.providers.publisher_packs.PublisherPack` per
platform you give it credentials for (see
:mod:`tish_video_sdk.internal.providers.publisher_packs`).

:class:`Publisher`
-------------------

Only platforms configured via ``__init__`` (a non-empty
``youtube_client_secret_filepath``, or both ``instagram_username`` and
``instagram_password``) are available -- :meth:`Publisher.available` lists
them, and calling :meth:`Publisher.publish_video` (or a named wrapper) for
an unconfigured platform raises.

YouTube and Instagram share almost no publishing surface beyond "upload a
video" -- YouTube wants title/description/tags/scheduling/category,
Instagram wants only a caption. :meth:`Publisher.publish_video` does not
force a common schema over that: its ``**kwargs`` pass straight through to
the target platform's pack. :meth:`Publisher.publish_to_youtube` and
:meth:`Publisher.publish_to_instagram` are thin named wrappers over the same
call, for IDE-friendliness.

YouTube alone has operations with no Instagram equivalent -- comments,
thumbnails, channel video listing, trending videos. Those live on
:attr:`Publisher.youtube` (a
:class:`~tish_video_sdk.internal.providers.publisher_packs.YouTubePublisherPack`,
or ``None`` if YouTube wasn't configured) rather than being reinvented on
:class:`Publisher` itself.

Authentication differs by platform: YouTube uses an OAuth "installed app"
credential file (``youtube_client_secret_filepath`` -- see
:doc:`../installation`) with a one-time browser consent flow, cached
afterward as a token file next to it. Instagram uses a username/password
login (``instagram_username``/``instagram_password``), optionally caching
the resulting session to ``instagram_session_file`` so later runs can skip
the login challenge.

.. automodule:: tish_video_sdk.publisher
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tish_video_sdk.publisher import Publisher

   publisher = Publisher(
       youtube_client_secret_filepath="client_secret.json",
       instagram_username="...", instagram_password="...",
   )

   publisher.publish_to_youtube(
       "video.mp4", title="My Video", description="...", tags=["tag1", "tag2"],
   )
   publisher.publish_to_instagram("video.mp4", caption="My caption")

   # Platform-specific operations
   trends = publisher.youtube.get_trending_videos(region_code="US")
