"""publisher: video publishing to YouTube and Instagram (see CONTEXT.md).

Single entry point for publishing, in the same shape as every other feature
module here (media.py -> ImageFetcher/ImageSearcher/ImageGenerator, music.py
-> MusicManager, tts.py -> TTSBuilder). Publisher holds one PublisherPack per
configured platform (see internal/providers/publisher_packs.py) and offers a
generic publish_video(provider, ...) dispatcher plus named per-platform
wrapper methods, the same fetch()/fetch_pexels_images() split ImageFetcher
uses.

YouTube and Instagram share almost no publishing surface beyond "upload a
video" -- YouTube wants title/description/tags/scheduling/category, Instagram
wants only a caption, and YouTube alone has comment/thumbnail/analytics
endpoints with no Instagram equivalent. Publisher does not try to force a
common schema over that; publish_video()'s **kwargs pass straight through to
the target PublisherPack, and platform-specific extras (add_comment,
get_trending_videos, ...) are reached via the youtube/instagram pack
properties directly rather than being reinvented here.

Publisher also owns publish-date tracking (next_unpublished_date/
mark_published, see ADR 0012): a consuming app's daily-content pipeline asks
the same object it publishes through whether today's content was already
handled, rather than reaching into date_managment.PublishingTracker directly.
This is deliberately per-language, not per-platform -- it gates the whole
day's content pipeline (fetch, TTS, video build, then whichever platforms are
configured), not just one platform's upload step.
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .date_managment import PublishingTracker, get_offset_date
from .internal.providers.publisher_packs import (
    InstagramPublisherPack, PublisherPack, YouTubePublisherPack, load_publisher_pack_entries,
)

_ALL_PUBLISHER_PACK_ENTRIES: Optional[List[dict]] = None


def _get_all_publisher_pack_entries() -> List[dict]:
    global _ALL_PUBLISHER_PACK_ENTRIES
    if _ALL_PUBLISHER_PACK_ENTRIES is None:
        path = os.getenv("PUBLISHER_PACKS_PATH")
        if not path:
            raise RuntimeError(
                "PUBLISHER_PACKS_PATH is not set. Point it at a publisher_packs.json listing "
                "the publishing credentials available to this process -- see "
                "configuration/publisher_packs.example.json."
            )
        _ALL_PUBLISHER_PACK_ENTRIES = load_publisher_pack_entries(path)
    return _ALL_PUBLISHER_PACK_ENTRIES


class Publisher:
    """Holds one PublisherPack per configured platform. Only platforms with
    credentials passed to __init__ are available -- calling publish_video()
    (or a named wrapper) for a platform that wasn't configured raises."""

    def __init__(self, youtube_client_secret_filepath: str = "", youtube_token_filename: str = "token.pickle",
                 youtube_category_id: int = 22, youtube_language_code: str = "en",
                 instagram_username: str = "", instagram_password: str = "", instagram_session_file: str = "",
                 language_code: Optional[str] = None):
        self._packs: Dict[str, PublisherPack] = {}
        # Independent of youtube_language_code (which only feeds YouTube's
        # defaultLanguage snippet field): needed by next_unpublished_date()/
        # mark_published() below, which are per-language regardless of which
        # platform(s) this Publisher is configured for. Optional so existing
        # credential-free/test construction (Publisher(), Publisher(youtube_...))
        # keeps working -- only next_unpublished_date()/mark_published() require it.
        self.language_code = language_code

        if youtube_client_secret_filepath:
            self._packs["youtube"] = YouTubePublisherPack(
                youtube_client_secret_filepath, token_filename=youtube_token_filename,
                youtube_category_id=youtube_category_id, language_code=youtube_language_code,
            )
        if instagram_username and instagram_password:
            self._packs["instagram"] = InstagramPublisherPack(
                instagram_username, instagram_password, session_file=instagram_session_file or None,
            )

    @classmethod
    def for_language(cls, language_code: str) -> "Publisher":
        """Build a Publisher for language_code entirely from
        PUBLISHER_PACKS_PATH -- the credential-free entry point; no
        client_secret_filepath/token_filename/username/password is ever
        passed in by caller code. __init__ remains available for explicit/
        advanced construction (e.g. tests) that don't want a config file."""
        entries = [e for e in _get_all_publisher_pack_entries() if e.get("language_code") == language_code]
        if not entries:
            raise ValueError(f"No publisher_packs.json entries found for language '{language_code}'.")

        youtube_client_secret_filepath = ""
        youtube_token_filename = "token.pickle"
        youtube_category_id = 22
        instagram_username = ""
        instagram_password = ""
        instagram_session_file = ""
        for entry in entries:
            provider = entry.get("provider")
            if provider == "youtube":
                youtube_client_secret_filepath = entry["client_secret_filepath"]
                youtube_token_filename = entry.get("token_filename", "token.pickle")
                youtube_category_id = entry.get("category_id", 22)
            elif provider == "instagram":
                instagram_username = entry["username"]
                instagram_password = entry["password"]
                instagram_session_file = entry.get("session_file", "")
            else:
                raise ValueError(
                    f"Unknown publisher provider '{provider}' for language '{language_code}' "
                    "in publisher_packs.json. Known providers: youtube, instagram."
                )
        return cls(
            youtube_client_secret_filepath=youtube_client_secret_filepath,
            youtube_token_filename=youtube_token_filename,
            youtube_category_id=youtube_category_id,
            youtube_language_code=language_code,
            instagram_username=instagram_username,
            instagram_password=instagram_password,
            instagram_session_file=instagram_session_file,
            language_code=language_code,
        )

    @property
    def youtube(self) -> Optional[YouTubePublisherPack]:
        """The configured YouTubePublisherPack, or None if youtube_client_secret_filepath
        wasn't given -- use this for YouTube-only operations (add_comment,
        get_trending_videos, set_thumbnail, ...) that have no cross-platform equivalent."""
        return self._packs.get("youtube")

    @property
    def instagram(self) -> Optional[InstagramPublisherPack]:
        """The configured InstagramPublisherPack, or None if instagram_username/password weren't given."""
        return self._packs.get("instagram")

    def available(self) -> List[str]:
        """Platforms this Publisher was actually configured for."""
        return sorted(self._packs)

    def publish_video(self, provider: str, video_filepath: str, **kwargs) -> Optional[dict]:
        """Upload video_filepath to provider ("youtube" or "instagram"),
        passing kwargs straight through to that platform's publish_video
        (title/description/tags/... for YouTube, caption/thumbnail_filepath
        for Instagram). Returns the platform's response dict, or None on
        failure -- see PublisherPack.publish_video."""
        pack = self._packs.get(provider)
        if pack is None:
            configured = ", ".join(self.available()) or "(none configured)"
            raise ValueError(f"Publisher was not configured for provider '{provider}'. Configured: {configured}.")
        return pack.publish_video(video_filepath, **kwargs)

    def publish_to_youtube(self, video_filepath: str, title: str, description: str = "",
                            tags: Optional[List[str]] = None, **kwargs) -> Optional[dict]:
        return self.publish_video("youtube", video_filepath, title=title, description=description,
                                   tags=tags, **kwargs)

    def publish_to_instagram(self, video_filepath: str, caption: str = "", **kwargs) -> Optional[dict]:
        return self.publish_video("instagram", video_filepath, caption=caption, **kwargs)

    def _require_language_code(self) -> str:
        if not self.language_code:
            raise ValueError(
                "Publisher was constructed without language_code -- publish-date tracking "
                "(next_unpublished_date/mark_published) needs one. Publisher.for_language() "
                "always supplies it; pass language_code explicitly to __init__ for other "
                "construction paths that need tracking."
            )
        return self.language_code

    def next_unpublished_date(self, offset_days: int = 1, force: bool = False) -> datetime:
        """Next content date not yet marked done for this Publisher's
        language_code, skipping past dates already marked published by
        mark_published(). Mirrors what a daily-content pipeline needs before
        starting any work: pick a date, know whether it's safe to proceed.

        Picks a base date from the last published date (falling back to
        today when nothing's been published yet, or when the last published
        date implies a publish date that's already in the past): the intended
        next date is last_published_date + offset_days; if that's already
        behind today, base off today instead so a gap (e.g. the pipeline
        didn't run for a while) doesn't pile up a backlog of dates to
        process one by one.

        force=True skips the already-published check entirely (still using
        the same base-date logic above), returning the first candidate date
        outright -- for debug/manual runs that want to reprocess a date
        regardless of tracked state."""
        tracker = PublishingTracker(language_code=self._require_language_code())
        last_published_date = tracker.get_last_published_date()

        base_date = None
        if last_published_date:
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            intended_publish_date = last_published_date + timedelta(days=offset_days)
            if intended_publish_date < current_date:
                base_date = current_date - timedelta(days=offset_days)
            else:
                base_date = last_published_date

        candidate_offset = offset_days
        while True:
            candidate_date = get_offset_date(candidate_offset, base_date)
            if tracker.has_date_been_published(candidate_date) and not force:
                candidate_offset += 1
            else:
                return candidate_date

    def mark_published(self, date_obj: datetime) -> None:
        """Records date_obj as done for this Publisher's language_code,
        regardless of whether publishing itself succeeded -- "done" means
        the caller decided not to retry this date (e.g. to avoid an infinite
        retry loop on a broken day), not "successfully live". Publisher has
        no visibility into whether the surrounding pipeline (content fetch,
        video build, the actual publish_video() calls) succeeded; the caller
        decides that and calls this only when it wants the date treated as
        settled."""
        PublishingTracker(language_code=self._require_language_code()).mark_date_as_published(date_obj)
