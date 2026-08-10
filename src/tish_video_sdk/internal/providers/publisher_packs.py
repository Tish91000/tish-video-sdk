"""publisher_packs: one publishing-platform's adapter (see CONTEXT.md), used
by publisher.py. Same shape as VoicePack/MediaPack -- an abstract base plus
concrete subclasses that auto-register themselves via __init_subclass__,
keyed by their `provider` class attribute. Unlike VoicePack there's no
fallback chain here -- publisher.py's Publisher calls a specific platform by
name, never "try YouTube then fall back to Instagram" (they're different
platforms, not alternative sources of the same content) -- so PublisherPack
only needs the one shared shape (publish_video()), not chain-selection
helpers.

Currently two providers: YouTube (Data API v3, OAuth "installed app" flow)
and Instagram (Reels, via instagrapi).

YouTubePublisherPack owns its OAuth token handling directly (loading/
refreshing/saving a cached user token, running the browser consent flow when
needed) rather than going through a separate generic Google-API-auth module
-- the only consumer of that logic is this one pack, so there's nothing to
share it with.
"""
import datetime
import os
import pickle
import random
import socket
import ssl
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

import httplib2


class PublisherPack(ABC):
    """Abstract base for a single publishing platform's adapter."""

    provider: str = ""  # set by each concrete subclass; doubles as its registry key
    _registry: Dict[str, Type["PublisherPack"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.provider:
            PublisherPack._registry[cls.provider] = cls

    @abstractmethod
    def publish_video(self, video_filepath: str, **kwargs) -> Optional[dict]:
        """Upload video_filepath to this platform. Returns the platform's
        response as a dict on success, or None on failure (missing/invalid
        credentials, network error, upload rejected) -- callers should treat
        a None return as "did not publish", not as a raised error."""

    @classmethod
    def get(cls, provider: str, **init_kwargs) -> "PublisherPack":
        """Instantiate the PublisherPack registered for provider (e.g.
        "youtube"), passing init_kwargs to its constructor."""
        pack_cls = cls._registry.get(provider)
        if pack_cls is None:
            known = ", ".join(sorted(cls._registry)) or "(none registered)"
            raise ValueError(f"Unknown publisher provider '{provider}'. Known providers: {known}.")
        return pack_cls(**init_kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Every registered provider name, sorted."""
        return sorted(cls._registry)


def _detect_credential_type(client_secret_filepath: str) -> str:
    """'installed' (OAuth desktop-app credentials), 'service_account', or
    'unknown', based on the JSON's top-level shape."""
    import json

    try:
        with open(client_secret_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading credential file {client_secret_filepath}: {e}")
        return "unknown"

    if "installed" in data or "web" in data:
        return "installed"
    if data.get("type") == "service_account":
        return "service_account"
    return "unknown"


def _build_google_service(client_secret_filepath: str, service_name: str, service_version: str,
                           api_scopes: List[str], token_filepath: str) -> Optional[object]:
    """Authenticate via the OAuth "installed app" flow and return a built
    googleapiclient service object, or None on failure. Caches the user
    token as a pickle file at token_filepath (refreshing it if expired)
    so a re-run doesn't need a fresh browser sign-in every time.

    Rejects service-account credentials outright: the YouTube Data API
    requires a real user's OAuth consent to act as a channel owner, which a
    service account can't provide -- failing fast here avoids a confusing
    permission error later.
    """
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    cred_type = _detect_credential_type(client_secret_filepath)
    if cred_type == "service_account":
        print(
            f"Error: '{client_secret_filepath}' is a service-account credential. "
            "The YouTube Data API needs an OAuth 'installed app' (Desktop) credential instead -- "
            "create one in the Google Cloud Console and sign in as the channel owner."
        )
        return None
    if cred_type == "unknown":
        print(f"Error: '{client_secret_filepath}' is not a recognized OAuth 'installed app' credential file.")
        return None

    credentials = None
    if os.path.exists(token_filepath):
        try:
            with open(token_filepath, "rb") as token_file:
                credentials = pickle.load(token_file)
        except Exception as e:
            print(f"Error loading cached token from {token_filepath}: {e}")
            credentials = None

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as e:
                print(f"Token refresh failed ({e}); starting a new authentication flow.")
                credentials = None

        if not credentials:
            try:
                token_dir = os.path.dirname(token_filepath)
                if token_dir:
                    os.makedirs(token_dir, exist_ok=True)
                flow = InstalledAppFlow.from_client_secrets_file(client_secret_filepath, api_scopes)
                credentials = flow.run_local_server(port=0)
            except Exception as e:
                print(f"Error during OAuth authentication flow: {e}")
                return None

        try:
            token_dir = os.path.dirname(token_filepath)
            if token_dir:
                os.makedirs(token_dir, exist_ok=True)
            with open(token_filepath, "wb") as token_file:
                pickle.dump(credentials, token_file)
        except Exception as e:
            print(f"Warning: could not cache token to {token_filepath}: {e}")

    try:
        return build(service_name, service_version, credentials=credentials)
    except Exception as e:
        print(f"Unable to build {service_name} v{service_version} service: {e}")
        return None


class YouTubePublisherPack(PublisherPack):
    """Publishes to YouTube via the YouTube Data API v3 (OAuth 'installed
    app' credentials -- see _build_google_service). Also exposes
    YouTube-specific operations beyond publish_video (comments, listing,
    trends, thumbnails) that have no Instagram equivalent, so they live here
    rather than on the shared PublisherPack interface."""

    provider = "youtube"

    def __init__(self, client_secret_filepath: str, token_filename: str = "token.pickle",
                 youtube_category_id: int = 22, language_code: str = "en"):
        self.client_secret_filepath = client_secret_filepath
        self.youtube_category_id = youtube_category_id
        self.language_code = language_code
        self.scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
        ]
        self.service = None

        secret_dir = os.path.dirname(client_secret_filepath)
        self.token_filepath = os.path.join(secret_dir, token_filename) if secret_dir else token_filename

    def _ensure_service(self) -> bool:
        if self.service is not None:
            return True
        self.service = _build_google_service(
            self.client_secret_filepath, "youtube", "v3", self.scopes, self.token_filepath
        )
        return self.service is not None

    @staticmethod
    def _create_publish_datetime(year: int, month: int, day: int, hour: int, minute: int = 0) -> Optional[str]:
        try:
            dt = datetime.datetime(year, month, day, hour, minute, 0, tzinfo=datetime.timezone.utc)
            return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except ValueError as e:
            print(f"Error creating publish datetime: {e}")
            return None

    def _build_request_body(self, title: str, description: str, tags: List[str], publish_at_iso: Optional[str],
                             privacy_status: str, notify_subscribers: bool, made_for_kids: bool,
                             category_id: Optional[int], language_code: Optional[str]) -> dict:
        status = {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": made_for_kids}
        if publish_at_iso:
            status["publishAt"] = publish_at_iso
        return {
            "snippet": {
                "categoryId": category_id or self.youtube_category_id,
                "title": title,
                "description": description,
                "tags": tags,
                "defaultLanguage": language_code or self.language_code,
            },
            "status": status,
            "notifySubscribers": notify_subscribers,
        }

    def publish_video(self, video_filepath: str, title: str = "", description: str = "",
                       tags: Optional[List[str]] = None, publish_year: Optional[int] = None,
                       publish_month: Optional[int] = None, publish_day: Optional[int] = None,
                       publish_hour: Optional[int] = None, publish_minute: int = 0,
                       privacy_status: str = "public", notify_subscribers: bool = True,
                       made_for_kids: bool = False, category_id: Optional[int] = None,
                       language_code: Optional[str] = None, **kwargs) -> Optional[dict]:
        """Resumable upload via the YouTube Data API. Scheduling (publish_at)
        only applies if publish_year/month/day/hour are all given; otherwise
        the video publishes per privacy_status immediately."""
        from googleapiclient.http import MediaFileUpload

        if not os.path.exists(video_filepath):
            print(f"Error: video file '{video_filepath}' not found.")
            return None
        if not self._ensure_service():
            return None

        publish_at_iso = None
        if None not in (publish_year, publish_month, publish_day, publish_hour):
            publish_at_iso = self._create_publish_datetime(publish_year, publish_month, publish_day,
                                                             publish_hour, publish_minute)

        request_body = self._build_request_body(
            title, description, tags or [], publish_at_iso, privacy_status,
            notify_subscribers, made_for_kids, category_id, language_code,
        )
        media_file = MediaFileUpload(video_filepath, mimetype="video/*", resumable=True, chunksize=4 * 1024 * 1024)

        try:
            request = self.service.videos().insert(part="snippet,status", body=request_body, media_body=media_file)
            response = None
            retries = 0
            max_retries = 10
            while response is None:
                try:
                    status, response = request.next_chunk()
                    retries = 0
                except (httplib2.HttpLib2Error, ssl.SSLError, socket.error, OSError) as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Max retries exceeded. Upload failed: {e}")
                        return None
                    sleep_time = random.uniform(1, 5) * retries
                    print(f"Network error: {e}. Retrying ({retries}/{max_retries}) in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
            return response
        except Exception as e:
            print(f"Error during YouTube video upload: {e}")
            return None

    def add_comment(self, video_id: str, comment_text: str) -> Optional[str]:
        if not self._ensure_service():
            return None
        try:
            response = self.service.commentThreads().insert(
                part="snippet",
                body={"snippet": {"videoId": video_id, "topLevelComment": {"snippet": {"textOriginal": comment_text}}}},
            ).execute()
            return response.get("id")
        except Exception as e:
            print(f"Error adding comment to video {video_id}: {e}")
            return None

    def pin_comment(self, comment_thread_id: str) -> bool:
        if not self._ensure_service():
            return False
        try:
            self.service.commentThreads().update(
                part="snippet", body={"id": comment_thread_id, "snippet": {"isPinned": True}}
            ).execute()
            return True
        except Exception as e:
            print(f"Error pinning comment thread {comment_thread_id}: {e}")
            return False

    def search_public_videos(self, query: str, max_results: int = 5) -> List[dict]:
        if not self._ensure_service():
            return []
        try:
            response = self.service.search().list(
                q=query, type="video", part="id,snippet", maxResults=max_results, videoDuration="short"
            ).execute()
            return [{"id": item["id"]["videoId"], "title": item["snippet"]["title"]} for item in response.get("items", [])]
        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def get_videos(self, limit: Optional[int] = None, days: Optional[int] = None) -> List[dict]:
        if not self._ensure_service():
            return []

        videos: List[dict] = []
        try:
            channels_response = self.service.channels().list(mine=True, part="contentDetails").execute()
            if not channels_response.get("items"):
                print("No channel found for the authenticated user.")
                return []
            uploads_playlist_id = channels_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            published_after = None
            if days:
                published_after = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

            request = self.service.playlistItems().list(playlistId=uploads_playlist_id, part="snippet,contentDetails", maxResults=50)
            while request:
                response = request.execute()
                stop = False
                for item in response.get("items", []):
                    published_at = datetime.datetime.fromisoformat(item["snippet"]["publishedAt"].replace("Z", "+00:00"))
                    if published_after and published_at < published_after:
                        stop = True
                        break
                    videos.append({
                        "id": item["contentDetails"]["videoId"],
                        "title": item["snippet"]["title"],
                        "publishedAt": published_at,
                        "description": item["snippet"]["description"],
                    })
                    if limit and len(videos) >= limit:
                        stop = True
                        break
                if stop:
                    break
                request = self.service.playlistItems().list_next(request, response)

            if not videos:
                return []

            video_ids = [v["id"] for v in videos]
            enriched = []
            for i in range(0, len(video_ids), 50):
                batch = video_ids[i:i + 50]
                stats_response = self.service.videos().list(part="statistics,contentDetails,status", id=",".join(batch)).execute()
                for details in stats_response.get("items", []):
                    match = next((v for v in videos if v["id"] == details["id"]), None)
                    if match:
                        match["statistics"] = details.get("statistics", {})
                        match["status"] = details.get("status", {})
                        match["duration"] = details.get("contentDetails", {}).get("duration")
                        enriched.append(match)

            enriched.sort(key=lambda v: v["publishedAt"], reverse=True)
            return enriched
        except Exception as e:
            print(f"Error retrieving videos: {e}")
            return []

    def get_trending_videos(self, region_code: str = "US", max_results: int = 10) -> List[dict]:
        if not self._ensure_service():
            return []
        try:
            response = self.service.videos().list(
                part="snippet,statistics,contentDetails", chart="mostPopular",
                regionCode=region_code, maxResults=max_results,
            ).execute()
            trends = []
            for item in response.get("items", []):
                snippet, stats = item["snippet"], item.get("statistics", {})
                trends.append({
                    "id": item["id"],
                    "title": snippet["title"],
                    "channel": snippet["channelTitle"],
                    "tags": snippet.get("tags", []),
                    "viewCount": stats.get("viewCount", 0),
                    "likeCount": stats.get("likeCount", 0),
                    "publishedAt": snippet["publishedAt"],
                    "description": snippet["description"][:200],
                })
            return trends
        except Exception as e:
            print(f"Error fetching trends: {e}")
            return []

    def update_video_details(self, video_id: str, title: Optional[str] = None, description: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> bool:
        if not self._ensure_service():
            return False
        if not title and not description and tags is None:
            return True
        try:
            video_response = self.service.videos().list(part="snippet", id=video_id).execute()
            if not video_response.get("items"):
                print(f"Video {video_id} not found.")
                return False
            snippet = video_response["items"][0]["snippet"]
            if title:
                snippet["title"] = title
            if description:
                snippet["description"] = description
            if tags is not None:
                snippet["tags"] = tags

            self.service.videos().update(
                part="snippet",
                body={"id": video_id, "snippet": {
                    "title": snippet["title"], "description": snippet["description"],
                    "categoryId": snippet["categoryId"], "tags": snippet.get("tags", []),
                }},
            ).execute()
            return True
        except Exception as e:
            print(f"Error updating video {video_id}: {e}")
            return False

    def set_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        from googleapiclient.http import MediaFileUpload

        if not os.path.exists(thumbnail_path):
            print(f"Thumbnail file not found: {thumbnail_path}")
            return False
        if not self._ensure_service():
            return False
        try:
            self.service.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
            return True
        except Exception as e:
            print(f"Error uploading thumbnail: {e}")
            return False


class InstagramPublisherPack(PublisherPack):
    """Publishes Reels to Instagram via instagrapi. Logs in lazily on first
    publish_video() call (mirroring YouTubePublisherPack's lazy service
    init), caching the session to session_file if given so later runs skip
    the login challenge when possible."""

    provider = "instagram"

    def __init__(self, username: str, password: str, session_file: Optional[str] = None):
        from instagrapi import Client

        self.username = username
        self.password = password
        self.session_file = session_file
        self._client = Client()
        self._logged_in = False

    def _ensure_login(self) -> bool:
        if self._logged_in:
            return True
        if self.session_file and os.path.exists(self.session_file):
            try:
                self._client.load_settings(self.session_file)
            except Exception as e:
                print(f"Error loading Instagram session: {e}")
        try:
            self._client.login(self.username, self.password)
        except Exception as e:
            print(f"Error logging in to Instagram: {e}")
            return False
        if self.session_file:
            try:
                self._client.dump_settings(self.session_file)
            except Exception as e:
                print(f"Error saving Instagram session: {e}")
        self._logged_in = True
        return True

    def publish_video(self, video_filepath: str, caption: str = "", thumbnail_filepath: Optional[str] = None,
                       **kwargs) -> Optional[dict]:
        if not os.path.exists(video_filepath):
            print(f"Error: video file '{video_filepath}' not found.")
            return None
        if not self._ensure_login():
            return None
        try:
            media = self._client.clip_upload(path=video_filepath, caption=caption, thumbnail=thumbnail_filepath)
        except Exception as e:
            print(f"Error during Instagram upload: {e}")
            return None
        if not media or not media.pk:
            print("Error: Reel upload failed (no media object returned).")
            return None
        return {"media_pk": media.pk, "url": f"https://www.instagram.com/p/{media.code}/"}


def load_publisher_pack_entries(publisher_packs_path: str) -> List[dict]:
    """Load and parse publisher_packs_path, returning every raw entry it
    describes (all languages, all providers) -- see
    configuration/publisher_packs.example.json for the expected shape.
    publisher.py's Publisher.for_language() filters this down to one
    language's entries and builds the matching PublisherPacks from them."""
    import json

    if not os.path.exists(publisher_packs_path):
        raise RuntimeError(
            f"PUBLISHER_PACKS_PATH points at '{publisher_packs_path}', which does not exist. "
            "See configuration/publisher_packs.example.json for the expected shape."
        )

    with open(publisher_packs_path, "r", encoding="utf-8") as f:
        return json.load(f)
