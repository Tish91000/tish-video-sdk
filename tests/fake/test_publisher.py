"""Mocked tests for publisher.py (Publisher) and
internal/providers/publisher_packs.py's PublisherPack subclasses -- no real
network/API calls, safe to run anywhere.

Run with: pytest tests/fake
"""
import json
import os
import pickle
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from tish_video_sdk import date_managment
from tish_video_sdk import publisher as publisher_module
from tish_video_sdk.publisher import Publisher
from tish_video_sdk.internal.providers import publisher_packs
from tish_video_sdk.internal.providers.publisher_packs import (
    InstagramPublisherPack, PublisherPack, YouTubePublisherPack, _build_google_service, _detect_credential_type,
)


class _FixedNow(datetime):
    """Stand-in for datetime with a frozen .now()/.today(), real everything
    else -- mirrors test_date_managment.py's _FixedDatetime. Patched into
    publisher.py's own `datetime` import for Publisher.next_unpublished_date's
    datetime.now() call, and/or date_managment's for get_offset_date's
    datetime.today() call, depending on which code path a test exercises."""

    _fixed_now = datetime(2024, 3, 10)

    @classmethod
    def now(cls, tz=None):
        return cls._fixed_now

    @classmethod
    def today(cls):
        return cls._fixed_now


# --------------------------------------------------------------------------
# PublisherPack registry
# --------------------------------------------------------------------------

class TestPublisherPackRegistry:
    def test_available_lists_every_registered_provider(self):
        assert set(PublisherPack.available()) == {"youtube", "instagram"}

    def test_get_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown publisher provider"):
            PublisherPack.get("tiktok")

    def test_get_instantiates_registered_provider(self):
        pack = PublisherPack.get("youtube", client_secret_filepath="secret.json")
        assert isinstance(pack, YouTubePublisherPack)


# --------------------------------------------------------------------------
# _detect_credential_type / _build_google_service
# --------------------------------------------------------------------------

class TestDetectCredentialType:
    def test_installed_app_credential(self, tmp_path):
        path = tmp_path / "secret.json"
        path.write_text(json.dumps({"installed": {"client_id": "x"}}))
        assert _detect_credential_type(str(path)) == "installed"

    def test_service_account_credential(self, tmp_path):
        path = tmp_path / "secret.json"
        path.write_text(json.dumps({"type": "service_account"}))
        assert _detect_credential_type(str(path)) == "service_account"

    def test_unrecognized_shape(self, tmp_path):
        path = tmp_path / "secret.json"
        path.write_text(json.dumps({"something_else": True}))
        assert _detect_credential_type(str(path)) == "unknown"

    def test_missing_file(self, tmp_path):
        assert _detect_credential_type(str(tmp_path / "missing.json")) == "unknown"


class _FakeCredentials:
    """A picklable stand-in for google.oauth2.credentials.Credentials --
    MagicMock instances can't be pickled, but _build_google_service caches
    real credentials to disk via pickle."""

    def __init__(self, valid=True, expired=False, refresh_token=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token

    def refresh(self, request):
        self.valid = True
        self.expired = False


class TestBuildGoogleService:
    def test_service_account_rejected(self, tmp_path):
        path = tmp_path / "secret.json"
        path.write_text(json.dumps({"type": "service_account"}))
        assert _build_google_service(str(path), "youtube", "v3", [], str(tmp_path / "token.pickle")) is None

    def test_unknown_credential_shape_rejected(self, tmp_path):
        path = tmp_path / "secret.json"
        path.write_text(json.dumps({"nope": True}))
        assert _build_google_service(str(path), "youtube", "v3", [], str(tmp_path / "token.pickle")) is None

    def test_valid_cached_token_skips_auth_flow(self, tmp_path):
        secret_path = tmp_path / "secret.json"
        secret_path.write_text(json.dumps({"installed": {"client_id": "x"}}))
        token_path = tmp_path / "token.pickle"
        with open(token_path, "wb") as f:
            pickle.dump(_FakeCredentials(), f)

        fake_service = object()
        with patch("googleapiclient.discovery.build", return_value=fake_service) as mock_build, \
             patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file") as mock_flow:
            result = _build_google_service(str(secret_path), "youtube", "v3", ["scope"], str(token_path))

        assert result is fake_service
        mock_flow.assert_not_called()
        mock_build.assert_called_once()

    def test_no_cached_token_runs_auth_flow_and_saves_it(self, tmp_path):
        secret_path = tmp_path / "secret.json"
        secret_path.write_text(json.dumps({"installed": {"client_id": "x"}}))
        token_path = tmp_path / "token.pickle"

        fake_credentials = MagicMock(valid=True)
        fake_flow = MagicMock()
        fake_flow.run_local_server.return_value = fake_credentials
        fake_service = object()

        with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file", return_value=fake_flow), \
             patch("googleapiclient.discovery.build", return_value=fake_service):
            result = _build_google_service(str(secret_path), "youtube", "v3", ["scope"], str(token_path))

        assert result is fake_service
        assert token_path.exists()

    def test_build_failure_returns_none(self, tmp_path):
        secret_path = tmp_path / "secret.json"
        secret_path.write_text(json.dumps({"installed": {"client_id": "x"}}))
        token_path = tmp_path / "token.pickle"
        with open(token_path, "wb") as f:
            pickle.dump(_FakeCredentials(), f)

        with patch("googleapiclient.discovery.build", side_effect=RuntimeError("boom")):
            result = _build_google_service(str(secret_path), "youtube", "v3", ["scope"], str(token_path))

        assert result is None


# --------------------------------------------------------------------------
# YouTubePublisherPack
# --------------------------------------------------------------------------

class TestYouTubePublisherPack:
    def test_token_filepath_placed_alongside_client_secret(self):
        pack = YouTubePublisherPack(os.path.join("creds", "secret.json"), token_filename="token.pickle")
        assert pack.token_filepath == os.path.join("creds", "token.pickle")

    def test_publish_video_missing_file_returns_none(self, tmp_path):
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        assert pack.publish_video(str(tmp_path / "missing.mp4"), title="t") is None

    def test_publish_video_service_init_failure_returns_none(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        with patch.object(publisher_packs, "_build_google_service", return_value=None):
            assert pack.publish_video(str(video), title="t") is None

    def test_publish_video_happy_path(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))

        fake_response = {"id": "abc123", "status": {"uploadStatus": "uploaded"}}
        fake_request = MagicMock()
        fake_request.next_chunk.return_value = (None, fake_response)
        fake_service = MagicMock()
        fake_service.videos.return_value.insert.return_value = fake_request

        with patch.object(publisher_packs, "_build_google_service", return_value=fake_service), \
             patch("googleapiclient.http.MediaFileUpload"):
            result = pack.publish_video(str(video), title="My Video", tags=["a", "b"])

        assert result == fake_response
        body = fake_service.videos.return_value.insert.call_args.kwargs["body"]
        assert body["snippet"]["title"] == "My Video"
        assert body["snippet"]["tags"] == ["a", "b"]

    def test_publish_video_reuses_existing_service(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        pack.service = MagicMock()
        pack.service.videos.return_value.insert.return_value.next_chunk.return_value = (None, {"id": "x"})

        with patch.object(publisher_packs, "_build_google_service") as mock_build, \
             patch("googleapiclient.http.MediaFileUpload"):
            pack.publish_video(str(video), title="t")

        mock_build.assert_not_called()

    def test_publish_video_content_date_and_hour_computes_schedule(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))

        fake_request = MagicMock()
        fake_request.next_chunk.return_value = (None, {"id": "abc123"})
        fake_service = MagicMock()
        fake_service.videos.return_value.insert.return_value = fake_request

        with patch.object(publisher_packs, "_build_google_service", return_value=fake_service), \
             patch("googleapiclient.http.MediaFileUpload"):
            pack.publish_video(
                str(video), title="t",
                content_date=datetime(2024, 3, 15), hour_of_day=21, utc_offset_hours=-2, publish_minute=5,
            )

        body = fake_service.videos.return_value.insert.call_args.kwargs["body"]
        assert body["status"]["publishAt"] == "2024-03-15T19:05:00.000Z"

    def test_publish_video_explicit_datetime_takes_precedence_over_content_date(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))

        fake_request = MagicMock()
        fake_request.next_chunk.return_value = (None, {"id": "x"})
        fake_service = MagicMock()
        fake_service.videos.return_value.insert.return_value = fake_request

        with patch.object(publisher_packs, "_build_google_service", return_value=fake_service), \
             patch("googleapiclient.http.MediaFileUpload"):
            pack.publish_video(
                str(video), title="t",
                publish_year=2025, publish_month=1, publish_day=1, publish_hour=0,
                content_date=datetime(2024, 3, 15), hour_of_day=21,
            )

        body = fake_service.videos.return_value.insert.call_args.kwargs["body"]
        assert body["status"]["publishAt"] == "2025-01-01T00:00:00.000Z"

    def test_publish_video_content_date_without_hour_of_day_leaves_unscheduled(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))

        fake_request = MagicMock()
        fake_request.next_chunk.return_value = (None, {"id": "x"})
        fake_service = MagicMock()
        fake_service.videos.return_value.insert.return_value = fake_request

        with patch.object(publisher_packs, "_build_google_service", return_value=fake_service), \
             patch("googleapiclient.http.MediaFileUpload"):
            pack.publish_video(str(video), title="t", content_date=datetime(2024, 3, 15))

        body = fake_service.videos.return_value.insert.call_args.kwargs["body"]
        assert "publishAt" not in body["status"]

    def test_add_comment(self, tmp_path):
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        pack.service = MagicMock()
        pack.service.commentThreads.return_value.insert.return_value.execute.return_value = {"id": "thread1"}
        assert pack.add_comment("vid1", "nice video") == "thread1"

    def test_add_comment_no_service_returns_none(self, tmp_path):
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        with patch.object(publisher_packs, "_build_google_service", return_value=None):
            assert pack.add_comment("vid1", "nice video") is None

    def test_get_trending_videos(self, tmp_path):
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        pack.service = MagicMock()
        pack.service.videos.return_value.list.return_value.execute.return_value = {
            "items": [{
                "id": "v1",
                "snippet": {"title": "T", "channelTitle": "C", "publishedAt": "2024-01-01T00:00:00Z", "description": "d" * 300},
                "statistics": {"viewCount": "100", "likeCount": "10"},
            }]
        }
        trends = pack.get_trending_videos()
        assert trends[0]["id"] == "v1"
        assert len(trends[0]["description"]) == 200

    def test_set_thumbnail_missing_file_returns_false(self, tmp_path):
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        assert pack.set_thumbnail("vid1", str(tmp_path / "missing.png")) is False

    def test_set_thumbnail_calls_thumbnails_api(self, tmp_path):
        thumb = tmp_path / "thumb.png"
        thumb.write_bytes(b"data")
        pack = YouTubePublisherPack(str(tmp_path / "secret.json"))
        pack.service = MagicMock()

        with patch("googleapiclient.http.MediaFileUpload"):
            result = pack.set_thumbnail("vid1", str(thumb))

        assert result is True
        pack.service.thumbnails.return_value.set.assert_called_once()
        assert pack.service.thumbnails.return_value.set.call_args.kwargs["videoId"] == "vid1"


# --------------------------------------------------------------------------
# InstagramPublisherPack
# --------------------------------------------------------------------------

class TestInstagramPublisherPack:
    def test_publish_video_missing_file_returns_none(self, tmp_path):
        with patch("instagrapi.Client"):
            pack = InstagramPublisherPack("user", "pass")
        assert pack.publish_video(str(tmp_path / "missing.mp4")) is None

    def test_publish_video_login_failure_returns_none(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        with patch("instagrapi.Client") as mock_client_cls:
            mock_client_cls.return_value.login.side_effect = RuntimeError("bad creds")
            pack = InstagramPublisherPack("user", "pass")
        assert pack.publish_video(str(video)) is None

    def test_publish_video_happy_path(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        fake_media = MagicMock(pk="123", code="abc")

        with patch("instagrapi.Client") as mock_client_cls:
            mock_client_cls.return_value.clip_upload.return_value = fake_media
            pack = InstagramPublisherPack("user", "pass")
            result = pack.publish_video(str(video), caption="hello")

        assert result == {"media_pk": "123", "url": "https://www.instagram.com/p/abc/"}
        mock_client_cls.return_value.clip_upload.assert_called_once_with(
            path=str(video), caption="hello", thumbnail=None
        )

    def test_login_only_happens_once(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        fake_media = MagicMock(pk="123", code="abc")

        with patch("instagrapi.Client") as mock_client_cls:
            mock_client_cls.return_value.clip_upload.return_value = fake_media
            pack = InstagramPublisherPack("user", "pass")
            pack.publish_video(str(video))
            pack.publish_video(str(video))

        mock_client_cls.return_value.login.assert_called_once()

    def test_session_file_saved_after_login(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"data")
        session_file = tmp_path / "session.json"
        fake_media = MagicMock(pk="123", code="abc")

        with patch("instagrapi.Client") as mock_client_cls:
            mock_client_cls.return_value.clip_upload.return_value = fake_media
            pack = InstagramPublisherPack("user", "pass", session_file=str(session_file))
            pack.publish_video(str(video))

        mock_client_cls.return_value.dump_settings.assert_called_once_with(str(session_file))


# --------------------------------------------------------------------------
# Publisher entry point
# --------------------------------------------------------------------------

class TestPublisher:
    def test_no_credentials_configures_nothing(self):
        publisher = Publisher()
        assert publisher.available() == []
        assert publisher.youtube is None
        assert publisher.instagram is None

    def test_youtube_credentials_configure_only_youtube(self):
        publisher = Publisher(youtube_client_secret_filepath="secret.json")
        assert publisher.available() == ["youtube"]
        assert isinstance(publisher.youtube, YouTubePublisherPack)
        assert publisher.instagram is None

    def test_instagram_credentials_configure_only_instagram(self):
        with patch("instagrapi.Client"):
            publisher = Publisher(instagram_username="u", instagram_password="p")
        assert publisher.available() == ["instagram"]
        assert isinstance(publisher.instagram, InstagramPublisherPack)
        assert publisher.youtube is None

    def test_both_configured(self):
        with patch("instagrapi.Client"):
            publisher = Publisher(
                youtube_client_secret_filepath="secret.json",
                instagram_username="u", instagram_password="p",
            )
        assert publisher.available() == ["instagram", "youtube"]

    def test_publish_video_unconfigured_provider_raises(self):
        publisher = Publisher(youtube_client_secret_filepath="secret.json")
        with pytest.raises(ValueError, match="not configured for provider 'instagram'"):
            publisher.publish_video("instagram", "video.mp4")

    def test_publish_video_dispatches_to_pack(self):
        publisher = Publisher(youtube_client_secret_filepath="secret.json")
        with patch.object(publisher.youtube, "publish_video", return_value={"id": "1"}) as mock_publish:
            result = publisher.publish_video("youtube", "video.mp4", title="t")

        assert result == {"id": "1"}
        mock_publish.assert_called_once_with("video.mp4", title="t")

    def test_publish_to_youtube_wrapper(self):
        publisher = Publisher(youtube_client_secret_filepath="secret.json")
        with patch.object(publisher.youtube, "publish_video", return_value={"id": "1"}) as mock_publish:
            publisher.publish_to_youtube("video.mp4", title="My Title", description="d", tags=["x"])

        mock_publish.assert_called_once_with("video.mp4", title="My Title", description="d", tags=["x"])

    def test_publish_to_instagram_wrapper(self):
        with patch("instagrapi.Client"):
            publisher = Publisher(instagram_username="u", instagram_password="p")
        with patch.object(publisher.instagram, "publish_video", return_value={"media_pk": "1"}) as mock_publish:
            publisher.publish_to_instagram("video.mp4", caption="hello")

        mock_publish.assert_called_once_with("video.mp4", caption="hello")


# --------------------------------------------------------------------------
# Publisher publish-date tracking (next_unpublished_date/mark_published)
# --------------------------------------------------------------------------

class TestPublisherPublishDateTracking:
    def test_language_code_none_by_default(self):
        publisher = Publisher(youtube_client_secret_filepath="secret.json")
        assert publisher.language_code is None

    def test_for_language_sets_language_code(self, tmp_path):
        packs_path = tmp_path / "publisher_packs.json"
        packs_path.write_text(json.dumps([
            {"language_code": "fr", "provider": "youtube", "client_secret_filepath": "secret.json"},
        ]))
        with patch.dict(os.environ, {"PUBLISHER_PACKS_PATH": str(packs_path)}):
            publisher_module._ALL_PUBLISHER_PACK_ENTRIES = None
            try:
                publisher = Publisher.for_language("fr")
            finally:
                publisher_module._ALL_PUBLISHER_PACK_ENTRIES = None
        assert publisher.language_code == "fr"

    def test_next_unpublished_date_without_language_code_raises(self):
        publisher = Publisher()
        with pytest.raises(ValueError, match="language_code"):
            publisher.next_unpublished_date()

    def test_mark_published_without_language_code_raises(self):
        publisher = Publisher()
        with pytest.raises(ValueError, match="language_code"):
            publisher.mark_published(datetime(2024, 3, 15))

    def test_next_unpublished_date_no_prior_publish_uses_today_plus_offset(self, monkeypatch, tmp_path):
        # base_date stays None on a fresh tracker, so get_offset_date falls
        # back to date_managment's own datetime.today() -- that's the one to freeze here.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(date_managment, "datetime", _FixedNow)
        publisher = Publisher(language_code="fr")
        assert publisher.next_unpublished_date(offset_days=1) == datetime(2024, 3, 11)

    def test_next_unpublished_date_and_mark_published_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(publisher_module, "datetime", _FixedNow)
        publisher = Publisher(language_code="fr")
        publisher.mark_published(datetime(2024, 3, 11))
        # last_published=3/11, offset_days=1 -> intended=3/12, "today" (frozen)=3/10,
        # 3/12 is not before 3/10 -> base_date=last_published_date=3/11 -> candidate=3/12.
        assert publisher.next_unpublished_date(offset_days=1) == datetime(2024, 3, 12)

    def test_next_unpublished_date_is_scoped_per_language(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(publisher_module, "datetime", _FixedNow)
        monkeypatch.setattr(date_managment, "datetime", _FixedNow)
        fr_publisher = Publisher(language_code="fr")
        en_publisher = Publisher(language_code="en")
        fr_publisher.mark_published(datetime(2024, 3, 20))
        # en's own tracker file is untouched by fr's mark_published -- falls
        # back to the no-prior-publish path (today + offset), not fr's state.
        assert en_publisher.next_unpublished_date(offset_days=1) == datetime(2024, 3, 11)

    def test_next_unpublished_date_skips_an_already_published_candidate(self, monkeypatch, tmp_path):
        # offset_days=0 deliberately: with the offset_days >= 1 every real
        # caller uses, base_date's own computation always already lands past
        # the tracked date on the first try (see next_unpublished_date's
        # docstring) -- 0 is the only way to actually force the
        # skip-and-advance loop to run, to test it directly.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(publisher_module, "datetime", _FixedNow)
        publisher = Publisher(language_code="fr")
        publisher.mark_published(datetime(2024, 3, 16))
        assert publisher.next_unpublished_date(offset_days=0) == datetime(2024, 3, 17)

    def test_next_unpublished_date_force_returns_already_published_candidate(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(publisher_module, "datetime", _FixedNow)
        publisher = Publisher(language_code="fr")
        publisher.mark_published(datetime(2024, 3, 16))
        assert publisher.next_unpublished_date(offset_days=0, force=True) == datetime(2024, 3, 16)
