"""Mocked tests for media.py (ImageFetcher, ImageSearcher, ImageGenerator,
ImageConverter) and internal/providers/media_packs.py's MediaPack
subclasses -- no real network/API calls, safe to run anywhere.

Run with: pytest tests/fake
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from tish_video_sdk.media import ImageConverter, ImageFetcher, ImageGenerator, ImageSearcher
from tish_video_sdk.internal.providers.media_packs import (
    DuckDuckGoMediaPack, MediaPack, MediaResult, PexelsMediaPack, WikipediaMediaPack,
)
from tish_video_sdk.internal.providers.reasoning_packs import ReasoningAPIError


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    # tts.py's module-level load_dotenv() can leak real PEXELS_API_KEY/
    # GEMINI_API_KEY/GOOGLE_API_KEY from .env into os.environ once any other
    # test module in this session imports it -- without this, bare
    # ImageFetcher()/ImageGenerator() calls below would risk making real,
    # unmocked network calls instead of the no-ops their tests expect.
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def _fake_urlopen_response(payload: dict):
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = json.dumps(payload).encode()
    return response


# --------------------------------------------------------------------------
# MediaPack registry + subclasses (internal/providers/media_packs.py)
# --------------------------------------------------------------------------

class TestMediaPackRegistry:
    def test_available_lists_every_registered_provider(self):
        assert set(MediaPack.available()) == {"pexels", "duckduckgo", "wikipedia"}

    def test_get_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown media provider"):
            MediaPack.get("flickr")

    def test_get_instantiates_registered_provider(self):
        pack = MediaPack.get("pexels", api_key="fake-key")
        assert isinstance(pack, PexelsMediaPack)
        assert pack.api_key == "fake-key"


class TestPexelsMediaPack:
    def test_no_api_key_returns_empty_list(self):
        assert PexelsMediaPack(api_key="").search("mountains") == []

    def test_env_var_used_when_no_explicit_key(self, monkeypatch):
        monkeypatch.setenv("PEXELS_API_KEY", "env-key")
        assert PexelsMediaPack().api_key == "env-key"

    def test_search_returns_image_results_with_attribution(self):
        pack = PexelsMediaPack(api_key="fake-key")
        payload = {"photos": [{
            "src": {"large2x": "https://images.pexels.com/1.jpg"},
            "photographer": "Jane Doe", "photographer_url": "https://pexels.com/@jane",
            "url": "https://pexels.com/photo/1",
        }]}

        with patch("tish_video_sdk.internal.providers.media_packs.urllib.request.urlopen",
                    return_value=_fake_urlopen_response(payload)):
            results = pack.search("mountains", count=5)

        assert results == [MediaResult(
            url="https://images.pexels.com/1.jpg",
            attribution={"photographer": "Jane Doe", "photographer_url": "https://pexels.com/@jane",
                         "page_url": "https://pexels.com/photo/1"},
        )]

    def test_search_videos_returns_mp4_results(self):
        pack = PexelsMediaPack(api_key="fake-key")
        payload = {"videos": [{
            "user": {"name": "John Smith", "url": "https://pexels.com/@john"},
            "url": "https://pexels.com/video/2",
            "video_files": [
                {"file_type": "video/mp4", "link": "https://videos.pexels.com/small.mp4", "width": 640},
                {"file_type": "video/mp4", "link": "https://videos.pexels.com/large.mp4", "width": 1920},
            ],
        }]}

        with patch("tish_video_sdk.internal.providers.media_packs.urllib.request.urlopen",
                    return_value=_fake_urlopen_response(payload)):
            results = pack.search_videos("ocean waves")

        assert results == [MediaResult(
            url="https://videos.pexels.com/large.mp4",  # highest-width mp4 picked
            attribution={"user": "John Smith", "user_url": "https://pexels.com/@john",
                         "page_url": "https://pexels.com/video/2"},
        )]

    def test_request_failure_returns_empty_list(self):
        import urllib.error
        pack = PexelsMediaPack(api_key="fake-key")

        with patch("tish_video_sdk.internal.providers.media_packs.urllib.request.urlopen",
                    side_effect=urllib.error.URLError("no network")):
            assert pack.search("mountains") == []


class TestDuckDuckGoMediaPack:
    def test_returns_image_results(self):
        pack = DuckDuckGoMediaPack()
        fake_results = [{"image": f"https://example.com/{i}.jpg"} for i in range(3)]

        with patch("tish_video_sdk.internal.providers.media_packs.DDGS") as mock_ddgs_cls:
            mock_ddgs_cls.return_value.__enter__.return_value.images.return_value = fake_results
            results = pack.search("sunset", count=3)

        assert results == [MediaResult(url=r["image"]) for r in fake_results]

    def test_caps_at_count(self):
        pack = DuckDuckGoMediaPack()
        fake_results = [{"image": f"https://example.com/{i}.jpg"} for i in range(10)]

        with patch("tish_video_sdk.internal.providers.media_packs.DDGS") as mock_ddgs_cls:
            mock_ddgs_cls.return_value.__enter__.return_value.images.return_value = fake_results
            results = pack.search("sunset", count=2)

        assert len(results) == 2

    def test_search_failure_returns_empty_list(self):
        pack = DuckDuckGoMediaPack()

        with patch("tish_video_sdk.internal.providers.media_packs.DDGS") as mock_ddgs_cls:
            mock_ddgs_cls.return_value.__enter__.side_effect = RuntimeError("network down")
            assert pack.search("sunset") == []


def _fake_wiki_page(title, html):
    return SimpleNamespace(title=title, html=lambda: html)


class TestWikipediaMediaPack:
    def test_returns_image_results_with_caption_and_page_title(self):
        pack = WikipediaMediaPack()
        html = """
        <div class="thumbinner">
          <img src="//upload.wikimedia.org/commons/a/ab/Sunrise.jpg" width="300" height="200">
          <div class="thumbcaption">The sun rising on Everest</div>
        </div>
        """
        fake_page = _fake_wiki_page("Mount Everest", html)

        with patch("tish_video_sdk.internal.providers.media_packs.wikipedia.page", return_value=fake_page):
            results = pack.search("Mount Everest", count=5)

        assert len(results) == 1
        assert results[0].url == "https://upload.wikimedia.org/commons/a/ab/Sunrise.jpg"
        assert results[0].attribution == {"caption": "The sun rising on Everest", "page_title": "Mount Everest"}

    def test_filters_out_icons_and_small_images(self):
        pack = WikipediaMediaPack()
        html = """
        <img src="//example.org/icon.svg" width="300" height="200">
        <img src="//example.org/tiny.jpg" width="50" height="50">
        <img src="//example.org/real_photo.jpg" width="300" height="200">
        """
        fake_page = _fake_wiki_page("Some Page", html)

        with patch("tish_video_sdk.internal.providers.media_packs.wikipedia.page", return_value=fake_page):
            results = pack.search("Some Page", count=5)

        assert [r.url for r in results] == ["https://example.org/real_photo.jpg"]

    def test_keyword_match_scores_higher(self):
        pack = WikipediaMediaPack()
        html = """
        <div class="thumbinner">
          <img src="//example.org/unrelated.jpg" width="300" height="200">
          <div class="thumbcaption">A random unrelated photo</div>
        </div>
        <div class="thumbinner">
          <img src="//example.org/ancient_temple.jpg" width="300" height="200">
          <div class="thumbcaption">An ancient temple ruin</div>
        </div>
        """
        fake_page = _fake_wiki_page("Some Page", html)

        with patch("tish_video_sdk.internal.providers.media_packs.wikipedia.page", return_value=fake_page):
            results = pack.search("Some Page", count=5, keywords=["ancient"])

        assert results[0].url == "https://example.org/ancient_temple.jpg"

    def test_search_failure_returns_empty_list(self):
        pack = WikipediaMediaPack()

        with patch("tish_video_sdk.internal.providers.media_packs.wikipedia.page",
                    side_effect=RuntimeError("boom")), \
             patch("tish_video_sdk.internal.providers.media_packs.wikipedia.search",
                    side_effect=RuntimeError("boom")):
            assert pack.search("Some Page") == []


# --------------------------------------------------------------------------
# ImageFetcher
# --------------------------------------------------------------------------

class TestImageFetcher:
    def test_fetch_dispatches_to_correct_pack(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))
        fake_result = [MediaResult(url="https://example.com/1.jpg", attribution={"caption": "hi"})]

        with patch.object(fetcher._packs["wikipedia"], "search", return_value=fake_result) as mock_search:
            results = fetcher.fetch("wikipedia", "mountains", count=3)

        mock_search.assert_called_once_with("mountains", count=3)
        assert results == [{"url": "https://example.com/1.jpg", "caption": "hi"}]

    def test_fetch_unknown_provider_raises(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))
        with pytest.raises(ValueError, match="Unknown media source"):
            fetcher.fetch("flickr", "mountains")

    def test_fetch_pexels_images_returns_dicts(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path), pexels_api_key="fake-key")
        fake_result = [MediaResult(url="https://images.pexels.com/1.jpg",
                                    attribution={"photographer": "Jane"})]

        with patch.object(fetcher._pexels, "search", return_value=fake_result):
            results = fetcher.fetch_pexels_images("mountains")

        assert results == [{"url": "https://images.pexels.com/1.jpg", "photographer": "Jane"}]

    def test_fetch_pexels_videos_returns_dicts(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path), pexels_api_key="fake-key")
        fake_result = [MediaResult(url="https://videos.pexels.com/1.mp4", attribution={"user": "John"})]

        with patch.object(fetcher._pexels, "search_videos", return_value=fake_result):
            results = fetcher.fetch_pexels_videos("ocean waves")

        assert results == [{"url": "https://videos.pexels.com/1.mp4", "user": "John"}]

    def test_fetch_duckduckgo_images_returns_urls_only(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))
        fake_result = [MediaResult(url="https://example.com/1.jpg")]

        with patch.object(fetcher._packs["duckduckgo"], "search", return_value=fake_result):
            urls = fetcher.fetch_duckduckgo_images("sunset")

        assert urls == ["https://example.com/1.jpg"]

    def test_fetch_wikipedia_images_returns_dicts(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))
        fake_result = [MediaResult(url="https://example.com/1.jpg",
                                    attribution={"caption": "A caption", "page_title": "Some Page"})]

        with patch.object(fetcher._packs["wikipedia"], "search", return_value=fake_result):
            results = fetcher.fetch_wikipedia_images("some topic", keywords=["ancient"])

        assert results == [{"url": "https://example.com/1.jpg", "caption": "A caption", "page_title": "Some Page"}]

    def test_pexels_api_key_property(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path), pexels_api_key="fake-key")
        assert fetcher.pexels_api_key == "fake-key"


class TestDownloadFile:
    def _fake_response(self, content_type="image/jpeg", chunks=(b"fake-image-bytes",)):
        response = MagicMock()
        response.headers = {"content-type": content_type}
        response.iter_content.return_value = list(chunks)
        response.raise_for_status.return_value = None
        return response

    def test_downloads_and_saves_with_guessed_extension(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))

        with patch("tish_video_sdk.media.requests.get", return_value=self._fake_response()):
            path = fetcher.download_file("https://example.com/photo", filename_prefix="test photo")

        assert path is not None
        assert path.endswith(".jpg")
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == b"fake-image-bytes"

    def test_reuses_existing_file_for_same_url(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))

        with patch("tish_video_sdk.media.requests.get", return_value=self._fake_response()) as mock_get:
            first = fetcher.download_file("https://example.com/photo", filename_prefix="test")
            second = fetcher.download_file("https://example.com/photo", filename_prefix="test")

        assert first == second
        assert mock_get.call_count == 2  # still requests headers each time, but doesn't re-write

    def test_download_failure_returns_none(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))

        with patch("tish_video_sdk.media.requests.get", side_effect=ConnectionError("no network")):
            path = fetcher.download_file("https://example.com/photo")

        assert path is None

    def test_extension_guessed_from_url_when_no_content_type_match(self, tmp_path):
        fetcher = ImageFetcher(download_dir=str(tmp_path))
        response = self._fake_response(content_type="application/octet-stream")

        with patch("tish_video_sdk.media.requests.get", return_value=response):
            path = fetcher.download_file("https://example.com/photo.webp?size=large")

        assert path.endswith(".webp")


# --------------------------------------------------------------------------
# ImageSearcher
# --------------------------------------------------------------------------

def _fake_fetch_dispatch(results_by_source):
    def _fetch(provider, query, count=5, **kwargs):
        return results_by_source.get(provider, [])
    return _fetch


def _fake_download_file(url, filename_prefix="download", timeout=30):
    import hashlib
    return f"/fake_cache/{filename_prefix}_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"


class TestImageSearcherSearch:
    def test_combines_sources_in_order_until_max_results(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        results_by_source = {
            "pexels": [{"url": "https://pexels.com/1.jpg", "photographer": "A"}],
            "duckduckgo": [{"url": "https://ddg.com/2.jpg"}, {"url": "https://ddg.com/3.jpg"}],
        }

        with patch.object(searcher.fetcher, "fetch", side_effect=_fake_fetch_dispatch(results_by_source)), \
             patch.object(searcher.fetcher, "download_file", side_effect=_fake_download_file):
            paths = searcher.search("mountains", max_results=3)

        assert len(paths) == 3

    def test_stops_once_max_results_reached_without_calling_later_sources(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        results_by_source = {
            "pexels": [{"url": "https://pexels.com/1.jpg"}, {"url": "https://pexels.com/2.jpg"}],
        }
        calls = []

        def fetch(provider, query, count=5, **kwargs):
            calls.append(provider)
            return results_by_source.get(provider, [])

        with patch.object(searcher.fetcher, "fetch", side_effect=fetch), \
             patch.object(searcher.fetcher, "download_file", side_effect=_fake_download_file):
            paths = searcher.search("mountains", max_results=2)

        assert len(paths) == 2
        assert "duckduckgo" not in calls and "wikipedia" not in calls

    def test_source_returning_nothing_is_skipped_not_an_error(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        results_by_source = {"duckduckgo": [{"url": "https://ddg.com/1.jpg"}]}

        with patch.object(searcher.fetcher, "fetch", side_effect=_fake_fetch_dispatch(results_by_source)), \
             patch.object(searcher.fetcher, "download_file", side_effect=_fake_download_file):
            paths = searcher.search("mountains", max_results=5)

        assert len(paths) == 1

    def test_custom_sources_restricts_which_are_tried(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        calls = []

        def fetch(provider, query, count=5, **kwargs):
            calls.append(provider)
            return [{"url": f"https://{provider}.com/1.jpg"}]

        with patch.object(searcher.fetcher, "fetch", side_effect=fetch), \
             patch.object(searcher.fetcher, "download_file", side_effect=_fake_download_file):
            searcher.search("mountains", max_results=5, sources=["duckduckgo"])

        assert calls == ["duckduckgo"]

    def test_source_error_is_caught_and_search_continues(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))

        def fetch(provider, query, count=5, **kwargs):
            if provider == "pexels":
                raise RuntimeError("boom")
            if provider == "duckduckgo":
                return [{"url": "https://ddg.com/1.jpg"}]
            return []

        with patch.object(searcher.fetcher, "fetch", side_effect=fetch), \
             patch.object(searcher.fetcher, "download_file", side_effect=_fake_download_file):
            paths = searcher.search("mountains", max_results=5)

        assert len(paths) == 1

    def test_failed_download_is_skipped(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        results_by_source = {"pexels": [{"url": "https://pexels.com/1.jpg"}]}

        with patch.object(searcher.fetcher, "fetch", side_effect=_fake_fetch_dispatch(results_by_source)), \
             patch.object(searcher.fetcher, "download_file", return_value=None):
            paths = searcher.search("mountains", max_results=5)

        assert paths == []


class TestImageSearcherMetadataSidecar:
    def test_writes_metadata_sidecar_next_to_downloaded_file(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        dest = str(tmp_path / "pexels_mountains_abc123.jpg")
        results_by_source = {"pexels": [{"url": "https://pexels.com/1.jpg", "photographer": "Jane Doe"}]}

        with patch.object(searcher.fetcher, "fetch", side_effect=_fake_fetch_dispatch(results_by_source)), \
             patch.object(searcher.fetcher, "download_file", return_value=dest):
            searcher.search("mountains", max_results=1, sources=["pexels"])

        metadata_path = dest + ".meta.json"
        assert os.path.exists(metadata_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["source"] == "pexels"
        assert metadata["query"] == "mountains"
        assert metadata["photographer"] == "Jane Doe"

    def test_get_metadata_reads_back_sidecar(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        dest = str(tmp_path / "image.jpg")
        searcher._write_metadata(dest, {"source": "pexels", "query": "test"})

        assert searcher.get_metadata(dest) == {"source": "pexels", "query": "test"}

    def test_get_metadata_returns_empty_dict_when_missing(self, tmp_path):
        searcher = ImageSearcher(cache_dir=str(tmp_path))
        assert searcher.get_metadata(str(tmp_path / "nonexistent.jpg")) == {}


# --------------------------------------------------------------------------
# ImageGenerator
# --------------------------------------------------------------------------

def _fake_png_bytes():
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (4, 4), (100, 100, 100)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_generate_images_result(num_images=1):
    images = [SimpleNamespace(image=SimpleNamespace(image_bytes=_fake_png_bytes())) for _ in range(num_images)]
    return SimpleNamespace(generated_images=images)


class TestImageGenerator:
    def test_no_api_key_returns_empty_list(self, tmp_path):
        generator = ImageGenerator(api_key="", output_directory=str(tmp_path))
        assert generator.generate_image("a quiet forest", "output") == []

    def test_generates_and_saves_image(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            paths = generator.generate_image("a quiet forest", "forest_scene")

        assert len(paths) == 1
        assert os.path.exists(paths[0])
        assert paths[0].endswith("forest_scene.png")

    def test_multiple_images_get_numbered_filenames(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(2)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            paths = generator.generate_image("a quiet forest", "forest_scene", num_images=2)

        assert sorted(os.path.basename(p) for p in paths) == ["forest_scene_1.png", "forest_scene_2.png"]

    def test_api_error_returns_empty_list(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.side_effect = RuntimeError("quota exceeded")

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            assert generator.generate_image("a quiet forest", "forest_scene") == []

    def test_no_generated_images_returns_empty_list(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = SimpleNamespace(generated_images=[])

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            assert generator.generate_image("a quiet forest", "forest_scene") == []

    def test_default_prompt_is_generic_not_domain_specific(self):
        # Regression check for the biblical-content cleanup: the default
        # prompt template must be plain/domain-neutral, not tied to any one
        # content domain (the original hardcoded biblical/Middle-East framing).
        from tish_video_sdk.media import DEFAULT_IMAGE_PROMPT_TEMPLATE
        assert "bibl" not in DEFAULT_IMAGE_PROMPT_TEMPLATE.lower()

    def test_custom_prompt_template_is_used(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_generate_images = mock_client_cls.return_value.models.generate_images
            mock_generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(
                api_key="fake-key", output_directory=str(tmp_path),
                prompt_template="Custom prompt about: {text}",
            )
            generator.generate_image("a quiet forest", "forest_scene")

        sent_prompt = mock_generate_images.call_args.kwargs["prompt"]
        assert sent_prompt == "Custom prompt about: a quiet forest"


class TestSearchOrGenerate:
    def test_uses_search_results_when_sufficient(self, tmp_path):
        generator = ImageGenerator(api_key="", output_directory=str(tmp_path))

        with patch.object(generator.image_searcher, "search", return_value=["/cache/a.jpg", "/cache/b.jpg"]) as mock_search:
            images = generator.search_or_generate("a quiet forest", "forest_scene", min_search_results=2)

        assert images == ["/cache/a.jpg", "/cache/b.jpg"]
        mock_search.assert_called_once()

    def test_falls_back_to_generation_when_search_insufficient(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            with patch.object(generator.image_searcher, "search", return_value=[]):
                images = generator.search_or_generate("a quiet forest", "forest_scene", min_search_results=2)

        assert len(images) == 1

    def test_prefer_generation_skips_search(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            with patch.object(generator.image_searcher, "search") as mock_search:
                generator.search_or_generate("a quiet forest", "forest_scene", prefer_generation=True)

        mock_search.assert_not_called()


class TestSearchOrGenerateKeywordExtraction:
    """Covers _gemini_extract_search_keywords() going through the reasoning
    module (see CONTEXT.md's reasoning.generate()/embed() entry) instead of
    ImageGenerator's own api_key-based client -- gated on language_code,
    not api_key, since keyword extraction is a text call unrelated to
    Imagen's own credential."""

    def test_no_language_code_falls_back_to_raw_text(self, tmp_path):
        generator = ImageGenerator(api_key="", output_directory=str(tmp_path))

        with patch("tish_video_sdk.reasoning.generate") as mock_generate, \
             patch.object(generator.image_searcher, "search", return_value=["/cache/a.jpg", "/cache/b.jpg"]) as mock_search:
            generator.search_or_generate("a quiet forest", "forest_scene", min_search_results=2)

        mock_generate.assert_not_called()
        mock_search.assert_called_once_with("a quiet forest", max_results=3)

    def test_language_code_configured_uses_extracted_keywords(self, tmp_path):
        generator = ImageGenerator(api_key="", output_directory=str(tmp_path), language_code="en")

        with patch("tish_video_sdk.reasoning.generate", return_value="forest, mist") as mock_generate, \
             patch.object(generator.image_searcher, "search", return_value=["/cache/a.jpg", "/cache/b.jpg"]) as mock_search:
            generator.search_or_generate("a quiet forest", "forest_scene", min_search_results=2)

        mock_generate.assert_called_once()
        mock_search.assert_called_once_with("forest, mist", max_results=3)

    def test_reasoning_error_falls_back_to_raw_text(self, tmp_path):
        generator = ImageGenerator(api_key="", output_directory=str(tmp_path), language_code="en")

        with patch("tish_video_sdk.reasoning.generate", side_effect=ReasoningAPIError("gemini is down")), \
             patch.object(generator.image_searcher, "search", return_value=["/cache/a.jpg", "/cache/b.jpg"]) as mock_search:
            generator.search_or_generate("a quiet forest", "forest_scene", min_search_results=2)

        mock_search.assert_called_once_with("a quiet forest", max_results=3)


class TestGenerateVariations:
    def test_generates_one_image_per_variation(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            paths = generator.generate_variations(
                "a wandering traveler", "traveler", variations=["smiling", "pensive"],
            )

        assert len(paths) == 2

    def test_caps_at_num_variations(self, tmp_path):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_images.return_value = _fake_generate_images_result(1)

            generator = ImageGenerator(api_key="fake-key", output_directory=str(tmp_path))
            paths = generator.generate_variations(
                "a wandering traveler", "traveler",
                variations=["smiling", "pensive", "surprised", "calm"], num_variations=2,
            )

        assert len(paths) == 2


# --------------------------------------------------------------------------
# ImageConverter
# --------------------------------------------------------------------------

class TestImageConverter:
    def test_convert_9_to_16_creates_wider_canvas(self, tmp_path):
        source_path = tmp_path / "portrait.png"
        Image.new("RGB", (900, 1600), (10, 20, 30)).save(source_path)
        output_path = tmp_path / "landscape.png"

        converter = ImageConverter()
        text_color = converter.convert_9_to_16_aspect_ratio(str(source_path), str(output_path))

        assert text_color is not None
        with Image.open(output_path) as result:
            assert result.width > result.height

    def test_convert_missing_file_returns_none(self, tmp_path):
        converter = ImageConverter()
        result = converter.convert_9_to_16_aspect_ratio(str(tmp_path / "missing.png"), str(tmp_path / "out.png"))
        assert result is None

    def test_resize_for_layer_maintains_aspect_by_default(self, tmp_path):
        source_path = tmp_path / "square.png"
        Image.new("RGB", (400, 200), (0, 0, 0)).save(source_path)
        output_path = tmp_path / "resized.png"

        converter = ImageConverter()
        converter.resize_for_layer(str(source_path), str(output_path), target_width=100, target_height=100)

        with Image.open(output_path) as result:
            assert result.width <= 100 and result.height <= 100
            assert result.width == 100  # 400x200 -> thumbnail(100,100) keeps 2:1 ratio -> 100x50

    def test_combine_with_background_produces_output(self, tmp_path):
        background_path = tmp_path / "bg.png"
        Image.new("RGBA", (200, 200), (255, 255, 255, 255)).save(background_path)
        foreground_path = tmp_path / "fg.png"
        Image.new("RGBA", (50, 50), (255, 0, 0, 255)).save(foreground_path)
        output_path = tmp_path / "combined.png"

        converter = ImageConverter()
        result = converter.combine_with_background(str(foreground_path), str(background_path), str(output_path))

        assert result == str(output_path)
        assert os.path.exists(output_path)

    def test_no_2_5d_walking_sequence_methods(self):
        # Regression check for the 2.5D-scope-cut decision: walking-sequence
        # / animated-GIF-with-movement-path compositing (2.5D-scene-specific,
        # already dropped elsewhere) must not resurface here.
        converter = ImageConverter()
        assert not hasattr(converter, "create_walking_video_sequence")
        assert not hasattr(converter, "combine_animated_gif_with_background")
