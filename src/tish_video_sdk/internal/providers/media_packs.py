"""MediaPack: one image/video source's adapter (see CONTEXT.md), used by
media.py. Same shape as VoicePack (see voice_packs.py) -- an abstract base
plus concrete subclasses that auto-register themselves via
__init_subclass__, keyed by their `provider` class attribute, so a source
name (e.g. "pexels") resolves to a class without a separate registry to
maintain. Unlike VoicePack there's no fallback chain here -- media.py's
ImageFetcher calls a specific provider by name, or ImageSearcher tries
several in order -- so MediaPack itself only needs one shared shape
(search()), not chain-selection helpers.

Currently three providers: Pexels (photos and videos, needs an API key),
DuckDuckGo image search, and Wikipedia article images (both need no
credentials).
"""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Type

import wikipedia
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# The `wikipedia` package (unmaintained since ~2014) sends no User-Agent on
# its API requests -- Wikimedia now rejects those with a 403 plaintext
# response instead of JSON (see https://phabricator.wikimedia.org/T400119),
# which surfaces as a confusing json.JSONDecodeError deep inside the package
# rather than an auth-looking error. Setting one (their robot policy just
# asks for something identifying, not a personal contact) is enough.
wikipedia.set_user_agent("tish-video-sdk/0.1 (https://github.com/Tish91000/tish-video-sdk)")


@dataclass
class MediaResult:
    """One search result, common across every MediaPack. attribution is a
    free-form dict since sources differ in what credit info they provide
    (Pexels: photographer/photographer_url/page_url; Wikipedia: caption/
    page_title; DuckDuckGo: none) -- several sources' licenses ask for
    credit where practical, so it travels with the result rather than only
    the bare URL."""
    url: str
    attribution: Dict[str, str] = field(default_factory=dict)


class MediaPack(ABC):
    """Abstract base for a single image/video source's adapter."""

    provider: str = ""  # set by each concrete subclass; doubles as its registry key
    _registry: Dict[str, Type["MediaPack"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.provider:
            MediaPack._registry[cls.provider] = cls

    @abstractmethod
    def search(self, query: str, count: int = 5, **kwargs) -> List[MediaResult]:
        """Search for up to count results matching query. Returns [] on any
        failure (missing credentials, network error, no matches) --
        callers should fall back to another source rather than propagate
        an error."""

    @classmethod
    def get(cls, provider: str, **init_kwargs) -> "MediaPack":
        """Instantiate the MediaPack registered for provider (e.g.
        "pexels"), passing init_kwargs to its constructor."""
        pack_cls = cls._registry.get(provider)
        if pack_cls is None:
            known = ", ".join(sorted(cls._registry)) or "(none registered)"
            raise ValueError(f"Unknown media provider '{provider}'. Known providers: {known}.")
        return pack_cls(**init_kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Every registered provider name, sorted."""
        return sorted(cls._registry)


class PexelsMediaPack(MediaPack):
    """Pexels (https://www.pexels.com/api/) photo and video search. Needs a
    free API key -- api_key/PEXELS_API_KEY; search() returns [] without one."""

    provider = "pexels"
    _API_BASE_URL = "https://api.pexels.com"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("PEXELS_API_KEY", "")

    def _get(self, path: str, params: dict, timeout: float) -> Optional[dict]:
        url = self._API_BASE_URL + path + "?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}
        )
        request = urllib.request.Request(url, headers={"Authorization": self.api_key})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            print(f"Pexels request failed: {e}")
            return None

    def search(self, query: str, count: int = 5, orientation: Optional[str] = None,
               size: Optional[str] = None, color: Optional[str] = None,
               timeout: float = 10.0, **kwargs) -> List[MediaResult]:
        if not self.api_key:
            return []

        payload = self._get("/v1/search", {
            "query": query, "per_page": min(count, 80),
            "orientation": orientation, "size": size, "color": color,
        }, timeout)
        if not payload:
            return []

        results = []
        for photo in payload.get("photos", []):
            src = photo.get("src", {})
            src_url = src.get("large2x") or src.get("large") or src.get("original")
            if not src_url:
                continue
            results.append(MediaResult(url=src_url, attribution={
                "photographer": photo.get("photographer", ""),
                "photographer_url": photo.get("photographer_url", ""),
                "page_url": photo.get("url", ""),
            }))
        return results

    def search_videos(self, query: str, count: int = 3, orientation: Optional[str] = None,
                       size: Optional[str] = None, timeout: float = 10.0) -> List[MediaResult]:
        """Pexels-specific: video (mp4) search, not part of every MediaPack."""
        if not self.api_key:
            return []

        payload = self._get("/videos/search", {
            "query": query, "per_page": count, "orientation": orientation, "size": size,
        }, timeout)
        if not payload:
            return []

        results = []
        for video in payload.get("videos", []):
            mp4_files = [f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4"]
            if not mp4_files:
                continue
            mp4_files.sort(key=lambda f: f.get("width", 0), reverse=True)
            best = mp4_files[0]
            results.append(MediaResult(url=best.get("link"), attribution={
                "user": video.get("user", {}).get("name", ""),
                "user_url": video.get("user", {}).get("url", ""),
                "page_url": video.get("url", ""),
            }))
        return results


class DuckDuckGoMediaPack(MediaPack):
    """DuckDuckGo image search. Needs no credentials; provides no
    attribution beyond the image URL itself."""

    provider = "duckduckgo"

    def search(self, query: str, count: int = 5, license_filter: Optional[str] = None,
               size: Optional[str] = None, type_image: Optional[str] = None,
               layout: Optional[str] = None, **kwargs) -> List[MediaResult]:
        try:
            with DDGS() as ddgs:
                results = ddgs.images(
                    query, region="wt-wt", safesearch="off", size=size, type_image=type_image,
                    layout=layout, license_image=license_filter, max_results=count * 2,
                )
                return [MediaResult(url=r["image"]) for r in results][:count]
        except Exception as e:
            print(f"Error fetching images from DuckDuckGo: {e}")
            return []


class WikipediaMediaPack(MediaPack):
    """Images embedded in Wikipedia article pages, ranked by relevance.
    Needs no credentials.

    Wikipedia's own API doesn't return a ranked "best image" for a page, so
    this scrapes the rendered article HTML (via the `wikipedia` package +
    BeautifulSoup) and scores each embedded <img> by caption/URL keyword
    matches, optional preferred/avoided terms, recency (year mentioned in
    caption or filename), and a resolution heuristic -- all caller-supplied
    via search()'s kwargs, no built-in content bias.
    """

    provider = "wikipedia"

    def search(self, query: str, count: int = 5, keywords: Optional[List[str]] = None,
               prefer_terms: Optional[List[str]] = None, avoid_terms: Optional[List[str]] = None,
               languages: Optional[List[str]] = None, **kwargs) -> List[MediaResult]:
        keywords = keywords or []
        prefer_terms = prefer_terms or []
        avoid_terms = avoid_terms or []
        languages = languages or ["en"]

        collected: List[dict] = []

        for lang in languages:
            if len(collected) >= count:
                break

            wikipedia.set_lang(lang)
            try:
                pages = self._resolve_pages(query)
            except Exception as e:
                print(f"Wikipedia search failed for language '{lang}': {e}")
                continue

            for page in pages:
                if len(collected) >= count:
                    break
                try:
                    soup = BeautifulSoup(page.html(), "html.parser")
                except Exception as e:
                    print(f"Error fetching/parsing Wikipedia page '{page.title}': {e}")
                    continue

                candidates = self._scan_page(soup, keywords, prefer_terms, avoid_terms, collected)
                candidates.sort(key=lambda c: c["score"], reverse=True)
                for cand in candidates:
                    if len(collected) >= count:
                        break
                    cand["page_title"] = page.title
                    collected.append(cand)

        return [
            MediaResult(url=c["url"], attribution={"caption": c["caption"], "page_title": c["page_title"]})
            for c in collected
        ]

    @staticmethod
    def _resolve_pages(query: str, max_pages: int = 3) -> List["wikipedia.WikipediaPage"]:
        pages = []
        try:
            pages.append(wikipedia.page(query, auto_suggest=True))
        except Exception:
            pass

        if not pages:
            for title in wikipedia.search(query, results=max_pages):
                try:
                    if not any(p.title == title for p in pages):
                        pages.append(wikipedia.page(title, auto_suggest=False))
                except Exception:
                    continue
        return pages

    def _scan_page(self, soup: BeautifulSoup, keywords: List[str], prefer_terms: List[str],
                    avoid_terms: List[str], already_collected: List[dict]) -> List[dict]:
        candidates = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = "https://wikipedia.org" + src

            width, height = img.get("width"), img.get("height")
            if (width and int(width) < 100) or (height and int(height) < 100):
                continue
            if src.lower().endswith(".svg") or any(x in src.lower() for x in ("icon", "logo", "static")):
                continue

            full_res_url = src
            if "/thumb/" in src:
                try:
                    base_part, rest = src.split("/thumb/")
                    path_parts = rest.split("/")
                    if len(path_parts) >= 2:
                        full_res_url = f"{base_part}/{'/'.join(path_parts[:-1])}"
                except Exception:
                    pass

            caption = self._extract_caption(img)

            if any(c["url"] == full_res_url for c in already_collected) or \
                    any(c["url"] == full_res_url for c in candidates):
                continue

            score = self._score_image(src.lower(), caption, keywords, prefer_terms, avoid_terms)
            candidates.append({"url": full_res_url, "caption": caption, "score": score})
        return candidates

    @staticmethod
    def _extract_caption(img) -> str:
        thumbinner = img.find_parent("div", class_="thumbinner")
        if thumbinner:
            caption_div = thumbinner.find("div", class_="thumbcaption")
            if caption_div:
                return caption_div.get_text(strip=True)
        figure = img.find_parent("figure")
        if figure:
            figcaption = figure.find("figcaption")
            if figcaption:
                return figcaption.get_text(strip=True)
        return img.get("alt", "")

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return int(match.group(0)) if match else None

    def _score_image(self, src_lower: str, caption: str, keywords: List[str],
                      prefer_terms: List[str], avoid_terms: List[str]) -> int:
        score = 0
        caption_lower = caption.lower()

        for kw in keywords:
            if kw.lower() in caption_lower:
                score += 20
            if kw.lower() in src_lower:
                score += 5

        for term in prefer_terms:
            if term.lower() in caption_lower or term.lower() in src_lower:
                score += 10

        for term in avoid_terms:
            if term.lower() in caption_lower or term.lower() in src_lower:
                score -= 50

        year = self._extract_year(caption) or self._extract_year(src_lower)
        if year:
            age = max(datetime.now().year - year, 0)
            if age <= 5:
                score += 15
            elif age <= 15:
                score += 5
            elif age > 40:
                score -= 5

        if len(caption) > 10:
            score += 2

        res_match = re.search(r"(\d+)px", src_lower)
        if res_match and int(res_match.group(1)) > 500:
            score += 2

        return score
