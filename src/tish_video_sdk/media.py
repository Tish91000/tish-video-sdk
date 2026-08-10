"""media: image/video discovery, local caching, and AI image generation (see
CONTEXT.md).

Single entry point for media tooling, in the same shape as every other
feature module here (music.py -> MusicManager, tts.py -> TTSBuilder,
subtitles.py -> SubtitleBuilder, video_maker.py -> VideoBuilder). Named
"media", not "image", since Pexels search covers both photos and videos --
see internal/providers/media_packs.py.

- ImageFetcher: low-level image/video lookup across several web sources
  (DuckDuckGo image search, Pexels, Wikipedia article images -- see
  internal/providers/media_packs.py's MediaPack subclasses), plus a generic
  URL downloader. Returns URLs and metadata; doesn't decide which result to
  use or cache anything.
- ImageSearcher: combines those sources behind one search() call,
  downloading and caching results locally with an attribution sidecar.
  Source order/selection is generic and domain-neutral -- no built-in query
  prefixing or content bias.
- ImageGenerator: AI image generation via Google's Imagen API, optionally
  searching first via ImageSearcher and only generating when search doesn't
  turn up enough (generation costs real API usage, search doesn't). The
  prompt template is plain content, not core SDK behavior -- generic by
  default, overridable (the same override-a-template approach music.py's
  mood-analysis prompt uses).
- ImageConverter: general-purpose aspect-ratio conversion, transparent
  cutouts, resizing, and foreground/background compositing -- independent
  of the other three.
"""
import hashlib
import json
import os
from io import BytesIO
from typing import Dict, List, Optional

import requests
from PIL import Image

from .internal.providers.media_packs import DuckDuckGoMediaPack, MediaPack, PexelsMediaPack, WikipediaMediaPack

DEFAULT_SOURCES = ["pexels", "duckduckgo", "wikipedia"]

DEFAULT_IMAGE_PROMPT_TEMPLATE = """Generate an ultra-realistic, detailed illustration for the
following text: {text}

This is a visual representation only: NO TEXT, NO WORDS, NO INSCRIPTIONS
should appear in the image. Use lighting and a color palette that fit the
text's tone and setting, and stage the scene so it visually conveys the
text's meaning -- visible emotion, meaningful interaction between subjects,
symbolic staging where it helps. The image should be suitable for use as a
video thumbnail or background.
"""

_SEARCH_KEYWORD_PROMPT_TEMPLATE = """
Extract a short (3-6 word) image search query capturing the key visual
subject of the following text, suitable for searching a stock-photo library.

Text:
"{text}"

Your task:
Respond with only the search query, nothing else.
"""


class ImageFetcher:
    """Fetches image/video URLs and downloads files, dispatching to the
    MediaPack registered for each source (see media_packs.py)."""

    def __init__(self, download_dir: str = "downloads", pexels_api_key: str = ""):
        """
        Args:
            download_dir: Directory downloaded files are saved into.
            pexels_api_key: Pexels API key (free, see
                https://www.pexels.com/api/). Falls back to the
                PEXELS_API_KEY env var; Pexels-backed methods return []
                when neither is set.
        """
        self.download_dir = download_dir
        self._pexels = PexelsMediaPack(api_key=pexels_api_key)
        self._packs: Dict[str, MediaPack] = {
            "pexels": self._pexels,
            "duckduckgo": DuckDuckGoMediaPack(),
            "wikipedia": WikipediaMediaPack(),
        }
        os.makedirs(self.download_dir, exist_ok=True)

    @property
    def pexels_api_key(self) -> str:
        return self._pexels.api_key

    def fetch(self, provider: str, query: str, count: int = 5, **kwargs) -> List[Dict[str, str]]:
        """Search provider (any of MediaPack.available(), e.g. 'pexels',
        'duckduckgo', 'wikipedia') for up to count results.

        Returns:
            List of {'url', **attribution} dicts -- attribution fields vary
            by provider (see each MediaPack subclass), empty when a source
            provides none (e.g. DuckDuckGo).
        """
        pack = self._packs.get(provider)
        if pack is None:
            known = ", ".join(sorted(self._packs))
            raise ValueError(f"Unknown media source '{provider}'. Known sources: {known}.")
        results = pack.search(query, count=count, **kwargs)
        return [{"url": r.url, **r.attribution} for r in results]

    def fetch_duckduckgo_images(self, query: str, num_images: int = 5, license_filter: Optional[str] = None,
                                 size: Optional[str] = None, type_image: Optional[str] = None,
                                 layout: Optional[str] = None) -> List[str]:
        """Search DuckDuckGo for image URLs.

        Args:
            query: Search query.
            num_images: Maximum number of image URLs to return.
            license_filter: One of DDGS's license_image values (e.g.
                'Public', 'Share', 'ShareCommercially', 'Modify',
                'ModifyCommercially'), or None for no filter.
            size: 'Small', 'Medium', 'Large', 'Wallpaper', or None.
            type_image: 'photo', 'clipart', 'gif', 'transparent', 'line', or None.
            layout: 'Square', 'Tall', 'Wide', or None.

        Returns:
            List of image URLs.
        """
        results = self._packs["duckduckgo"].search(query, count=num_images, license_filter=license_filter,
                                                     size=size, type_image=type_image, layout=layout)
        return [r.url for r in results]

    def fetch_pexels_images(self, query: str, num_images: int = 5, orientation: Optional[str] = None,
                             size: Optional[str] = None, color: Optional[str] = None) -> List[Dict[str, str]]:
        """Search Pexels for photos. Returns [] if no Pexels API key is
        configured.

        Returns:
            List of {'url', 'photographer', 'photographer_url', 'page_url'}
            dicts -- Pexels asks that photographer be credited where
            practical, so it travels with the result rather than only the
            bare image URL.
        """
        results = self._pexels.search(query, count=num_images, orientation=orientation, size=size, color=color)
        return [{"url": r.url, **r.attribution} for r in results]

    def fetch_pexels_videos(self, query: str, num_videos: int = 3, orientation: Optional[str] = None,
                             size: Optional[str] = None) -> List[Dict[str, str]]:
        """Search Pexels for videos (mp4). Returns [] if no Pexels API key is
        configured.

        Returns:
            List of {'url', 'user', 'user_url', 'page_url'} dicts.
        """
        results = self._pexels.search_videos(query, count=num_videos, orientation=orientation, size=size)
        return [{"url": r.url, **r.attribution} for r in results]

    def fetch_wikipedia_images(self, query: str, num_images: int = 5, keywords: Optional[List[str]] = None,
                                prefer_terms: Optional[List[str]] = None, avoid_terms: Optional[List[str]] = None,
                                languages: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Search Wikipedia article pages for embedded images, ranked by
        relevance to keywords/prefer_terms/avoid_terms (all optional --
        unset, images are ranked by recency/caption-quality/resolution
        heuristics alone).

        Returns:
            List of {'url', 'caption', 'page_title'} dicts, highest score first.
        """
        results = self._packs["wikipedia"].search(query, count=num_images, keywords=keywords,
                                                    prefer_terms=prefer_terms, avoid_terms=avoid_terms,
                                                    languages=languages)
        return [{"url": r.url, **r.attribution} for r in results]

    def download_file(self, url: str, filename_prefix: str = "download", timeout: int = 30) -> Optional[str]:
        """Download a file (image or video) from url into download_dir.
        Reuses an already-downloaded file with the same url instead of
        re-fetching it.

        Returns:
            Path to the downloaded file, or None on failure.
        """
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        safe_prefix = "".join(c for c in filename_prefix if c.isalnum() or c in (" ", "_", "-")).strip()
        safe_prefix = safe_prefix.replace(" ", "_") or "download"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            ext = self._guess_extension(response.headers.get("content-type", ""), url)
            filename = f"{safe_prefix}_{url_hash}{ext}"
            filepath = os.path.join(self.download_dir, filename)

            if os.path.exists(filepath):
                return filepath

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded: {filepath}")
            return filepath
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None

    @staticmethod
    def _guess_extension(content_type: str, url: str) -> str:
        by_content_type = {
            "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
            "image/gif": ".gif", "video/mp4": ".mp4", "video/webm": ".webm",
        }
        for prefix, ext in by_content_type.items():
            if prefix in content_type:
                return ext

        tail = url.split("/")[-1]
        if "." in tail:
            possible_ext = "." + tail.split(".")[-1].split("?")[0]
            if len(possible_ext) <= 5:
                return possible_ext

        return ".mp4" if "video" in content_type else ".jpg"


class ImageSearcher:
    """Searches multiple web image sources (via ImageFetcher) and caches
    results locally."""

    def __init__(self, cache_dir: str = "./image_cache", pexels_api_key: str = ""):
        """
        Args:
            cache_dir: Directory downloaded images (and their metadata
                sidecars) are stored in.
            pexels_api_key: Pexels API key (free, see
                https://www.pexels.com/api/). Falls back to the
                PEXELS_API_KEY env var; the 'pexels' source is skipped
                (not an error) when neither is set.
        """
        self.cache_dir = cache_dir
        self.fetcher = ImageFetcher(download_dir=cache_dir, pexels_api_key=pexels_api_key)
        os.makedirs(cache_dir, exist_ok=True)

    def search(self, query: str, max_results: int = 5,
               sources: Optional[List[str]] = None) -> List[str]:
        """Search sources in order, downloading until max_results images are
        collected (or every source is exhausted).

        Args:
            query: Search query, passed to every source as-is.
            max_results: Maximum number of images to download.
            sources: Which sources to try, in order -- any of 'pexels',
                'duckduckgo', 'wikipedia'. Defaults to all three; a source
                with no results (or, for 'pexels', no configured API key)
                is simply skipped, not an error.

        Returns:
            List of local file paths, in source order. Re-downloading a URL
            already fetched before reuses the cached file (see
            ImageFetcher.download_file).
        """
        sources = sources or DEFAULT_SOURCES
        downloaded: List[str] = []

        for source in sources:
            if len(downloaded) >= max_results:
                break
            remaining = max_results - len(downloaded)
            try:
                results = self.fetcher.fetch(source, query, count=remaining)
            except Exception as e:
                print(f"Error fetching from source '{source}': {e}")
                continue

            for result in results:
                if len(downloaded) >= max_results:
                    break
                path = self.fetcher.download_file(result["url"], filename_prefix=f"{source}_{query[:20]}")
                if path:
                    self._write_metadata(path, {"source": source, "query": query, **result})
                    downloaded.append(path)

        return downloaded

    @staticmethod
    def _metadata_path(file_path: str) -> str:
        return file_path + ".meta.json"

    @classmethod
    def _write_metadata(cls, file_path: str, metadata: dict) -> None:
        try:
            with open(cls._metadata_path(file_path), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except Exception:
            pass  # best-effort -- a read-only cache_dir shouldn't break search

    @classmethod
    def get_metadata(cls, file_path: str) -> dict:
        """Read back a downloaded file's metadata sidecar (source, query,
        and whatever attribution that source provided). Returns {} if none
        exists."""
        metadata_path = cls._metadata_path(file_path)
        if not os.path.exists(metadata_path):
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


class ImageGenerator:
    """Generates images from text via Google's Imagen API, optionally
    preferring a web search (see ImageSearcher) over generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_directory: str = "images",
        model_name: str = "imagen-4.0-generate-001",
        aspect_ratio: str = "9:16",
        prompt_template: str = DEFAULT_IMAGE_PROMPT_TEMPLATE,
        image_searcher: Optional[ImageSearcher] = None,
    ):
        """
        Args:
            api_key: Google GenAI API key. Falls back to GEMINI_API_KEY,
                then GOOGLE_API_KEY. generate_image raises if none is set.
            output_directory: Directory generated images are saved into.
            model_name: Imagen model name.
            aspect_ratio: Aspect ratio for generated images.
            prompt_template: Prompt sent to Imagen, formatted with {text}.
                Generic/domain-neutral by default; override for a specific
                project's visual style or content domain.
            image_searcher: ImageSearcher used by search_or_generate() to
                try a web search before generating. Defaults to a new
                ImageSearcher(cache_dir=output_directory).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.output_directory = output_directory
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.prompt_template = prompt_template
        self.image_searcher = image_searcher or ImageSearcher(cache_dir=output_directory)

        self._client = None
        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def generate_image(self, text_prompt: str, output_name: str, num_images: int = 1) -> List[str]:
        """Generate image(s) illustrating text_prompt via Imagen.

        Args:
            text_prompt: Text to illustrate -- formatted into prompt_template.
            output_name: Base filename (without extension) for the saved
                image(s); a numeric suffix is added when num_images > 1.
            num_images: Number of images to generate.

        Returns:
            List of file paths to the generated images. Empty on any
            failure (no api_key, API error, nothing generated).
        """
        if not self._client:
            print("Warning: No Google GenAI API key provided (GEMINI_API_KEY/GOOGLE_API_KEY). Skipping generation.")
            return []

        os.makedirs(self.output_directory, exist_ok=True)
        prompt = self.prompt_template.format(text=text_prompt)
        print(f"Generating {num_images} image(s) for: '{text_prompt[:60]}...'")

        from google.genai.types import PersonGeneration
        try:
            result = self._client.models.generate_images(
                model=self.model_name,
                prompt=prompt,
                config=dict(
                    number_of_images=num_images,
                    person_generation=PersonGeneration.ALLOW_ADULT,
                    aspect_ratio=self.aspect_ratio,
                ),
            )
        except Exception as e:
            print(f"Error during image generation API call: {e}")
            return []

        if not result.generated_images:
            print("No images were generated by the API.")
            return []

        return self._save_images(result, output_name)

    def _save_images(self, result, output_name: str) -> List[str]:
        saved_paths = []
        for idx, generated in enumerate(result.generated_images):
            try:
                image_bytes = getattr(getattr(generated, "image", None), "image_bytes", None)
                if not isinstance(image_bytes, (bytes, bytearray)):
                    print(f"Warning: generated image {idx + 1} is missing image data.")
                    continue

                image = Image.open(BytesIO(image_bytes))
                filename = f"{output_name}.png" if len(result.generated_images) == 1 else f"{output_name}_{idx + 1}.png"
                file_path = os.path.join(self.output_directory, filename)
                image.save(file_path)
                print(f"Image saved to: {file_path}")
                saved_paths.append(file_path)
            except Exception as e:
                print(f"Error processing/saving generated image {idx + 1}: {e}")
                continue
        return saved_paths

    def search_or_generate(self, text: str, output_name: str, prefer_generation: bool = False,
                            min_search_results: int = 2, max_search_results: int = 3) -> List[str]:
        """Try a web search for images matching text first (cheaper, no
        generation cost); only generate when search doesn't find enough.

        Args:
            text: Text to find/generate an image for.
            output_name: Base filename for a generated image, if generation
                is used.
            prefer_generation: Skip search and generate directly.
            min_search_results: Search results at or above this count are
                considered sufficient -- generation is skipped.
            max_search_results: Maximum images to request from search.

        Returns:
            List of local file paths (searched and/or generated).
        """
        images: List[str] = []

        if not prefer_generation:
            search_query = self._gemini_extract_search_keywords(text) or text
            images = self.image_searcher.search(search_query, max_results=max_search_results)
            if len(images) >= min_search_results:
                return images

        if prefer_generation or len(images) < min_search_results:
            generated = self.generate_image(text, output_name, num_images=1)
            images.extend(generated)

        return images

    def _gemini_extract_search_keywords(self, text: str) -> Optional[str]:
        """Ask Gemini for a short image-search query capturing text's key
        visual subject. Returns None on any failure (including no api_key)
        so the caller can fall back to using text itself as the query."""
        if not self._client:
            return None
        prompt = _SEARCH_KEYWORD_PROMPT_TEMPLATE.format(text=text)
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            keywords = (response.text or "").strip()
            return keywords or None
        except Exception as e:
            print(f"Gemini search-keyword extraction failed: {e}")
            return None

    def generate_variations(self, base_prompt: str, output_name: str,
                             variations: Optional[List[str]] = None, num_variations: int = 3) -> List[str]:
        """Generate several variations of base_prompt, one per entry in
        variations (e.g. different expressions, poses, or framings of the
        same subject).

        Args:
            base_prompt: The subject description shared by every variation.
            output_name: Base filename; each variation adds its own suffix.
            variations: Short descriptors appended to base_prompt, one image
                generated per entry. Defaults to ["", "close-up", "wide shot"].
            num_variations: Caps how many of variations are used.

        Returns:
            List of file paths for the generated variations.
        """
        variations = (variations if variations is not None else ["", "close-up", "wide shot"])[:num_variations]

        all_paths = []
        for i, variation in enumerate(variations):
            prompt = f"{base_prompt}. {variation}." if variation else base_prompt
            paths = self.generate_image(prompt, f"{output_name}_{i + 1}", num_images=1)
            all_paths.extend(paths)
        return all_paths


class ImageConverter:
    """General-purpose image aspect-ratio conversion, transparent cutouts,
    resizing, and foreground/background compositing."""

    def convert_9_to_16_aspect_ratio(
        self,
        image_path: str,
        horizontal_image_path: str,
        image_position: str = "left",
        background_color: Optional[tuple] = None,
    ) -> Optional[str]:
        """Convert a 9:16 image to 16:9 by placing it on a wider canvas.

        Args:
            image_path: Path to the input (9:16) image.
            horizontal_image_path: Output path for the converted image.
            image_position: Where to place the original image on the new
                canvas -- 'left', 'right', or 'center'.
            background_color: RGB tuple or color name/hex string for the
                canvas background. Defaults to the input image's top-left
                pixel color.

        Returns:
            A contrasting hex text color computed from background_color, or
            None on failure.
        """
        try:
            original_image = Image.open(image_path)
            original_width, original_height = original_image.size

            if background_color is None:
                background_color = original_image.getpixel((0, 0))
            elif isinstance(background_color, str):
                from PIL import ImageColor
                background_color = ImageColor.getrgb(background_color)

            new_height = original_height
            new_width = int(new_height * (16 / 9))
            new_image = Image.new("RGB", (new_width, new_height), background_color)

            x_paste = self._calculate_paste_position(image_position, new_width, original_width)
            new_image.paste(original_image, (x_paste, 0))
            new_image.save(horizontal_image_path)

            text_color_rgb = tuple(255 - c for c in background_color)
            return f"#{text_color_rgb[0]:02x}{text_color_rgb[1]:02x}{text_color_rgb[2]:02x}"
        except FileNotFoundError:
            print(f"Error: The file at {image_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def _calculate_paste_position(position: str, new_width: int, original_width: int) -> int:
        if position == "left":
            return 0
        if position == "right":
            return new_width - original_width
        return (new_width - original_width) // 2

    def create_transparent_cutout(self, image_path: str, output_path: str,
                                   background_threshold: int = 30) -> str:
        """Remove a roughly-uniform background (detected from the image's
        corner pixels) and save the result as a transparent PNG.

        Returns:
            output_path on success, "" on failure.
        """
        try:
            import numpy as np
            from collections import Counter

            image = Image.open(image_path).convert("RGBA")
            data = np.array(image)

            corner_colors = [tuple(data[0, 0][:3]), tuple(data[0, -1][:3]),
                              tuple(data[-1, 0][:3]), tuple(data[-1, -1][:3])]
            most_common_bg = Counter(corner_colors).most_common(1)[0][0]

            bg_mask = np.all(np.abs(data[:, :, :3].astype(int) - most_common_bg) < background_threshold, axis=2)
            data[bg_mask] = [0, 0, 0, 0]

            Image.fromarray(data, "RGBA").save(output_path, "PNG")
            return output_path
        except Exception as e:
            print(f"Error creating transparent cutout: {e}")
            return ""

    def resize_for_layer(self, image_path: str, output_path: str, target_width: int,
                          target_height: int, maintain_aspect: bool = True) -> str:
        """Resize an image, optionally preserving aspect ratio.

        Returns:
            output_path on success, "" on failure.
        """
        try:
            image = Image.open(image_path)
            if maintain_aspect:
                image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
            else:
                image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            image.save(output_path)
            return output_path
        except Exception as e:
            print(f"Error resizing image: {e}")
            return ""

    def combine_with_background(
        self,
        foreground_path: str,
        background_path: str,
        output_path: str,
        foreground_position: tuple = (0, 0),
        foreground_scale: float = 1.0,
        blend_mode: str = "normal",
    ) -> str:
        """Composite a foreground image (PNG/GIF, using its own transparency)
        onto a background image.

        Args:
            foreground_path: Path to the foreground image.
            background_path: Path to the background image.
            output_path: Path to save the composited result to.
            foreground_position: (x, y) position of the foreground on the
                background.
            foreground_scale: Scale factor applied to the foreground before
                compositing.
            blend_mode: 'normal', 'multiply', 'screen', or 'overlay'.

        Returns:
            output_path on success, "" on failure.
        """
        try:
            background = Image.open(background_path).convert("RGBA")

            foreground = Image.open(foreground_path)
            if getattr(foreground, "is_animated", False):
                foreground.seek(0)
            foreground = foreground.convert("RGBA")

            if foreground_scale != 1.0:
                new_size = (int(foreground.width * foreground_scale), int(foreground.height * foreground_scale))
                foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

            positioned = self._position_image(foreground, background.size, foreground_position)
            if blend_mode == "normal":
                combined = Image.alpha_composite(background, positioned)
            else:
                combined = self._apply_blend_mode(background, positioned, blend_mode)

            combined.convert("RGB").save(output_path)
            print(f"Combined images saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error combining images: {e}")
            return ""

    @staticmethod
    def _position_image(foreground: Image.Image, background_size: tuple, position: tuple) -> Image.Image:
        canvas = Image.new("RGBA", background_size, (0, 0, 0, 0))
        x, y = position
        x = max(0, min(x, background_size[0] - foreground.width))
        y = max(0, min(y, background_size[1] - foreground.height))
        canvas.paste(foreground, (x, y), foreground)
        return canvas

    @staticmethod
    def _apply_blend_mode(background: Image.Image, positioned_foreground: Image.Image, mode: str) -> Image.Image:
        from PIL import ImageChops
        if mode == "multiply":
            return ImageChops.multiply(background, positioned_foreground)
        if mode == "screen":
            return ImageChops.screen(background, positioned_foreground)
        if mode == "overlay":
            return ImageChops.overlay(background, positioned_foreground)
        return Image.alpha_composite(background, positioned_foreground)
