"""Example: search and generate images with ImageSearcher/ImageGenerator.

Demonstrates:
- DuckDuckGo image search: no credentials needed, always runs. Still a real
  network call to a third-party service every time this example runs.
- Pexels image search: gated on PEXELS_API_KEY (see ../.env.template) --
  skipped otherwise, same pattern as the other examples' provider checks.
- Wikipedia image search: no credentials needed, always runs (real network call).
- AI image generation via Imagen: gated on GEMINI_API_KEY/GOOGLE_API_KEY --
  this calls a real, costed API, so it's opt-in like the other paid-provider
  flows in these examples, not exercised by default.
- search_or_generate(): tries a web search first, only generates if search
  doesn't find enough -- gated the same way as plain generation, since it
  can still fall through to a real Imagen call.
- ImageConverter utilities: fully local (PIL only), no network, always runs.

Uses ImageGenerator's built-in generic prompt template -- a project with its
own visual style/content domain would override it via prompt_template.
"""
import os

from dotenv import load_dotenv
from PIL import Image

from tish_video_sdk.media import ImageConverter, ImageGenerator, ImageSearcher

load_dotenv()

SEARCH_QUERY = "mountain sunrise"
GENERATION_PROMPT = "A quiet mountain sunrise, mist rising from a valley below."


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    cache_dir = os.path.join(output_dir, "image_cache")
    os.makedirs(output_dir, exist_ok=True)

    searcher = ImageSearcher(cache_dir=cache_dir, pexels_api_key=os.getenv("PEXELS_API_KEY", ""))

    print("--- DuckDuckGo image search (no credentials needed) ---")
    ddg_paths = searcher.search(SEARCH_QUERY, max_results=2, sources=["duckduckgo"])
    print(f"Downloaded {len(ddg_paths)} image(s): {ddg_paths}")

    print("--- Pexels image search ---")
    if os.getenv("PEXELS_API_KEY"):
        pexels_paths = searcher.search(SEARCH_QUERY, max_results=2, sources=["pexels"])
        for path in pexels_paths:
            print(f"{path} -> {searcher.get_metadata(path)}")
    else:
        print("No PEXELS_API_KEY configured; skipping Pexels search.")

    print("--- Wikipedia image search (no credentials needed) ---")
    wiki_paths = searcher.search("Mount Everest", max_results=2, sources=["wikipedia"])
    for path in wiki_paths:
        print(f"{path} -> {searcher.get_metadata(path)}")

    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    generator = ImageGenerator(
        api_key=gemini_api_key,
        output_directory=os.path.join(output_dir, "generated"),
        image_searcher=searcher,
    )

    print("--- AI image generation (Imagen, real costed API call) ---")
    if gemini_api_key:
        generated_paths = generator.generate_image(GENERATION_PROMPT, output_name="sunrise_scene")
        print(f"Generated: {generated_paths}")
    else:
        print("No GEMINI_API_KEY/GOOGLE_API_KEY configured; skipping generation.")

    print("--- search_or_generate (search first, generate only if search comes up short) ---")
    if gemini_api_key:
        images = generator.search_or_generate(GENERATION_PROMPT, output_name="sunrise_scene_v2", min_search_results=2)
        print(f"Result: {images}")
    else:
        print("No GEMINI_API_KEY/GOOGLE_API_KEY configured; skipping.")

    print("--- ImageConverter utilities (local, no network) ---")
    converter = ImageConverter()
    portrait_path = os.path.join(output_dir, "placeholder_portrait.png")
    Image.new("RGB", (900, 1600), (80, 110, 140)).save(portrait_path)
    landscape_path = os.path.join(output_dir, "placeholder_landscape.png")
    text_color = converter.convert_9_to_16_aspect_ratio(portrait_path, landscape_path)
    print(f"Converted {portrait_path} -> {landscape_path} (suggested text color: {text_color})")


if __name__ == "__main__":
    main()
