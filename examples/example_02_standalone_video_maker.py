"""Example: build a video from a single image with VideoBuilder.

Generates its own placeholder background image with PIL rather than requiring
a bundled asset file, so this example runs standalone with no extra input.

Demonstrates two flows:
- A silent video with a fixed duration (VideoBuilder.from_single_image).
- A TTS-narrated video (VideoBuilder.from_single_image_with_tts), reusing the
  same .env / voice_packs.json configuration as example_01_standalone_tts.py.

Subtitle support isn't available yet (see source/api/video_maker.rst) -- it
returns once the subtitles module lands.
"""
import os

from PIL import Image, ImageDraw

from tish_video_sdk.tts import TTSBuilder
from tish_video_sdk.video_maker import VideoBuilder

NARRATION_TEXT = "Welcome to the Tish Video SDK video maker demonstration."


def _make_placeholder_image(path: str, width: int = 1080, height: int = 1920) -> str:
    # A mid-tone color with a contrasting center rectangle, so the output is
    # visibly a real image rather than looking like a broken black screen.
    img = Image.new("RGB", (width, height), color=(70, 110, 170))
    draw = ImageDraw.Draw(img)
    margin_x, margin_y = width // 6, height // 3
    draw.rectangle((margin_x, margin_y, width - margin_x, height - margin_y), fill=(230, 200, 90))
    img.save(path)
    return path


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    image_path = _make_placeholder_image(os.path.join(output_dir, "placeholder.png"))

    print("--- Silent video, fixed duration ---")
    builder = VideoBuilder.from_single_image(image_path, duration=5.0)
    if builder.build():
        builder.save(os.path.join(output_dir, "silent_video.mp4"))

    print("--- TTS-narrated video ---")
    language_code = os.getenv("TTS_LANGUAGE_CODE")
    if not TTSBuilder.available_providers(language_code):
        print(f"No TTS provider configured for language '{language_code}'; skipping.")
        return

    # No provider= pin here: leaving it unset uses the full ordered fallback
    # chain configured for this language, so a failure on one provider (e.g.
    # Gemini rate-limited) automatically retries the next (e.g. Google Cloud).
    builder = VideoBuilder.from_single_image_with_tts(image_path, NARRATION_TEXT, language_code)
    if builder.build():
        builder.save(os.path.join(output_dir, "narrated_video.mp4"))


if __name__ == "__main__":
    main()
