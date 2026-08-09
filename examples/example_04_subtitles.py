"""Example: subtitles, both hand-written and auto-generated from TTS audio.

Generates its own placeholder background image with PIL, same as the other
standalone examples, so this runs with no extra input.

Demonstrates two flows:
- Manual subtitles: hand-written timed segments via VideoBuilder.with_subtitles(),
  no TTS or forced alignment needed (mirrors the original's
  example_05_manual_subtitles.py).
- Auto-generated subtitles: VideoBuilder.with_tts_subtitles() extracts segments
  from the synthesized narration via tish_video_sdk.subtitles.SubtitleBuilder,
  styled from the SubtitlePack configured for the language (see
  ../configuration/subtitle_packs.example.json), reusing the same .env / voice_packs.json
  configuration as example_01_standalone_tts.py.

Forced alignment via Montreal Forced Aligner (MFA, the default -- see
tish_video_sdk.subtitles) requires a separate MFA install most environments
won't have, so this example forces time-based splitting instead
(USE_MFA_ALIGNMENT = False) purely so it runs standalone; real projects with
MFA installed can leave the default as-is for word-level accuracy.
"""
import os

from PIL import Image, ImageDraw

import tish_video_sdk.subtitles as subtitles
from tish_video_sdk.tts import TTSBuilder
from tish_video_sdk.video_maker import VideoBuilder, TextStyle

subtitles.USE_MFA_ALIGNMENT = False

NARRATION_TEXT = "Welcome to the Tish Video SDK subtitles demonstration. This sentence is auto-aligned."


def _make_placeholder_image(path: str, width: int = 1080, height: int = 1920) -> str:
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

    print("--- Manual subtitles (no TTS/alignment needed) ---")
    manual_segments = [
        {"text": "Welcome to the Tish Video SDK.", "start": 0.5, "end": 2.5},
        {"text": "These subtitles were hand-written.", "start": 3.0, "end": 6.0},
    ]
    builder = VideoBuilder.from_single_image(image_path, duration=7.0)
    builder.with_subtitles(manual_segments, style=TextStyle(font_color="white", bg_color="black"))
    if builder.build():
        builder.save(os.path.join(output_dir, "manual_subtitles.mp4"))

    print("--- Auto-generated subtitles from TTS narration ---")
    language_code = os.getenv("TTS_LANGUAGE_CODE")
    if not TTSBuilder.available_providers(language_code):
        print(f"No TTS provider configured for language '{language_code}'; skipping.")
        return

    # No provider= pin here: leaving it unset uses the full ordered fallback
    # chain configured for this language (see example_02's TTS flow).
    builder = VideoBuilder.from_single_image_with_tts(image_path, NARRATION_TEXT, language_code)
    builder.with_tts_subtitles()  # style defaults to the language's SubtitlePack
    if builder.build():
        builder.save(os.path.join(output_dir, "auto_subtitles.mp4"))


if __name__ == "__main__":
    main()
