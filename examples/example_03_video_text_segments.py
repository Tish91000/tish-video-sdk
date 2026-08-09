"""Example: overlay timed text (a title, then manual captions) onto a video
with VideoBuilder.with_text_segments().

Generates its own placeholder background image with PIL, same as
example_02_standalone_video_maker.py, so this runs standalone with no extra
input and no TTS/API calls.

This is the generic text-in-video building block: a single segment spanning
the whole clip renders as a static title; multiple segments (optionally with
per-word timestamps) render as word-highlighted captions -- what subtitle
support will build on top of once tish_video_sdk.subtitles lands.
"""
import os

from PIL import Image, ImageDraw

from tish_video_sdk.video_maker import VideoBuilder, TextStyle


def _make_placeholder_image(path: str, width: int = 1080, height: int = 1920) -> str:
    img = Image.new("RGB", (width, height), color=(40, 80, 120))
    draw = ImageDraw.Draw(img)
    margin_x, margin_y = width // 6, height // 3
    draw.rectangle((margin_x, margin_y, width - margin_x, height - margin_y), fill=(230, 200, 90))
    img.save(path)
    return path


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    image_path = _make_placeholder_image(os.path.join(output_dir, "placeholder.png"))

    print("--- Title overlay (single segment spanning the whole clip) ---")
    builder = VideoBuilder.from_single_image(image_path, duration=5.0)
    builder.with_text_segments(
        [{"text": "Chapter One", "start": 0.0, "end": 5.0}],
        style=TextStyle(font_size=90, text_position=("center", "top")),
    )
    if builder.build():
        builder.save(os.path.join(output_dir, "title_overlay.mp4"))

    print("--- Manual captions (multiple timed segments, no TTS/alignment needed) ---")
    captions = [
        {"text": "Welcome to the Tish Video SDK.", "start": 0.5, "end": 2.5},
        {"text": "This overlay works from plain timed segments.", "start": 3.0, "end": 6.0},
    ]
    builder = VideoBuilder.from_single_image(image_path, duration=7.0)
    builder.with_text_segments(
        captions,
        style=TextStyle(font_color="white", bg_color="black", text_position=("center", "bottom")),
    )
    if builder.build():
        builder.save(os.path.join(output_dir, "manual_captions.mp4"))


if __name__ == "__main__":
    main()
