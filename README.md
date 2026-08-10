# Tish Video SDK

A Python SDK for generating videos with narration, subtitles, music, and imagery.

## Features

- **Text-to-speech narration** via Google Cloud TTS or Gemini, with automatic SSML generation
- **Video generation** from a single image or timestamped image/video segments, paired with provided audio or TTS narration, with image/video overlay support
- **Timed text overlays** (titles, captions, karaoke-style word-highlighted text) composited onto the video
- **Subtitle generation** from narration audio via forced alignment (Montreal Forced Aligner) or time-based splitting, styled per language via SubtitlePack config
- **Background music** selection and mixing, by mood name, Gemini-driven text analysis, or a direct file path
- **Image search and generation**: multi-source web image/video search (Pexels, DuckDuckGo, Wikipedia) with local caching, AI image generation via Imagen (search-first with generation as fallback), and general-purpose aspect-ratio/compositing utilities

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from tish_video_sdk.tts import TTSBuilder

tts = TTSBuilder.from_text("Bonjour le monde", language_code="fr").build()
tts.save("narration.wav")
```

```python
from tish_video_sdk.video_maker import VideoBuilder

video = VideoBuilder.from_single_image_with_audio("scene.png", "narration.wav")
video.build()
video.save("output.mp4")
```

```python
from tish_video_sdk.music import MusicManager

manager = MusicManager(bgm_directory="./music/bgm")
manager.add_background_music("narration.wav", "narration_with_music.wav", music_input="calm")
```

```python
from tish_video_sdk.media import ImageGenerator

generator = ImageGenerator(output_directory="./images")
paths = generator.search_or_generate("a quiet mountain sunrise", output_name="sunrise_scene")
```

See the full API reference under `source/api/`.

## License

MIT License - see [LICENSE](LICENSE) for details.
