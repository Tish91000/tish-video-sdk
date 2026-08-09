# Tish Video SDK

A Python SDK for generating videos with narration, subtitles, music, and imagery.

## Features

- **Text-to-speech narration** via Google Cloud TTS or Gemini, with automatic SSML generation
- **Video generation** from a single image or timestamped image/video segments, paired with provided audio or TTS narration, with image/video overlay support

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

See the full API reference under `source/api/`.

## License

MIT License - see [LICENSE](LICENSE) for details.
