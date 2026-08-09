# Tish Video SDK

A Python SDK for generating videos with narration, subtitles, music, and imagery.

## Features

- **Text-to-speech narration** via Google Cloud TTS or Gemini, with automatic SSML generation

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

See the full API reference under `source/api/`.

## License

MIT License - see [LICENSE](LICENSE) for details.
