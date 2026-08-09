"""Real API tests rely on the process's real .env (TTS_LANGUAGE_CODE,
TTS_VOICE_PACKS_PATH) the same way tts.py itself loads it at import time --
there's no separate config-path injection here, unlike the old
TTS_LANGUAGE_CONFIG_PATH mechanism.

Two constraints inherited from the "config loads once, at import" design:
- tish_video_sdk.tts raises RuntimeError at import time if TTS_LANGUAGE_CODE or
  TTS_VOICE_PACKS_PATH aren't set at all. Graceful skipping (see
  test_tts_builder.py) only covers the case where they're set but point at
  something invalid/unavailable -- not their total absence.
- Because the loaded config is cached in sys.modules on first import, running
  `pytest tests/fake tests/real` (or bare `pytest`) together in one process
  makes tests/real inherit whichever suite imported tish_video_sdk.tts first --
  run these suites in separate pytest invocations, as their own docstrings say.

test_tts_builder.py's skip conditions check os.getenv("TTS_LANGUAGE_CODE")
*before* deciding whether to import tish_video_sdk.tts (importing it is what
normally triggers load_dotenv()) -- load .env here first, so that check sees
real values instead of the raw OS environment.
"""
from dotenv import load_dotenv

load_dotenv()
