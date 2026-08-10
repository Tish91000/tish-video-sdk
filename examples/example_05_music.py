"""Example: mix background music under a speech track with MusicManager.

Generates its own placeholder speech clip -- via Google Cloud TTS when
configured (see ../.env.template), falling back to a synthetic pydub tone
otherwise so this still runs standalone with no external services required --
and mood-tagged background tracks with pydub's tone generator.

Demonstrates five flows:
- Mood-name selection: music_input="calm" picks straight from bgm_directory,
  no Gemini call involved.
- Text-driven mood analysis: reference_text_for_mood asks Gemini (via the
  reasoning module -- see ../CONTEXT.md's reasoning.generate()/embed() entry)
  to pick the mood, gated on REASONING_PACKS_PATH being configured with an
  "en" entry (see ../.env.template) -- skipped otherwise, same pattern as the
  other examples' provider checks.
- Jamendo fallback: mixing for "meditative", a mood with no local placeholder
  track, gated on JAMENDO_CLIENT_ID being configured -- skipped otherwise.
- Gemini + Jamendo combined: reference_text_for_mood describing a scene with
  no local placeholder mood, gated on both REASONING_PACKS_PATH and
  JAMENDO_CLIENT_ID. Gemini classifies the mood *and* -- since the reference
  text is long prose, not a short tag -- condenses it into one well-chosen
  search tag for Jamendo (see MusicManager._refine_search_query); the
  downloaded track still gets filed under the classified mood either way.
  This and the plain Jamendo fallback both download a real track from a real
  third-party service when they run, so they're opt-in like the Gemini flow,
  not exercised by default.
- Explicit pool growth + random pick: fetch_from_jamendo() called directly a
  few times with different search_query text (standing in for a few days'
  worth of a mostly-single-mood daily content series, e.g. Daily_Readings),
  growing "meditative"'s local pool beyond the single track the earlier
  Jamendo-fallback flow already fetched -- then a few plain select_audio_music
  calls show it picking among that pool instead of always the same track.

Uses MusicManager's built-in generic moods (see music.py's DEFAULT_MOODS) --
a project with its own real bgm library and mood subfolders would instead
point MUSIC_MOODS_PATH/mood_packs_path at its own music_moods.json (see
../configuration/music_moods.example.json).
"""
import os

from dotenv import load_dotenv
from pydub.generators import Sine

from tish_video_sdk.music import MusicManager
from tish_video_sdk.tts import TTSBuilder

load_dotenv()

SPEECH_TEXT = "Welcome to the Tish Video SDK background music demonstration."
SPEECH_TEXT_FOR_MOOD = "The sun broke through the clouds and everyone in the crowd cheered."
# Deliberately long (well past _refine_search_query's short-tag threshold)
# and themed toward "adventurous"/"dark" -- moods with no local placeholder
# below -- so this actually exercises Gemini's keyword condensation and the
# Jamendo fallback together, not just mood classification on its own.
ADVENTURE_TEXT_FOR_MOOD = (
    "Thunder rolled across the ridge as the climbers pushed higher, ice axes "
    "biting into the frozen rock, racing the storm to reach the summit before nightfall."
)
# Stand-in for a few days' worth of reading text in a mostly-"meditative"
# daily content series -- each becomes part of that day's Jamendo query (see
# MusicManager.fetch_from_jamendo), so different days search differently.
DAILY_READING_TEXTS = [
    "Be still and know the silence between each breath.",
    "The mountain does not move, yet everything moves around it.",
    "In stillness, the mind finds what it was always seeking.",
]


def _make_placeholder_speech(path: str, duration_ms: int = 4000) -> str:
    language_code = os.getenv("TTS_LANGUAGE_CODE")
    if "google_cloud" in TTSBuilder.available_providers(language_code):
        tts_builder = TTSBuilder.from_text(SPEECH_TEXT, language_code, provider="google_cloud")
        if tts_builder.build() and tts_builder.save(path):
            return path
        print("Google Cloud TTS synthesis failed; falling back to a synthetic tone.")

    # No Google Cloud TTS configured (or it failed) -- fall back to a
    # synthetic tone so this example still runs standalone with no external
    # services required.
    Sine(220).to_audio_segment(duration=duration_ms).export(path, format="wav")
    return path


def _make_placeholder_bgm(bgm_directory: str, mood: str, freq: int, duration_ms: int = 8000) -> None:
    mood_dir = os.path.join(bgm_directory, mood)
    os.makedirs(mood_dir, exist_ok=True)
    Sine(freq).to_audio_segment(duration=duration_ms).export(
        os.path.join(mood_dir, f"{mood}_placeholder.wav"), format="wav"
    )


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    bgm_directory = os.path.join(output_dir, "bgm")
    os.makedirs(output_dir, exist_ok=True)

    speech_path = _make_placeholder_speech(os.path.join(output_dir, "placeholder_speech.wav"))
    _make_placeholder_bgm(bgm_directory, "calm", freq=330)
    _make_placeholder_bgm(bgm_directory, "joy", freq=440)

    # Mood analysis needs a REASONING_PACKS_PATH configured with an "en"
    # entry (see ../CONTEXT.md's ReasoningPack entry) -- no raw key is ever
    # passed to MusicManager directly.
    language_code = "en" if os.getenv("REASONING_PACKS_PATH") else None

    manager = MusicManager(
        bgm_directory=bgm_directory,
        # A real, already-generated fallback -- the class default
        # ("./music/bgm/default_music.mp3") doesn't exist in this example,
        # so if a Jamendo search below genuinely finds nothing, mixing has
        # something real to fall back to instead of crashing on a missing file.
        default_music_file=os.path.join(bgm_directory, "calm", "calm_placeholder.wav"),
        language_code=language_code,
        jamendo_client_id=os.getenv("JAMENDO_CLIENT_ID", ""),
    )

    print("--- Mood-name selection (no Gemini call) ---")
    manager.add_background_music(
        speech_path,
        os.path.join(output_dir, "speech_with_calm_bgm.wav"),
        music_input="calm",
    )

    print("--- Text-driven mood analysis ---")
    if manager.language_code:
        manager.add_background_music(
            speech_path,
            os.path.join(output_dir, "speech_with_analyzed_bgm.wav"),
            reference_text_for_mood=SPEECH_TEXT_FOR_MOOD,
        )
    else:
        print("No REASONING_PACKS_PATH configured; skipping mood analysis.")

    print("--- Jamendo fallback (mood with no local track) ---")
    if manager.jamendo_client_id:
        manager.add_background_music(
            speech_path,
            os.path.join(output_dir, "speech_with_jamendo_bgm.wav"),
            music_input="meditative",  # no local placeholder generated for this mood
        )
    else:
        print("No JAMENDO_CLIENT_ID configured; skipping Jamendo fallback.")

    print("--- Gemini + Jamendo combined (mood classification, then keyword-condensed search) ---")
    if manager.language_code and manager.jamendo_client_id:
        manager.add_background_music(
            speech_path,
            os.path.join(output_dir, "speech_with_gemini_jamendo_bgm.wav"),
            reference_text_for_mood=ADVENTURE_TEXT_FOR_MOOD,
        )
    else:
        print("Need both REASONING_PACKS_PATH and JAMENDO_CLIENT_ID configured; skipping.")

    print("--- Explicit pool growth (fetch_from_jamendo) + random local pick ---")
    if manager.jamendo_client_id:
        for day, text in enumerate(DAILY_READING_TEXTS, start=1):
            fetched_path = manager.fetch_from_jamendo("meditative", search_query=text)
            print(f"Day {day}: fetch_from_jamendo -> {fetched_path}")

        meditative_dir = os.path.join(bgm_directory, "meditative")
        pool = sorted(f for f in os.listdir(meditative_dir) if f.endswith(".mp3"))
        print(f"'meditative' local pool now has {len(pool)} track(s): {pool}")

        print("Picking from that pool across a few calls (should vary if pool > 1):")
        for i in range(4):
            picked = manager.select_audio_music("meditative", duration=5)
            print(f"  call {i + 1}: {os.path.basename(picked)}")
    else:
        print("No JAMENDO_CLIENT_ID configured; skipping pool-growth demo.")


if __name__ == "__main__":
    main()
