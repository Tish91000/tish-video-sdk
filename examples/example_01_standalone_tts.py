"""Example: synthesize speech from text with TTSBuilder and save it to a WAV file.

Requires a configured .env (see ../.env.template) -- TTS_LANGUAGE_CODE (an
optional default language) and TTS_VOICE_PACKS_PATH -- pointing at a
voice_packs.json (see ../configuration/voice_packs.example.json) that lists the VoicePacks
available to this process. This demo's text is illustrative; it's synthesized
in whatever language is configured, not necessarily English.

Demonstrates TTSBuilder(..., provider=...), which pins synthesis to exactly
one provider and disables fallback -- useful for testing/comparing providers
individually rather than always going through the full ordered chain. Each
provider configured for the language is tried and saved to its own file.
"""
import os

from tish_video_sdk.tts import TTSBuilder

TEXT = """
Welcome to the Tish Video SDK demonstration. This powerful toolkit allows you to create
professional videos from images, audio, and text. The SDK supports multiple languages
and provides flexible styling options.
"""


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    language_code = os.getenv("TTS_LANGUAGE_CODE")

    for provider in TTSBuilder.available_providers(language_code):
        print(f"--- Testing provider: {provider} ---")
        tts_builder = TTSBuilder.from_text(TEXT, language_code, provider=provider)
        if not tts_builder.build():
            print(f"TTS build failed for provider '{provider}'.")
            continue

        output_path = os.path.join(output_dir, f"standalone_tts_{provider}.wav")
        saved_path = tts_builder.save(output_path)
        if saved_path:
            print(f"Standalone TTS audio ({provider}) saved to {saved_path}")


if __name__ == "__main__":
    main()
