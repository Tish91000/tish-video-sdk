# Tish Video SDK

A Python library for generating videos with subtitles, narration, music, and imagery. Consumed as a dependency by content-specific applications (e.g. Daily_Readings) — it does not itself contain content-strategy logic for any one channel.

## Language

**SDK boundary**:
A module belongs in TishSDK only if it's reusable across different content channels/apps — video/audio/image mechanics, not any one channel's content strategy. Logic tied to a specific channel's content (e.g. generating CTA copy for a particular kind of video, scraping a particular content source's readings) belongs in the consuming application.
_Avoid_: adding channel-specific features to the SDK "because it's convenient" for the one app that needs them today

**VoicePack**:
A single TTS provider's (Gemini, Google Cloud, Amazon Polly, Deepgram, Gradium, ...) adapter for one language — pairs that provider's config (voice name, credentials, character limits, SSML support) with its own model-loading and synthesis logic. A process loads every VoicePack it might need (any number of languages and providers); `TTSBuilder(language_code, provider=None)` filters to the ordered chain for one call, falling back to the next VoicePack on a provider-level failure unless `provider` pins it to exactly one.
_Avoid_: provider, TTS backend, TTS engine
