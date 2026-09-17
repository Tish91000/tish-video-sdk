# Tish Video SDK

A Python library for generating videos with subtitles, narration, music, and imagery. Consumed as a dependency by content-specific applications (e.g. Daily_Readings) — it does not itself contain content-strategy logic for any one channel.

## Language

**SDK boundary**:
A module belongs in TishSDK only if it's reusable across different content channels/apps — video/audio/image mechanics, not any one channel's content strategy. Logic tied to a specific channel's content (e.g. generating CTA copy for a particular kind of video, scraping a particular content source's readings) belongs in the consuming application.
_Avoid_: adding channel-specific features to the SDK "because it's convenient" for the one app that needs them today

**VoicePack**:
A single TTS provider's (Gemini, Google Cloud, Amazon Polly, Deepgram, Gradium, ...) adapter for one language — pairs that provider's config (voice name, credentials, character limits, SSML support) with its own model-loading and synthesis logic. A process loads every VoicePack it might need (any number of languages and providers); `TTSBuilder(language_code, provider=None)` filters to the ordered chain for one call, falling back to the next VoicePack on a provider-level failure unless `provider` pins it to exactly one.
_Avoid_: provider, TTS backend, TTS engine

**publisher_packs.json**:
Per-language, per-provider YouTube/Instagram publishing credentials (client_secret_filepath/token_filename for YouTube, username/password/session_file for Instagram), loaded from `PUBLISHER_PACKS_PATH`. `Publisher.for_language(language_code)` is the credential-free entry point that reads this file directly; `Publisher.__init__` still takes raw credentials for explicit/advanced construction (e.g. tests) that don't want a config file.
_Avoid_: passing `youtube_client_secret_filepath`/`youtube_token_filename` from application code when `for_language()` would do

**Publish-date tracking**:
Per-language state, owned by `Publisher` and backed internally by `date_managment.PublishingTracker`, recording the most recent date a channel's content was fully processed — lets a scheduling run pick the next date to work on and skip dates already done. Exposed to consuming apps only via `Publisher.next_unpublished_date()`/`Publisher.mark_published()`; application code never constructs `PublishingTracker` or calls `date_managment` directly for this (see ADR 0012).
_Avoid_: reaching into `PublishingTracker`/`date_managment` from application code — go through `Publisher`

**mark_published (Publisher)**:
Records a date as done regardless of whether publishing itself succeeded — "done" means the app decided not to retry this date, not "successfully live." The caller decides that (e.g. mark done anyway to avoid an infinite retry loop on a broken day); `Publisher` has no visibility into whether the surrounding pipeline (content fetch, video build) actually succeeded.
_Avoid_: "published" as a synonym for "upload succeeded" — a date can be marked published after a failed run

**Publish scheduling** (`date_managment.compute_publish_datetime`, `Publisher.publish_video`'s `content_date`/`hour_of_day`/`utc_offset_hours`):
Pure, stateless conversion from (a content date, an hour-of-day, a UTC-offset-in-hours) to the UTC datetime a video should go live. The offset table and base hour are always caller-supplied — a channel's own publishing policy (when *this* audience wants content), not domain-neutral content the SDK should default, unlike `DateFormatter`'s month names.
_Avoid_: adding a built-in per-language schedule default to the SDK the way month names have one

**Credential sourcing rule**:
No SDK method or constructor ever takes a raw API key, service-account path, or OAuth credential file as a required argument from the consuming app. Every credential is read by the SDK itself, either from a per-language JSON config file pointed to by an env var (VoicePack via `TTS_VOICE_PACKS_PATH`, ReasoningPack via `REASONING_PACKS_PATH`, publisher_packs.json via `PUBLISHER_PACKS_PATH`) or, for a single global secret with no per-language variant, directly from an env var (`ImageGenerator`'s `GEMINI_API_KEY`/`GOOGLE_API_KEY` fallback). A consuming app passes `language_code`, never the secret itself.
_Avoid_: adding a `*_api_key` or `*_credential_path` constructor parameter without also giving it a config-file or env-var source the SDK reads on its own

**ReasoningPack**:
One provider's adapter for one language's generic LLM/text calls that aren't TTS synthesis or image generation — mood analysis (`MusicManager`), meditation text, image-prompt descriptions, visual planning, image-search-keyword extraction, and embeddings (image retrieval), across both the SDK's own internal calls and a consuming app's bespoke ones. An `ABC` with a `provider` class attribute and a `__init_subclass__` self-registry, same shape as VoicePack (see ADR 0011, mirroring ADR 0006) — `GeminiReasoningPack` is the first concrete subclass. Loaded from `REASONING_PACKS_PATH`, lazily (only on first actual lookup, not at import time) so importing `reasoning.py` never forces the env var to be set for projects that don't use it.
_Avoid_: TTS key, LLM key (say "reasoning credential" — it's deliberately not the TTS or image-gen key, even though it's often the same underlying Gemini key value)

**reasoning.generate() / reasoning.embed()**:
The only way any code — SDK-internal (`MusicManager`, `ImageGenerator`) or a consuming app (Daily_Readings) — calls an LLM outside TTS synthesis and Imagen generation. No caller ever receives a raw credential or holds a client: `generate(language_code, template, values, tier="fast", provider=None, response_format="text")` and `embed(language_code, texts, task_type=..., provider=None)` look up and inject credentials internally, per call. `provider=None` walks the ordered ReasoningPack chain configured for that language (falling back on a `ReasoningError`, with backoff retry against the *current* provider first on a `ReasoningRateLimitError` before falling through — see ADR 0011); a given `provider` string pins to one, disabling fallback. The "default" provider is just whichever entry is first in that language's chain in `reasoning_packs.json` — today that's always Gemini (the only registered provider), but swapping or reordering the default later (e.g. a provider partnership) is a config change, not a call-site change. Raises `ReasoningError` (or a typed subclass) rather than returning `None`/`""` when the whole chain is exhausted — every call site must decide explicitly how to degrade.
_Avoid_: building a `genai.Client`/`ChatGoogleGenerativeAI` directly anywhere outside `reasoning.py`, threading a `google_api_key`-shaped parameter through app code, or checking a credential string for a "dummy" marker to detect test mode (use a `provider="mock"` ReasoningPack entry and `reasoning.is_mock(language_code)` instead)

**reasoning.is_mock()**:
`reasoning.is_mock(language_code)` -- whether language_code's chain currently resolves to the mock provider. Lets a consuming app gate its *own* mock behavior (e.g. swapping in a fake `MusicManager` for a coherent dummy end-to-end test run) on the same test-mode signal reasoning.py itself uses, without ever inspecting a credential string. The mock response *content* for an app's own bespoke calls (e.g. Daily_Readings' canned meditation text/image descriptions) stays app-specific and lives in the app, guarded by this check -- only the generic "am I in mock mode" question belongs in the SDK.
_Avoid_: reconstructing this by checking a credential value yourself

**Per-frame effect purity**:
Every per-frame visual/audio effect `VideoBuilder` renders (karaoke word-highlight, glow, scrolling-lyrics transition, audio-reactive pulse) must be a pure function of the clip's absolute timestamp `t` — no state carried across successive frames. This is what makes `VideoBuilder.save()`'s chunked parallel rendering (ADR 0013) correct: each worker process renders an independent absolute-time range of the same clip, and the seams are invisible only because no effect depends on what came before it.
_Avoid_: adding an effect that reads or mutates instance state across successive `get_frame(t)` calls (e.g. an accumulating counter, a running average) — it will look correct under today's single-process serial write and then render visibly wrong at chunk boundaries the moment it's split across parallel workers.

**tier**:
`"fast" | "balanced" | "deep"` — the speed/capability level `reasoning.generate()` asks for, not a model name. The mapping to an actual model lives once per provider in code (e.g. `GeminiReasoningPack._TIER_MODELS`), not per language in `reasoning_packs.json`, since it's a property of the model family, not of the language — this is what lets a future model or provider get wired in without any call site changing.
_Avoid_: passing a raw model name (`"gemini-2.5-pro"`, ...) from application code — ask for a tier instead
