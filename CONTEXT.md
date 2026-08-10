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
