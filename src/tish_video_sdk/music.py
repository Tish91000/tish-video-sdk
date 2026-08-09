"""music: background-music selection and mixing (see CONTEXT.md).

MusicManager picks a background-music file by mood -- an explicit mood name,
a Gemini-driven text mood analysis, or a direct file path -- then mixes it
under a speech track via pydub. The set of moods and the mood-analysis prompt
are plain content, not core SDK behavior: MusicManager ships generic,
domain-neutral defaults and lets a project override both via a mood-config
JSON file (see music_moods.example.json), the same MUSIC_MOODS_PATH-style
override every other per-project config in this SDK uses (TTS_VOICE_PACKS_PATH,
SUBTITLE_PACKS_PATH).

When a mood has no local track configured, and a Jamendo API client_id is
configured (opt-in -- see JAMENDO_CLIENT_ID), MusicManager downloads one
free, Creative-Commons-licensed track from Jamendo into bgm_directory so
every later call for that mood finds it locally like any other file. Jamendo
isn't only an automatic fallback: fetch_from_jamendo() exposes it as an
explicit action too, e.g. for a mostly-single-mood daily content series that
wants its bgm_directory pool to keep growing rather than settle on one track
forever. Once a mood has more than one local track, select_audio_music picks
randomly among the ones that satisfy the requested duration.
"""
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

from pydub import AudioSegment

from .internal.providers import music_packs

DEFAULT_MOODS: Dict[str, str] = {
    "calm": "Peaceful, reflective, reassuring -- suited to quiet or contemplative moments.",
    "joy": "Upbeat, celebratory, warm -- suited to positive, uplifting moments.",
    "adventurous": "Energetic, driven, triumphant -- suited to moments of action or overcoming odds.",
    "dark": "Somber, tense, subdued -- suited to serious or sorrowful moments.",
}

DEFAULT_MOOD_PROMPT_TEMPLATE = """
Analyze the following text and determine its primary mood.
Choose one of the following moods: {mood_list}.

Here are descriptions for each mood to guide your choice:
{mood_descriptions}

Text:
"{text}"

Your task:
Respond with only the single word for the chosen mood in lowercase. For example: calm
"""

# Jamendo's fuzzytags search wants one or two short, tag-like words -- too
# many combined tags over-constrains the match and can return nothing (seen
# in practice). A search_query longer than this (a whole paragraph of
# reference_text, or even one of the longer mood descriptions above) gets
# condensed to a single tag via Gemini before being sent, rather than passed
# through as-is (see MusicManager._refine_search_query).
_MAX_TAG_QUERY_CHARS = 40

_KEYWORD_EXTRACTION_PROMPT_TEMPLATE = """
Pick ONE short music genre or mood tag (lowercase, no punctuation) for
searching a royalty-free music library by tag, for an instrumental
background-music track fitting the following text.

The tag must correspond to the text, but also be a real, commonly-used
music genre/mood tag -- not too vague (a generic word like "music" won't
narrow anything down) and not too precise (an unusual or overly specific
phrase is unlikely to have a matching track in the library). Only add a
second tag, comma-separated, if it genuinely adds something the first
tag alone can't capture -- most of the time, one tag is enough.

Text:
"{text}"

Your task:
Respond with only the tag (or two, comma-separated), nothing else. For example: ambient
"""


def _load_mood_config(mood_packs_path: Optional[str]) -> Tuple[Dict[str, str], str]:
    """Load {moods, prompt_template} from mood_packs_path (see
    music_moods.example.json for the shape), falling back to the built-in
    generic defaults for whichever of the two the file doesn't override.
    mood_packs_path itself falls back to the MUSIC_MOODS_PATH env var, then
    to the defaults alone when neither is set."""
    path = mood_packs_path or os.getenv("MUSIC_MOODS_PATH")
    if not path:
        return dict(DEFAULT_MOODS), DEFAULT_MOOD_PROMPT_TEMPLATE

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    moods = config.get("moods", DEFAULT_MOODS)
    prompt_template = config.get("prompt_template", DEFAULT_MOOD_PROMPT_TEMPLATE)
    return moods, prompt_template


class MusicManager:
    """Manages background-music selection and mixing for narrated audio."""

    def __init__(self,
                 bgm_directory: str = "./music/bgm",
                 default_music_file: str = "./music/bgm/default_music.mp3",
                 default_mood: str = "calm",
                 gemini_api_key: str = "",
                 mood_packs_path: Optional[str] = None,
                 jamendo_client_id: str = ""):
        """
        Args:
            bgm_directory: Directory of mood subfolders containing background music files.
            default_music_file: Fallback music file when no mood match is found.
            default_mood: Mood used when no mood can be determined.
            gemini_api_key: Gemini API key for mood analysis. Mood analysis falls
                back to default_mood when omitted.
            mood_packs_path: Path to a JSON file overriding the available moods
                and/or the mood-analysis prompt template (see
                music_moods.example.json). Falls back to the MUSIC_MOODS_PATH
                env var, then to generic built-in defaults.
            jamendo_client_id: Jamendo API client_id (free, see
                https://devportal.jamendo.com/). When set, a mood with no
                usable local track downloads one free, Creative-Commons track
                from Jamendo into bgm_directory instead of falling back to
                default_music_file. Falls back to the JAMENDO_CLIENT_ID env
                var; the feature is simply off when neither is set.
        """
        self.bgm_directory = bgm_directory
        self.default_music_file = default_music_file
        self.default_mood = default_mood
        self.gemini_api_key = gemini_api_key
        self.jamendo_client_id = jamendo_client_id or os.getenv("JAMENDO_CLIENT_ID", "")

        self.available_moods, self._mood_prompt_template = _load_mood_config(mood_packs_path)
        self._loudness_cache: Dict[str, float] = {}

        self._client = None
        if self.gemini_api_key:
            from google import genai
            self._client = genai.Client(api_key=self.gemini_api_key)

    def _load_audio(self, file_path: str) -> AudioSegment:
        """
        Loads an audio file into an AudioSegment.
        If loading fails (e.g., due to missing ffmpeg for mp3),
        attempts to convert the file to WAV format using moviepy.
        """
        # Try normal pydub load first
        try:
            if file_path.lower().endswith(".wav"):
                return AudioSegment.from_wav(file_path)
            else:
                return AudioSegment.from_file(file_path)
        except Exception as e:
            # Fallback to moviepy conversion for compressed formats
            if file_path.lower().endswith((".mp3", ".ogg", ".m4a", ".aac")):
                base, _ = os.path.splitext(file_path)
                cached_wav_path = base + ".wav"

                if os.path.exists(cached_wav_path):
                    try:
                        return AudioSegment.from_wav(cached_wav_path)
                    except Exception:
                        pass  # If corrupt, re-generate

                print(f"pydub failed to load '{file_path}' (likely due to missing ffmpeg/ffprobe): {e}")
                print("Attempting to load/convert using moviepy...")
                try:
                    from moviepy import AudioFileClip
                    clip = AudioFileClip(file_path)
                    clip.write_audiofile(cached_wav_path, logger=None)
                    clip.close()
                    print(f"Successfully converted and saved to '{cached_wav_path}'")
                    return AudioSegment.from_wav(cached_wav_path)
                except Exception as moviepy_err:
                    print(f"Fallback to moviepy failed: {moviepy_err}")
                    raise e
            else:
                raise e

    def _measured_dbfs(self, file_path: str, segment: AudioSegment) -> float:
        """segment's own loudness (dBFS). Cached both in memory for this
        process and in file_path's sidecar cache (see _sidecar_path -- same
        file that Jamendo attribution lives in), so a future call -- in this
        process or a later one, e.g. a fresh batch-processing run reusing the
        same "calm" track -- doesn't re-measure it. The cached value is
        invalidated automatically if file_path's size or mtime changes."""
        if file_path in self._loudness_cache:
            return self._loudness_cache[file_path]

        dbfs = self._read_cached_dbfs(file_path)
        if dbfs is None:
            dbfs = segment.dBFS
            try:
                stat = os.stat(file_path)
                self._write_sidecar(file_path, {"dbfs": dbfs, "size": stat.st_size, "mtime": stat.st_mtime})
            except Exception:
                pass  # best-effort -- a read-only bgm directory shouldn't break mixing

        self._loudness_cache[file_path] = dbfs
        return dbfs

    @staticmethod
    def _sidecar_path(file_path: str) -> str:
        return file_path + ".cache.json"

    @classmethod
    def _read_sidecar(cls, file_path: str) -> dict:
        """Volume analysis and Jamendo attribution share one sidecar file
        per audio file -- returns {} if it doesn't exist yet or is corrupt."""
        sidecar_path = cls._sidecar_path(file_path)
        if not os.path.exists(sidecar_path):
            return {}
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    @classmethod
    def _write_sidecar(cls, file_path: str, updates: dict) -> None:
        """Merge updates into file_path's sidecar cache and write it back --
        read-modify-write so writing loudness data doesn't clobber
        previously-written attribution data, or vice versa."""
        existing = cls._read_sidecar(file_path)
        existing.update(updates)
        with open(cls._sidecar_path(file_path), "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

    @classmethod
    def _read_cached_dbfs(cls, file_path: str) -> Optional[float]:
        cached = cls._read_sidecar(file_path)
        if "dbfs" not in cached:
            return None
        try:
            stat = os.stat(file_path)
            if cached.get("size") != stat.st_size or cached.get("mtime") != stat.st_mtime:
                return None  # source file changed since this was measured
            return cached["dbfs"]
        except Exception:
            return None  # corrupt/unreadable sidecar -- just re-measure

    @staticmethod
    def _music_gain_db(speech_dbfs: float, music_dbfs: float, below_speech_db: float) -> float:
        """Gain (dB) that puts a background-music track's loudness
        below_speech_db under the speech track's own measured loudness --
        replaces a flat subtraction from whatever the track's own source
        volume happens to be, which produces wildly inconsistent relative
        balance across a real bgm library (source tracks commonly vary by
        15+ dB from each other). Falls back to a flat cut when either side
        can't be measured (a fully silent segment measures as -inf dBFS)."""
        if speech_dbfs == float("-inf") or music_dbfs == float("-inf"):
            return -below_speech_db
        return (speech_dbfs - below_speech_db) - music_dbfs

    def get_mood_from_text(self, text: str) -> str:
        """
        Analyzes a text to determine its primary mood using the Gemini API.

        Args:
            text: The passage to be analyzed.

        Returns:
            The determined mood (one of self.available_moods' keys).
            Returns default_mood if the mood cannot be determined or an error occurs.
        """
        if not self._client:
            print("Warning: No Gemini API key provided. Using default mood.")
            return self.default_mood

        # Prepare the mood descriptions for the prompt
        mood_descriptions = "\n".join([f"- {mood.capitalize()}: {desc}" for mood, desc in self.available_moods.items()])
        mood_list = ", ".join(self.available_moods.keys())

        prompt = self._mood_prompt_template.format(
            mood_list=mood_list, mood_descriptions=mood_descriptions, text=text
        )

        # Call the Gemini API
        print("Calling Gemini API for mood analysis...")
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )

            # Clean up the response to get a single word
            determined_mood = (response.text or "").strip().lower()

            # Validate the response
            if determined_mood in self.available_moods:
                print(f"Mood determined: {determined_mood}")
                return determined_mood
            else:
                print(f"Warning: Gemini returned an unexpected mood: '{determined_mood}'")
                return self.default_mood

        except Exception as e:
            print(f"Error calling Gemini API for mood analysis: {e}")
            return self.default_mood

    def select_audio_music(self, mood: str, duration: int, search_query: Optional[str] = None,
                            _allow_jamendo: bool = True) -> str:
        """
        Selects a background music file based on mood and duration.

        Args:
            mood: The mood of the background music -- also the local cache
                classification: wherever the Jamendo fallback below finds a
                track, it's filed under this mood's own folder, regardless
                of what search_query actually found it.
            duration: The desired duration of the background music, in seconds.
            search_query: Extra context to search Jamendo with, if the
                fallback triggers -- combined with mood itself (see
                _refine_search_query), e.g. the original text a mood was
                classified from via get_mood_from_text. Optional; mood alone
                is always a valid Jamendo tag on its own.
            _allow_jamendo: Internal -- caps the Jamendo fallback (see
                jamendo_client_id) to one attempt per call, so a downloaded
                track that somehow still doesn't satisfy the scan below can't
                recurse indefinitely.

        Returns:
            The file path of the selected background music file. When more
            than one local track satisfies duration, one is picked at random
            -- so a mood with a pool of tracks (see fetch_from_jamendo)
            doesn't always return the same one.
        """
        mood_directory = os.path.join(self.bgm_directory, mood)

        if not os.path.exists(mood_directory):
            if _allow_jamendo and self._fetch_from_jamendo(mood, duration, mood_directory, search_query):
                return self.select_audio_music(mood, duration, _allow_jamendo=False)
            print(f"Warning: Mood directory does not exist: {mood_directory}. Using default music.")
            return self.default_music_file

        # Select a music file based on duration and mood
        longest_music_file = None
        longest_duration = 0
        duration_matches: List[str] = []

        files = os.listdir(mood_directory)
        audio_files = []
        for f in files:
            if f.endswith(".wav"):
                audio_files.append(f)
            elif f.endswith(".mp3"):
                wav_counterpart = f[:-4] + ".wav"
                if wav_counterpart not in files:
                    audio_files.append(f)

        for filename in audio_files:
            music_file_path = os.path.join(mood_directory, filename)
            try:
                audio_segment = self._load_audio(music_file_path)

                # If loading converted the file to wav, use the wav path
                resolved_path = music_file_path
                if music_file_path.lower().endswith(".mp3"):
                    wav_path = music_file_path[:-4] + ".wav"
                    if os.path.exists(wav_path):
                        resolved_path = wav_path

                if len(audio_segment) > longest_duration:
                    longest_duration = len(audio_segment)
                    longest_music_file = resolved_path

                if len(audio_segment) >= duration * 1000:  # Check duration in milliseconds
                    duration_matches.append(resolved_path)
            except Exception as e:
                print(f"Error loading audio file {music_file_path}: {e}")
                continue

        if duration_matches:
            return random.choice(duration_matches)

        if longest_music_file:
            print(f"No suitable music found for duration '{duration}s'. Using longest available: {longest_music_file}")
            return longest_music_file

        if _allow_jamendo and self._fetch_from_jamendo(mood, duration, mood_directory, search_query):
            return self.select_audio_music(mood, duration, _allow_jamendo=False)

        default_resolved = self.default_music_file
        if default_resolved.lower().endswith(".mp3"):
            wav_path = default_resolved[:-4] + ".wav"
            if os.path.exists(wav_path):
                default_resolved = wav_path
        return default_resolved

    def fetch_from_jamendo(self, mood: str, duration: int = 60, search_query: Optional[str] = None) -> Optional[str]:
        """Explicitly search Jamendo and download one more track for mood
        into bgm_directory -- unlike select_audio_music's automatic
        fallback (which only reaches Jamendo when nothing local already
        satisfies a lookup), this always attempts a fresh search regardless
        of what's already local. Meant to be called deliberately, separately
        from ordinary playback/mixing calls -- e.g. a mostly-single-mood
        daily content series (a "meditative" reading published once a day,
        say) calling this occasionally to grow that mood's local pool,
        rather than settling on whichever single track the first automatic
        fallback happened to find. Combined with a pool of more than one
        local track, select_audio_music then picks among them at random.

        Args:
            mood: Which mood's local pool (bgm_directory/mood/) to add to.
            duration: Minimum track length to search for, in seconds.
            search_query: Extra context to search with, combined with mood
                itself (see _refine_search_query) -- e.g. today's specific
                reading text, so the query (and likely the result) varies
                by day instead of always resolving to the same tags.

        Returns:
            The downloaded track's local path, or None if jamendo_client_id
            isn't configured, nothing matched, the download failed, or the
            search happened to land on a track already downloaded before.
        """
        if not self.jamendo_client_id:
            return None

        mood_directory = os.path.join(self.bgm_directory, mood)
        return self._fetch_from_jamendo(mood, duration, mood_directory, search_query)

    def _refine_search_query(self, search_query: Optional[str], mood: str) -> str:
        """Build the tag list Jamendo is actually searched with. Always
        includes mood itself -- a reliable plain tag on its own, and keeps
        results grounded in the right general category -- plus extra context
        when search_query carries more than that: as-is when it's already
        short/tag-like (an explicit custom query), or condensed to one
        well-chosen tag via Gemini when it's long prose (e.g. reference_text),
        skipped entirely if that condensing isn't available. Combining tags
        this way is what actually varies results per input text -- two
        different reference texts classified to the same mood search
        differently -- rather than relying on picking randomly from one
        static query's results. Deliberately stays to at most a couple of
        tags total: combining too many over-constrains Jamendo's match and
        can return nothing, even when each tag individually has results."""
        tags = [mood]

        extra = (search_query or "").strip()
        if extra and extra != mood:
            if len(extra) <= _MAX_TAG_QUERY_CHARS:
                tags.append(extra)
            elif self._client:
                keywords = self._gemini_extract_search_keywords(extra)
                if keywords:
                    tags.append(keywords)

        return ", ".join(tags)

    def _gemini_extract_search_keywords(self, text: str) -> Optional[str]:
        """Ask Gemini for one (occasionally two) music genre/mood tag
        describing text -- specific enough to correspond to the text, but
        common enough to actually have matches in a real music library, not
        an invented or overly narrow phrase. Returns None on any failure so
        the caller can fall back to something safer."""
        prompt = _KEYWORD_EXTRACTION_PROMPT_TEMPLATE.format(text=text)
        try:
            response = self._client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
            )
            keywords = (response.text or "").strip()
            if keywords:
                print(f"Gemini condensed the search text into a tag: '{keywords}'")
            return keywords or None
        except Exception as e:
            print(f"Gemini search-keyword extraction failed: {e}")
            return None

    def _fetch_from_jamendo(self, mood: str, duration: int, mood_directory: str,
                             search_query: Optional[str]) -> Optional[str]:
        """Look up and download one Jamendo track into mood_directory when no
        local track satisfies mood -- opt-in via jamendo_client_id/
        JAMENDO_CLIENT_ID, a no-op otherwise. search_query drives the actual
        Jamendo lookup and can be richer than the bare mood name; the result
        is classified into mood_directory (mood) regardless of what query
        found it, so every later local lookup for this mood finds it too.
        Downloaded tracks land directly in mood_directory so the normal local
        scan above picks them up; nothing here re-fetches once a file already
        exists on disk. Never raises -- any failure (network, no match, bad
        download) just returns None so the caller falls through to its
        existing default-music behavior.

        Returns the downloaded (or already-cached) track's local path, or
        None on any failure.
        """
        if not self.jamendo_client_id:
            return None

        query = self._refine_search_query(search_query, mood)

        try:
            track = music_packs.search_track(query, duration, self.jamendo_client_id)
        except Exception as e:
            print(f"Jamendo lookup failed for mood '{mood}' (query '{query}'): {e}")
            return None
        if track is None:
            return None

        safe_name = re.sub(r"[^\w.-]+", "_", f"{track.id}_{track.name}")
        dest_path = os.path.join(mood_directory, f"jamendo_{safe_name}.mp3")
        if os.path.exists(dest_path):
            return dest_path  # already downloaded by an earlier call

        try:
            os.makedirs(mood_directory, exist_ok=True)
            music_packs.download_track(track, dest_path)
        except Exception as e:
            print(f"Jamendo download failed for mood '{mood}': {e}")
            return None

        print(f"Downloaded '{track.name}' by {track.artist_name} from Jamendo for mood '{mood}'.")
        self._write_attribution(dest_path, track)
        return dest_path

    @classmethod
    def _write_attribution(cls, file_path: str, track: "music_packs.JamendoTrack") -> None:
        """Record a downloaded track's credit/license info in file_path's
        sidecar cache (shared with the loudness cache -- see _sidecar_path)
        -- CC licenses commonly require attribution, so this keeps it
        discoverable alongside the audio rather than only in a log line."""
        try:
            cls._write_sidecar(file_path, {
                "attribution": {
                    "source": "jamendo",
                    "track_id": track.id,
                    "name": track.name,
                    "artist_name": track.artist_name,
                    "license_ccurl": track.license_ccurl,
                },
            })
        except Exception:
            pass  # best-effort -- a read-only bgm directory shouldn't break mixing

    def get_music_path(self,
                        music_input: Union[str, None] = None,
                        reference_text: str = "",
                        duration: int = 60) -> str:
        """
        Unified method to get a music file path based on different input types.

        Args:
            music_input: A file path, or a mood name -- not limited to
                available_moods' pre-registered keys; select_audio_music
                already handles an unregistered mood gracefully (local
                lookup by that name, then the Jamendo fallback if
                configured, then default_music_file), so any name works.
            reference_text: Text to analyze for mood if music_input is None.
            duration: Duration in seconds for music selection.

        Returns:
            Path to the selected music file.
        """
        # If music_input is a file path that exists, return it
        if music_input and os.path.isfile(music_input):
            print(f"Using provided music file: {music_input}")
            return music_input

        # Any other non-empty music_input is used as a mood name directly.
        if music_input:
            mood = music_input.lower()
            print(f"Using provided mood: {mood}")
            return self.select_audio_music(mood, duration)

        # If reference_text is provided, analyze it for mood -- if the
        # Jamendo fallback ends up triggering, search with this richer text
        # rather than just the classified mood name, while still filing
        # whatever it finds under that mood for future local reuse.
        if reference_text:
            mood = self.get_mood_from_text(reference_text)
            return self.select_audio_music(mood, duration, search_query=reference_text)

        # Fallback to default mood
        print(f"No valid input provided. Using default mood: {self.default_mood}")
        return self.select_audio_music(self.default_mood, duration)

    def add_background_music(self,
                              speech_audio_filepath: str,
                              output_audio_filepath: str,
                              background_music_volume_reduction_db: float = 10.0,
                              music_input: Union[str, None] = None,
                              reference_text_for_mood: str = "") -> None:
        """
        Overlays background music onto a speech audio file and saves the combined audio.

        Args:
            speech_audio_filepath: The file path to the main speech audio.
            output_audio_filepath: The file path where the mixed audio will be saved.
            background_music_volume_reduction_db: Target loudness gap, in dB, between the
                background music and the speech track's own measured loudness (not a flat
                cut from the track's own volume -- each file's loudness is measured and
                normalized to sit this far under the speech).
            music_input: Music file path, mood name, or None for auto-selection.
            reference_text_for_mood: Reference text to determine mood if needed.
        """
        print(f"Attempting to add background music to '{speech_audio_filepath}'...")

        # Load speech audio to get duration
        try:
            speech_audio = AudioSegment.from_file(speech_audio_filepath)
            speech_duration_seconds = len(speech_audio) // 1000
            print(f"Loaded speech audio: '{speech_audio_filepath}' (Duration: {len(speech_audio)} ms)")
        except Exception as e:
            raise Exception(f"Error loading speech audio '{speech_audio_filepath}': {e}")

        # Get background music path using unified method
        background_music_filepath = self.get_music_path(
            music_input=music_input,
            reference_text=reference_text_for_mood,
            duration=speech_duration_seconds
        )

        # Load background music
        try:
            background_music = self._load_audio(background_music_filepath)
            print(f"Loaded background music: '{background_music_filepath}' (Duration: {len(background_music)} ms)")
        except Exception as e:
            raise Exception(f"Error loading background music '{background_music_filepath}': {e}")

        # Measure loudness before any looping/trimming distorts it
        speech_dbfs = self._measured_dbfs(speech_audio_filepath, speech_audio)
        music_dbfs = self._measured_dbfs(background_music_filepath, background_music)

        # Adjust background music length
        target_music_length = len(speech_audio) + 1000  # 1 second buffer
        if len(background_music) < target_music_length:
            loops_needed = (target_music_length // len(background_music)) + 1
            background_music = background_music * loops_needed
            print(f"Background music looped {loops_needed} times to match speech duration.")

        # Trim the background music to exactly match the target length
        background_music = background_music[:target_music_length]
        print(f"Background music trimmed to {len(background_music)} ms.")

        # Adjust background music volume: land it background_music_volume_reduction_db
        # below the speech's own measured loudness, not a flat cut from whatever the
        # track's own source volume happens to be.
        gain_db = self._music_gain_db(speech_dbfs, music_dbfs, background_music_volume_reduction_db)
        background_music = background_music.apply_gain(gain_db)
        print(f"Background music gain adjusted by {gain_db:.1f} dB (targeting {background_music_volume_reduction_db} dB under measured speech loudness).")

        # Overlay audio
        mixed_audio = speech_audio.overlay(background_music)
        print(f"Audio mixed. Combined duration: {len(mixed_audio)} ms.")

        # Export final audio
        try:
            output_dir = os.path.dirname(output_audio_filepath)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")

            mixed_audio.export(output_audio_filepath, format="wav")
            print(f"Final mixed audio successfully saved to: '{output_audio_filepath}'")
        except Exception as e:
            raise Exception(f"Error exporting mixed audio to '{output_audio_filepath}': {e}")

    def get_audio_with_bgm_from_mood(self,
                                      speech_audio_filepath: str,
                                      mood: str,
                                      background_music_volume_reduction_db: float = 10.0) -> AudioSegment:
        """
        Get audio with background music applied based on mood.

        Args:
            speech_audio_filepath: Path to the speech audio file.
            mood: The mood for background music selection.
            background_music_volume_reduction_db: Target loudness gap, in dB, between the
                background music and the speech track's own measured loudness (not a flat
                cut from the track's own volume -- each file's loudness is measured and
                normalized to sit this far under the speech).

        Returns:
            The mixed audio with background music.
        """
        # Load speech audio
        speech_audio = AudioSegment.from_file(speech_audio_filepath)
        speech_duration_seconds = len(speech_audio) // 1000

        # Get background music for the specified mood
        background_music_filepath = self.select_audio_music(mood, speech_duration_seconds)

        # Load and process background music
        background_music = self._load_audio(background_music_filepath)

        # Measure loudness before any looping/trimming distorts it
        speech_dbfs = self._measured_dbfs(speech_audio_filepath, speech_audio)
        music_dbfs = self._measured_dbfs(background_music_filepath, background_music)

        # Adjust background music length and volume
        target_music_length = len(speech_audio) + 1000
        if len(background_music) < target_music_length:
            loops_needed = (target_music_length // len(background_music)) + 1
            background_music = background_music * loops_needed

        background_music = background_music[:target_music_length]
        gain_db = self._music_gain_db(speech_dbfs, music_dbfs, background_music_volume_reduction_db)
        background_music = background_music.apply_gain(gain_db)

        # Mix and return
        return speech_audio.overlay(background_music)

    def get_audio_with_bgm_from_text(self,
                                      speech_audio_filepath: str,
                                      reference_text: str,
                                      background_music_volume_reduction_db: float = 10.0) -> AudioSegment:
        """
        Get audio with background music applied based on text analysis.

        Args:
            speech_audio_filepath: Path to the speech audio file.
            reference_text: Text to analyze for mood determination.
            background_music_volume_reduction_db: Target loudness gap, in dB, between the
                background music and the speech track's own measured loudness (not a flat
                cut from the track's own volume -- each file's loudness is measured and
                normalized to sit this far under the speech).

        Returns:
            The mixed audio with background music.
        """
        # Analyze text to determine mood
        mood = self.get_mood_from_text(reference_text)

        # Use the mood-based method
        return self.get_audio_with_bgm_from_mood(
            speech_audio_filepath,
            mood,
            background_music_volume_reduction_db
        )

    def get_audio_with_bgm_from_file(self,
                                      speech_audio_filepath: str,
                                      background_music_filepath: str,
                                      background_music_volume_reduction_db: float = 10.0) -> AudioSegment:
        """
        Get audio with background music applied from a specific file.

        Args:
            speech_audio_filepath: Path to the speech audio file.
            background_music_filepath: Path to the background music file.
            background_music_volume_reduction_db: Target loudness gap, in dB, between the
                background music and the speech track's own measured loudness (not a flat
                cut from the track's own volume -- each file's loudness is measured and
                normalized to sit this far under the speech).

        Returns:
            The mixed audio with background music.
        """
        # Load both audio files
        speech_audio = self._load_audio(speech_audio_filepath)
        background_music = self._load_audio(background_music_filepath)

        # Measure loudness before any looping/trimming distorts it
        speech_dbfs = self._measured_dbfs(speech_audio_filepath, speech_audio)
        music_dbfs = self._measured_dbfs(background_music_filepath, background_music)

        # Adjust background music length and volume
        target_music_length = len(speech_audio) + 1000
        if len(background_music) < target_music_length:
            loops_needed = (target_music_length // len(background_music)) + 1
            background_music = background_music * loops_needed

        background_music = background_music[:target_music_length]
        gain_db = self._music_gain_db(speech_dbfs, music_dbfs, background_music_volume_reduction_db)
        background_music = background_music.apply_gain(gain_db)

        # Mix and return
        return speech_audio.overlay(background_music)

    def get_audio_with_bgm(self,
                            speech_audio_filepath: str,
                            music_input: Union[str, None] = None,
                            reference_text: str = "",
                            background_music_volume_reduction_db: float = 10.0) -> AudioSegment:
        """
        Unified method to get audio with background music applied based on different input types.

        Args:
            speech_audio_filepath: Path to the speech audio file.
            music_input: A file path, a mood name, or None.
            reference_text: Text to analyze for mood if music_input is None.
            background_music_volume_reduction_db: Target loudness gap, in dB, between the
                background music and the speech track's own measured loudness (not a flat
                cut from the track's own volume -- each file's loudness is measured and
                normalized to sit this far under the speech).

        Returns:
            The mixed audio with background music.
        """
        # If music_input is a file path that exists, use file-based method
        if music_input and os.path.isfile(music_input):
            return self.get_audio_with_bgm_from_file(
                speech_audio_filepath,
                music_input,
                background_music_volume_reduction_db
            )

        # If music_input is a valid mood, use mood-based method
        if music_input and music_input.lower() in self.available_moods:
            return self.get_audio_with_bgm_from_mood(
                speech_audio_filepath,
                music_input.lower(),
                background_music_volume_reduction_db
            )

        # If reference_text is provided, use text-based method
        if reference_text:
            return self.get_audio_with_bgm_from_text(
                speech_audio_filepath,
                reference_text,
                background_music_volume_reduction_db
            )

        # Fallback to default mood
        return self.get_audio_with_bgm_from_mood(
            speech_audio_filepath,
            self.default_mood,
            background_music_volume_reduction_db
        )
