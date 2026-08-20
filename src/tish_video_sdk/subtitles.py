"""Subtitle generation: align narration text to synthesized audio, either via
Montreal Forced Aligner (MFA, word-level accuracy, requires an external MFA
install) or proportional time-based splitting (no external dependency, lower
accuracy). Produces timed segments in the shape
:meth:`tish_video_sdk.video_maker.VideoBuilder.with_text_segments` expects.

Per-language config (which MFA model to align with, default text styling) is
a :class:`~tish_video_sdk.internal.providers.subtitle_packs.SubtitlePack`,
loaded once at import -- see that module and ``configuration/subtitle_packs.example.json``.
"""
import os
import json
import wave
import contextlib
import re
import shutil
import subprocess
import tempfile
from bs4 import BeautifulSoup
from typing import Tuple, Optional, List, Dict

from .internal.providers.subtitle_packs import SubtitlePack, DEFAULT_SUBTITLE_PACKS, load_subtitle_packs


def _load_all_subtitle_packs() -> Dict[str, SubtitlePack]:
    """Start from the built-in defaults (so subtitle generation works out of
    the box for fr/en/es/ta) and layer on SUBTITLE_PACKS_PATH (optional --
    see configuration/subtitle_packs.example.json) if set, letting a process add languages
    or override a pack's model/styling without editing this module."""
    packs = dict(DEFAULT_SUBTITLE_PACKS)
    subtitle_packs_path = os.getenv("SUBTITLE_PACKS_PATH")
    if subtitle_packs_path:
        packs.update(load_subtitle_packs(subtitle_packs_path))
    return packs


_ALL_SUBTITLE_PACKS = _load_all_subtitle_packs()


def get_subtitle_pack(language_code: str) -> Optional[SubtitlePack]:
    """The configured SubtitlePack for `language_code`, or None if none is
    configured (via built-in defaults or SUBTITLE_PACKS_PATH)."""
    return _ALL_SUBTITLE_PACKS.get(language_code)


MFA_ERROR_MESSAGE = (
    "Montreal Forced Aligner (MFA) is not installed or configured correctly on this system.\n"
    "To use MFA forced alignment for subtitle generation, you must:\n"
    "1. Install a conda/mamba distribution, e.g. Miniforge:\n"
    "   https://github.com/conda-forge/miniforge\n"
    "2. Create an environment with MFA:\n"
    "   conda create -n mfa -c conda-forge montreal-forced-aligner\n"
    "3. Download the acoustic model and dictionary for each language you use, e.g.:\n"
    "   conda run -n mfa mfa model download acoustic french_mfa\n"
    "   conda run -n mfa mfa model download dictionary french_mfa\n"
    "4. Set the MFA_ENV_PATH environment variable to that environment's root directory\n"
    "   (e.g. C:\\Users\\you\\miniforge3\\envs\\mfa or ~/miniforge3/envs/mfa).\n"
)

# Configuration constants
USE_MFA_ALIGNMENT = True  # Set to True to use MFA forced alignment, False for time-based splitting

# MFA tokens that aren't real words (skipped words, epsilon transitions) --
# shared between fragment-matching and the raw aligned-words list.
_NOISE_TOKENS = {"", "[bracketed]", "<unk>", "<eps>"}


class SubtitleBuilder:
    """Builder for generating subtitles from audio, optionally aligning with reference text."""

    def __init__(self, language_code: str):
        self.language_code = language_code
        self._audio_filepath: Optional[str] = None
        self._reference_text: Optional[str] = None
        self._transcribed_segments: List[Dict] = []
        self._is_built = False
        self._beam: Optional[int] = None
        self._retry_beam: Optional[int] = None
        self._aligned_words: List[Dict] = []
        self._segments: Optional[List[Dict]] = None

    @classmethod
    def from_audio(cls, audio_filepath: str, language_code: str) -> 'SubtitleBuilder':
        builder = cls(language_code)
        builder._audio_filepath = audio_filepath
        return builder

    @classmethod
    def from_audio_with_reference(
        cls,
        audio_filepath: str,
        reference_text: str,
        language_code: str,
        beam: Optional[int] = None,
        retry_beam: Optional[int] = None,
    ) -> 'SubtitleBuilder':
        """
        Args:
            beam, retry_beam: Passed through to `mfa align --beam --retry_beam`.
                Omit to use MFA's own defaults, which can fail to align long
                audio treated as one utterance; widening the beam fixes it
                at the cost of slower alignment.
        """
        builder = cls(language_code)
        builder._audio_filepath = audio_filepath
        builder._reference_text = reference_text
        builder._beam = beam
        builder._retry_beam = retry_beam
        return builder

    @classmethod
    def from_audio_segments_with_reference(
        cls,
        audio_filepath: str,
        segments: List[Dict],
        language_code: str,
        beam: Optional[int] = None,
        retry_beam: Optional[int] = None,
    ) -> 'SubtitleBuilder':
        """
        Align many known-boundary segments of one audio file (e.g. one per
        song line) in a single MFA pass, constraining each segment's search
        to its own window instead of the whole recording -- improves both
        word-level accuracy and placement within long audio.

        Args:
            audio_filepath: One audio file covering all segments.
            segments (List[Dict]): `{"start", "end", "text"}` dicts (seconds),
                in the audio's own timeline, non-overlapping.
            beam, retry_beam: See from_audio_with_reference.
        """
        builder = cls(language_code)
        builder._audio_filepath = audio_filepath
        builder._segments = segments
        builder._beam = beam
        builder._retry_beam = retry_beam
        return builder

    def build(self) -> Optional['SubtitleBuilder']:
        """Align audio with reference text using forced alignment."""
        if self._is_built:
            return self
        if self._audio_filepath is None:
            print("Error: No audio file provided for subtitle generation.")
            return None

        print(f"--- Starting subtitle generation for language '{self.language_code}' ---")

        # Determine generation method
        use_mfa = USE_MFA_ALIGNMENT
        subtitle_pack = get_subtitle_pack(self.language_code)

        if use_mfa and (not subtitle_pack or not subtitle_pack.mfa_model):
            print(
                f"No MFA acoustic model configured for language '{self.language_code}' "
                f"(see SubtitlePack.mfa_model); skipping subtitle generation. Add it via "
                f"SUBTITLE_PACKS_PATH (see configuration/subtitle_packs.example.json) once its model is "
                f"downloaded, or set USE_MFA_ALIGNMENT = False to use time-based splitting instead."
            )
            self._transcribed_segments = []
            self._is_built = True
            return self

        if not use_mfa:
            print(f"Using time-based text splitting for '{self.language_code}' (skipping MFA)...")
            if self._generate_time_based_subtitles():
                self._is_built = True
                print(f"Subtitle generation completed. Generated {len(self._transcribed_segments)} segments.")
                return self
            else:
                print("Error: Time-based subtitle generation failed.")
                return None

        # MFA forced alignment
        if self._segments is not None:
            if not self._segments:
                raise ValueError("MFA forced alignment requires at least one segment. None were provided.")
            mfa_segments = self._align_audio_with_mfa_segments()
        else:
            if not self._reference_text:
                raise ValueError("MFA forced alignment requires reference text. None was provided.")
            mfa_segments = self._align_audio_with_mfa()

        if mfa_segments is None:
            print("Error: MFA alignment failed.")
            return None

        self._transcribed_segments = mfa_segments
        self._is_built = True
        print(f"Subtitle generation completed. Generated {len(self._transcribed_segments)} segments.")
        return self

    def get_segments(self) -> List[Dict]:
        if not self._is_built:
            print("Warning: SubtitleBuilder not built yet. Call build() first.")
            return []
        return self._transcribed_segments

    def get_words(self) -> List[Dict]:
        """The flat, time-ordered `{"word", "start", "end"}` list MFA itself
        produced, unlike get_segments()'s lossy sentence-punctuation
        grouping. Use this when doing your own segmentation over already
        aligned words. Empty until build() has run a real MFA alignment."""
        if not self._is_built:
            print("Warning: SubtitleBuilder not built yet. Call build() first.")
            return []
        return self._aligned_words

    def save_segments_to_json(self, output_filepath: str) -> Optional[str]:
        if not self._is_built:
            print("Error: SubtitleBuilder not built yet. Call build() first.")
            return None
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(self._transcribed_segments, f, ensure_ascii=False, indent=2)
            print(f"Subtitle segments saved to: {output_filepath}")
            return output_filepath
        except Exception as e:
            print(f"Error saving subtitle segments to JSON: {e}")
            return None

    @staticmethod
    def _mfa_executable_and_env(mfa_env_path: str) -> Tuple[str, Dict[str, str]]:
        """Build the mfa executable path and a subprocess environment with the
        conda environment's binary directories on PATH (mirrors what
        `conda activate` would do, without requiring conda itself on PATH)."""
        is_windows = os.name == "nt"
        mfa_exe = (
            os.path.join(mfa_env_path, "Scripts", "mfa.exe")
            if is_windows
            else os.path.join(mfa_env_path, "bin", "mfa")
        )
        bin_dirs = [
            mfa_env_path,
            os.path.join(mfa_env_path, "Library", "mingw-w64", "bin"),
            os.path.join(mfa_env_path, "Library", "usr", "bin"),
            os.path.join(mfa_env_path, "Library", "bin"),
            os.path.join(mfa_env_path, "Scripts"),
            os.path.join(mfa_env_path, "bin"),
        ]
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + env.get("PATH", "")
        return mfa_exe, env

    @staticmethod
    def _normalize_word_for_matching(word: str) -> str:
        """Strip punctuation for matching while keeping any script's letters
        (not just Latin) -- str.isalpha() is Unicode-aware, so this also
        works for Tamil, Cyrillic, CJK, etc."""
        return "".join(c for c in word.lower() if c.isalpha())

    @classmethod
    def _map_mfa_words_to_fragments(cls, fragments: List[str], mfa_words: List[Tuple[float, float, str]]) -> List[Dict]:
        """Reconstruct sentence-level segments from MFA's word-level timestamps
        by walking each fragment's own tokens against MFA's word sequence in
        order. MFA can tokenize differently than a plain .split() (splitting
        hyphenated compounds, tagging unrecognized words as '[bracketed]'), so
        this resyncs on local mismatches rather than assuming a strict 1:1
        count between fragment words and MFA words."""
        frag_tokens: List[Tuple[int, str]] = []
        for frag_idx, frag in enumerate(fragments):
            for w in frag.split():
                n = cls._normalize_word_for_matching(w)
                if n:
                    frag_tokens.append((frag_idx, n))

        mfa_norm = [
            "NOISE" if raw in _NOISE_TOKENS else cls._normalize_word_for_matching(raw)
            for _, _, raw in mfa_words
        ]

        matches: Dict[int, List[Tuple[float, float, str]]] = {}
        i = j = 0
        while i < len(frag_tokens) and j < len(mfa_words):
            frag_idx, g = frag_tokens[i]
            m = mfa_norm[j]
            if g == m or m == "NOISE":
                matches.setdefault(frag_idx, []).append(mfa_words[j])
                i += 1
                j += 1
            elif i + 1 < len(frag_tokens) and frag_tokens[i + 1][1] == m:
                i += 1
            elif j + 1 < len(mfa_words) and mfa_norm[j + 1] == g:
                j += 1
            else:
                i += 1
                j += 1

        segments = []
        for frag_idx, frag_text in enumerate(fragments):
            frag_matches = matches.get(frag_idx)
            if not frag_matches:
                try:
                    print(f"Warning: no MFA alignment found for fragment, skipping: {frag_text!r}")
                except UnicodeEncodeError:
                    print(f"Warning: no MFA alignment found for fragment {frag_idx} (non-ASCII text, {len(frag_text)} chars), skipping")
                continue
            segments.append({
                "text": frag_text,
                "start": frag_matches[0][0],
                "end": frag_matches[-1][1],
                "words": [
                    {"word": w, "start": s, "end": e}
                    for s, e, w in frag_matches if w not in _NOISE_TOKENS
                ],
            })
        return segments

    @staticmethod
    def _clean_plain_text(text: str) -> str:
        """Strip [SFX:...] tags and SSML markup down to plain text and
        collapse whitespace -- shared by the whole-file and segmented MFA
        alignment paths."""
        cleaned_source = re.sub(r'\[SFX:.*?\]', '', text, flags=re.IGNORECASE)

        is_ssml = "<speak>" in cleaned_source
        if is_ssml:
            try:
                soup = BeautifulSoup(cleaned_source, "xml")
                plain_text = soup.get_text(separator=" ", strip=True)
            except Exception as e:
                print(f"Error parsing SSML: {e}. Falling back to plain regex text cleaning.")
                plain_text = re.sub(r'<[^>]+>', '', cleaned_source)
        else:
            plain_text = cleaned_source

        return re.sub(r'\s+', ' ', plain_text).strip()

    @staticmethod
    def _build_utterances_textgrid(duration: float, segments: List[Tuple[float, float, str]]) -> str:
        """A minimal Praat TextGrid with one IntervalTier ("utterances"),
        one interval per segment. Intervals must be contiguous and span
        [0, duration], so gaps between segments get empty-text intervals."""
        intervals: List[Tuple[float, float, str]] = []
        cursor = 0.0
        for start, end, text in segments:
            if start > cursor:
                intervals.append((cursor, start, ""))
            intervals.append((start, end, text))
            cursor = end
        if cursor < duration:
            intervals.append((cursor, duration, ""))

        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            "",
            "xmin = 0",
            f"xmax = {duration}",
            "tiers? <exists>",
            "size = 1",
            "item []:",
            "    item [1]:",
            '        class = "IntervalTier"',
            '        name = "utterances"',
            "        xmin = 0",
            f"        xmax = {duration}",
            f"        intervals: size = {len(intervals)}",
        ]
        for idx, (xmin, xmax, text) in enumerate(intervals, start=1):
            escaped_text = text.replace('"', '""')
            lines += [
                f"        intervals [{idx}]:",
                f"            xmin = {xmin}",
                f"            xmax = {xmax}",
                f'            text = "{escaped_text}"',
            ]
        return "\n".join(lines) + "\n"

    def _align_audio_with_mfa_segments(self) -> Optional[List[Dict]]:
        """Align audio against many known-boundary segments of reference
        text via an MFA TextGrid corpus (one utterance per segment, all
        against the same whole-file WAV) -- see
        from_audio_segments_with_reference()."""
        print(f"DEBUG: Starting segmented MFA forced alignment on audio file: {self._audio_filepath}")
        if not self._audio_filepath or not os.path.exists(self._audio_filepath):
            print(f"Error: Audio file '{self._audio_filepath}' does not exist.")
            return None

        mfa_env_path = os.getenv("MFA_ENV_PATH")
        if not mfa_env_path or not os.path.isdir(mfa_env_path):
            print(MFA_ERROR_MESSAGE)
            return None

        model_name = get_subtitle_pack(self.language_code).mfa_model

        duration = self._get_audio_duration()
        if duration <= 0:
            print("Error: Could not determine audio duration for segmented alignment.")
            return None

        cleaned_segments: List[Tuple[float, float, str]] = []
        for seg in sorted(self._segments, key=lambda s: s["start"]):
            text = self._clean_plain_text(seg["text"])
            start = max(0.0, min(float(seg["start"]), duration))
            end = max(start, min(float(seg["end"]), duration))
            if text and end > start:
                cleaned_segments.append((start, end, text))

        if not cleaned_segments:
            print("Error: No usable segments found for alignment.")
            return None

        mfa_exe, env = self._mfa_executable_and_env(mfa_env_path)
        if not os.path.exists(mfa_exe):
            print(MFA_ERROR_MESSAGE)
            return None

        with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir:
            speaker_dir = os.path.join(corpus_dir, "speaker1")
            os.makedirs(speaker_dir, exist_ok=True)
            utterance_name = "utterance"
            shutil.copy(self._audio_filepath, os.path.join(speaker_dir, f"{utterance_name}.wav"))
            textgrid = self._build_utterances_textgrid(duration, cleaned_segments)
            with open(os.path.join(speaker_dir, f"{utterance_name}.TextGrid"), "w", encoding="utf-8") as f:
                f.write(textgrid)

            output_dir = os.path.join(corpus_dir, "output")
            align_cmd = [
                mfa_exe, "align", corpus_dir, model_name, model_name, output_dir,
                "--clean", "--output_format", "json",
            ]
            if self._beam is not None:
                align_cmd += ["--beam", str(self._beam)]
            if self._retry_beam is not None:
                align_cmd += ["--retry_beam", str(self._retry_beam)]

            print("Running segmented MFA forced alignment...")
            result = subprocess.run(align_cmd, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error during MFA forced alignment:\n{result.stderr[-3000:]}")
                return None
            print("MFA forced alignment completed.")

            json_path = os.path.join(output_dir, "speaker1", f"{utterance_name}.json")
            if not os.path.exists(json_path):
                print(f"Error: MFA did not produce expected output at '{json_path}'.")
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                mfa_data = json.load(f)

            mfa_words = [
                (float(entry[0]), float(entry[1]), entry[2])
                for entry in mfa_data.get("tiers", {}).get("words", {}).get("entries", [])
                if entry[2].strip()
            ]

        if not mfa_words:
            print("Error: MFA produced no word-level alignments.")
            return None

        # Bucket raw (pre-noise-filter) entries by midpoint into their segment.
        segments_out = []
        for start, end, text in cleaned_segments:
            raw_entries = [w for w in mfa_words if start <= (w[0] + w[1]) / 2 <= end]
            expected_words = text.split()
            if len(raw_entries) == len(expected_words):
                # Word count matches, so trust our own text over MFA's label
                # (which can be an uncertain tag like <unk>) and keep its timing.
                seg_words = [
                    {"word": expected_words[i], "start": raw_entries[i][0], "end": raw_entries[i][1]}
                    for i in range(len(expected_words))
                ]
            else:
                seg_words = [
                    {"word": w, "start": s, "end": e}
                    for s, e, w in raw_entries if w not in _NOISE_TOKENS
                ]
            segments_out.append({"text": text, "start": start, "end": end, "words": seg_words})

        self._aligned_words = [w for seg in segments_out for w in seg["words"]]
        return segments_out

    def _align_audio_with_mfa(self) -> Optional[List[Dict]]:
        """Align audio with reference text using Montreal Forced Aligner (MFA)."""
        print(f"DEBUG: Starting MFA forced alignment on audio file: {self._audio_filepath}")
        if not self._audio_filepath or not os.path.exists(self._audio_filepath):
            print(f"Error: Audio file '{self._audio_filepath}' does not exist.")
            return None

        mfa_env_path = os.getenv("MFA_ENV_PATH")
        if not mfa_env_path or not os.path.isdir(mfa_env_path):
            print(MFA_ERROR_MESSAGE)
            return None

        # build() already confirmed a SubtitlePack with an mfa_model exists for
        # this language before ever calling this method.
        model_name = get_subtitle_pack(self.language_code).mfa_model

        plain_text = self._clean_plain_text(self._reference_text)

        # Split into sentence-level fragments (by sentence-ending punctuation)
        sentences = re.split(r'(?<=[.?!])\s+', plain_text)
        fragments = [s.strip() for s in sentences if s.strip()]

        if not fragments:
            print("Error: No text fragments found for alignment.")
            return None

        mfa_exe, env = self._mfa_executable_and_env(mfa_env_path)
        if not os.path.exists(mfa_exe):
            print(MFA_ERROR_MESSAGE)
            return None

        with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir:
            speaker_dir = os.path.join(corpus_dir, "speaker1")
            os.makedirs(speaker_dir, exist_ok=True)
            utterance_name = "utterance"
            shutil.copy(self._audio_filepath, os.path.join(speaker_dir, f"{utterance_name}.wav"))
            with open(os.path.join(speaker_dir, f"{utterance_name}.txt"), "w", encoding="utf-8") as f:
                f.write(plain_text)

            output_dir = os.path.join(corpus_dir, "output")
            align_cmd = [
                mfa_exe, "align", corpus_dir, model_name, model_name, output_dir,
                "--clean", "--output_format", "json",
            ]
            if self._beam is not None:
                align_cmd += ["--beam", str(self._beam)]
            if self._retry_beam is not None:
                align_cmd += ["--retry_beam", str(self._retry_beam)]

            print("Running MFA forced alignment...")
            result = subprocess.run(
                align_cmd,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"Error during MFA forced alignment:\n{result.stderr[-3000:]}")
                return None
            print("MFA forced alignment completed.")

            json_path = os.path.join(output_dir, "speaker1", f"{utterance_name}.json")
            if not os.path.exists(json_path):
                print(f"Error: MFA did not produce expected output at '{json_path}'.")
                return None

            with open(json_path, "r", encoding="utf-8") as f:
                mfa_data = json.load(f)

            mfa_words = [
                (float(entry[0]), float(entry[1]), entry[2])
                for entry in mfa_data.get("tiers", {}).get("words", {}).get("entries", [])
                if entry[2].strip()
            ]

        if not mfa_words:
            print("Error: MFA produced no word-level alignments.")
            return None

        self._aligned_words = [
            {"word": w, "start": s, "end": e}
            for s, e, w in mfa_words if w not in _NOISE_TOKENS
        ]
        return self._map_mfa_words_to_fragments(fragments, mfa_words)

    def _get_audio_duration(self) -> float:
        """Get the duration of the audio file in seconds."""
        if not self._audio_filepath or not os.path.exists(self._audio_filepath):
            return 0.0

        try:
            with contextlib.closing(wave.open(self._audio_filepath, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 0.0

    def _generate_time_based_subtitles(self) -> bool:
        """
        Generate subtitles by splitting reference text and distributing it over audio duration.
        This includes handling SSML pauses (<break>) and removing SFX tags.
        """
        if not self._reference_text:
            print("Error: Reference text is required for time-based subtitle generation.")
            return False

        duration = self._get_audio_duration()
        if duration <= 0:
            print("Error: Invalid audio duration.")
            return False

        # 1. Strip [SFX: ...] tags regardless of SSML or text
        # These are visual cues for generation but shouldn't appear in subtitles
        cleaned_source = re.sub(r'\[SFX:.*?\]', '', self._reference_text, flags=re.IGNORECASE)

        # 2. Check for SSML and Extract Pauses
        total_pause_duration = 0.0
        ssml_blocks = []  # List of {'text': str, 'pause_after': float}

        is_ssml = "<speak>" in cleaned_source

        if is_ssml:
            try:
                # Use BeautifulSoup to parse fragments and breaks
                soup = BeautifulSoup(cleaned_source, "xml")

                # We need to flatten the structure but keep order of text and breaks
                # Since BS4 flattens automatically with .strings, we need to be careful about <break> tags
                # Let's iterate over elements

                # Recursive function to extract text and breaks
                def parse_element(element):
                    items = []
                    for child in element.children:
                        if child.name == 'break':
                            time_str = child.get('time', '0s')
                            # Parse "1.5s" or "500ms"
                            pause_val = 0.0
                            if time_str.endswith('s'):
                                try:
                                    pause_val = float(time_str[:-1])
                                except:
                                    pass
                            elif time_str.endswith('ms'):
                                try:
                                    pause_val = float(time_str[:-2]) / 1000.0
                                except:
                                    pass

                            # Append a break marker
                            items.append({'type': 'break', 'duration': pause_val})

                        elif child.name is None:
                            # NavigableString (Text)
                            text = str(child).strip()
                            if text:
                                items.append({'type': 'text', 'content': text})
                        else:
                            # Other tags (p, prosody, emphasis, etc.) -> recurse
                            items.extend(parse_element(child))
                    return items

                # If there's a speak root, use it, otherwise use soup
                root = soup.find('speak')
                if not root:
                    root = soup

                parsed_items = parse_element(root)

                # Now coalesce text and breaks
                current_text = ""
                for item in parsed_items:
                    if item['type'] == 'text':
                        current_text += " " + item['content']
                    elif item['type'] == 'break':
                        if current_text.strip():
                            ssml_blocks.append({'text': current_text.strip(), 'pause_after': item['duration']})
                            current_text = ""
                            total_pause_duration += item['duration']
                        else:
                            # Break at start or consecutive breaks
                            # Add to last block if exists, otherwise just track total time?
                            # Optimally we need to account for it.
                            # If we have no text yet, it's pre-roll pause.
                            # If we have text, it's post-roll.
                            if ssml_blocks:
                                ssml_blocks[-1]['pause_after'] += item['duration']
                            total_pause_duration += item['duration']

                # Add remaining text
                if current_text.strip():
                    ssml_blocks.append({'text': current_text.strip(), 'pause_after': 0.0})

            except Exception as e:
                print(f"Error parsing SSML: {e}. Falling back to plain text cleaning.")
                # Fallback to simple strip
                soup = BeautifulSoup(cleaned_source, "xml")
                plain_text = soup.get_text(separator=" ", strip=True)
                ssml_blocks = [{'text': plain_text, 'pause_after': 0.0}]

        else:
            # Plain Text
            clean_text = re.sub(r'\s+', ' ', cleaned_source).strip()
            ssml_blocks = [{'text': clean_text, 'pause_after': 0.0}]

        # 3. Time Distribution
        # Effective speaking time
        speaking_duration = duration - total_pause_duration
        if speaking_duration < 0.1:
            speaking_duration = 0.1  # Avoid division by zero or negative time

        # Calculate total characters in all blocks
        total_chars = sum(len(b['text']) for b in ssml_blocks)
        if total_chars == 0:
            print("Error: No text content found after cleaning.")
            return False

        current_time = 0.0
        final_segments = []

        for block in ssml_blocks:
            block_text = block['text']
            pause = block['pause_after']

            # How much time does this block take?
            block_speaking_time = (len(block_text) / total_chars) * speaking_duration

            # Now we split this block into sentence-level chunks for better subtitle granularity
            sentences = re.split(r'(?<=[.?!])\s+', block_text)
            block_chunks = [s.strip() for s in sentences if s.strip()]

            chunk_total_chars = sum(len(c) for c in block_chunks)
            if chunk_total_chars == 0:
                current_time += pause
                continue

            block_start_time = current_time

            for chunk in block_chunks:
                # Proportional time within the block
                chunk_duration = (len(chunk) / chunk_total_chars) * block_speaking_time

                final_segments.append({
                    "text": chunk,
                    "start": current_time,
                    "end": current_time + chunk_duration
                })

                current_time += chunk_duration

            # Add the pause after the block is done
            current_time += pause

        self._transcribed_segments = final_segments
        return True
