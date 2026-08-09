import os
import re
import json
import io
import hashlib
from typing import Optional, List
from dotenv import load_dotenv
from pydub import AudioSegment

from .internal.providers.voice_packs import VoicePack, VoicePackSynthesisError, SynthesisResult, load_voice_packs

# --- Configuration ---
# TTS_VOICE_PACKS_PATH points at a JSON file listing every VoicePack (provider
# x language combination) this process might use -- see
# voice_packs.example.json for the expected shape. All entries are loaded once
# at import time; each TTSBuilder call filters them down to the language (and
# optionally a single pinned provider) it actually wants, via
# VoicePack.filter_chain() -- see TTSBuilder.__init__. TTS_LANGUAGE_CODE is an
# optional default language_code for callers that don't pass one explicitly.

load_dotenv()


def _load_all_voice_packs() -> tuple:
    default_language_code = os.getenv("TTS_LANGUAGE_CODE")

    voice_packs_path = os.getenv("TTS_VOICE_PACKS_PATH")
    if not voice_packs_path:
        raise RuntimeError(
            "TTS_VOICE_PACKS_PATH is not set. Point it at a voice_packs.json listing "
            "the VoicePacks available to this process -- see voice_packs.example.json."
        )

    all_packs = load_voice_packs(voice_packs_path)
    return default_language_code, all_packs


_DEFAULT_LANGUAGE_CODE, _ALL_VOICE_PACKS = _load_all_voice_packs()


def _normalize_to_wav(raw_bytes: bytes, mime_type: str) -> bytes:
    """Convert a VoicePack's raw synthesis output to a consistent WAV format
    (24kHz, 16-bit, mono), regardless of which provider produced it."""
    try:
        if "wav" in mime_type:
            audio_segment = AudioSegment.from_wav(io.BytesIO(raw_bytes))
        elif "mpeg" in mime_type or "mp3" in mime_type:
            audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes), format="mp3")
        elif "pcm" in mime_type or "l16" in mime_type:
            audio_segment = AudioSegment(data=raw_bytes, sample_width=2, frame_rate=24000, channels=1)
        else:
            audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes))

        audio_segment = audio_segment.set_frame_rate(24000).set_channels(1).set_sample_width(2)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        return wav_io.getvalue()
    except Exception as e:
        print(f"Error normalizing audio (mime_type={mime_type}): {e}. Saving raw bytes as a fallback.")
        return raw_bytes


def _chunk_content(content: str, is_ssml: bool, synthesis_character_limit: int, max_sentence_character_limit: int) -> List[str]:
    """Divide content into chunks that fit the current VoicePack's own
    character limits."""
    chunks = []

    if is_ssml:
        for i in range(0, len(content), synthesis_character_limit):
            chunks.append(content[i:i + synthesis_character_limit])
        return chunks

    paragraphs = content.split('\n\n')
    current_chunk = ""

    for paragraph in paragraphs:
        stripped_paragraph = paragraph.strip()
        if not stripped_paragraph:
            continue

        if (len(current_chunk) + len(stripped_paragraph) + 2 > synthesis_character_limit and current_chunk) or \
           len(stripped_paragraph) > synthesis_character_limit:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = ""

        processed_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', stripped_paragraph)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_sentence_character_limit:
                words = sentence.split(' ')
                temp_part = ""
                for word in words:
                    if len(temp_part) + len(word) + 1 > max_sentence_character_limit and temp_part:
                        processed_sentences.append(temp_part.strip())
                        temp_part = word + ' '
                    else:
                        temp_part += word + ' '
                if temp_part:
                    processed_sentences.append(temp_part.strip())
            else:
                processed_sentences.append(sentence)

        processed_paragraph = " ".join(processed_sentences)

        if current_chunk:
            current_chunk += "\n\n"
        current_chunk += processed_paragraph

        if len(current_chunk) > synthesis_character_limit:
            chunks.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class TTSBuilder:
    """
    A flexible TTS (Text-to-Speech) builder that synthesizes speech from text
    by trying an ordered chain of VoicePacks (providers) for a given language,
    falling back to the next on a provider-level failure. One process can
    build TTSBuilders for multiple languages -- language_code (and optionally
    provider) are filters applied per instance, not a process-wide constraint.
    """

    def __init__(self, language_code: Optional[str] = None, provider: Optional[str] = None, use_llm_ssml: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize the TTSBuilder.

        Args:
            language_code (str, optional): The language code. Defaults to TTS_LANGUAGE_CODE if not given.
            provider (str, optional): Pin to exactly one provider (e.g. "gemini"), disabling fallback to any other VoicePack for this language. Omit to use the full ordered chain.
            use_llm_ssml (bool): Whether to attempt SSML conversion on VoicePacks that support it. Default True.
            cache_dir (str, optional): Directory to store cached audio files.
        """
        language_code = language_code or _DEFAULT_LANGUAGE_CODE
        if not language_code:
            raise ValueError(
                "No language_code was given and TTS_LANGUAGE_CODE is not set as a default. "
                "Pass language_code explicitly or set TTS_LANGUAGE_CODE in .env."
            )

        self.language_code = language_code
        self.provider = provider
        self.use_llm_ssml = use_llm_ssml
        self.cache_dir = cache_dir
        self._voice_pack_chain = VoicePack.filter_chain(_ALL_VOICE_PACKS, language_code, provider)

        self._content_text: Optional[str] = None
        self._processed_content: Optional[str] = None
        self._audio_chunks: List[bytes] = []
        self._is_built = False
        self.used_provider: Optional[str] = None
        self._used_voice_pack: Optional[VoicePack] = None

    @classmethod
    def from_text(cls, text: str, language_code: Optional[str] = None, provider: Optional[str] = None, use_llm_ssml: bool = True, cache_dir: Optional[str] = None) -> 'TTSBuilder':
        """
        Create a TTSBuilder from text input.

        Args:
            text (str): The input text to be synthesized.
            language_code (str, optional): The language code. Defaults to TTS_LANGUAGE_CODE if not given.
            provider (str, optional): Pin to exactly one provider, disabling fallback.
            use_llm_ssml (bool): Whether to attempt SSML conversion on VoicePacks that support it.
            cache_dir (str, optional): Directory to store cached audio.

        Returns:
            TTSBuilder: Configured TTSBuilder instance.
        """
        builder = cls(language_code, provider=provider, use_llm_ssml=use_llm_ssml, cache_dir=cache_dir)
        builder._content_text = text
        return builder

    @classmethod
    def available_providers(cls, language_code: Optional[str] = None) -> List[str]:
        """
        List the providers configured for `language_code` (or the default
        TTS_LANGUAGE_CODE if not given), in voice_packs.json order.

        Args:
            language_code (str, optional): The language code. Defaults to TTS_LANGUAGE_CODE if not given.

        Returns:
            List[str]: Provider names, e.g. ["gemini", "google_cloud"].
        """
        language_code = language_code or _DEFAULT_LANGUAGE_CODE
        if not language_code:
            raise ValueError(
                "No language_code was given and TTS_LANGUAGE_CODE is not set as a default."
            )
        return [pack.provider for pack in VoicePack.filter_chain(_ALL_VOICE_PACKS, language_code)]

    @classmethod
    def concatenate(cls, tts_builders: List['TTSBuilder']) -> Optional['TTSBuilder']:
        """
        Concatenate multiple built TTSBuilder instances into a single instance.

        Args:
            tts_builders (List[TTSBuilder]): List of built TTSBuilder instances to concatenate.
                                             Must all use the same language_code.

        Returns:
            Optional[TTSBuilder]: A new TTSBuilder instance with concatenated content,
                                 or None if concatenation fails.
        """
        if not tts_builders:
            raise ValueError("Cannot concatenate empty list of TTSBuilder instances.")

        for i, builder in enumerate(tts_builders):
            if not builder._is_built:
                raise ValueError(f"TTSBuilder at index {i} is not built. Call build() first.")

        first_language = tts_builders[0].language_code
        for i, builder in enumerate(tts_builders):
            if builder.language_code != first_language:
                raise ValueError(
                    f"Language mismatch: TTSBuilder at index {i} has language '{builder.language_code}' "
                    f"but expected '{first_language}'. All instances must use the same language."
                )

        print(f"Concatenating {len(tts_builders)} TTSBuilder instances...")

        new_builder = cls(first_language)

        all_pcm_data = []

        for builder_idx, builder in enumerate(tts_builders):
            for chunk_idx, chunk in enumerate(builder._audio_chunks):
                if chunk[:4] == b'RIFF':
                    pos = 12
                    pcm_data = None

                    while pos < len(chunk):
                        if pos + 8 > len(chunk):
                            break

                        chunk_id = chunk[pos:pos+4]
                        chunk_size = int.from_bytes(chunk[pos+4:pos+8], 'little')

                        if chunk_id == b'data':
                            pcm_data = chunk[pos+8:pos+8+chunk_size]
                            break
                        else:
                            pos += 8 + chunk_size

                    if pcm_data:
                        all_pcm_data.append(pcm_data)
                        print(f"  Extracted {len(pcm_data):,} bytes of PCM data from builder {builder_idx+1}, chunk {chunk_idx+1}")
                    else:
                        print(f"  Warning: Could not find data chunk in builder {builder_idx+1}, chunk {chunk_idx+1}")
                else:
                    all_pcm_data.append(chunk)
                    print(f"  Using raw PCM data from builder {builder_idx+1}, chunk {chunk_idx+1}")

        combined_pcm = b''.join(all_pcm_data)
        total_pcm_size = len(combined_pcm)
        print(f"  Total PCM data: {total_pcm_size:,} bytes")

        sample_rate = 24000
        bits_per_sample = 16
        num_channels = 1
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

        wav_header = io.BytesIO()
        wav_header.write(b'RIFF')
        wav_header.write((36 + total_pcm_size).to_bytes(4, 'little'))
        wav_header.write(b'WAVE')
        wav_header.write(b'fmt ')
        wav_header.write((16).to_bytes(4, 'little'))
        wav_header.write((1).to_bytes(2, 'little'))
        wav_header.write(num_channels.to_bytes(2, 'little'))
        wav_header.write(sample_rate.to_bytes(4, 'little'))
        wav_header.write(byte_rate.to_bytes(4, 'little'))
        wav_header.write(block_align.to_bytes(2, 'little'))
        wav_header.write(bits_per_sample.to_bytes(2, 'little'))
        wav_header.write(b'data')
        wav_header.write(total_pcm_size.to_bytes(4, 'little'))

        complete_wav = wav_header.getvalue() + combined_pcm
        new_builder._audio_chunks = [complete_wav]

        print(f"  Created single WAV file: {len(complete_wav):,} bytes")

        processed_contents = []
        is_ssml = False
        if tts_builders and tts_builders[0]._processed_content and tts_builders[0]._processed_content.strip().startswith('<speak>'):
            is_ssml = True

        for b in tts_builders:
            if b._processed_content:
                content = b._processed_content.strip()
                if is_ssml and content.startswith('<speak>') and content.endswith('</speak>'):
                    content = content[7:-8].strip()
                processed_contents.append(content)

        if processed_contents:
            combined_content = "\n\n".join(processed_contents)
            if is_ssml:
                new_builder._processed_content = f"<speak>{combined_content}</speak>"
            else:
                new_builder._processed_content = combined_content

        new_builder._is_built = True

        print("Concatenation completed successfully.")
        return new_builder

    def build(self) -> Optional['TTSBuilder']:
        """
        Build the TTS synthesis by trying each VoicePack in the configured
        chain, in order, until one succeeds.

        Returns:
            Optional[TTSBuilder]: Self if successful, None if every VoicePack failed.
        """
        if self._content_text is None:
            print("Error: No text content provided for synthesis.")
            return None

        if self.cache_dir:
            cache_hash = self._calculate_current_hash()
            print(f"Checking cache (Hash: {cache_hash})...")

            cached_builder = self.load_from_cache(self.cache_dir, cache_hash)
            if cached_builder:
                print("Cache Hit! Loading TTS from cache.")
                self._audio_chunks = cached_builder._audio_chunks
                self._processed_content = cached_builder._processed_content
                self._is_built = True
                self.used_provider = "cache"
                return self

        print(f"--- Starting TTS synthesis for language '{self.language_code}' ---")

        for voice_pack in self._voice_pack_chain:
            provider = voice_pack.provider
            try:
                print(f"Trying VoicePack '{provider}'...")
                voice_pack.load_model()

                is_ssml = voice_pack.supports_ssml and self.use_llm_ssml
                if is_ssml:
                    print(f"Converting text to SSML via '{provider}'...")
                    try:
                        self._processed_content = voice_pack.convert_to_ssml(self._content_text)
                    except VoicePackSynthesisError:
                        raise
                    except Exception as e:
                        print(f"SSML conversion failed ({e}), falling back to plain text for this VoicePack.")
                        is_ssml = False
                        self._processed_content = self._content_text
                else:
                    self._processed_content = self._content_text

                content_chunks = _chunk_content(
                    self._processed_content if is_ssml else self._content_text,
                    is_ssml,
                    voice_pack.synthesis_character_limit,
                    voice_pack.max_sentence_character_limit,
                )
                if not content_chunks:
                    print(f"Error: No valid content chunks generated for '{provider}'.")
                    continue

                audio_chunks = []
                for i, chunk in enumerate(content_chunks):
                    if not chunk.strip():
                        continue
                    print(f"Synthesizing chunk {i+1}/{len(content_chunks)} via '{provider}'...")
                    result = voice_pack.synthesize(chunk, is_ssml)
                    audio_chunks.append(_normalize_to_wav(result.audio_bytes, result.mime_type))

                if not audio_chunks:
                    print(f"Error: No audio chunks were synthesized via '{provider}'.")
                    continue

                self._audio_chunks = audio_chunks
                self.used_provider = provider
                self._used_voice_pack = voice_pack
                self._is_built = True

                if self.cache_dir:
                    print("Saving to TTS Cache...")
                    self.save_to_cache(self.cache_dir, cache_filename=self._calculate_current_hash())

                print(f"TTS synthesis completed successfully using VoicePack: {provider}")
                return self

            except VoicePackSynthesisError as e:
                print(f"VoicePack '{provider}' failed ({type(e).__name__}: {e}). Trying next VoicePack...")
                continue

        print("Error: All VoicePacks in the chain failed.")
        return None

    def save(self, output_filepath: str) -> Optional[str]:
        """
        Save the synthesized audio to a file.

        Args:
            output_filepath (str): The path where the audio file will be saved.

        Returns:
            Optional[str]: The path to the saved audio file if successful, None otherwise.
        """
        if not self._is_built:
            print("Error: TTS not built yet. Call build() first.")
            return None

        if not self._audio_chunks:
            print("Error: No audio content to save.")
            return None

        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        print(f"Saving {len(self._audio_chunks)} audio chunks to '{output_filepath}'...")
        try:
            if len(self._audio_chunks) == 1:
                with open(output_filepath, "wb") as out_file:
                    out_file.write(self._audio_chunks[0])
            else:
                combined_audio = None
                for chunk in self._audio_chunks:
                    try:
                        segment = AudioSegment.from_file(io.BytesIO(chunk))
                        if combined_audio is None:
                            combined_audio = segment
                        else:
                            combined_audio += segment
                    except Exception as e:
                        print(f"Error combining audio chunk: {e}")

                if combined_audio:
                    combined_audio.export(output_filepath, format="wav")
                else:
                    print("Error: Failed to combine any audio chunks.")
                    return None
            print(f"Audio successfully saved to '{output_filepath}'")
            return output_filepath
        except IOError as e:
            print(f"Error saving audio file '{output_filepath}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error while saving audio: {e}")
            return None

    def get_processed_content(self) -> Optional[str]:
        """
        Get the processed content (SSML or original text) that was used for synthesis.

        Returns:
            Optional[str]: The processed content if available, None otherwise.
        """
        return self._processed_content

    def _calculate_audio_hash(self) -> str:
        """Calculate a basic hash of the content for caching (legacy/basic)."""
        hasher = hashlib.md5()
        if self._content_text:
            hasher.update(self._content_text.encode('utf-8'))
        hasher.update(self.language_code.encode('utf-8'))
        return hasher.hexdigest()

    def _calculate_current_hash(self) -> str:
        """
        Calculate a comprehensive hash for the current synthesis request,
        based on the content and the deterministic VoicePack chain that will
        be attempted (not just whichever one ends up succeeding, since the
        cache check happens before any VoicePack is tried).
        """
        hasher = hashlib.md5()
        if self._content_text:
            hasher.update(self._content_text.encode('utf-8'))

        params = [self.language_code, str(self.use_llm_ssml)]
        for voice_pack in self._voice_pack_chain:
            params.append(voice_pack.provider)
            params.append(voice_pack.voice_name)

        for param in params:
            hasher.update(param.encode('utf-8'))

        return hasher.hexdigest()

    def save_to_cache(self, cache_dir: str, cache_filename: Optional[str] = None) -> Optional[str]:
        """
        Save the current TTS build to a cache directory.

        Args:
            cache_dir (str): Directory to save cache files.
            cache_filename (str, optional): Specific filename (without extension) for the cache file.
                                          If None, a hash of the content will be used.

        Returns:
            Optional[str]: Path to the saved cache file, or None if failed.
        """
        if not self._is_built:
            return None

        os.makedirs(cache_dir, exist_ok=True)

        if cache_filename:
            cache_path = os.path.join(cache_dir, f"{cache_filename}.json")
        else:
            audio_hash = self._calculate_audio_hash()
            cache_path = os.path.join(cache_dir, f"{audio_hash}.json")

        cache_data = {
            "content_text": self._content_text,
            "language_code": self.language_code,
            "used_provider": self.used_provider,
            "audio_chunks": [chunk.hex() for chunk in self._audio_chunks],
            "processed_content": self._processed_content,
        }

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            return cache_path
        except Exception as e:
            print(f"Error saving to cache: {e}")
            return None

    @classmethod
    def load_from_cache(cls, cache_dir: str, cache_key: str) -> Optional['TTSBuilder']:
        """
        Try to load a TTSBuilder from cache using a specific key/filename.

        Args:
            cache_dir (str): Directory where cache files are stored.
            cache_key (str): The filename/key for the cache file (without .json extension).

        Returns:
            Optional[TTSBuilder]: Reconstructed TTSBuilder if found in cache, None otherwise.
        """
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            language_code = cache_data.get("language_code")
            if not language_code:
                print(f"Error: Cache file {cache_path} missing language_code.")
                return None

            builder = cls(language_code)
            builder._content_text = cache_data.get("content_text")
            builder._processed_content = cache_data.get("processed_content")
            builder.used_provider = cache_data.get("used_provider")

            if "audio_chunks" in cache_data:
                builder._audio_chunks = [bytes.fromhex(chunk) for chunk in cache_data["audio_chunks"]]

            builder._is_built = True

            print(f"Loaded TTS from cache: {cache_path}")
            return builder
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return None
