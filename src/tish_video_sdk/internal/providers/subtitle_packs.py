"""SubtitlePack: per-language config bundling the MFA acoustic model name and
text styling (font, color, position, ...) for subtitle generation/rendering,
in the same spirit as VoicePack -- a process loads every SubtitlePack it might
need once at import, keyed by language_code, so config lives in JSON, not code.

Deliberately has no dependency on tish_video_sdk.video_maker.TextStyle (which
would create an import cycle: subtitles.py -> this module -> video_maker.py ->
tts.py -> subtitles.py). Instead, style_kwargs() returns a plain dict shaped
for TextStyle(**kwargs); video_maker.py does that conversion itself.
"""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SubtitlePack:
    """One language's subtitle configuration: which MFA model to align with,
    and the default text styling to render segments with. mfa_model is only
    read when tish_video_sdk.subtitles.USE_MFA_ALIGNMENT is True; a pack with
    mfa_model=None (or no pack at all for a language) makes SubtitleBuilder
    skip MFA alignment for that language rather than failing."""
    language_code: str
    mfa_model: Optional[str] = None
    font_size: int = 50
    font_color: str = 'black'
    font_family: str = 'Arial-Bold'
    stroke_color: str = 'black'
    stroke_width: int = 3
    bg_color: Optional[str] = None
    text_position: Tuple = (0.5, 0.5)
    box_size: Tuple[Optional[float], Optional[float]] = (0.8, None)
    highlight_color: str = '#FFD700'
    timing_offset: float = 0.0

    def style_kwargs(self) -> Dict:
        """This pack's styling fields, shaped for
        tish_video_sdk.video_maker.TextStyle(**kwargs). Color/size validation
        happens there (TextStyle.__post_init__), not duplicated here."""
        return {
            'font_size': self.font_size,
            'font_color': self.font_color,
            'font_family': self.font_family,
            'stroke_color': self.stroke_color,
            'stroke_width': self.stroke_width,
            'bg_color': self.bg_color,
            'text_position': self.text_position,
            'box_size': self.box_size,
            'highlight_color': self.highlight_color,
            'timing_offset': self.timing_offset,
            'language_code': self.language_code,
        }


# Built-in defaults so subtitle generation (and MFA alignment specifically)
# works out of the box for these languages without requiring a
# SUBTITLE_PACKS_PATH file -- SUBTITLE_PACKS_PATH only needs to add languages
# or override entries. 'ta' uses tamil_cv (Common Voice) since MFA has no
# tamil_mfa model -- likely lower accuracy than the *_mfa models.
DEFAULT_SUBTITLE_PACKS: Dict[str, SubtitlePack] = {
    'fr': SubtitlePack(language_code='fr', mfa_model='french_mfa'),
    'en': SubtitlePack(language_code='en', mfa_model='english_mfa'),
    'es': SubtitlePack(language_code='es', mfa_model='spanish_mfa'),
    'ta': SubtitlePack(
        language_code='ta',
        mfa_model='tamil_cv',
        font_family=(
            'Nirmala-UI-&-Nirmala-UI-Bold-&-Nirmala-UI-Semilight-&-'
            'Nirmala-Text-&-Nirmala-Text-Bold-&-Nirmala-Text-Semilight'
        ),
    ),
}


def load_subtitle_packs(path: str) -> Dict[str, SubtitlePack]:
    """Load a SUBTITLE_PACKS_PATH JSON file -- a list of objects, one per
    language, in the shape of SubtitlePack's fields (see
    configuration/subtitle_packs.example.json)."""
    with open(path, 'r', encoding='utf-8') as f:
        entries: List[Dict] = json.load(f)
    return {entry['language_code']: SubtitlePack(**entry) for entry in entries}
