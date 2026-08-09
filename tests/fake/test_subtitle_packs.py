"""Mocked SubtitlePack tests -- no real MFA/network calls, safe to run
anywhere.

Run with: pytest tests/fake
"""
import json

from tish_video_sdk.internal.providers.subtitle_packs import (
    SubtitlePack,
    DEFAULT_SUBTITLE_PACKS,
    load_subtitle_packs,
)


class TestDefaultPacks:
    def test_default_packs_cover_fr_en_es_ta(self):
        assert set(DEFAULT_SUBTITLE_PACKS.keys()) == {'fr', 'en', 'es', 'ta'}

    def test_default_packs_have_an_mfa_model(self):
        for pack in DEFAULT_SUBTITLE_PACKS.values():
            assert pack.mfa_model


class TestStyleKwargs:
    def test_style_kwargs_shape_matches_text_style_fields(self):
        # Cross-check against the actual TextStyle constructor this feeds,
        # rather than hardcoding the field list twice.
        from tish_video_sdk.video_maker import TextStyle

        pack = SubtitlePack(language_code='fr', mfa_model='french_mfa')
        kwargs = pack.style_kwargs()

        style = TextStyle(**kwargs)  # must not raise
        assert style.language_code == 'fr'

    def test_style_kwargs_excludes_mfa_model(self):
        pack = SubtitlePack(language_code='fr', mfa_model='french_mfa')
        assert 'mfa_model' not in pack.style_kwargs()


class TestLoadSubtitlePacks:
    def test_load_from_json_file(self, tmp_path):
        path = tmp_path / "subtitle_packs.json"
        path.write_text(json.dumps([
            {"language_code": "de", "mfa_model": "german_mfa", "font_color": "white"},
        ]), encoding="utf-8")

        packs = load_subtitle_packs(str(path))

        assert set(packs.keys()) == {'de'}
        assert packs['de'].mfa_model == 'german_mfa'
        assert packs['de'].font_color == 'white'
