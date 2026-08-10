"""Mocked reasoning.py/reasoning_packs.py tests -- no real network/API
calls, safe to run anywhere. Covers the ReasoningPack ABC+registry chain
(mirroring VoicePack, see ADR 0011) and reasoning.generate()/embed()'s
chain-walking, retry-then-fallback, and JSON-response handling.

conftest.py configures a "xx" sentinel language with two providers (gemini,
then mock) and a "yy" sentinel language with just gemini -- same shape
test_tts_builder.py uses for VoicePack's chain, letting fallback tests
exercise a real second provider (mock, which never touches the network)
without inventing a fake third-party provider class.

Run with: pytest tests/fake
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tish_video_sdk import reasoning
from tish_video_sdk.internal.providers.reasoning_packs import ReasoningError


def _fake_gemini_text_response(text):
    return SimpleNamespace(text=text)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Rate-limit retry backs off with time.sleep() -- tests exercising it
    # shouldn't actually wait.
    monkeypatch.setattr(reasoning.time, "sleep", lambda seconds: None)


class TestLanguageAndProviderFilter:
    def test_unconfigured_language_raises(self):
        with pytest.raises(ValueError):
            reasoning.generate("not-a-configured-language", "hello")

    def test_pinning_to_a_provider_restricts_the_chain(self):
        # "xx" also has a mock entry, but pinning to gemini must disable
        # fallback to it -- a gemini failure should propagate, not silently
        # succeed via mock.
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("gemini is down")
            with pytest.raises(ReasoningError):
                reasoning.generate("xx", "hello", provider="gemini")

    def test_unknown_provider_for_language_raises(self):
        with pytest.raises(ValueError):
            reasoning.generate("yy", "hello", provider="mock")  # yy only has a gemini entry


class TestIsMock:
    def test_gemini_first_in_chain_is_not_mock(self):
        assert reasoning.is_mock("xx", provider="gemini") is False

    def test_mock_provider_pinned_is_mock(self):
        assert reasoning.is_mock("xx", provider="mock") is True

    def test_default_chain_for_xx_is_not_mock(self):
        # "xx"'s chain is [gemini, mock] in that order -- the *first* entry
        # is what's currently in effect, so an app gating its own mock
        # behavior on this should see False even though a mock entry exists
        # further down the fallback chain.
        assert reasoning.is_mock("xx") is False


class TestGenerate:
    def test_returns_stripped_text(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_text_response("  joy  ")
            assert reasoning.generate("xx", "classify: {text}", {"text": "hooray"}, provider="gemini") == "joy"

    def test_template_is_formatted_with_values(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_generate = mock_client_cls.return_value.models.generate_content
            mock_generate.return_value = _fake_gemini_text_response("ok")

            reasoning.generate(
                "xx", "Hello {name}, mood is {mood}.", {"name": "World", "mood": "calm"}, provider="gemini"
            )

            assert mock_generate.call_args.kwargs["contents"] == "Hello World, mood is calm."

    def test_unknown_response_format_raises(self):
        with pytest.raises(ValueError):
            reasoning.generate("xx", "hello", response_format="xml")


class TestFallbackChain:
    def test_provider_failure_falls_back_to_next_provider(self):
        # gemini (first in "xx"'s chain) fails outright; mock (second) never
        # touches the network and always succeeds.
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("permission denied")
            result = reasoning.generate("xx", "hello")

        assert result == "This is a mock reasoning response."

    def test_all_providers_failing_raises(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("permission denied")
            # yy only has gemini -- nothing to fall back to.
            with pytest.raises(ReasoningError):
                reasoning.generate("yy", "hello")

    def test_rate_limit_retries_same_provider_before_falling_back(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_generate = mock_client_cls.return_value.models.generate_content
            # Fails with a rate-limit-shaped error every time -- exhausts
            # gemini's own retry budget before falling through to mock.
            mock_generate.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")

            result = reasoning.generate("xx", "hello")

        assert result == "This is a mock reasoning response."
        assert mock_generate.call_count == reasoning._RATE_LIMIT_RETRIES

    def test_rate_limit_recovers_without_falling_back(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_generate = mock_client_cls.return_value.models.generate_content
            mock_generate.side_effect = [
                RuntimeError("429 RESOURCE_EXHAUSTED"),
                _fake_gemini_text_response("recovered"),
            ]

            result = reasoning.generate("xx", "hello", provider="gemini")

        assert result == "recovered"
        assert mock_generate.call_count == 2


class TestJsonResponseFormat:
    def test_parses_fenced_json(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_text_response(
                '```json\n{"mood": "calm"}\n```'
            )
            result = reasoning.generate("xx", "hello", provider="gemini", response_format="json")

        assert result == {"mood": "calm"}

    def test_retries_on_malformed_json_then_succeeds(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_generate = mock_client_cls.return_value.models.generate_content
            mock_generate.side_effect = [
                _fake_gemini_text_response("not json at all"),
                _fake_gemini_text_response('{"mood": "calm"}'),
            ]

            result = reasoning.generate("xx", "hello", provider="gemini", response_format="json")

        assert result == {"mood": "calm"}
        assert mock_generate.call_count == 2

    def test_malformed_json_exhausted_falls_back_then_still_raises(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.generate_content.return_value = _fake_gemini_text_response("not json")
            # mock provider's own canned text isn't valid JSON either --
            # exhausting the whole chain must still raise, not return
            # unparsed text silently.
            with pytest.raises(ReasoningError):
                reasoning.generate("xx", "hello", response_format="json")


class TestEmbed:
    def test_returns_one_vector_per_text(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client_cls.return_value.models.embed_content.return_value = SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.1, 0.2]), SimpleNamespace(values=[0.3, 0.4])]
            )
            result = reasoning.embed("xx", ["a", "b"], provider="gemini")

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_task_type_is_uppercased_for_gemini(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_embed = mock_client_cls.return_value.models.embed_content
            mock_embed.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1])])

            reasoning.embed("xx", ["a"], task_type="retrieval_query", provider="gemini")

            assert mock_embed.call_args.kwargs["config"].task_type == "RETRIEVAL_QUERY"
