"""Tests for app.features.normalizer — RequestNormalizer."""

from __future__ import annotations

import pytest

from app.features.normalizer import NormalizedPrompt, RequestNormalizer


@pytest.fixture()
def normalizer() -> RequestNormalizer:
    return RequestNormalizer()


class TestFillerRemoval:
    """Filler phrases are stripped from normalised output."""

    def test_please_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("Please summarize this")
        assert "please" not in result.normalized
        assert result.normalized == "summarize this"

    def test_can_you_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("Can you explain quantum physics?")
        assert result.normalized == "explain quantum physics"

    def test_could_you_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("Could you list the capitals?")
        assert result.normalized == "list the capitals"

    def test_compound_fillers_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize(
            "Please, could you kindly help me write a summary?"
        )
        # All filler words gone; only meaningful content remains.
        assert "please" not in result.normalized
        assert "could you" not in result.normalized
        assert "kindly" not in result.normalized
        assert "help me" not in result.normalized
        assert "write a summary" in result.normalized

    def test_i_want_you_to_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("I want you to translate this sentence")
        assert result.normalized == "translate this sentence"

    def test_i_would_like_you_to_removed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("I would like you to solve this equation")
        assert result.normalized == "solve this equation"


class TestWhitespace:
    """Whitespace normalisation."""

    def test_leading_trailing_stripped(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("   hello world   ")
        assert result.normalized == "hello world"

    def test_multiple_spaces_collapsed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("hello    world    foo")
        assert result.normalized == "hello world foo"

    def test_tabs_and_newlines_collapsed(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("hello\t\tworld\n\nfoo")
        assert result.normalized == "hello world foo"


class TestHashDeterminism:
    """SHA-256 hash must be deterministic."""

    def test_same_input_same_hash(self, normalizer: RequestNormalizer) -> None:
        a = normalizer.normalize("Hello World")
        b = normalizer.normalize("Hello World")
        assert a.prompt_hash == b.prompt_hash

    def test_equivalent_inputs_same_hash(self, normalizer: RequestNormalizer) -> None:
        """Inputs that differ only in case / whitespace produce identical hashes."""
        a = normalizer.normalize("  Hello   World  ")
        b = normalizer.normalize("hello world")
        assert a.prompt_hash == b.prompt_hash

    def test_different_inputs_different_hash(self, normalizer: RequestNormalizer) -> None:
        a = normalizer.normalize("Hello World")
        b = normalizer.normalize("Goodbye World")
        assert a.prompt_hash != b.prompt_hash


class TestEdgeCases:
    """Edge-case handling."""

    def test_empty_string(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("")
        assert result.normalized == ""
        assert isinstance(result.prompt_hash, str)
        assert len(result.prompt_hash) == 64  # SHA-256 hex digest length

    def test_only_filler(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("Please kindly")
        assert result.normalized == ""

    def test_raw_preserved(self, normalizer: RequestNormalizer) -> None:
        raw = "  Please summarize THIS!  "
        result = normalizer.normalize(raw)
        assert result.raw == raw

    def test_trailing_punctuation_stripped(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("Summarize this??!...")
        assert result.normalized == "summarize this"

    def test_hash_is_64_hex_chars(self, normalizer: RequestNormalizer) -> None:
        result = normalizer.normalize("test")
        assert len(result.prompt_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.prompt_hash)
