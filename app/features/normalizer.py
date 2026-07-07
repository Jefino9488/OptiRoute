"""Prompt normalizer — canonical form, filler removal, and hashing.

Converts raw user prompts into a deterministic canonical form suitable
for caching and consistent feature extraction.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedPrompt:
    """Immutable result of prompt normalization.

    Attributes:
        raw: The original, unmodified prompt string.
        normalized: Canonical form after cleaning.
        prompt_hash: SHA-256 hex digest of *normalized* text.
    """

    raw: str
    normalized: str
    prompt_hash: str

    def __repr__(self) -> str:
        return (
            f"NormalizedPrompt(hash={self.prompt_hash[:12]}…, "
            f"normalized={self.normalized!r})"
        )


# Pre-compiled patterns --------------------------------------------------

# Filler phrases to strip (order matters — longer phrases first to avoid
# partial matches).  Each entry becomes a regex alternative.
_FILLER_PHRASES: list[str] = [
    r"i\s+would\s+like\s+you\s+to",
    r"i\s+need\s+you\s+to",
    r"i\s+want\s+you\s+to",
    r"could\s+you",
    r"can\s+you",
    r"assist\s+me",
    r"help\s+me",
    r"please",
    r"kindly",
]

_FILLER_RE = re.compile(
    r"\b(?:" + "|".join(_FILLER_PHRASES) + r")\b",
    re.IGNORECASE,
)

# Trailing punctuation clutter: repeated or mixed sentence-end marks.
_TRAILING_PUNCT_RE = re.compile(r"[?!.,;:\s]+$")

# Collapse multiple whitespace characters into a single space.
_MULTI_WS_RE = re.compile(r"\s+")


class RequestNormalizer:
    """Stateless prompt normalizer.

    Normalisation pipeline:
    1. Strip leading / trailing whitespace.
    2. Collapse runs of whitespace to a single space.
    3. Lower-case the entire string.
    4. Remove filler phrases (``please``, ``can you``, …).
    5. Remove trailing punctuation clutter.
    6. Generate a SHA-256 hash of the final normalised form.
    """

    def normalize(self, prompt: str) -> NormalizedPrompt:
        """Return the normalised representation of *prompt*.

        Args:
            prompt: Raw user prompt (may contain arbitrary whitespace,
                filler words, mixed case, etc.).

        Returns:
            A :class:`NormalizedPrompt` with the cleaned text and its
            deterministic hash.
        """
        text = prompt.strip()
        text = _MULTI_WS_RE.sub(" ", text)
        text = text.lower()
        text = _FILLER_RE.sub("", text)
        # Collapse whitespace again after filler removal.
        text = _MULTI_WS_RE.sub(" ", text).strip()
        text = _TRAILING_PUNCT_RE.sub("", text)

        prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return NormalizedPrompt(
            raw=prompt,
            normalized=text,
            prompt_hash=prompt_hash,
        )
