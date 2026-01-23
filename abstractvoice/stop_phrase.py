"""Stop phrase matching utilities (no heavy deps).

Keep this module dependency-free so it can be used in:
- core unit tests
- recognition pipeline (without forcing VAD/STT imports)
"""

from __future__ import annotations

import re
from typing import Iterable


def normalize_stop_phrase(text: str) -> str:
    """Normalize text for conservative stop-phrase matching."""
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_stop_phrase(text: str, phrases: Iterable[str]) -> bool:
    """Return True if text matches any configured stop phrase.

    Matching is intentionally:
    - conservative about normalization (no fancy text transforms)
    - but tolerant to common STT variations like "stop." / "stop please"

    We match phrases as whole-word sequences inside the normalized text.
    """
    normalized = normalize_stop_phrase(text)
    if not normalized:
        return False
    phrase_set = {normalize_stop_phrase(p) for p in phrases if p}
    for phrase in phrase_set:
        if not phrase:
            continue
        # We only match:
        # - exact (stop)
        # - prefix (stop please)
        # - suffix (please stop)
        # This avoids false positives like "don't stop now" when "stop" is a phrase.
        if normalized == phrase:
            return True
        if normalized.startswith(phrase + " "):
            return True
        if normalized.endswith(" " + phrase):
            return True
    return False

