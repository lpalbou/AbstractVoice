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
    """Return True if text exactly matches any configured stop phrase."""
    normalized = normalize_stop_phrase(text)
    if not normalized:
        return False
    phrase_set = {normalize_stop_phrase(p) for p in phrases if p}
    return normalized in phrase_set

