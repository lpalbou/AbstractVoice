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


def _levenshtein_leq(a: str, b: str, *, max_dist: int) -> bool:
    """Return True if Levenshtein(a,b) <= max_dist (small, early-exit).

    This is intentionally tiny and only used for short tokens like "ok"/"okay".
    """
    a = a or ""
    b = b or ""
    if a == b:
        return True
    if max_dist <= 0:
        return False
    # Fast bounds.
    if abs(len(a) - len(b)) > max_dist:
        return False

    # DP with early exit.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur_val = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
            cur.append(cur_val)
            row_min = min(row_min, cur_val)
        if row_min > max_dist:
            return False
        prev = cur
    return prev[-1] <= max_dist


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
        # Special-case: tolerate common STT variants for "ok/okay stop"
        # (e.g. "okay stop", "okey stop", "oh stop").
        # Keep it conservative:
        # - require "stop" at the end
        # - require an ok-like token right before it (or one token earlier with "please")
        phrase_toks = phrase.split()
        toks = normalized.split()

        if phrase_toks == ["ok", "stop"] or phrase_toks == ["okay", "stop"]:
            if len(toks) in (2, 3) and toks[-1] == "stop":
                candidates = [toks[-2]]
                if len(toks) == 3:
                    candidates.append(toks[-3])
                for t in candidates:
                    if _levenshtein_leq(t, "ok", max_dist=1) or _levenshtein_leq(t, "okay", max_dist=1):
                        return True

        # Default rule:
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

