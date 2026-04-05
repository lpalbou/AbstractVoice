"""Speech duration estimation utilities (engine-agnostic).

We deliberately keep this heuristic simple:
- default: average units-per-second by language
- units: whitespace words for most languages, CJK characters for zh

This is used by TTS/cloning engines that require an explicit target duration
(e.g. diffusion models that need a frame count).
"""

from __future__ import annotations

import re


DEFAULT_UNITS_PER_SECOND: dict[str, float] = {
    # Rough defaults; tune per-engine/model later.
    # Typical conversational English ~150–180 wpm => ~2.5–3.0 words/sec.
    "en": 2.7,
    "fr": 2.8,
    "de": 2.5,
    "es": 2.8,
    "ru": 2.4,
    # For zh we treat each CJK character as a "unit".
    # 4–6 chars/sec is a common conversational range; start conservative.
    "zh": 4.5,
}


_RE_CJK = re.compile(r"[\u4e00-\u9fff]")


def count_speech_units(text: str, *, language: str) -> int:
    """Count speech 'units' for duration estimation.

    - zh: count CJK characters (falls back to word count if none are present)
    - other languages: count whitespace-delimited words
    """
    s = " ".join(str(text or "").strip().split())
    if not s:
        return 0

    lang = str(language or "").strip().lower()
    if lang == "zh":
        cjk = len(_RE_CJK.findall(s))
        if cjk > 0:
            return int(cjk)

    return int(len(s.split()))


def estimate_duration_s(
    text: str,
    *,
    language: str,
    units_per_second: float | None = None,
    min_s: float = 0.2,
    max_s: float | None = None,
) -> float:
    """Estimate spoken duration in seconds for `text`.

    This is a heuristic used to choose a *default* target duration for models that
    require a length parameter. It is intentionally conservative and clamped.
    """
    lang = str(language or "").strip().lower()
    ups = float(units_per_second) if units_per_second is not None else float(DEFAULT_UNITS_PER_SECOND.get(lang, 2.6))
    ups = max(0.1, ups)

    units = count_speech_units(text, language=lang)
    if units <= 0:
        dur = 0.0
    else:
        dur = float(units) / float(ups)

    dur = max(float(min_s), float(dur))
    if max_s is not None:
        dur = min(float(max_s), float(dur))
    return float(dur)

