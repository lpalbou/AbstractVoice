"""The spoken languages the STT engines support — THE one list (round 18).

Every local STT adapter (faster-whisper, mlx-whisper, transformers ASR) advertises this list
through ``get_supported_languages()``; remote OpenAI-compatible engines pass the code through.
AbstractGateway validates an account's ``spoken_language`` preference and a request's
``language`` hint against it, so a client never keeps a list of its own.

``AUTO`` ("auto") means: let the engine detect the language. ``None`` is how the adapters spell it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: The wire value for "let the engine detect it".
AUTO = "auto"

#: ISO 639-1 codes, in the order the adapters have always listed them (all are Whisper languages).
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "fr", "de", "es", "ru", "zh", "it", "pt", "ja", "ko", "ar", "hi", "nl", "pl", "vi", "el",
)

#: The English name shown for each code (the label clients display, served, never rewritten).
LANGUAGE_LABELS: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "zh": "Chinese",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "vi": "Vietnamese",
    "el": "Greek",
}

AUTO_LABEL = "Auto (detected)"


def supported_languages() -> List[str]:
    """The supported ISO 639-1 codes (a copy)."""
    return list(SUPPORTED_LANGUAGES)


def language_label(code: Optional[str]) -> str:
    """The display label of a code; ``None``/"auto" -> "Auto (detected)"; an unknown code is shown as is."""
    if code is None or str(code).strip().lower() == AUTO:
        return AUTO_LABEL
    key = str(code).strip().lower()
    return LANGUAGE_LABELS.get(key, key)


def refusal_sentence(value: Any) -> str:
    codes = ", ".join(sorted(SUPPORTED_LANGUAGES))
    return f"spoken_language = {value!r} refused: not a language the speech engines support. Choose auto or one of: {codes}."


def normalize_language(value: Any) -> Optional[str]:
    """``None``, "", "auto" (any case) -> ``None`` (auto); a supported code (any case, ``en-US``
    accepted as ``en``) -> the lower-case code; anything else raises ``ValueError`` with ONE
    user-readable sentence."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(refusal_sentence(value))
    text = value.strip().lower()
    if not text or text == AUTO:
        return None
    base = text.replace("_", "-").split("-", 1)[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    raise ValueError(refusal_sentence(value))


def choices() -> List[Dict[str, str]]:
    """``[{"value": "auto", "label": "Auto (detected)"}, {"value": "en", "label": "English"}, ...]`` —
    auto first, then the languages by label."""
    rows = sorted(({"value": code, "label": LANGUAGE_LABELS[code]} for code in SUPPORTED_LANGUAGES), key=lambda r: r["label"])
    return [{"value": AUTO, "label": AUTO_LABEL}, *rows]


__all__ = [
    "AUTO",
    "AUTO_LABEL",
    "LANGUAGE_LABELS",
    "SUPPORTED_LANGUAGES",
    "choices",
    "language_label",
    "normalize_language",
    "refusal_sentence",
    "supported_languages",
]
