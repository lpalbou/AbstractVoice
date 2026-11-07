"""STT package namespace.

The supported STT engines live in `abstractvoice.adapters`.
"""

from .languages import (  # noqa: F401 - the public list
    AUTO,
    AUTO_LABEL,
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    choices,
    language_label,
    normalize_language,
    refusal_sentence,
    supported_languages,
)

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
