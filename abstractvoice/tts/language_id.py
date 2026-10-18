"""The language of a whole text, identified once, so every streamed segment of one reply is
spoken in the same language.

A multilingual engine told the wrong language reads a sentence in it: on Supertonic 3 with
the default `en`, "Marie enverra les diapositives ce soir," came out in English, while `fr`
gave French (measured with Whisper, 2026-10-17). Segments are too short to identify one by
one ("Ah oui," and "Oui." score 12-19 % for any language), so the reply is identified as a
whole and the result held for all its segments; below `MIN_CONFIDENCE` the configured
language stays.
"""

from __future__ import annotations

import threading
from typing import Iterable, Optional

MIN_CONFIDENCE = 0.8
MIN_CHARS = 20

_lock = threading.Lock()
_identifiers: dict[tuple[str, ...], object] = {}


def _identifier(languages: tuple[str, ...]):
    with _lock:
        ident = _identifiers.get(languages)
        if ident is None:
            from py3langid.langid import MODEL_FILE, LanguageIdentifier

            # py3langid 0.4 (Python 3.10+) renamed the loader; 0.3 serves Python 3.9.
            load = getattr(LanguageIdentifier, "from_model_file", None) or LanguageIdentifier.from_pickled_model
            ident = load(MODEL_FILE, norm_probs=True)
            ident.set_languages(list(languages))
            _identifiers[languages] = ident
        return ident


def identify_language(text: str, languages: Iterable[str]) -> Optional[str]:
    """The language of `text` among `languages` (ISO 639-1 codes), or None when the text is too
    short or the identification is not confident."""
    s = " ".join(str(text or "").split())
    langs = tuple(sorted({str(x).strip().lower() for x in languages if str(x or "").strip()}))
    if len(s) < MIN_CHARS or len(langs) < 2:
        return None
    lang, prob = _identifier(langs).classify(s)
    return str(lang) if float(prob) >= MIN_CONFIDENCE else None
