"""Languages each engine accepts, read without loading any model or framework.

Every list comes from the same source the engine's adapter uses (its static
language table), so this is cheap enough for discovery. Remote OpenAI and
OpenAI-compatible providers decide their own languages: they are reported as
``open`` with an empty list. An engine that declares no list reports ``None``.
"""
from __future__ import annotations

import ast
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

KINDS = ("tts", "stt", "cloning")
_REMOTE = {"openai", "openai-compatible"}
_EN_ZH = ["en", "zh"]
_OMNIVOICE_FALLBACK = ["en", "fr", "de", "es", "ru", "zh", "ja", "ko"]


def _norm(provider: Any) -> str:
    text = str(provider or "").strip().lower().replace("_", "-")
    return {"faster-whisper": "faster-whisper", "f5-tts": "f5_tts", "f5tts": "f5_tts"}.get(text, text)


@lru_cache(maxsize=1)
def omnivoice_languages() -> List[str]:
    """OmniVoice's language ids, parsed from its package file without importing it (no torch)."""
    try:
        spec = importlib.util.find_spec("omnivoice")
        if spec is None or not spec.origin:
            return list(_OMNIVOICE_FALLBACK)
        path = Path(spec.origin).parent / "utils" / "lang_map.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(isinstance(t, ast.Name) and t.id == "LANG_NAME_TO_ID" for t in targets):
                ids = ast.literal_eval(node.value).values()
                return sorted({str(i).strip().lower() for i in ids if str(i).strip()})
    except Exception:
        pass
    return list(_OMNIVOICE_FALLBACK)


def _qwen3() -> List[str]:
    from .adapters.tts_qwen3_tts import _LANGUAGE_NAMES

    return list(_LANGUAGE_NAMES)


def _tts(provider: str) -> Optional[List[str]]:
    if provider == "supertonic":
        from .adapters.tts_supertonic import SUPERTONIC_LANGUAGES

        return list(SUPERTONIC_LANGUAGES)
    if provider == "piper":
        from .adapters.tts_piper import PiperTTSAdapter

        return list(PiperTTSAdapter.PIPER_MODELS)
    if provider == "qwen3-tts":
        return _qwen3()
    if provider == "omnivoice":
        return omnivoice_languages()
    if provider == "audiodit":
        return list(_EN_ZH)
    if provider in _REMOTE:
        return []
    raise KeyError(provider)


def _stt(provider: str) -> Optional[List[str]]:
    if provider in {"faster-whisper", "whisper"}:
        from .adapters.stt_faster_whisper import FasterWhisperAdapter

        return list(FasterWhisperAdapter.LANGUAGES)
    if provider == "mlx-whisper":
        from .adapters.stt_mlx_whisper import MLXWhisperAdapter

        return list(MLXWhisperAdapter.LANGUAGES)
    if provider == "transformers-asr":
        from .adapters.stt_transformers_asr import TransformersASRAdapter

        return list(TransformersASRAdapter.LANGUAGES)
    if provider in _REMOTE:
        return []
    raise KeyError(provider)


def _cloning(provider: str) -> Optional[List[str]]:
    if provider == "qwen3-tts":
        return _qwen3()
    if provider == "omnivoice":
        return omnivoice_languages()
    if provider in {"f5_tts", "audiodit"}:
        return list(_EN_ZH)
    if provider == "chroma":
        return None  # no language list declared upstream
    if provider in _REMOTE:
        return []
    raise KeyError(provider)


_PROVIDERS = {
    "tts": ("supertonic", "piper", "qwen3-tts", "omnivoice", "audiodit", "openai", "openai-compatible"),
    "stt": ("faster-whisper", "mlx-whisper", "transformers-asr", "openai", "openai-compatible"),
    "cloning": ("qwen3-tts", "omnivoice", "f5_tts", "chroma", "audiodit", "openai", "openai-compatible"),
}
_READERS = {"tts": _tts, "stt": _stt, "cloning": _cloning}


def language_entry(kind: str, provider: str) -> Dict[str, Any]:
    """``{"provider", "languages", "open", "declared"}`` for one engine; KeyError for an unknown one."""
    k = str(kind or "").strip().lower()
    if k not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)} (got {kind!r})")
    p = _norm(provider)
    langs = _READERS[k](p)
    return {
        "provider": p,
        "languages": list(langs or []),
        "open": p in _REMOTE,
        "declared": langs is not None,
    }


def language_catalog(kind: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """``{kind: {provider: entry}}`` for every known engine, optionally narrowed.

    With ``provider`` alone, only the kinds that engine serves are returned (Piper:
    ``tts``; Qwen3-TTS: ``tts`` and ``cloning``). An engine no requested kind knows
    raises KeyError naming the known ones; an unknown kind raises ValueError.
    """
    if kind:
        k = str(kind).strip().lower()
        if k not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)} (got {kind!r})")
        kinds = [k]
    else:
        kinds = list(KINDS)
    out: Dict[str, Dict[str, Any]] = {}
    for k in kinds:
        entries: Dict[str, Any] = {}
        for p in [_norm(provider)] if provider else list(_PROVIDERS[k]):
            try:
                entries[p] = language_entry(k, p)
            except KeyError:
                continue
        if entries or not provider:
            out[k] = entries
    if provider and not out:
        known = sorted({p for k in kinds for p in _PROVIDERS[k]})
        scope = f"{kinds[0]} " if len(kinds) == 1 else ""
        raise KeyError(f"unknown {scope}provider {provider!r}; known: {', '.join(known)}")
    return out


def languages_by_provider(kind: str, providers: Any) -> Dict[str, List[str]]:
    """``{provider: languages}`` for the given providers of one kind.

    A provider this module does not know (a third-party engine) is left out
    rather than reported with an empty list, which would read as "speaks nothing".
    """
    out: Dict[str, List[str]] = {}
    for provider in providers or []:
        try:
            entry = language_entry(kind, provider)
        except KeyError:
            continue
        out[str(provider)] = entry["languages"]
    return out


# Native language of each Qwen3-TTS preset speaker (Qwen3-TTS model card). Every
# preset speaks all ten Qwen3-TTS languages; this is the one it was voiced in.
QWEN3_TTS_PRESET_LANGUAGES = {
    "vivian": "zh",
    "serena": "zh",
    "uncle_fu": "zh",
    "dylan": "zh",
    "eric": "zh",
    "ryan": "en",
    "aiden": "en",
    "ono_anna": "ja",
    "sohee": "ko",
}


def speakable_languages(provider: str) -> List[str]:
    """Languages a voice of this engine can speak: its TTS list, else its cloning list.

    Empty when the engine does not restrict them (remote providers) or is unknown here.
    """
    for kind in ("tts", "cloning"):
        try:
            return language_entry(kind, provider)["languages"]
        except KeyError:
            continue
    return []
