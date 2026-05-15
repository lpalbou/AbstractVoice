"""Interactive example defaults for TTS engine selection.

The library API remains remote-first (`VoiceManager()` defaults to OpenAI).
Interactive examples are different: a plain lightweight install should use
OpenAI, while an install profile that includes Supertonic should make
Supertonic the default TTS engine.
"""

from __future__ import annotations

import importlib.util
from typing import Callable


def normalize_tts_engine_name(engine: str | None) -> str:
    name = str(engine or "").strip().lower().replace("_", "-")
    if name in ("remote", "compatible", "proxy"):
        return "openai-compatible"
    return name or "auto"


def _supertonic_installed() -> bool:
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except Exception:
        return False


def _piper_installed() -> bool:
    try:
        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


_INSTALL_CHECKS: dict[str, Callable[[], bool]] = {
    "supertonic": _supertonic_installed,
    "piper": _piper_installed,
}


def resolve_interactive_tts_engine(engine: str | None, *, language: str | None = None) -> str:
    """Resolve example `auto` TTS defaults.

    Priority for interactive examples:
    1. Explicit engine names are respected.
    2. `auto` uses installed Supertonic first, then installed Piper, then OpenAI.
       This makes `abstractvoice[all-apple]` / `abstractvoice[all-gpu]`
       interactive defaults Supertonic, while plain `abstractvoice` remains
       remote/OpenAI.
    """
    requested = normalize_tts_engine_name(engine)
    if requested != "auto":
        return requested

    _ = language

    # Supertonic is the preferred local base TTS when the runtime dependency is
    # present. Cache readiness is deliberately not part of the default decision:
    # all-* installs should default to Supertonic and fail with an explicit
    # prefetch hint if artifacts are missing, not fall back to remote audio.
    for candidate in ("supertonic", "piper"):
        if _INSTALL_CHECKS[candidate]():
            return candidate

    return "openai"
