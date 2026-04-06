"""TTS adapter registry (engine selection).

Design goals
------------
- Keep AbstractVoice "Piper-first" by default (engine="auto" => Piper).
- Allow opt-in engines (e.g. diffusion/LMM TTS) without importing heavy deps
  unless explicitly selected.
- Keep the API surface small and stable: VoiceManager routes engine selection
  through this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import TTSAdapter

_Factory = Callable[..., TTSAdapter | None]


def _normalize_engine_name(engine: str | None) -> str:
    name = str(engine or "").strip().lower()
    return name or "auto"


def _resolve_auto_engine(engine: str) -> str:
    # Policy: keep "auto" deterministic and lightweight.
    # If users want a heavy engine, they must request it explicitly.
    if engine == "auto":
        return "piper"
    return engine


def _piper_factory(*, language: str, allow_downloads: bool, auto_load: bool, **_kwargs: Any) -> TTSAdapter | None:
    try:
        from .tts_piper import PiperTTSAdapter
    except Exception:
        return None
    try:
        adapter = PiperTTSAdapter(
            language=str(language),
            allow_downloads=bool(allow_downloads),
            auto_load=bool(auto_load),
        )
        # Return the adapter if the runtime is importable even when no voice is
        # loaded yet (offline-first). This keeps audio output available for
        # cloning backends.
        return adapter if bool(getattr(adapter, "_piper_available", False)) else None
    except Exception:
        return None


def _audiodit_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    try:
        from .tts_audiodit import AudioDiTTTSAdapter
    except Exception as e:
        raise RuntimeError(
            "AudioDiT engine requires optional dependencies.\n"
            "Install with:\n"
            "  pip install \"abstractvoice[audiodit]\""
        ) from e
    return AudioDiTTTSAdapter(
        language=str(language),
        allow_downloads=bool(allow_downloads),
        auto_load=bool(auto_load),
        debug_mode=bool(debug_mode),
        model_id=kwargs.get("model_id"),
        revision=kwargs.get("revision"),
        device=kwargs.get("device", "auto"),
    )


def _omnivoice_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    try:
        from .tts_omnivoice import OmniVoiceTTSAdapter
    except Exception as e:
        raise RuntimeError(
            "OmniVoice engine requires optional dependencies.\n"
            "Install with:\n"
            "  pip install \"abstractvoice[omnivoice]\""
        ) from e
    return OmniVoiceTTSAdapter(
        language=str(language),
        allow_downloads=bool(allow_downloads),
        auto_load=bool(auto_load),
        debug_mode=bool(debug_mode),
        model_id=kwargs.get("model_id"),
        revision=kwargs.get("revision"),
        device=kwargs.get("device", "auto"),
    )


_TTS_ADAPTER_FACTORIES: dict[str, _Factory] = {
    "piper": _piper_factory,
    "audiodit": _audiodit_factory,
    "omnivoice": _omnivoice_factory,
}


def register_tts_adapter(engine: str, factory: _Factory) -> None:
    """Register a new TTS adapter factory.

    Intended for internal engines first; integrators can also register custom
    adapters in-process if needed.
    """
    name = _normalize_engine_name(engine)
    if name in ("auto",):
        raise ValueError("Cannot register factory for reserved engine name: auto")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _TTS_ADAPTER_FACTORIES[name] = factory


def get_supported_tts_engines() -> list[str]:
    """Return known engine names (including 'auto')."""
    out = ["auto"]
    out.extend(sorted(_TTS_ADAPTER_FACTORIES.keys()))
    return out


def create_tts_adapter(
    *,
    engine: str | None,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    **kwargs: Any,
) -> tuple[TTSAdapter | None, str]:
    """Create a TTS adapter for `engine`.

    Returns (adapter_or_none, resolved_engine_name).
    """
    requested = _normalize_engine_name(engine)
    resolved = _resolve_auto_engine(requested)

    if resolved not in _TTS_ADAPTER_FACTORIES:
        supported = ", ".join(get_supported_tts_engines())
        raise ValueError(f"Unknown tts_engine: {requested}. Supported: {supported}")

    adapter = _TTS_ADAPTER_FACTORIES[resolved](
        language=str(language),
        allow_downloads=bool(allow_downloads),
        auto_load=bool(auto_load),
        **kwargs,
    )
    if adapter is None and requested != "auto":
        raise RuntimeError(
            f"TTS engine '{requested}' is not available in this environment.\n"
            f"Install the required optional dependencies (or pick a different engine).\n"
            f"Supported engines: {', '.join(get_supported_tts_engines())}"
        )
    return adapter, resolved

