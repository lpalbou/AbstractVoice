"""TTS adapter registry (engine selection).

Design goals
------------
- Keep AbstractVoice remote-first by default (engine="auto" => OpenAI).
- Allow opt-in local/heavy engines without importing heavy deps unless
  explicitly selected.
- Keep the API surface small and stable: VoiceManager routes engine selection
  through this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from .base import TTSAdapter

_Factory = Callable[..., Optional[TTSAdapter]]


def _normalize_engine_name(engine: str | None) -> str:
    name = str(engine or "").strip().lower().replace("_", "-")
    if name in ("remote", "compatible", "proxy"):
        return "openai-compatible"
    return name or "auto"


def _resolve_auto_engine(engine: str) -> str:
    # Policy: keep "auto" deterministic and lightweight/remote.
    # If users want local inference, they must request the local engine.
    if engine == "auto":
        return "openai"
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


def _supertonic_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    try:
        from .tts_supertonic import SupertonicTTSAdapter
    except Exception as e:
        raise RuntimeError(
            "Supertonic engine requires optional dependencies.\n"
            "Install with:\n"
            "  pip install \"abstractvoice[supertonic]\""
        ) from e
    try:
        adapter = SupertonicTTSAdapter(
            language=str(language),
            allow_downloads=bool(allow_downloads),
            auto_load=bool(auto_load),
            debug_mode=bool(debug_mode),
            model_id=kwargs.get("model_id"),
            revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
        )
        return adapter if bool(getattr(adapter, "_onnx_available", False)) else None
    except RuntimeError:
        raise
    except Exception:
        return None


def _openai_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    _ = allow_downloads, auto_load
    from .tts_openai_compatible import OpenAICompatibleTTSAdapter

    return OpenAICompatibleTTSAdapter(
        provider="openai",
        language=str(language),
        base_url=kwargs.get("base_url"),
        api_key=kwargs.get("api_key"),
        model_id=kwargs.get("model_id"),
        voice=kwargs.get("voice"),
        instructions=kwargs.get("instructions"),
        timeout_s=kwargs.get("timeout_s"),
        session=kwargs.get("session"),
        debug_mode=bool(debug_mode),
    )


def _openai_compatible_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    _ = allow_downloads, auto_load
    from .tts_openai_compatible import OpenAICompatibleTTSAdapter

    return OpenAICompatibleTTSAdapter(
        provider="openai-compatible",
        language=str(language),
        base_url=kwargs.get("base_url"),
        api_key=kwargs.get("api_key"),
        model_id=kwargs.get("model_id"),
        voice=kwargs.get("voice"),
        instructions=kwargs.get("instructions"),
        timeout_s=kwargs.get("timeout_s"),
        session=kwargs.get("session"),
        debug_mode=bool(debug_mode),
    )


def _qwen3_tts_factory(
    *,
    language: str,
    allow_downloads: bool,
    auto_load: bool,
    debug_mode: bool = False,
    **kwargs: Any,
) -> TTSAdapter | None:
    try:
        from .tts_qwen3_tts import Qwen3TTSAdapter
    except Exception as e:
        raise RuntimeError(
            "Qwen3-TTS engine requires optional dependencies.\n"
            "Install with:\n"
            "  pip install \"abstractvoice[qwen3-tts]\""
        ) from e
    return Qwen3TTSAdapter(
        language=str(language),
        allow_downloads=bool(allow_downloads),
        auto_load=bool(auto_load),
        debug_mode=bool(debug_mode),
        model_id=kwargs.get("model_id"),
        revision=kwargs.get("revision"),
        device=kwargs.get("device", "auto"),
    )


_TTS_ADAPTER_FACTORIES: dict[str, _Factory] = {
    "openai": _openai_factory,
    "openai-compatible": _openai_compatible_factory,
    "piper": _piper_factory,
    "supertonic": _supertonic_factory,
    "audiodit": _audiodit_factory,
    "omnivoice": _omnivoice_factory,
    "qwen3-tts": _qwen3_tts_factory,
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
        if requested == "piper":
            raise RuntimeError(
                "TTS engine 'piper' requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[piper]\"\n"
                "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                "  pip install \"abstractvoice[gpu]\"    # GPU profile"
            )
        if requested == "supertonic":
            raise RuntimeError(
                "TTS engine 'supertonic' requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[supertonic]\"\n"
                "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                "  pip install \"abstractvoice[gpu]\"    # GPU profile\n"
                "  pip install \"abstractvoice[all-apple]\"  # Apple profile + web\n"
                "  pip install \"abstractvoice[all-gpu]\"    # GPU profile + web"
            )
        raise RuntimeError(
            f"TTS engine '{requested}' is not available in this environment.\n"
            f"Install the required optional dependencies (or pick a different engine).\n"
            f"Supported engines: {', '.join(get_supported_tts_engines())}"
        )
    return adapter, resolved
