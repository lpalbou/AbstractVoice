"""Is a voice engine's runtime installed? One public, cheap answer per engine.

Every local AbstractVoice engine runs on optional Python packages that come
from an install extra (``abstractvoice[supertonic]`` brings ``onnxruntime``,
``abstractvoice[stt]`` brings ``faster-whisper``, and so on). Whether those
packages are importable is a different fact from whether a model is
downloaded, and callers that report engine status -- AbstractCore routes,
gateway consoles, installers -- need it on its own, in plain words, without
paying for the import.

The answer comes from :func:`importlib.util.find_spec` only: no engine module,
ML framework or adapter is imported, so asking about every engine costs
milliseconds. The table below is the single source of truth for which
distributions an engine needs; the AbstractCore plugin reads it too.

Public API (stable)::

    from abstractvoice.engine_runtime import engine_runtime_status

    status = engine_runtime_status("supertonic")
    status.installed          # False when onnxruntime is missing
    status.missing_modules    # ("onnxruntime",)
    status.install_command    # 'pip install "abstractvoice[supertonic]"'
    status.reason             # plain-words sentence, None when installed
    status.to_dict()          # JSON-safe record

An unknown engine id raises :class:`ValueError`; the ids are listed by
:func:`known_engines`. Remote engines (``openai``, ``openai-compatible``) need
nothing beyond the core install and always report ``installed=True``; whether
they are *configured* (an API key, a base URL) is not a runtime question.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "EngineRuntimeStatus",
    "engine_runtime_installed",
    "engine_runtime_status",
    "known_engines",
    "normalize_engine_id",
]


@dataclass(frozen=True)
class _EngineRequirement:
    label: str
    kinds: Tuple[str, ...]
    # Each group is satisfied when ANY module in it is importable; every group
    # must be satisfied. Piper ships either as ``piper`` or ``piper_phonemize``.
    module_groups: Tuple[Tuple[str, ...], ...]
    extra: Optional[str]
    remote: bool = False


_ENGINES: Dict[str, _EngineRequirement] = {
    "openai": _EngineRequirement("OpenAI", ("tts", "stt", "cloning"), (), None, remote=True),
    "openai-compatible": _EngineRequirement(
        "OpenAI-compatible server", ("tts", "stt", "cloning"), (), None, remote=True
    ),
    "supertonic": _EngineRequirement("Supertonic", ("tts",), (("onnxruntime",),), "supertonic"),
    "piper": _EngineRequirement("Piper", ("tts",), (("piper", "piper_phonemize"),), "piper"),
    "audiodit": _EngineRequirement(
        "LongCat-AudioDiT", ("tts", "cloning"), (("torch",), ("transformers",)), "audiodit"
    ),
    "qwen3-tts": _EngineRequirement(
        "Qwen3-TTS", ("tts", "cloning"), (("torch",), ("transformers",), ("soundfile",)), "qwen3-tts"
    ),
    "omnivoice": _EngineRequirement("OmniVoice", ("tts", "cloning"), (("omnivoice",),), "omnivoice"),
    "f5_tts": _EngineRequirement("OpenF5 (F5-TTS)", ("cloning",), (("f5_tts",),), "cloning"),
    "chroma": _EngineRequirement("Chroma", ("cloning",), (("torch",), ("transformers",)), "chroma"),
    "faster-whisper": _EngineRequirement("faster-whisper", ("stt",), (("faster_whisper",),), "stt"),
    "transformers-asr": _EngineRequirement(
        "Transformers ASR (Whisper, Qwen3-ASR)",
        ("stt",),
        (("torch",), ("transformers",), ("soundfile",)),
        "stt-hf",
    ),
}

_ALIASES = {
    "remote": "openai-compatible",
    "compatible": "openai-compatible",
    "proxy": "openai-compatible",
    "f5-tts": "f5_tts",
    "f5tts": "f5_tts",
    "openf5": "f5_tts",
    "open-f5": "f5_tts",
}


@dataclass(frozen=True)
class EngineRuntimeStatus:
    """Whether one engine's Python runtime is importable on this machine."""

    engine: str
    label: str
    kinds: Tuple[str, ...]
    remote: bool
    installed: bool
    required_modules: Tuple[str, ...]
    missing_modules: Tuple[str, ...]
    extra: Optional[str]
    install_command: Optional[str]
    reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "label": self.label,
            "kinds": list(self.kinds),
            "remote": self.remote,
            "installed": self.installed,
            "required_modules": list(self.required_modules),
            "missing_modules": list(self.missing_modules),
            "extra": self.extra,
            "install_command": self.install_command,
            "reason": self.reason,
        }


def normalize_engine_id(engine: Any) -> str:
    """Canonical engine id (``"F5-TTS"`` -> ``"f5_tts"``, ``"qwen3_tts"`` -> ``"qwen3-tts"``)."""
    text = str(engine or "").strip().lower().replace("_", "-")
    return _ALIASES.get(text, text)


def known_engines(kind: Optional[str] = None) -> Tuple[str, ...]:
    """Every engine id this module answers for, optionally only those serving ``kind``
    (``"tts"``, ``"stt"`` or ``"cloning"``)."""
    if kind is None:
        return tuple(_ENGINES)
    wanted = _require_kind(kind)
    return tuple(engine for engine, spec in _ENGINES.items() if wanted in spec.kinds)


def engine_runtime_status(engine: Any, *, kind: Optional[str] = None) -> EngineRuntimeStatus:
    """Report whether ``engine``'s runtime packages are importable.

    ``kind`` is optional; when given, an engine that does not serve that kind
    (``faster-whisper`` for ``"tts"``) raises :class:`ValueError` rather than
    answering a question nobody should be asking.
    """
    engine_id = normalize_engine_id(engine)
    spec = _ENGINES.get(engine_id)
    if spec is None:
        raise ValueError(
            f"unknown AbstractVoice engine {engine!r}; known engines: {', '.join(_ENGINES)}"
        )
    if kind is not None and _require_kind(kind) not in spec.kinds:
        raise ValueError(f"engine {engine_id!r} does not serve {kind!r} (it serves {', '.join(spec.kinds)})")

    required = tuple(module for group in spec.module_groups for module in group)
    missing_groups = [group for group in spec.module_groups if not any(_importable(m) for m in group)]
    missing = tuple(" or ".join(group) for group in missing_groups)
    installed = not missing
    install_command = f'pip install "abstractvoice[{spec.extra}]"' if spec.extra else None
    reason = None
    if not installed:
        reason = (
            f"{spec.label} is not installed: the Python package"
            f"{'s' if len(missing) > 1 else ''} {', '.join(missing)} "
            f"{'are' if len(missing) > 1 else 'is'} missing. Install it with: {install_command}"
        )
    return EngineRuntimeStatus(
        engine=engine_id,
        label=spec.label,
        kinds=spec.kinds,
        remote=spec.remote,
        installed=installed,
        required_modules=required,
        missing_modules=missing,
        extra=spec.extra,
        install_command=install_command,
        reason=reason,
    )


def engine_runtime_installed(engine: Any, *, kind: Optional[str] = None) -> bool:
    """Shorthand for ``engine_runtime_status(engine, kind=kind).installed``."""
    return engine_runtime_status(engine, kind=kind).installed


def _require_kind(kind: str) -> str:
    text = str(kind or "").strip().lower()
    if text not in {"tts", "stt", "cloning"}:
        raise ValueError(f"kind must be 'tts', 'stt' or 'cloning', not {kind!r}")
    return text


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A parent package that is present but broken, or a module whose
        # ``__spec__`` is unset, cannot be imported either.
        return False
