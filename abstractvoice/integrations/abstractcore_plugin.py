from __future__ import annotations

import importlib.util
import os
import threading
import weakref
from typing import Any, Dict, Optional, Union

from ..artifacts import RuntimeArtifactStoreAdapter, is_artifact_ref, get_artifact_id


_VM_CACHE_LOCK = threading.Lock()
_VM_CACHE: dict[tuple, Any] = {}
_VM_LOCKS: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()
_TRUE_BOOL_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_BOOL_VALUES = {"0", "false", "no", "n", "off"}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(str(key), None)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = _env(str(key))
        if value is not None:
            return value
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in _TRUE_BOOL_VALUES:
        return True
    if text in _FALSE_BOOL_VALUES:
        return False
    return bool(default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(str(key))
    return _coerce_bool(raw, default)


def _env_float(*keys: str) -> Optional[float]:
    raw = _env_first(*keys)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _voice_profile_to_dict(profile: Any) -> Dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, dict):
        return _json_safe(profile)

    return {
        "engine_id": _json_safe(getattr(profile, "engine_id", None)),
        "profile_id": _json_safe(getattr(profile, "profile_id", None)),
        "label": _json_safe(getattr(profile, "label", None)),
        "description": _json_safe(getattr(profile, "description", None)),
        "params": _json_safe(getattr(profile, "params", None) or {}),
        "tags": _json_safe(getattr(profile, "tags", None)),
        "provenance": _json_safe(getattr(profile, "provenance", None)),
    }


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _dedupe_provider_ids(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        norm = _norm_engine_id(text) or text.lower()
        key = norm.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(norm or text)
    return out


def _norm_engine_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"remote", "compatible", "proxy"}:
        return "openai-compatible"
    if text == "faster-whisper":
        return "faster-whisper"
    return text


def _engine_aliases(value: Any) -> set[str]:
    engine = _norm_engine_id(value)
    if not engine:
        return set()
    aliases = {engine, engine.replace("-", "_")}
    if engine == "openai-compatible":
        aliases.update({"remote", "compatible", "proxy", "openai_compatible"})
    if engine in {"faster-whisper", "faster_whisper"}:
        aliases.update({"faster-whisper", "faster_whisper", "whisper", "local"})
    return aliases


def _local_tts_engines() -> list[str]:
    try:
        from ..adapters.tts_registry import get_supported_tts_engines

        engines = get_supported_tts_engines()
    except Exception:
        engines = []
    return [
        engine
        for engine in _dedupe_strings(engines)
        if _norm_engine_id(engine) not in {"", "auto", "openai", "openai-compatible"}
    ]


def _catalog_safe_local_tts_engines() -> list[str]:
    """Local engines whose catalog/profile listing is side-effect-light."""

    return [
        engine
        for engine in _local_tts_engines()
        if _norm_engine_id(engine) in {"piper", "supertonic"}
    ]


def _engine_runtime_available(engine: Any, configured_providers: Any = ()) -> bool:
    normalized = _norm_engine_id(engine)
    if not normalized:
        return False
    configured_aliases: set[str] = set()
    for provider in list(configured_providers or []):
        configured_aliases.update(_engine_aliases(provider))
    if _engine_aliases(normalized) & configured_aliases:
        return True
    if normalized in {"openai", "openai-compatible"}:
        return False
    if normalized == "piper":
        return importlib.util.find_spec("piper") is not None or importlib.util.find_spec("piper_phonemize") is not None
    if normalized == "supertonic":
        return importlib.util.find_spec("onnxruntime") is not None
    if normalized == "omnivoice":
        return importlib.util.find_spec("omnivoice") is not None
    if normalized == "f5-tts":
        return importlib.util.find_spec("f5_tts") is not None
    if normalized == "audiodit":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    return normalized in {_norm_engine_id(provider) for provider in _local_tts_engines()}


def _extract_tts_model_ids(catalog: Any) -> list[str]:
    model_ids: list[str] = []

    def add_many(values: Any) -> None:
        if isinstance(values, str):
            model_ids.append(values)
        elif isinstance(values, (list, tuple, set)):
            for value in values:
                if isinstance(value, str):
                    model_ids.append(value)
                elif isinstance(value, dict):
                    for key in ("id", "model", "model_id", "name"):
                        item = value.get(key)
                        if isinstance(item, str):
                            model_ids.append(item)
                            break

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            usable = node.get("cached") is not False or node.get("remote") is True
            if usable:
                for key in ("available_models", "tts_models", "speech_models", "audio_speech_models"):
                    add_many(node.get(key))
                for key in ("model", "model_id", "model_filename"):
                    value = node.get(key)
                    if isinstance(value, str):
                        model_ids.append(value)
            for value in node.values():
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item)

    visit(catalog)
    return _dedupe_strings(model_ids)


def _extract_provider_ids(*values: Any) -> list[str]:
    providers: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            providers.append(value.strip())

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            tags = node.get("tags")
            if isinstance(tags, dict):
                add(tags.get("provider"))
                add(tags.get("engine_id"))
            for key in ("provider", "provider_id", "engine", "engine_id", "backend", "backend_id"):
                add(node.get(key))
            for value in node.values():
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item)

    for value in values:
        visit(value)
    return _dedupe_strings(providers)


def _extract_tts_provider_ids(vm: Any, catalog: Any, profiles: Any) -> list[str]:
    adapter = getattr(vm, "tts_adapter", None)
    return _extract_provider_ids(
        {
            "engine_id": getattr(adapter, "engine_id", None),
            "provider": getattr(adapter, "provider", None),
            "backend_id": getattr(adapter, "backend_id", None),
        },
        {
            "engine_id": getattr(vm, "tts_engine", None),
            "provider_id": getattr(vm, "_abstractvoice_tts_engine", None),
            "provider": getattr(vm, "_tts_engine_name", None),
            "backend_id": getattr(vm, "tts_backend_id", None),
        },
        profiles,
        catalog,
    )


def _extract_stt_provider_ids(vm: Any) -> list[str]:
    adapter = getattr(vm, "stt_adapter", None)
    return _extract_provider_ids(
        {
            "engine_id": getattr(adapter, "engine_id", None),
            "provider": getattr(adapter, "provider", None),
            "backend_id": getattr(adapter, "backend_id", None),
        },
        {
            "engine_id": getattr(vm, "stt_engine", None),
            "provider_id": getattr(vm, "_abstractvoice_stt_engine", None),
            "provider": getattr(vm, "_stt_engine_name", None),
            "backend_id": getattr(vm, "stt_backend_id", None),
        },
    )


def _extract_stt_model_ids(vm: Any) -> list[str]:
    model_ids: list[str] = []
    for key in (
        "ABSTRACTVOICE_STT_MODEL",
        "ABSTRACTVOICE_OPENAI_STT_MODEL",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
        "ABSTRACTVOICE_REMOTE_STT_MODEL",
    ):
        value = _env(key)
        if isinstance(value, str) and value.strip():
            model_ids.extend(value.split(","))

    for target in (vm, getattr(vm, "stt_adapter", None)):
        if target is None:
            continue
        for attr in ("stt_model", "model_id", "model", "model_size", "_model_size"):
            value = getattr(target, attr, None)
            if isinstance(value, str) and value.strip():
                model_ids.append(value.strip())

    engine = _norm_engine_id(
        getattr(vm, "_abstractvoice_stt_engine", None)
        or getattr(vm, "_stt_engine_name", None)
        or getattr(vm, "_stt_engine_preference", None)
        or getattr(vm, "stt_engine", None)
        or _env("ABSTRACTVOICE_STT_ENGINE", "openai")
        or "openai"
    )
    if engine in {"openai", "openai-compatible", "remote"}:
        model_ids.extend(["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
    if engine in {"faster_whisper", "faster-whisper", "whisper", "local"}:
        try:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            model_ids.extend(list(getattr(FasterWhisperAdapter, "MODELS", {}).keys()))
        except Exception:
            model_ids.extend(["tiny", "base", "small", "medium", "large-v3"])
    return _dedupe_strings(model_ids)


def _active_tts_model(vm: Any, catalog: Any, model_ids: list[str]) -> Optional[str]:
    adapter = getattr(vm, "tts_adapter", None)
    model = getattr(adapter, "model_id", None)
    if isinstance(model, str) and model.strip():
        return model.strip()

    if isinstance(catalog, dict):
        for engine_catalog in catalog.values():
            if not isinstance(engine_catalog, dict):
                continue
            for item in engine_catalog.values():
                if isinstance(item, dict):
                    model = item.get("model") or item.get("model_id")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
    return model_ids[0] if model_ids else None


def _profiles_from_tts_catalog(catalog: Any, *, engine_id: str) -> list[Dict[str, Any]]:
    """Build profile-like records from adapter catalogs that expose cached voices."""
    engine = _norm_engine_id(engine_id)
    if not engine or not isinstance(catalog, dict):
        return []

    profiles: list[Dict[str, Any]] = []

    for group_key, group_value in catalog.items():
        if not isinstance(group_value, dict):
            continue
        for voice_key, raw in group_value.items():
            if not isinstance(raw, dict):
                continue
            if raw.get("cached") is False and raw.get("remote") is not True:
                continue
            model_id = raw.get("model") or raw.get("model_id") or raw.get("model_filename")
            profile_id = str(model_id or f"{group_key}.{voice_key}").strip()
            if not profile_id:
                continue
            label = str(raw.get("name") or raw.get("label") or profile_id).strip() or profile_id
            profiles.append(
                {
                    "engine_id": engine,
                    "profile_id": profile_id,
                    "id": profile_id,
                    "label": label,
                    "description": _json_safe(raw.get("description")),
                    "params": {
                        "provider": engine,
                        "model": profile_id,
                        "voice": str(raw.get("voice") or voice_key).strip(),
                        "language": str(group_key).strip(),
                    },
                    "tags": {
                        "provider": engine,
                        "engine_id": engine,
                        "cached": "true",
                    },
                    "provenance": "abstractvoice.adapter_catalog",
                }
            )

    return profiles


def _piper_language_for_model(model_id: str) -> Optional[str]:
    requested = str(model_id or "").strip().lower()
    if not requested:
        return None
    try:
        from ..adapters.tts_piper import PiperTTSAdapter
    except Exception:
        return None

    for language, raw in getattr(PiperTTSAdapter, "PIPER_MODELS", {}).items():
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        hf_path, model_filename = str(raw[0] or ""), str(raw[1] or "")
        parts = hf_path.split("/")
        voice_id = parts[2] if len(parts) >= 3 else ""
        candidates = {
            str(language).lower(),
            model_filename.lower(),
            f"{language}:{model_filename}".lower(),
            f"{language}/{model_filename}".lower(),
        }
        if voice_id:
            candidates.update(
                {
                    voice_id.lower(),
                    f"{language}.{voice_id}".lower(),
                    f"{language}:{voice_id}".lower(),
                }
            )
        if requested in candidates:
            return str(language)
    return None


def _tts_formats_for_provider(provider: Any) -> list[str]:
    engine = _norm_engine_id(provider)
    if engine == "piper":
        return ["wav"]
    if engine in {"openai", "openai-compatible", "remote"}:
        return ["mp3", "opus", "aac", "flac", "wav", "pcm"]
    return ["wav"]


def _stt_formats_for_provider(provider: Any) -> list[str]:
    engine = _norm_engine_id(provider)
    if engine in {"openai", "openai-compatible", "remote"}:
        return ["json", "text", "verbose_json", "srt", "vtt"]
    return ["json", "text"]


def _profile_provider_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    tags = profile.get("tags")
    params = profile.get("params")
    if not isinstance(tags, dict):
        tags = {}
    if not isinstance(params, dict):
        params = {}
    return _norm_engine_id(
        profile.get("provider")
        or profile.get("engine_id")
        or tags.get("provider")
        or tags.get("engine_id")
        or params.get("provider")
        or params.get("engine_id")
    )


def _profile_model_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    params = profile.get("params")
    if not isinstance(params, dict):
        params = {}
    for value in (
        params.get("model"),
        params.get("model_id"),
        params.get("model_filename"),
        profile.get("model"),
        profile.get("model_id"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profile_voice_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    params = profile.get("params")
    if not isinstance(params, dict):
        params = {}
    for value in (
        profile.get("voice_id"),
        params.get("voice"),
        profile.get("voice"),
        profile.get("profile_id"),
        profile.get("id"),
        profile.get("name"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profile_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    for value in (profile.get("profile_id"), profile.get("id"), profile.get("name"), _profile_voice_id(profile)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _add_provider_value(mapping: dict[str, list[str]], provider: Any, value: Any) -> None:
    provider_id = _norm_engine_id(provider)
    text = str(value or "").strip()
    if not provider_id or not text:
        return
    values = mapping.setdefault(provider_id, [])
    if text.lower() not in {item.lower() for item in values}:
        values.append(text)


class _BaseVoice:
    def __init__(self, owner: Any):
        self._owner = owner
        self._vm = None

    def _vm_lock(self, vm: Any) -> threading.Lock:
        """Per-VoiceManager lock for synthesis/metrics consistency."""
        with _VM_CACHE_LOCK:
            lk = _VM_LOCKS.get(vm)
            if lk is None:
                lk = threading.Lock()
                _VM_LOCKS[vm] = lk
            return lk

    def _get_vm(self):
        if self._vm is not None:
            return self._vm

        # Injection hook (tests / advanced embedding).
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                inst = cfg.get("voice_manager_instance")
                if inst is not None:
                    self._vm = inst
                    return self._vm
                factory = cfg.get("voice_manager_factory")
                if callable(factory):
                    self._vm = factory(self._owner)
                    return self._vm
        except Exception:
            pass

        # Lazy import (keeps plugin import-light).
        from ..voice_manager import VoiceManager

        # Best-effort config overrides (optional). Hosted integrations and the
        # public VoiceManager default both select OpenAI remote engines.
        language = _env("ABSTRACTVOICE_LANGUAGE", "en") or "en"
        allow_downloads = _env_bool("ABSTRACTVOICE_ALLOW_DOWNLOADS", True)
        tts_engine = _env("ABSTRACTVOICE_TTS_ENGINE", "openai") or "openai"
        stt_engine = _env("ABSTRACTVOICE_STT_ENGINE", "openai") or "openai"
        whisper_model = _env("ABSTRACTVOICE_WHISPER_MODEL", "base") or "base"
        stt_model = _env_first(
            "ABSTRACTVOICE_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
            "ABSTRACTVOICE_REMOTE_STT_MODEL",
        )
        tts_model = _env_first(
            "ABSTRACTVOICE_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL",
        )
        cloning_engine = _env("ABSTRACTVOICE_CLONING_ENGINE", "f5_tts") or "f5_tts"
        cloned_tts_streaming = _env_bool("ABSTRACTVOICE_CLONED_TTS_STREAMING", True)
        tts_delivery_mode = _env("ABSTRACTVOICE_TTS_DELIVERY_MODE")
        remote_base_url = _env_first(
            "ABSTRACTVOICE_REMOTE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        )
        remote_api_key = _env_first(
            "ABSTRACTVOICE_REMOTE_API_KEY",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
            "ABSTRACTVOICE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
        remote_timeout_s = _env_float(
            "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TIMEOUT_S",
        )
        debug_mode = _env_bool("ABSTRACTVOICE_DEBUG", False)
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                if isinstance(cfg.get("voice_language"), str) and cfg["voice_language"].strip():
                    language = str(cfg["voice_language"]).strip().lower()
                if "voice_allow_downloads" in cfg:
                    allow_downloads = _coerce_bool(cfg.get("voice_allow_downloads"), allow_downloads)
                if isinstance(cfg.get("voice_tts_engine"), str) and str(cfg["voice_tts_engine"]).strip():
                    tts_engine = str(cfg["voice_tts_engine"]).strip().lower()
                if isinstance(cfg.get("voice_stt_engine"), str) and str(cfg["voice_stt_engine"]).strip():
                    stt_engine = str(cfg["voice_stt_engine"]).strip().lower()
                if isinstance(cfg.get("voice_whisper_model"), str) and str(cfg["voice_whisper_model"]).strip():
                    whisper_model = str(cfg["voice_whisper_model"]).strip()
                if isinstance(cfg.get("voice_tts_model"), str) and str(cfg["voice_tts_model"]).strip():
                    tts_model = str(cfg["voice_tts_model"]).strip()
                if isinstance(cfg.get("voice_stt_model"), str) and str(cfg["voice_stt_model"]).strip():
                    stt_model = str(cfg["voice_stt_model"]).strip()
                if isinstance(cfg.get("voice_cloning_engine"), str) and str(cfg["voice_cloning_engine"]).strip():
                    cloning_engine = str(cfg["voice_cloning_engine"]).strip().lower()
                if isinstance(cfg.get("voice_remote_base_url"), str) and str(cfg["voice_remote_base_url"]).strip():
                    remote_base_url = str(cfg["voice_remote_base_url"]).strip()
                if isinstance(cfg.get("voice_remote_api_key"), str) and str(cfg["voice_remote_api_key"]).strip():
                    remote_api_key = str(cfg["voice_remote_api_key"]).strip()
                if cfg.get("voice_remote_timeout_s") not in (None, ""):
                    try:
                        remote_timeout_s = float(cfg.get("voice_remote_timeout_s"))
                    except Exception:
                        remote_timeout_s = None
                if "voice_cloned_tts_streaming" in cfg:
                    cloned_tts_streaming = _coerce_bool(
                        cfg.get("voice_cloned_tts_streaming"),
                        cloned_tts_streaming,
                    )
                # Unified override for delivery mode (applies to base + clone).
                # Accept either a mode string (buffered|streamed) or a bool-ish flag.
                if "voice_tts_delivery_mode" in cfg:
                    raw = cfg.get("voice_tts_delivery_mode")
                    if raw is not None and str(raw).strip():
                        try:
                            from ..tts.delivery_mode import normalize_audio_delivery_mode

                            tts_delivery_mode = normalize_audio_delivery_mode(str(raw))
                        except Exception:
                            tts_delivery_mode = None
                elif "voice_tts_streaming" in cfg:
                    raw = cfg.get("voice_tts_streaming")
                    try:
                        from ..tts.delivery_mode import normalize_audio_delivery_mode

                        tts_delivery_mode = normalize_audio_delivery_mode(raw)
                    except Exception:
                        tts_delivery_mode = None
                if "voice_debug_mode" in cfg:
                    debug_mode = _coerce_bool(cfg.get("voice_debug_mode"), debug_mode)
        except Exception:
            pass

        key = (
            str(language),
            bool(allow_downloads),
            str(tts_engine),
            str(stt_engine),
            str(whisper_model),
            str(tts_model or ""),
            str(stt_model or ""),
            str(cloning_engine),
            bool(cloned_tts_streaming),
            str(tts_delivery_mode) if tts_delivery_mode else "",
            str(remote_base_url or ""),
            str(remote_api_key or ""),
            str(remote_timeout_s or ""),
            bool(debug_mode),
        )

        with _VM_CACHE_LOCK:
            cached = _VM_CACHE.get(key)
            if cached is None:
                cached = VoiceManager(
                    language=language,
                    allow_downloads=allow_downloads,
                    debug_mode=bool(debug_mode),
                    tts_engine=str(tts_engine),
                    stt_engine=str(stt_engine),
                    whisper_model=str(whisper_model),
                    tts_model=str(tts_model) if tts_model else None,
                    stt_model=str(stt_model) if stt_model else None,
                    cloning_engine=str(cloning_engine),
                    cloned_tts_streaming=bool(cloned_tts_streaming),
                    tts_delivery_mode=str(tts_delivery_mode) if tts_delivery_mode else None,
                    remote_base_url=str(remote_base_url) if remote_base_url else None,
                    remote_api_key=str(remote_api_key) if remote_api_key else None,
                    remote_timeout_s=remote_timeout_s,
                )
                _VM_CACHE[key] = cached
                _VM_LOCKS[cached] = threading.Lock()
            try:
                setattr(cached, "_abstractvoice_tts_engine", str(tts_engine))
                setattr(cached, "_abstractvoice_stt_engine", str(stt_engine))
            except Exception:
                pass
            self._vm = cached
            return self._vm

    def _vm_engine_values(self, vm: Any, *, kind: str) -> set[str]:
        adapter = getattr(vm, "tts_adapter", None) if kind == "tts" else getattr(vm, "stt_adapter", None)
        values = {
            getattr(adapter, "engine_id", None),
            getattr(adapter, "provider", None),
        }
        if kind == "tts":
            values.update(
                {
                    getattr(vm, "_abstractvoice_tts_engine", None),
                    getattr(vm, "_tts_engine_name", None),
                    getattr(vm, "_tts_engine_preference", None),
                }
            )
        else:
            values.update(
                {
                    getattr(vm, "_abstractvoice_stt_engine", None),
                    getattr(vm, "_stt_engine_name", None),
                    getattr(vm, "_stt_engine_preference", None),
                }
            )
        out: set[str] = set()
        for value in values:
            out.update(_engine_aliases(value))
        return out

    def _config_text(self, *keys: str) -> Optional[str]:
        cfg = getattr(self._owner, "config", None)
        if isinstance(cfg, dict):
            for key in keys:
                value = cfg.get(str(key))
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _configured_remote_tts_engines(self) -> list[str]:
        engines: list[str] = []

        def add(value: Any) -> None:
            engine = _norm_engine_id(value)
            if engine in {"openai", "openai-compatible"}:
                engines.append(engine)

        requested = _norm_engine_id(
            self._config_text("voice_tts_engine")
            or _env_first("ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTGATEWAY_VOICE_TTS_ENGINE")
        )
        add(requested)

        remote_base_url = self._config_text("voice_remote_base_url") or _env_first(
            "ABSTRACTVOICE_REMOTE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        )
        remote_api_key = self._config_text("voice_remote_api_key") or _env_first(
            "ABSTRACTVOICE_REMOTE_API_KEY",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
            "ABSTRACTVOICE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
        openai_specific = _env_first(
            "ABSTRACTVOICE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_TTS_MODELS",
            "ABSTRACTVOICE_OPENAI_TTS_VOICE",
            "ABSTRACTVOICE_OPENAI_TTS_VOICES",
        )
        compatible_specific = _env_first(
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
            "ABSTRACTVOICE_REMOTE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
            "ABSTRACTVOICE_REMOTE_API_KEY",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODELS",
            "ABSTRACTVOICE_REMOTE_TTS_MODELS",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICE",
            "ABSTRACTVOICE_REMOTE_TTS_VOICE",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICES",
            "ABSTRACTVOICE_REMOTE_TTS_VOICES",
        )

        base_url_s = str(remote_base_url or "").strip().lower()
        if requested == "openai" or openai_specific or (remote_api_key and ("api.openai.com" in base_url_s or not base_url_s)):
            add("openai")
        if requested == "openai-compatible" or compatible_specific or (remote_base_url and "api.openai.com" not in base_url_s):
            add("openai-compatible")

        return _dedupe_strings(engines)

    def _configured_tts_model_ids(self, engine: str) -> list[str]:
        provider = _norm_engine_id(engine)
        values: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.extend(value.split(","))

        add(self._config_text("voice_tts_model"))
        if provider == "openai":
            for key in (
                "ABSTRACTVOICE_OPENAI_TTS_MODEL",
                "ABSTRACTVOICE_OPENAI_TTS_MODELS",
                "ABSTRACTVOICE_TTS_MODEL",
                "ABSTRACTVOICE_REMOTE_TTS_MODEL",
            ):
                add(_env(key))
            values.extend(["gpt-4o-mini-tts", "tts-1", "tts-1-hd"])
        elif provider == "openai-compatible":
            for key in (
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODELS",
                "ABSTRACTVOICE_REMOTE_TTS_MODEL",
                "ABSTRACTVOICE_REMOTE_TTS_MODELS",
                "ABSTRACTVOICE_TTS_MODEL",
            ):
                add(_env(key))
        return _dedupe_strings(values)

    def _configured_remote_stt_engines(self) -> list[str]:
        engines: list[str] = []

        def add(value: Any) -> None:
            engine = _norm_engine_id(value)
            if engine in {"openai", "openai-compatible"}:
                engines.append(engine)

        requested = _norm_engine_id(
            self._config_text("voice_stt_engine")
            or _env_first("ABSTRACTVOICE_STT_ENGINE", "ABSTRACTGATEWAY_VOICE_STT_ENGINE")
        )
        add(requested)

        remote_base_url = self._config_text("voice_remote_base_url") or _env_first(
            "ABSTRACTVOICE_REMOTE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_BASE_URL",
            "OPENAI_BASE_URL",
        )
        remote_api_key = self._config_text("voice_remote_api_key") or _env_first(
            "ABSTRACTVOICE_REMOTE_API_KEY",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
            "ABSTRACTVOICE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        )
        openai_specific = _env_first(
            "ABSTRACTVOICE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
            "ABSTRACTVOICE_OPENAI_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_STT_MODELS",
        )
        compatible_specific = _env_first(
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
            "ABSTRACTVOICE_REMOTE_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
            "ABSTRACTVOICE_REMOTE_API_KEY",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
            "ABSTRACTVOICE_REMOTE_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODELS",
            "ABSTRACTVOICE_REMOTE_STT_MODELS",
        )

        base_url_s = str(remote_base_url or "").strip().lower()
        if requested == "openai" or openai_specific or (remote_api_key and ("api.openai.com" in base_url_s or not base_url_s)):
            add("openai")
        if requested == "openai-compatible" or compatible_specific or (remote_base_url and "api.openai.com" not in base_url_s):
            add("openai-compatible")

        return _dedupe_strings(engines)

    def _configured_stt_model_ids(self, engine: str) -> list[str]:
        provider = _norm_engine_id(engine)
        values: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.extend(value.split(","))

        add(self._config_text("voice_stt_model"))
        if provider == "openai":
            for key in (
                "ABSTRACTVOICE_OPENAI_STT_MODEL",
                "ABSTRACTVOICE_OPENAI_STT_MODELS",
                "ABSTRACTVOICE_STT_MODEL",
                "ABSTRACTVOICE_REMOTE_STT_MODEL",
            ):
                add(_env(key))
            values.extend(["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
        elif provider == "openai-compatible":
            for key in (
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODELS",
                "ABSTRACTVOICE_REMOTE_STT_MODEL",
                "ABSTRACTVOICE_REMOTE_STT_MODELS",
                "ABSTRACTVOICE_STT_MODEL",
            ):
                add(_env(key))
        return _dedupe_strings(values)

    def _get_vm_for_provider(
        self,
        *,
        tts_provider: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
        stt_model: Optional[str] = None,
    ):
        tts_engine = _norm_engine_id(tts_provider)
        stt_engine = _norm_engine_id(stt_provider)
        if tts_engine == "cloned":
            tts_engine = ""
        if stt_engine == "cloned":
            stt_engine = ""

        if not tts_engine and not stt_engine:
            return self._get_vm()

        current = None
        try:
            current = self._get_vm()
        except Exception:
            current = None

        if current is not None:
            tts_ok = not tts_engine or bool(_engine_aliases(tts_engine) & self._vm_engine_values(current, kind="tts"))
            stt_ok = not stt_engine or bool(_engine_aliases(stt_engine) & self._vm_engine_values(current, kind="stt"))
            if tts_ok and stt_ok:
                return current

        cfg = getattr(self._owner, "config", None)
        override_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        # Provider-specific routing must not be short-circuited by injected active
        # VoiceManager instances. The module-level VoiceManager cache still keeps
        # repeated provider/model requests cheap.
        override_cfg.pop("voice_manager_instance", None)
        override_cfg.pop("voice_manager_factory", None)

        if tts_engine:
            override_cfg["voice_tts_engine"] = tts_engine
        if stt_engine:
            override_cfg["voice_stt_engine"] = stt_engine
        if isinstance(tts_model, str) and tts_model.strip():
            override_cfg["voice_tts_model"] = tts_model.strip()
        if isinstance(stt_model, str) and stt_model.strip():
            override_cfg["voice_stt_model"] = stt_model.strip()
            if stt_engine in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                override_cfg["voice_whisper_model"] = stt_model.strip()

        owner = type("_AbstractVoiceProviderOverride", (), {"config": override_cfg})()
        cap = self.__class__(owner)
        return cap._get_vm()

    def _maybe_store_audio(
        self,
        audio_bytes: bytes,
        *,
        artifact_store: Any,
        fmt: str,
        run_id: Optional[str],
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
    ):
        if artifact_store is None:
            return bytes(audio_bytes)
        store = RuntimeArtifactStoreAdapter(artifact_store)
        merged_tags: Dict[str, str] = {"kind": "generated_media", "modality": "audio", "task": "tts"}
        if isinstance(tags, dict):
            merged_tags.update({str(k): str(v) for k, v in tags.items()})
        return store.store_bytes(
            bytes(audio_bytes),
            content_type=f"audio/{str(fmt).lower()}",
            filename=f"tts.{str(fmt).lower()}",
            run_id=str(run_id) if run_id else None,
            tags=merged_tags,
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def _resolve_audio_bytes(self, audio: Union[bytes, Dict[str, Any], str], *, artifact_store: Any) -> bytes:
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        if isinstance(audio, dict):
            if not is_artifact_ref(audio):
                raise ValueError("Expected an artifact ref dict like {'$artifact': '...'}")
            if artifact_store is None:
                raise ValueError("artifact_store is required to resolve artifact refs to bytes")
            store = RuntimeArtifactStoreAdapter(artifact_store)
            return store.load_bytes(get_artifact_id(audio))
        if isinstance(audio, str):
            from pathlib import Path

            p = Path(audio).expanduser()
            if p.exists() and p.is_file():
                return p.read_bytes()
            raise FileNotFoundError(f"File not found: {audio}")
        raise TypeError("Unsupported input type; expected bytes, artifact-ref dict, or file path")

    def _suffix_for_audio_ref(self, audio: Dict[str, Any], *, artifact_store: Any) -> str:
        """Pick a best-effort file suffix for an audio artifact-ref dict."""
        import mimetypes
        from pathlib import Path

        # Prefer explicit filename when provided (most clients include it).
        try:
            filename = audio.get("filename")
            if isinstance(filename, str) and filename.strip():
                suf = Path(filename.strip()).suffix
                if isinstance(suf, str) and suf and len(suf) <= 10:
                    return suf
        except Exception:
            pass

        # Next: content_type from ref (or artifact metadata when available).
        content_type: Optional[str] = None
        try:
            ct = audio.get("content_type")
            if isinstance(ct, str) and ct.strip():
                content_type = ct.strip()
        except Exception:
            content_type = None

        if content_type is None and artifact_store is not None:
            try:
                store = RuntimeArtifactStoreAdapter(artifact_store)
                meta = store.get_metadata(get_artifact_id(audio))
                if isinstance(meta, dict):
                    ct2 = meta.get("content_type")
                    if isinstance(ct2, str) and ct2.strip():
                        content_type = ct2.strip()
                    fn2 = meta.get("filename")
                    if isinstance(fn2, str) and fn2.strip():
                        suf = Path(fn2.strip()).suffix
                        if isinstance(suf, str) and suf and len(suf) <= 10:
                            return suf
            except Exception:
                pass

        if isinstance(content_type, str) and content_type.strip():
            # Drop charset/params (e.g. "audio/wav; codecs=...").
            base = content_type.split(";", 1)[0].strip().lower()
            ext = mimetypes.guess_extension(base) or ""
            if ext:
                return ext

        return ".bin"


class _VoiceCapability(_BaseVoice):
    backend_id = "abstractvoice:default"

    def list_profiles(self, *, kind: str = "tts") -> list[Dict[str, Any]]:
        """List active-engine voice profiles through the plugin boundary."""
        vm = self._get_vm()
        profiles = []
        if hasattr(vm, "get_profiles"):
            profiles = list(vm.get_profiles(kind=str(kind or "tts")) or [])
        return [_voice_profile_to_dict(profile) for profile in profiles]

    def list_tts_models(self) -> list[str]:
        """List deduplicated TTS model ids from serveable AbstractVoice engines."""
        model_ids: list[str] = []
        active_vm = self._get_vm()
        active_catalog: Any = {}
        if hasattr(active_vm, "list_available_models"):
            active_catalog = active_vm.list_available_models()
        model_ids.extend(_extract_tts_model_ids(active_catalog))

        active_engines = self._vm_engine_values(active_vm, kind="tts")
        for engine in self._configured_remote_tts_engines():
            model_ids.extend(self._configured_tts_model_ids(engine))
            if _engine_aliases(engine) & active_engines:
                continue
            try:
                vm = self._get_vm_for_provider(tts_provider=engine)
                catalog = vm.list_available_models() if hasattr(vm, "list_available_models") else {}
                model_ids.extend(_extract_tts_model_ids(catalog))
            except Exception:
                continue

        for engine in _catalog_safe_local_tts_engines():
            if _engine_aliases(engine) & active_engines:
                continue
            try:
                vm = self._get_vm_for_provider(tts_provider=engine)
                catalog = vm.list_available_models() if hasattr(vm, "list_available_models") else {}
                catalog_engines = set()
                for provider_id in _extract_provider_ids(catalog):
                    catalog_engines.update(_engine_aliases(provider_id))
                if _norm_engine_id(engine) != "piper" and not (_engine_aliases(engine) & catalog_engines):
                    continue
                model_ids.extend(_extract_tts_model_ids(catalog))
            except Exception:
                continue
        return _dedupe_strings(model_ids)

    def list_stt_models(self) -> list[str]:
        """List deduplicated STT model ids from serveable AbstractVoice engines."""
        model_ids: list[str] = []
        vm = self._get_vm()
        model_ids.extend(_extract_stt_model_ids(vm))
        for engine in self._configured_remote_stt_engines():
            model_ids.extend(self._configured_stt_model_ids(engine))
        try:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            model_ids.extend(list(getattr(FasterWhisperAdapter, "MODELS", {}).keys()))
        except Exception:
            pass
        return _dedupe_strings(model_ids)

    def voice_catalog(self) -> Dict[str, Any]:
        """Return JSON-safe profile/model discovery data for Core/Gateway."""
        vm = self._get_vm()

        profiles = self.list_profiles(kind="tts")
        active_profile = None
        if hasattr(vm, "get_active_profile"):
            active_profile = vm.get_active_profile(kind="tts")

        catalog: Any = {}
        if hasattr(vm, "list_available_models"):
            catalog = vm.list_available_models()
        tts_models = _extract_tts_model_ids(catalog)
        stt_models = _extract_stt_model_ids(vm)
        tts_providers = _extract_tts_provider_ids(vm, catalog, profiles)
        stt_providers = _extract_stt_provider_ids(vm)
        catalogs: Dict[str, Any] = {}
        active_engine = _norm_engine_id(tts_providers[0] if tts_providers else getattr(vm, "_abstractvoice_tts_engine", None))
        if active_engine:
            catalogs[active_engine] = _json_safe(catalog)
            profiles.extend(_profiles_from_tts_catalog(catalog, engine_id=active_engine))

        active_tts_engines = self._vm_engine_values(vm, kind="tts")
        for engine in self._configured_remote_tts_engines():
            tts_providers.append(engine)
            tts_models.extend(self._configured_tts_model_ids(engine))
            if _engine_aliases(engine) & active_tts_engines:
                continue
            try:
                engine_vm = self._get_vm_for_provider(tts_provider=engine)
                engine_catalog = engine_vm.list_available_models() if hasattr(engine_vm, "list_available_models") else {}
                engine_models = _extract_tts_model_ids(engine_catalog)
                engine_profiles: list[Dict[str, Any]] = []
                if hasattr(engine_vm, "get_profiles"):
                    try:
                        engine_profiles.extend(_voice_profile_to_dict(p) for p in list(engine_vm.get_profiles(kind="tts") or []))
                    except Exception:
                        pass
                engine_profiles.extend(_profiles_from_tts_catalog(engine_catalog, engine_id=engine))
                tts_models.extend(engine_models)
                profiles.extend(engine_profiles)
                catalogs[_norm_engine_id(engine)] = _json_safe(engine_catalog)
            except Exception:
                continue

        for engine in _catalog_safe_local_tts_engines():
            if _engine_aliases(engine) & active_tts_engines:
                continue
            try:
                engine_vm = self._get_vm_for_provider(tts_provider=engine)
                engine_catalog = engine_vm.list_available_models() if hasattr(engine_vm, "list_available_models") else {}
                engine_models = _extract_tts_model_ids(engine_catalog)
                engine_profiles: list[Dict[str, Any]] = []
                if hasattr(engine_vm, "get_profiles"):
                    try:
                        engine_profiles.extend(_voice_profile_to_dict(p) for p in list(engine_vm.get_profiles(kind="tts") or []))
                    except Exception:
                        pass
                engine_profiles.extend(_profiles_from_tts_catalog(engine_catalog, engine_id=engine))
                catalog_engines = set()
                for provider_id in _extract_provider_ids(engine_catalog):
                    catalog_engines.update(_engine_aliases(provider_id))
                if _norm_engine_id(engine) != "piper" and not (_engine_aliases(engine) & catalog_engines):
                    continue
                if not engine_models and not engine_profiles:
                    continue
                tts_models.extend(engine_models)
                tts_providers.append(engine)
                profiles.extend(engine_profiles)
                catalogs[_norm_engine_id(engine)] = _json_safe(engine_catalog)
            except Exception:
                continue

        try:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            stt_providers.append("faster-whisper")
            stt_models.extend(list(getattr(FasterWhisperAdapter, "MODELS", {}).keys()))
        except Exception:
            pass

        for engine in self._configured_remote_stt_engines():
            stt_providers.append(engine)
            stt_models.extend(self._configured_stt_model_ids(engine))

        tts_models = _dedupe_strings(tts_models)
        stt_models = _dedupe_strings(stt_models)
        tts_providers = _dedupe_provider_ids(tts_providers)
        stt_providers = _dedupe_provider_ids(stt_providers)
        adapter = getattr(vm, "tts_adapter", None)
        cloned_voices: list[Dict[str, Any]] = []
        if hasattr(vm, "list_cloned_voices"):
            try:
                raw_clones = vm.list_cloned_voices()
            except Exception:
                raw_clones = []
            if isinstance(raw_clones, list):
                for item in raw_clones:
                    if isinstance(item, dict):
                        clone_id = item.get("voice_id") or item.get("id") or item.get("name")
                        clone_name = item.get("name") or item.get("label") or clone_id
                        if isinstance(clone_id, str) and clone_id.strip():
                            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                            clone_engine = _norm_engine_id(
                                item.get("engine")
                                or item.get("engine_id")
                                or meta.get("engine")
                                or meta.get("tts_engine")
                                or getattr(vm, "_abstractvoice_tts_engine", None)
                                or getattr(vm, "tts_engine", None)
                                or active_engine
                            )
                            if clone_engine and not _engine_runtime_available(clone_engine, tts_providers):
                                continue
                            if clone_engine:
                                tts_providers.append(clone_engine)
                            label = str(clone_name or clone_id).strip() or clone_id.strip()
                            cloned_voices.append(
                                {
                                    "id": clone_id.strip(),
                                    "voice_id": clone_id.strip(),
                                    "profile_id": clone_id.strip(),
                                    "label": label,
                                    "display_name": label,
                                    "name": label,
                                    "kind": "clone",
                                    "provider": clone_engine,
                                    "engine_id": clone_engine,
                                    "engine": clone_engine,
                                    "tags": {
                                        "provider": clone_engine,
                                        "engine_id": clone_engine,
                                        "engine": clone_engine,
                                        "source": "abstractvoice_clone_store",
                                    },
                                    "params": _json_safe(item),
                                }
                            )
                    elif isinstance(item, str) and item.strip():
                        clone_engine = _norm_engine_id(
                            getattr(vm, "_abstractvoice_tts_engine", None)
                            or getattr(vm, "tts_engine", None)
                            or active_engine
                        )
                        if clone_engine and not _engine_runtime_available(clone_engine, tts_providers):
                            continue
                        if clone_engine:
                            tts_providers.append(clone_engine)
                        cloned_voices.append(
                            {
                                "id": item.strip(),
                                "voice_id": item.strip(),
                                "profile_id": item.strip(),
                                "label": item.strip(),
                                "display_name": item.strip(),
                                "name": item.strip(),
                                "kind": "clone",
                                "provider": clone_engine,
                                "engine_id": clone_engine,
                                "engine": clone_engine,
                            }
                        )
        if cloned_voices:
            tts_providers = _dedupe_provider_ids(tts_providers)

        tts_models_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        tts_voices_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        tts_profiles_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        for profile in profiles:
            provider = _profile_provider_id(profile)
            _add_provider_value(tts_models_by_provider, provider, _profile_model_id(profile))
            _add_provider_value(tts_voices_by_provider, provider, _profile_voice_id(profile))
            _add_provider_value(tts_profiles_by_provider, provider, _profile_id(profile))

        for engine in self._configured_remote_tts_engines():
            provider = _norm_engine_id(engine)
            for model_id in self._configured_tts_model_ids(engine):
                _add_provider_value(tts_models_by_provider, provider, model_id)

        for model_id in tts_models:
            if len(tts_providers) == 1:
                _add_provider_value(tts_models_by_provider, tts_providers[0], model_id)

        for clone in cloned_voices:
            clone_provider = _profile_provider_id(clone) or active_engine
            _add_provider_value(tts_voices_by_provider, clone_provider, _profile_voice_id(clone))
            _add_provider_value(tts_profiles_by_provider, clone_provider, _profile_id(clone))

        stt_models_by_provider: dict[str, list[str]] = {provider: [] for provider in stt_providers}
        for engine in self._configured_remote_stt_engines():
            provider = _norm_engine_id(engine)
            for model_id in self._configured_stt_model_ids(engine):
                _add_provider_value(stt_models_by_provider, provider, model_id)
        if "faster-whisper" in stt_providers:
            try:
                from ..adapters.stt_faster_whisper import FasterWhisperAdapter

                for model_id in getattr(FasterWhisperAdapter, "MODELS", {}).keys():
                    _add_provider_value(stt_models_by_provider, "faster-whisper", model_id)
            except Exception:
                pass

        return {
            "kind": "tts",
            "engine_id": getattr(adapter, "engine_id", None) or getattr(vm, "_tts_engine_name", None),
            "active_profile": _voice_profile_to_dict(active_profile) if active_profile is not None else None,
            "active_model": _active_tts_model(vm, catalog, tts_models),
            "active_tts_provider": tts_providers[0] if tts_providers else None,
            "active_stt_provider": stt_providers[0] if stt_providers else None,
            "profiles": profiles,
            "voices": profiles + cloned_voices,
            "cloned_voices": cloned_voices,
            "tts_providers": tts_providers,
            "stt_providers": stt_providers,
            "tts_models": tts_models,
            "stt_models": stt_models,
            "tts_models_by_provider": tts_models_by_provider,
            "stt_models_by_provider": stt_models_by_provider,
            "tts_voices_by_provider": tts_voices_by_provider,
            "tts_profiles_by_provider": tts_profiles_by_provider,
            "tts_formats_by_provider": {provider: _tts_formats_for_provider(provider) for provider in tts_providers},
            "stt_formats_by_provider": {provider: _stt_formats_for_provider(provider) for provider in stt_providers},
            "controls": {
                "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
                "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
                "instructions": {"supported": True},
                "profile": {"supported": True},
                "voice_clone": {"supported": True},
            },
            "catalog": _json_safe(catalog),
            "catalogs": catalogs,
        }

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ):
        vm = self._get_vm_for_provider(tts_provider=provider, tts_model=model)
        lk = self._vm_lock(vm)
        with lk:
            model_name = str(model or "").strip() if isinstance(model, str) else ""
            profile_name = str(_kwargs.get("profile") or "").strip() if isinstance(_kwargs.get("profile"), str) else ""
            voice_name = str(voice).strip() if isinstance(voice, str) and voice.strip() else None
            quality_preset = str(_kwargs.get("quality_preset") or _kwargs.get("quality") or "").strip()
            speed_value = _kwargs.get("speed")
            provider_id = _norm_engine_id(
                provider
                or getattr(vm, "_abstractvoice_tts_engine", None)
                or getattr(vm, "_tts_engine_name", None)
                or getattr(getattr(vm, "tts_adapter", None), "engine_id", None)
            )
            piper_voice_is_profile = False
            sentinel = object()
            old_vm_model = getattr(vm, "tts_model", sentinel)
            old_language = getattr(vm, "language", sentinel)
            old_profile_id = ""
            adapter = getattr(vm, "tts_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            old_speed = getattr(vm, "speed", sentinel)
            old_tts_quality = sentinel
            old_cloned_quality = sentinel
            applied_profile = False
            try:
                if speed_value is not None and hasattr(vm, "set_speed"):
                    try:
                        vm.set_speed(float(speed_value))
                    except Exception:
                        pass
                if (profile_name or voice_name) and hasattr(vm, "get_active_profile"):
                    try:
                        old_profile = vm.get_active_profile(kind="tts")
                        old_profile_id = str(getattr(old_profile, "profile_id", "") or "").strip()
                    except Exception:
                        old_profile_id = ""
                if model_name:
                    try:
                        setattr(vm, "tts_model", model_name)
                    except Exception:
                        pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                    if _norm_engine_id(provider) == "piper":
                        language = _piper_language_for_model(model_name)
                        if language and hasattr(vm, "set_language"):
                            vm.set_language(language)
                if provider_id == "piper":
                    for candidate in (profile_name, model_name, voice_name):
                        language = _piper_language_for_model(str(candidate or ""))
                        if not language:
                            continue
                        if hasattr(vm, "set_language"):
                            vm.set_language(language)
                        if voice_name and str(candidate) == voice_name:
                            piper_voice_is_profile = True
                        break
                is_cloned_voice = False
                if voice_name and hasattr(vm, "get_cloned_voice"):
                    try:
                        is_cloned_voice = bool(vm.get_cloned_voice(voice_name))
                    except Exception:
                        is_cloned_voice = False
                if quality_preset:
                    if not is_cloned_voice and hasattr(vm, "get_tts_quality_preset"):
                        try:
                            old_tts_quality = vm.get_tts_quality_preset()
                        except Exception:
                            old_tts_quality = sentinel
                    if is_cloned_voice and hasattr(vm, "get_cloned_tts_quality_preset"):
                        try:
                            old_cloned_quality = vm.get_cloned_tts_quality_preset()
                        except Exception:
                            old_cloned_quality = sentinel
                    if not is_cloned_voice and hasattr(vm, "set_tts_quality_preset"):
                        try:
                            vm.set_tts_quality_preset(quality_preset)
                        except Exception:
                            pass
                    if is_cloned_voice and hasattr(vm, "set_cloned_tts_quality"):
                        try:
                            vm.set_cloned_tts_quality(quality_preset)
                        except Exception:
                            pass
                profile_candidate = "" if is_cloned_voice else (profile_name or voice_name or "")
                if profile_candidate and hasattr(vm, "set_profile"):
                    try:
                        applied_profile = bool(vm.set_profile(profile_candidate, kind="tts"))
                    except TypeError:
                        try:
                            applied_profile = bool(vm.set_profile(profile_candidate))
                        except Exception:
                            applied_profile = False
                    except Exception:
                        applied_profile = False
                audio = vm.speak_to_bytes(
                    str(text),
                    format=str(format),
                    voice=None if applied_profile or piper_voice_is_profile else voice_name,
                )
            finally:
                if applied_profile and old_profile_id and hasattr(vm, "set_profile"):
                    try:
                        vm.set_profile(old_profile_id, kind="tts")
                    except TypeError:
                        try:
                            vm.set_profile(old_profile_id)
                        except Exception:
                            pass
                    except Exception:
                        pass
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "tts_model", old_vm_model)
                    except Exception:
                        pass
                if old_language is not sentinel and getattr(vm, "language", None) != old_language:
                    try:
                        vm.set_language(old_language)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass
                if old_speed is not sentinel and hasattr(vm, "set_speed"):
                    try:
                        vm.set_speed(float(old_speed))
                    except Exception:
                        try:
                            setattr(vm, "speed", old_speed)
                        except Exception:
                            pass
                if old_tts_quality is not sentinel and hasattr(vm, "set_tts_quality_preset"):
                    try:
                        vm.set_tts_quality_preset(old_tts_quality)
                    except Exception:
                        pass
                if old_cloned_quality is not sentinel and hasattr(vm, "set_cloned_tts_quality"):
                    try:
                        vm.set_cloned_tts_quality(old_cloned_quality)
                    except Exception:
                        pass
            tts_metrics = None
            try:
                if hasattr(vm, "pop_last_tts_metrics"):
                    tts_metrics = vm.pop_last_tts_metrics()
            except Exception:
                tts_metrics = None

        merged_meta: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            merged_meta.update(metadata)
        if isinstance(tts_metrics, dict) and tts_metrics:
            merged_meta["abstractvoice_tts"] = dict(tts_metrics)

        return self._maybe_store_audio(
            audio,
            artifact_store=artifact_store,
            fmt=str(format),
            run_id=run_id,
            tags=tags,
            metadata=merged_meta if merged_meta else None,
        )

    def stt(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        vm = self._get_vm_for_provider(stt_provider=provider, stt_model=model)
        lk = self._vm_lock(vm)
        with lk:
            model_name = str(model or "").strip() if isinstance(model, str) else ""
            sentinel = object()
            old_vm_model = getattr(vm, "stt_model", sentinel)
            old_whisper_model = getattr(vm, "whisper_model", sentinel)
            adapter = getattr(vm, "stt_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            try:
                if model_name:
                    try:
                        setattr(vm, "stt_model", model_name)
                    except Exception:
                        pass
                    if _norm_engine_id(provider) in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                        try:
                            setattr(vm, "whisper_model", model_name)
                            setattr(vm, "stt_adapter", None)
                            adapter = None
                        except Exception:
                            pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                if isinstance(audio, str):
                    return vm.transcribe_file(str(audio), language=language)

                if isinstance(audio, dict):
                    import os
                    import tempfile

                    audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                    suffix = self._suffix_for_audio_ref(audio, artifact_store=artifact_store)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                        tmp_file.write(bytes(audio_bytes))
                        tmp_path = tmp_file.name
                    try:
                        return vm.transcribe_file(tmp_path, language=language)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                return vm.transcribe_from_bytes(bytes(audio_bytes), language=language)
            finally:
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "stt_model", old_vm_model)
                    except Exception:
                        pass
                if old_whisper_model is not sentinel:
                    try:
                        setattr(vm, "whisper_model", old_whisper_model)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass


class _AudioCapability(_BaseVoice):
    backend_id = "abstractvoice:stt"

    def transcribe(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        vm = self._get_vm_for_provider(stt_provider=provider, stt_model=model)
        lk = self._vm_lock(vm)
        with lk:
            model_name = str(model or "").strip() if isinstance(model, str) else ""
            sentinel = object()
            old_vm_model = getattr(vm, "stt_model", sentinel)
            old_whisper_model = getattr(vm, "whisper_model", sentinel)
            adapter = getattr(vm, "stt_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            try:
                if model_name:
                    try:
                        setattr(vm, "stt_model", model_name)
                    except Exception:
                        pass
                    if _norm_engine_id(provider) in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                        try:
                            setattr(vm, "whisper_model", model_name)
                            setattr(vm, "stt_adapter", None)
                            adapter = None
                        except Exception:
                            pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                if isinstance(audio, str):
                    return vm.transcribe_file(str(audio), language=language)

                if isinstance(audio, dict):
                    import os
                    import tempfile

                    audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                    suffix = self._suffix_for_audio_ref(audio, artifact_store=artifact_store)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                        tmp_file.write(bytes(audio_bytes))
                        tmp_path = tmp_file.name
                    try:
                        return vm.transcribe_file(tmp_path, language=language)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                return vm.transcribe_from_bytes(bytes(audio_bytes), language=language)
            finally:
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "stt_model", old_vm_model)
                    except Exception:
                        pass
                if old_whisper_model is not sentinel:
                    try:
                        setattr(vm, "whisper_model", old_whisper_model)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass


def register(registry: Any) -> None:
    """Register AbstractVoice as an AbstractCore capability plugin."""

    registry.register_voice_backend(
        backend_id=_VoiceCapability.backend_id,
        factory=lambda owner: _VoiceCapability(owner),
        priority=0,
        description="AbstractVoice VoiceManager (TTS+STT).",
        config_hint=(
            "Defaults to OpenAI remote TTS in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_tts_engine/ABSTRACTVOICE_TTS_ENGINE. Use piper or supertonic with "
            "abstractvoice[apple]/[gpu] for local offline TTS."
        ),
    )
    registry.register_audio_backend(
        backend_id=_AudioCapability.backend_id,
        factory=lambda owner: _AudioCapability(owner),
        priority=0,
        description="AbstractVoice STT (speech-to-text).",
        config_hint=(
            "Defaults to OpenAI remote STT in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_stt_engine/ABSTRACTVOICE_STT_ENGINE. Use faster_whisper with abstractvoice[stt], [apple], or [gpu] for local offline STT."
        ),
    )
