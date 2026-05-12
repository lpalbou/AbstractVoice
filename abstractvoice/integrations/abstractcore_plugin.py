from __future__ import annotations

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

    engine = str(getattr(vm, "stt_engine", "") or _env("ABSTRACTVOICE_STT_ENGINE", "openai") or "").strip().lower()
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
            self._vm = cached
            return self._vm

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
        """List deduplicated TTS model ids from the active VoiceManager catalog."""
        vm = self._get_vm()
        catalog: Any = {}
        if hasattr(vm, "list_available_models"):
            catalog = vm.list_available_models()
        return _extract_tts_model_ids(catalog)

    def list_stt_models(self) -> list[str]:
        """List deduplicated STT model ids from the active VoiceManager configuration."""
        vm = self._get_vm()
        return _extract_stt_model_ids(vm)

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
                            cloned_voices.append(
                                {
                                    "id": clone_id.strip(),
                                    "voice_id": clone_id.strip(),
                                    "profile_id": clone_id.strip(),
                                    "label": str(clone_name or clone_id).strip() or clone_id.strip(),
                                    "kind": "clone",
                                    "engine_id": "cloned",
                                    "params": _json_safe(item),
                                }
                            )
                    elif isinstance(item, str) and item.strip():
                        cloned_voices.append(
                            {
                                "id": item.strip(),
                                "voice_id": item.strip(),
                                "profile_id": item.strip(),
                                "label": item.strip(),
                                "kind": "clone",
                                "engine_id": "cloned",
                            }
                        )

        return {
            "kind": "tts",
            "engine_id": getattr(adapter, "engine_id", None) or getattr(vm, "_tts_engine_name", None),
            "active_profile": _voice_profile_to_dict(active_profile) if active_profile is not None else None,
            "active_model": _active_tts_model(vm, catalog, tts_models),
            "profiles": profiles,
            "voices": profiles + cloned_voices,
            "cloned_voices": cloned_voices,
            "tts_models": tts_models,
            "stt_models": stt_models,
            "catalog": _json_safe(catalog),
        }

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        artifact_store: Any = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ):
        vm = self._get_vm()
        lk = self._vm_lock(vm)
        with lk:
            audio = vm.speak_to_bytes(str(text), format=str(format), voice=voice)
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
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        vm = self._get_vm()
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


class _AudioCapability(_BaseVoice):
    backend_id = "abstractvoice:stt"

    def transcribe(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        language: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        vm = self._get_vm()
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


def register(registry: Any) -> None:
    """Register AbstractVoice as an AbstractCore capability plugin."""

    registry.register_voice_backend(
        backend_id=_VoiceCapability.backend_id,
        factory=lambda owner: _VoiceCapability(owner),
        priority=0,
        description="AbstractVoice VoiceManager (TTS+STT).",
        config_hint=(
            "Defaults to OpenAI remote TTS in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_tts_engine/ABSTRACTVOICE_TTS_ENGINE. Use piper with abstractvoice[local] for local offline TTS."
        ),
    )
    registry.register_audio_backend(
        backend_id=_AudioCapability.backend_id,
        factory=lambda owner: _AudioCapability(owner),
        priority=0,
        description="AbstractVoice STT (speech-to-text).",
        config_hint=(
            "Defaults to OpenAI remote STT in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_stt_engine/ABSTRACTVOICE_STT_ENGINE. Use faster_whisper with abstractvoice[local] for local offline STT."
        ),
    )
