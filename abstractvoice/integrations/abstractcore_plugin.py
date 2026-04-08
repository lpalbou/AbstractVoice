from __future__ import annotations

import threading
import weakref
from typing import Any, Dict, Optional, Union

from ..artifacts import RuntimeArtifactStoreAdapter, is_artifact_ref, get_artifact_id


_VM_CACHE_LOCK = threading.Lock()
_VM_CACHE: dict[tuple, Any] = {}
_VM_LOCKS: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()


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

        # Best-effort config overrides (optional).
        language = "en"
        allow_downloads = True
        tts_engine = "auto"
        stt_engine = "auto"
        whisper_model = "base"
        cloning_engine = "f5_tts"
        cloned_tts_streaming = True
        tts_delivery_mode = None
        debug_mode = False
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                if isinstance(cfg.get("voice_language"), str) and cfg["voice_language"].strip():
                    language = str(cfg["voice_language"]).strip().lower()
                if "voice_allow_downloads" in cfg:
                    allow_downloads = bool(cfg.get("voice_allow_downloads"))
                if isinstance(cfg.get("voice_tts_engine"), str) and str(cfg["voice_tts_engine"]).strip():
                    tts_engine = str(cfg["voice_tts_engine"]).strip().lower()
                if isinstance(cfg.get("voice_stt_engine"), str) and str(cfg["voice_stt_engine"]).strip():
                    stt_engine = str(cfg["voice_stt_engine"]).strip().lower()
                if isinstance(cfg.get("voice_whisper_model"), str) and str(cfg["voice_whisper_model"]).strip():
                    whisper_model = str(cfg["voice_whisper_model"]).strip()
                if isinstance(cfg.get("voice_cloning_engine"), str) and str(cfg["voice_cloning_engine"]).strip():
                    cloning_engine = str(cfg["voice_cloning_engine"]).strip().lower()
                if "voice_cloned_tts_streaming" in cfg:
                    cloned_tts_streaming = bool(cfg.get("voice_cloned_tts_streaming"))
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

                        tts_delivery_mode = normalize_audio_delivery_mode(bool(raw))
                    except Exception:
                        tts_delivery_mode = None
                if "voice_debug_mode" in cfg:
                    debug_mode = bool(cfg.get("voice_debug_mode"))
        except Exception:
            pass

        key = (
            str(language),
            bool(allow_downloads),
            str(tts_engine),
            str(stt_engine),
            str(whisper_model),
            str(cloning_engine),
            bool(cloned_tts_streaming),
            str(tts_delivery_mode) if tts_delivery_mode else "",
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
                    cloning_engine=str(cloning_engine),
                    cloned_tts_streaming=bool(cloned_tts_streaming),
                    tts_delivery_mode=str(tts_delivery_mode) if tts_delivery_mode else None,
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
        config_hint="Install voices/models with `abstractvoice-prefetch` for offline use (or allow downloads).",
    )
    registry.register_audio_backend(
        backend_id=_AudioCapability.backend_id,
        factory=lambda owner: _AudioCapability(owner),
        priority=0,
        description="AbstractVoice STT (speech-to-text).",
        config_hint="Install STT models with `abstractvoice-prefetch --stt <size>` for offline use (or allow downloads).",
    )
