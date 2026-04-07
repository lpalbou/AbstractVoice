from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional

from ..artifacts import MediaStore, RuntimeArtifactStoreAdapter, get_artifact_id, is_artifact_ref


def _require_abstractcore_tool():
    try:
        from abstractcore import tool  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("AbstractCore is required for this integration. Install it via: pip install abstractcore") from e
    return tool


def _decode_base64_bytes(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        return b""
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1].strip()
    raw = "".join(raw.split())
    pad = (-len(raw)) % 4
    if pad:
        raw = raw + ("=" * pad)
    return base64.b64decode(raw, validate=False)


def _require_store(store: Any) -> MediaStore:
    # If the caller passed an AbstractRuntime ArtifactStore, adapt it.
    if hasattr(store, "store") and hasattr(store, "load") and not hasattr(store, "store_bytes"):
        return RuntimeArtifactStoreAdapter(store)
    if not hasattr(store, "store_bytes") or not hasattr(store, "load_bytes"):
        raise TypeError("store must be a MediaStore-like object or an AbstractRuntime-like ArtifactStore")
    return store  # type: ignore[return-value]


def _resolve_audio_bytes(
    *,
    store: MediaStore,
    artifact: Optional[Dict[str, Any]],
    b64: Optional[str],
    required: bool,
) -> Optional[bytes]:
    if artifact is not None:
        if not is_artifact_ref(artifact):
            raise ValueError("audio_artifact: expected an artifact ref dict like {'$artifact': '...'}")
        return store.load_bytes(get_artifact_id(artifact))
    if b64 is not None:
        out = _decode_base64_bytes(b64)
        if required and not out:
            raise ValueError("audio_b64: decoded to empty bytes")
        return out
    if required:
        raise ValueError("Either audio_artifact or audio_b64 is required")
    return None


def make_voice_tools(
    *,
    voice_manager: Any,
    store: Any,
) -> List[Callable[..., Any]]:
    """Create AbstractCore tools for TTS/STT (artifact-first outputs)."""

    tool = _require_abstractcore_tool()
    media_store = _require_store(store)

    @tool(
        name="voice_tts",
        description="Synthesize speech from text and return an audio artifact ref.",
        tags=["voice", "tts", "audio"],
        when_to_use="Use when you need to generate an audio rendition of text (TTS).",
    )
    def voice_tts(
        text: str,
        voice: Optional[str] = None,
        format: str = "wav",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio = voice_manager.speak_to_bytes(str(text), format=str(format), voice=voice)
        tts_metrics = None
        try:
            if hasattr(voice_manager, "pop_last_tts_metrics"):
                tts_metrics = voice_manager.pop_last_tts_metrics()
        except Exception:
            tts_metrics = None
        return media_store.store_bytes(
            bytes(audio),
            content_type=f"audio/{str(format).lower()}",
            filename=f"tts.{str(format).lower()}",
            run_id=str(run_id) if run_id else None,
            tags={"kind": "generated_media", "modality": "audio", "task": "tts"},
            metadata={"abstractvoice_tts": dict(tts_metrics)} if isinstance(tts_metrics, dict) and tts_metrics else None,
        )

    def _profile_to_dict(p: Any) -> Dict[str, Any]:
        try:
            return {
                "engine_id": getattr(p, "engine_id", None),
                "profile_id": getattr(p, "profile_id", None),
                "label": getattr(p, "label", None),
                "description": getattr(p, "description", None),
                "params": getattr(p, "params", None),
                "tags": getattr(p, "tags", None),
                "provenance": getattr(p, "provenance", None),
            }
        except Exception:
            return {}

    @tool(
        name="voice_profile_list",
        description="List available base-TTS voice profiles for the active engine.",
        tags=["voice", "tts", "profiles"],
        when_to_use="Use when you need to list preset voice profiles for the currently selected base TTS engine.",
    )
    def voice_profile_list(kind: str = "tts") -> Dict[str, Any]:
        profiles = []
        try:
            if hasattr(voice_manager, "get_profiles"):
                profiles = list(voice_manager.get_profiles(kind=str(kind)))
        except Exception:
            profiles = []
        active = None
        try:
            if hasattr(voice_manager, "get_active_profile"):
                active = voice_manager.get_active_profile(kind=str(kind))
        except Exception:
            active = None
        return {
            "kind": str(kind or "tts"),
            "active_profile": _profile_to_dict(active) if active is not None else None,
            "profiles": [_profile_to_dict(p) for p in profiles],
        }

    @tool(
        name="voice_profile_set",
        description="Select/apply a base-TTS voice profile by id for the active engine.",
        tags=["voice", "tts", "profiles"],
        when_to_use="Use when you need to select a preset voice profile for the current base TTS engine before synthesizing speech.",
    )
    def voice_profile_set(profile_id: str, kind: str = "tts") -> Dict[str, Any]:
        ok = False
        try:
            if hasattr(voice_manager, "set_profile"):
                ok = bool(voice_manager.set_profile(str(profile_id), kind=str(kind)))
        except Exception as e:
            return {"ok": False, "error": str(e), "profile_id": str(profile_id), "kind": str(kind or "tts")}

        active = None
        try:
            if hasattr(voice_manager, "get_active_profile"):
                active = voice_manager.get_active_profile(kind=str(kind))
        except Exception:
            active = None
        return {"ok": bool(ok), "kind": str(kind or "tts"), "active_profile": _profile_to_dict(active) if active is not None else None}

    @tool(
        name="audio_transcribe",
        description="Transcribe audio (speech-to-text) and return text plus a transcript artifact ref.",
        tags=["audio", "stt", "transcribe"],
        when_to_use="Use when you need to convert speech audio into text (STT).",
    )
    def audio_transcribe(
        audio_artifact: Optional[Dict[str, Any]] = None,
        audio_b64: Optional[str] = None,
        language: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio_bytes = _resolve_audio_bytes(store=media_store, artifact=audio_artifact, b64=audio_b64, required=True)
        text = voice_manager.transcribe_from_bytes(bytes(audio_bytes or b""), language=language)

        transcript_ref = media_store.store_bytes(
            str(text).encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            filename="transcript.txt",
            run_id=str(run_id) if run_id else None,
            tags={"kind": "derived_text", "modality": "audio", "task": "stt"},
        )
        return {"text": text, "transcript_artifact": transcript_ref}

    return [voice_tts, voice_profile_list, voice_profile_set, audio_transcribe]

