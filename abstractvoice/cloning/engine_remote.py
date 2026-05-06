"""Remote voice cloning engine for OpenAI-compatible services.

The OpenAI-compatible audio spec standardizes speech/transcription endpoints,
but cloning/custom-voice creation is provider-specific. This engine supports a
small configurable HTTP contract:

- POST `<base_url>/<clone_path>` (default `/voice/clone` for compatible servers)
- multipart file field (default `file`)
- fields: `name`, `reference_text` when provided
- response: `{"voice_id": "..."}` or `{"id": "..."}` (also accepts nested forms)

For OpenAI's custom voices, set `cloning_engine="openai"` and provide the
required consent id via `ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID`; the default
path is `/audio/voices` and the default file field is `audio_sample`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..adapters.openai_compatible_http import (
    RemoteAudioHTTPClient,
    RemoteVoiceProviderError,
    coerce_timeout_s,
    decode_audio_bytes_to_array,
    env_first,
    guess_audio_content_type,
    normalize_remote_provider,
    require_provider_ready,
    resolve_api_key,
    resolve_base_url,
    response_json,
    safe_filename,
)
from ..adapters.tts_openai_compatible import OpenAICompatibleTTSAdapter


def _clone_path(provider: str, configured: str | None = None) -> str:
    if configured and str(configured).strip():
        return str(configured).strip()
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_VOICE_CREATE_PATH") or "/audio/voices"
    return (
        env_first(
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_CLONE_PATH",
            "ABSTRACTVOICE_REMOTE_VOICE_CLONE_PATH",
        )
        or "/voice/clone"
    )


def _file_field(provider: str) -> str:
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_VOICE_AUDIO_FIELD") or "audio_sample"
    return (
        env_first(
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_FILE_FIELD",
            "ABSTRACTVOICE_REMOTE_VOICE_FILE_FIELD",
        )
        or "file"
    )


def _extract_remote_voice_id(payload: dict[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("voice_id"),
        payload.get("id"),
        payload.get("voice"),
    ]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.extend([data.get("voice_id"), data.get("id"), data.get("voice")])
    voice = payload.get("voice")
    if isinstance(voice, dict):
        candidates.extend([voice.get("voice_id"), voice.get("id")])

    for item in candidates:
        if isinstance(item, str) and item.strip():
            return item.strip()
    raise RemoteVoiceProviderError("Remote clone response did not include a voice id")


class RemoteVoiceCloningEngine:
    """Provider-specific remote voice-cloning bridge."""

    def __init__(
        self,
        *,
        provider: str = "openai-compatible",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        tts_model: str | None = None,
        clone_path: str | None = None,
        session: Any = None,
        debug: bool = False,
    ) -> None:
        self.provider = normalize_remote_provider(provider)
        self.engine_id = "openai" if self.provider == "openai" else "openai-compatible"
        self.base_url = resolve_base_url(self.provider, base_url)
        self.api_key = resolve_api_key(self.provider, api_key)
        require_provider_ready(self.provider, base_url=self.base_url, api_key=self.api_key)
        self.timeout_s = coerce_timeout_s(timeout_s)
        self.tts_model = str(tts_model).strip() if isinstance(tts_model, str) and tts_model.strip() else None
        self.clone_path = _clone_path(self.provider, clone_path)
        self.session = session
        self.debug = bool(debug)
        self._client = RemoteAudioHTTPClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            session=session,
        )

    def _clone_fields(self, *, name: str | None, reference_text: str | None) -> dict[str, str]:
        fields: dict[str, str] = {}
        if name and str(name).strip():
            fields["name"] = str(name).strip()

        if self.provider == "openai":
            consent_id = env_first("ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID")
            if not consent_id and self.clone_path.rstrip("/") == "/audio/voices":
                raise RemoteVoiceProviderError(
                    "OpenAI custom voice creation requires consent. Set "
                    "ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID, or use "
                    "cloning_engine='openai-compatible' with a compatible custom clone endpoint."
                )
            if consent_id:
                fields["consent"] = consent_id
        elif reference_text and str(reference_text).strip():
            fields["reference_text"] = str(reference_text).strip()
        return fields

    def clone_voice_from_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "reference.wav",
        content_type: str = "audio/wav",
        name: str | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("audio_bytes must be non-empty")

        response = self._client.request(
            "POST",
            self.clone_path,
            endpoint_name="voice/clone",
            data=self._clone_fields(name=name, reference_text=reference_text),
            files={_file_field(self.provider): (filename, bytes(audio_bytes), content_type or "audio/wav")},
        )
        payload = response_json(response)
        remote_voice_id = _extract_remote_voice_id(payload)
        return {
            "remote_voice_id": remote_voice_id,
            "remote_provider": self.provider,
            "remote_base_url": self.base_url,
            "remote_clone_path": self.clone_path,
            "remote_response": payload,
        }

    def clone_voice(
        self,
        reference_audio_path: str,
        *,
        name: str | None = None,
        reference_text: str | None = None,
    ) -> dict[str, Any]:
        p = Path(reference_audio_path).expanduser()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(reference_audio_path))
        return self.clone_voice_from_bytes(
            p.read_bytes(),
            filename=safe_filename(p),
            content_type=guess_audio_content_type(p, default="audio/wav"),
            name=name,
            reference_text=reference_text,
        )

    def speak_to_bytes_for_voice(
        self,
        text: str,
        *,
        voice: dict[str, Any],
        format: str = "wav",
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> bytes:
        meta = dict(voice.get("meta") or {}) if isinstance(voice, dict) else {}
        remote_voice_id = str(meta.get("remote_voice_id") or "").strip()
        if not remote_voice_id:
            raise RemoteVoiceProviderError("Stored remote cloned voice is missing meta.remote_voice_id")

        tts = OpenAICompatibleTTSAdapter(
            provider=self.provider,
            language=str(language or "en"),
            base_url=self.base_url,
            api_key=self.api_key,
            model_id=self.tts_model,
            voice=remote_voice_id,
            timeout_s=self.timeout_s,
            session=self.session,
            debug_mode=self.debug,
        )
        return tts.synthesize_to_bytes_with_voice(
            str(text),
            format=str(format or "wav"),
            voice=remote_voice_id,
            speed=speed,
        )

    def audio_chunks_for_voice(
        self,
        text: str,
        *,
        voice: dict[str, Any],
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ):
        wav_bytes = self.speak_to_bytes_for_voice(
            str(text),
            voice=voice,
            format="wav",
            speed=speed,
            language=language,
        )
        arr, sr = decode_audio_bytes_to_array(wav_bytes)
        yield np.asarray(arr, dtype=np.float32).reshape(-1), int(sr or 0)

    def set_quality_preset(self, preset: str) -> None:
        _ = preset

    def runtime_info(self) -> dict[str, Any]:
        return {
            "engine": self.engine_id,
            "provider": self.provider,
            "base_url": self.base_url,
            "clone_path": self.clone_path,
            "tts_model": self.tts_model,
        }
