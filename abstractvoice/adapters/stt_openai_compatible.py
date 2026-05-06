"""OpenAI-compatible remote STT adapter.

Use `stt_engine="openai"` for OpenAI's hosted transcription API, or
`stt_engine="openai-compatible"` for any compatible `/v1/audio/transcriptions`
endpoint, including an AbstractCore Server instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import STTAdapter
from .openai_compatible_http import (
    RemoteAudioHTTPClient,
    coerce_timeout_s,
    env_first,
    extract_transcription_text,
    guess_audio_content_type,
    normalize_remote_provider,
    require_provider_ready,
    resolve_api_key,
    resolve_base_url,
    safe_filename,
    wav_bytes_from_array,
)


def _default_stt_model(provider: str, model_id: str | None) -> str | None:
    if model_id and str(model_id).strip():
        return str(model_id).strip()
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_STT_MODEL") or "gpt-4o-mini-transcribe"
    return env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
        "ABSTRACTVOICE_REMOTE_STT_MODEL",
        "ABSTRACTVOICE_OPENAI_STT_MODEL",
    )


class OpenAICompatibleSTTAdapter(STTAdapter):
    """STT adapter backed by an OpenAI-compatible HTTP transcription endpoint."""

    def __init__(
        self,
        *,
        provider: str = "openai-compatible",
        language: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        prompt: str | None = None,
        response_format: str | None = "json",
        temperature: float | None = None,
        timeout_s: float | None = None,
        session: Any = None,
        debug_mode: bool = False,
    ) -> None:
        self.provider = normalize_remote_provider(provider)
        self.engine_id = "openai" if self.provider == "openai" else "openai-compatible"
        self.language = str(language).strip().lower() if isinstance(language, str) and language.strip() else None
        self.model_id = _default_stt_model(self.provider, model_id)
        self.prompt = str(prompt).strip() if isinstance(prompt, str) and prompt.strip() else None
        self.response_format = (
            str(response_format).strip() if isinstance(response_format, str) and response_format.strip() else None
        )
        self.temperature = temperature
        self.debug_mode = bool(debug_mode)

        self.base_url = resolve_base_url(self.provider, base_url)
        self.api_key = resolve_api_key(self.provider, api_key)
        require_provider_ready(self.provider, base_url=self.base_url, api_key=self.api_key)
        self.timeout_s = coerce_timeout_s(timeout_s)
        self._client = RemoteAudioHTTPClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
            session=session,
        )

    def _fields(self, language: str | None) -> dict[str, str]:
        fields: dict[str, str] = {}
        if self.model_id:
            fields["model"] = str(self.model_id)
        lang = str(language).strip() if isinstance(language, str) and language.strip() else self.language
        if lang:
            fields["language"] = str(lang)
        if self.prompt:
            fields["prompt"] = str(self.prompt)
        if self.response_format:
            fields["response_format"] = str(self.response_format)
        if self.temperature is not None:
            try:
                fields["temperature"] = str(float(self.temperature))
            except Exception:
                pass
        return fields

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        path = Path(audio_path).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(str(audio_path))
        content = path.read_bytes()
        return self._transcribe_bytes(
            content,
            filename=safe_filename(path),
            content_type=guess_audio_content_type(path),
            language=language,
        )

    def _transcribe_bytes(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        content_type: str = "application/octet-stream",
        language: Optional[str] = None,
    ) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes must be non-empty")
        response = self._client.request(
            "POST",
            "/audio/transcriptions",
            endpoint_name="audio/transcriptions",
            data=self._fields(language),
            files={"file": (filename, bytes(audio_bytes), content_type or "application/octet-stream")},
        )
        return extract_transcription_text(response)

    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        return self._transcribe_bytes(
            bytes(audio_bytes),
            filename="audio.wav",
            content_type="audio/wav",
            language=language,
        )

    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
    ) -> str:
        wav_bytes = wav_bytes_from_array(np.asarray(audio_array, dtype=np.float32), int(sample_rate))
        return self.transcribe_from_bytes(wav_bytes, language=language)

    def set_language(self, language: str) -> bool:
        self.language = str(language or "").strip().lower() or None
        return True

    def get_supported_languages(self) -> list[str]:
        # Remote providers vary widely; treat language as a pass-through hint.
        return []

    def is_available(self) -> bool:
        try:
            require_provider_ready(self.provider, base_url=self.base_url, api_key=self.api_key)
            return True
        except Exception:
            return False

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "OpenAI-compatible remote STT",
                "provider": self.provider,
                "base_url": self.base_url,
                "model_id": self.model_id,
                "current_language": self.language,
            }
        )
        return info
