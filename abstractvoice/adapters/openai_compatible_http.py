"""Small HTTP helpers for OpenAI-compatible audio providers.

This module intentionally depends only on `requests` (already a core
AbstractVoice dependency). It is used by remote TTS/STT adapters and by the
remote cloning engine without importing AbstractCore provider classes.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import wave
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import requests


class RemoteVoiceProviderError(RuntimeError):
    """Raised when a remote audio provider request fails."""


def env_first(*keys: str) -> Optional[str]:
    """Return the first non-empty environment value from `keys`."""
    for key in keys:
        raw = os.environ.get(str(key), None)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return None


def normalize_remote_provider(provider: str | None) -> str:
    p = str(provider or "").strip().lower().replace("_", "-")
    if p in {"openai", "oa"}:
        return "openai"
    if p in {"openai-compatible", "compatible", "remote", "proxy"}:
        return "openai-compatible"
    return p or "openai-compatible"


def coerce_timeout_s(value: Any, *, default: float = 60.0) -> float:
    if value is None or value == "":
        raw = env_first(
            "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TIMEOUT_S",
        )
        value = raw if raw is not None else default
    try:
        out = float(value)
        return out if out > 0 else float(default)
    except Exception:
        return float(default)


def join_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("base_url is required")
    p = str(path or "").strip() or "/"
    if not p.startswith("/"):
        p = "/" + p
    return base.rstrip("/") + p


def resolve_base_url(provider: str, base_url: str | None = None) -> str:
    p = normalize_remote_provider(provider)
    if base_url and str(base_url).strip():
        return str(base_url).strip()
    if p == "openai":
        return (
            env_first("ABSTRACTVOICE_OPENAI_BASE_URL", "OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
    out = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL",
        "ABSTRACTVOICE_REMOTE_BASE_URL",
        "OPENAI_BASE_URL",
    )
    if not out:
        raise ValueError(
            "Missing remote audio base URL. Set `remote_base_url=...` or "
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_BASE_URL / ABSTRACTVOICE_REMOTE_BASE_URL."
        )
    return str(out).strip()


def resolve_api_key(provider: str, api_key: str | None = None) -> Optional[str]:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_API_KEY", "OPENAI_API_KEY")
    return env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_API_KEY",
        "ABSTRACTVOICE_REMOTE_API_KEY",
        "OPENAI_API_KEY",
    )


def require_provider_ready(provider: str, *, base_url: str, api_key: Optional[str]) -> None:
    p = normalize_remote_provider(provider)
    if not str(base_url or "").strip():
        raise ValueError("Remote audio base_url is required")
    if p == "openai" and not str(api_key or "").strip():
        raise ValueError("OpenAI audio requires OPENAI_API_KEY or remote_api_key=...")


def response_json(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        content = bytes(getattr(response, "content", b"") or b"")
        if not content:
            return {}
        data = json.loads(content.decode("utf-8"))
    if isinstance(data, dict):
        return data
    return {}


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    try:
        return bytes(getattr(response, "content", b"") or b"").decode("utf-8", errors="replace")
    except Exception:
        return ""


def raise_for_status(response: Any, *, endpoint: str) -> None:
    status = int(getattr(response, "status_code", 0) or 0)
    if status < 400:
        return

    detail = response_text(response).strip()
    try:
        payload = response_json(response)
        err = payload.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or err.get("detail") or detail)
        elif isinstance(err, str) and err.strip():
            detail = err.strip()
        else:
            msg = payload.get("message") or payload.get("detail")
            if isinstance(msg, str) and msg.strip():
                detail = msg.strip()
    except Exception:
        pass

    if not detail:
        detail = "remote provider request failed"
    raise RemoteVoiceProviderError(f"{endpoint} failed ({status}): {detail}")


class RemoteAudioHTTPClient:
    """Thin wrapper around `requests` with provider-friendly errors."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_s: float = 60.0,
        session: Any = None,
    ) -> None:
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key).strip() if api_key else None
        self.timeout_s = coerce_timeout_s(timeout_s)
        self.session = session if session is not None else requests.Session()

    def _headers(self, extra: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update({str(k): str(v) for k, v in extra.items()})
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        endpoint_name: str,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        url = join_url(self.base_url, path)
        try:
            response = self.session.request(
                str(method).upper(),
                url,
                timeout=float(self.timeout_s),
                headers=self._headers(headers),
                **kwargs,
            )
        except requests.RequestException as e:
            raise RemoteVoiceProviderError(f"{endpoint_name} request failed: {e}") from e
        raise_for_status(response, endpoint=endpoint_name)
        return response


def strip_content_type_params(content_type: str | None) -> str:
    return str(content_type or "").split(";", 1)[0].strip().lower()


def is_json_response(response: Any) -> bool:
    headers = getattr(response, "headers", {}) or {}
    try:
        ctype = headers.get("content-type") or headers.get("Content-Type")
    except Exception:
        ctype = ""
    media = strip_content_type_params(ctype)
    return media == "application/json" or media.endswith("+json")


def decode_base64_audio(value: str) -> bytes:
    raw = str(value or "").strip()
    if not raw:
        return b""
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1].strip()
    raw = "".join(raw.split())
    pad = (-len(raw)) % 4
    if pad:
        raw += "=" * pad
    return base64.b64decode(raw, validate=False)


def extract_audio_bytes_from_response(response: Any) -> bytes:
    """Return raw audio bytes from an OpenAI-compatible speech response.

    OpenAI returns raw audio bytes. Some compatible endpoints return JSON with a
    base64 field; accepting both makes this adapter usable against lightweight
    local proxies.
    """
    if not is_json_response(response):
        return bytes(getattr(response, "content", b"") or b"")

    payload = response_json(response)
    candidates: list[Any] = [
        payload.get("audio"),
        payload.get("audio_b64"),
        payload.get("b64_json"),
        payload.get("bytes"),
    ]
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            candidates.extend(
                [
                    first.get("audio"),
                    first.get("audio_b64"),
                    first.get("b64_json"),
                    first.get("bytes"),
                ]
            )
    elif isinstance(data, dict):
        candidates.extend([data.get("audio"), data.get("audio_b64"), data.get("b64_json")])

    for item in candidates:
        if isinstance(item, str) and item.strip():
            out = decode_base64_audio(item)
            if out:
                return out
    raise RemoteVoiceProviderError("audio/speech returned JSON without audio bytes")


def extract_transcription_text(response: Any) -> str:
    if is_json_response(response):
        payload = response_json(response)
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            return str(data.get("text") or "").strip()
        return ""
    return bytes(getattr(response, "content", b"") or b"").decode("utf-8", errors="replace").strip()


def decode_audio_bytes_to_array(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio bytes into mono float32 samples."""
    b = bytes(audio_bytes or b"")
    if not b:
        return np.zeros((0,), dtype=np.float32), 0

    try:
        import soundfile as sf

        arr, sr = sf.read(io.BytesIO(b), always_2d=True, dtype="float32")
        mono = np.mean(arr, axis=1).astype(np.float32).reshape(-1)
        return mono, int(sr)
    except Exception:
        pass

    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            sr = int(w.getframerate())
            channels = int(w.getnchannels())
            width = int(w.getsampwidth())
            frames = w.readframes(w.getnframes())
        if width == 2:
            x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            x = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            x = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {width}")
        if channels > 1:
            x = x.reshape(-1, channels).mean(axis=1)
        return x.astype(np.float32).reshape(-1), int(sr)
    except Exception as e:
        raise RemoteVoiceProviderError(f"Failed to decode remote audio bytes: {e}") from e


def wav_bytes_from_array(audio_array: np.ndarray, sample_rate: int) -> bytes:
    x = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def guess_audio_content_type(path: str | Path, *, default: str = "application/octet-stream") -> str:
    p = Path(path)
    ctype, _ = mimetypes.guess_type(str(p))
    return str(ctype or default)


def safe_filename(path: str | Path, *, default: str = "audio.wav") -> str:
    try:
        name = Path(path).name
        return name if name else default
    except Exception:
        return default
