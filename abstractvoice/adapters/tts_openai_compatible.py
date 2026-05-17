"""OpenAI-compatible remote TTS adapter.

This adapter lets AbstractVoice call remote `/v1/audio/speech` endpoints
directly, without depending on AbstractCore provider classes. Use
`tts_engine="openai"` for OpenAI's hosted API, or
`tts_engine="openai-compatible"` for any compatible endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..voice_profiles import VoiceProfile
from .base import TTSAdapter
from .openai_compatible_http import (
    RemoteAudioHTTPClient,
    coerce_timeout_s,
    decode_audio_bytes_to_array,
    env_first,
    extract_audio_bytes_from_response,
    normalize_remote_provider,
    require_provider_ready,
    resolve_api_key,
    resolve_base_url,
    response_json,
)


_OPENAI_BUILTIN_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)

_OPENAI_KNOWN_TTS_MODELS = (
    "gpt-4o-mini-tts",
    "tts-1",
    "tts-1-hd",
)


def _default_tts_model(provider: str, model_id: str | None) -> str | None:
    if model_id and str(model_id).strip():
        return str(model_id).strip()
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_TTS_MODEL") or "gpt-4o-mini-tts"
    return env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
        "ABSTRACTVOICE_REMOTE_TTS_MODEL",
        "ABSTRACTVOICE_OPENAI_TTS_MODEL",
    )


def _default_voice(provider: str, voice: str | None) -> str | None:
    if voice and str(voice).strip():
        return str(voice).strip()
    p = normalize_remote_provider(provider)
    if p == "openai":
        return env_first("ABSTRACTVOICE_OPENAI_TTS_VOICE") or "alloy"
    return env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICE",
        "ABSTRACTVOICE_REMOTE_TTS_VOICE",
    )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in str(value).replace("\n", ",").split(","):
        s = item.strip()
        if s:
            out.append(s)
    return out


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _configured_tts_models(provider: str) -> list[str]:
    p = normalize_remote_provider(provider)
    if p == "openai":
        raw = env_first("ABSTRACTVOICE_OPENAI_TTS_MODELS")
        return _dedupe(_split_csv(raw) + list(_OPENAI_KNOWN_TTS_MODELS))
    raw = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODELS",
        "ABSTRACTVOICE_REMOTE_TTS_MODELS",
        "ABSTRACTVOICE_OPENAI_TTS_MODELS",
    )
    return _dedupe(_split_csv(raw))


def _remote_model_paths(provider: str) -> list[str]:
    p = normalize_remote_provider(provider)
    if p == "openai":
        raw = env_first(
            "ABSTRACTVOICE_OPENAI_MODEL_PATHS",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL_PATHS",
            "ABSTRACTVOICE_REMOTE_MODEL_PATHS",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATHS",
        )
        if raw:
            return _split_csv(raw)
        path = env_first(
            "ABSTRACTVOICE_OPENAI_MODEL_PATH",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL_PATH",
            "ABSTRACTVOICE_REMOTE_MODEL_PATH",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATH",
        )
        return [path] if path else ["/models"]

    raw = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_MODEL_PATHS",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL_PATHS",
        "ABSTRACTVOICE_REMOTE_MODEL_PATHS",
        "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATHS",
    )
    if raw:
        return _split_csv(raw)

    path = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_MODEL_PATH",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL_PATH",
        "ABSTRACTVOICE_REMOTE_MODEL_PATH",
        "ABSTRACTVOICE_REMOTE_TTS_MODEL_PATH",
    )
    return [path] if path else ["/models"]


def _remote_model_id(item: Any) -> str | None:
    if isinstance(item, str):
        value = item.strip()
        return value or None
    if not isinstance(item, dict):
        return None
    value = item.get("id") or item.get("model") or item.get("model_id") or item.get("name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _looks_like_tts_model(model_id: str, item: Any, *, explicit_tts_list: bool) -> bool:
    if explicit_tts_list:
        return True
    mid = str(model_id or "").strip().lower()
    if "tts" in mid or "speech" in mid:
        return True
    if isinstance(item, dict):
        kind = str(item.get("kind") or item.get("type") or item.get("capability") or "").strip().lower()
        if kind in {"tts", "speech", "audio_speech", "audio.speech"}:
            return True
        modalities = item.get("modalities")
        if isinstance(modalities, list) and any(str(v).strip().lower() in {"tts", "speech"} for v in modalities):
            return True
    return False


def _models_from_remote_payload(payload: dict[str, Any]) -> list[str]:
    models: list[str] = []

    def add_many(items: Any, *, explicit_tts_list: bool) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            model_id = _remote_model_id(item)
            if model_id and _looks_like_tts_model(model_id, item, explicit_tts_list=explicit_tts_list):
                models.append(model_id)

    add_many(payload.get("tts_models"), explicit_tts_list=True)
    add_many(payload.get("speech_models"), explicit_tts_list=True)
    add_many(payload.get("audio_speech_models"), explicit_tts_list=True)
    add_many(payload.get("models"), explicit_tts_list=False)

    data = payload.get("data")
    if isinstance(data, list):
        add_many(data, explicit_tts_list=False)
    elif isinstance(data, dict):
        models.extend(_models_from_remote_payload(data))

    return _dedupe(models)


def _remote_profile_paths(provider: str) -> list[str]:
    p = normalize_remote_provider(provider)
    if p == "openai":
        raw = env_first(
            "ABSTRACTVOICE_OPENAI_VOICE_PROFILE_PATHS",
            "ABSTRACTVOICE_OPENAI_VOICE_LIST_PATHS",
            "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATHS",
            "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATHS",
        )
        if raw:
            return _split_csv(raw)
        # Built-in voices are available locally; this discovery path adds
        # account/org-specific custom voices when the API exposes them.
        path = env_first(
            "ABSTRACTVOICE_OPENAI_VOICE_PROFILE_PATH",
            "ABSTRACTVOICE_OPENAI_VOICE_LIST_PATH",
            "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATH",
            "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATH",
        )
        return [path] if path else ["/audio/voices"]

    raw = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_PROFILE_PATHS",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_LIST_PATHS",
        "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATHS",
        "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATHS",
    )
    if raw:
        return _split_csv(raw)

    path = env_first(
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_PROFILE_PATH",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_VOICE_LIST_PATH",
        "ABSTRACTVOICE_REMOTE_VOICE_PROFILE_PATH",
        "ABSTRACTVOICE_REMOTE_VOICE_LIST_PATH",
    )
    if path:
        return [path]

    # `/audio/voices` is the AbstractVoice-compatible extension exposed by the
    # local web example. `/voices` keeps simple custom gateways easy to support.
    return ["/audio/voices", "/voices"]


def _profile_from_remote_item(
    item: Any,
    *,
    engine_id: str,
    provider: str,
    kind: str = "voice",
) -> VoiceProfile | None:
    if isinstance(item, str):
        pid = item.strip()
        if not pid:
            return None
        return VoiceProfile(
            engine_id=engine_id,
            profile_id=pid,
            label=pid,
            params={"voice": pid},
            tags={"provider": provider, "kind": kind},
        )

    if not isinstance(item, dict):
        return None

    pid = (
        item.get("profile_id")
        or item.get("voice_id")
        or item.get("id")
        or item.get("voice")
        or item.get("name")
    )
    if not isinstance(pid, str) or not pid.strip():
        return None
    profile_id = pid.strip()
    label = str(item.get("label") or item.get("name") or profile_id).strip() or profile_id
    desc = item.get("description")
    description = str(desc).strip() if isinstance(desc, str) and desc.strip() else None

    params: dict[str, Any] = {}
    raw_params = item.get("params")
    if isinstance(raw_params, dict):
        params.update(raw_params)
    params.setdefault("voice", str(item.get("voice") or item.get("voice_id") or profile_id).strip() or profile_id)

    tags = {"provider": provider, "kind": str(item.get("kind") or kind)}
    raw_tags = item.get("tags")
    if isinstance(raw_tags, dict):
        for k, v in raw_tags.items():
            tags[str(k)] = str(v)

    provenance = item.get("provenance") if isinstance(item.get("provenance"), dict) else None
    return VoiceProfile(
        engine_id=engine_id,
        profile_id=profile_id,
        label=label,
        description=description,
        params=params,
        tags=tags,
        provenance=provenance,
    )


def _profiles_from_remote_payload(payload: dict[str, Any], *, engine_id: str, provider: str) -> list[VoiceProfile]:
    profiles: list[VoiceProfile] = []

    def add_many(items: Any, *, kind: str) -> None:
        if isinstance(items, list):
            for item in items:
                profile = _profile_from_remote_item(item, engine_id=engine_id, provider=provider, kind=kind)
                if profile is not None:
                    profiles.append(profile)

    add_many(payload.get("profiles"), kind="profile")
    add_many(payload.get("voice_profiles"), kind="profile")
    add_many(payload.get("voices"), kind="voice")
    add_many(payload.get("cloned_voices"), kind="clone")

    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            kind = "voice"
            if isinstance(item, dict):
                kind = str(item.get("kind") or item.get("object") or kind)
            profile = _profile_from_remote_item(item, engine_id=engine_id, provider=provider, kind=kind)
            if profile is not None:
                profiles.append(profile)
    elif isinstance(data, dict):
        profiles.extend(_profiles_from_remote_payload(data, engine_id=engine_id, provider=provider))

    return profiles


class OpenAICompatibleTTSAdapter(TTSAdapter):
    """TTS adapter backed by an OpenAI-compatible HTTP speech endpoint."""

    def __init__(
        self,
        *,
        provider: str = "openai-compatible",
        language: str = "en",
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        timeout_s: float | None = None,
        session: Any = None,
        debug_mode: bool = False,
    ) -> None:
        self.provider = normalize_remote_provider(provider)
        self.engine_id = "openai" if self.provider == "openai" else "openai-compatible"
        self.language = str(language or "en").strip().lower() or "en"
        self.model_id = _default_tts_model(self.provider, model_id)
        self.voice = _default_voice(self.provider, voice)
        if isinstance(instructions, str) and instructions.strip():
            self.instructions = str(instructions).strip()
        elif self.provider == "openai":
            self.instructions = env_first("ABSTRACTVOICE_OPENAI_TTS_INSTRUCTIONS")
        else:
            self.instructions = env_first(
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_INSTRUCTIONS",
                "ABSTRACTVOICE_REMOTE_TTS_INSTRUCTIONS",
            )
        self.debug_mode = bool(debug_mode)
        self._sample_rate = 24000
        self._remote_profiles_loaded = False
        self._remote_profiles: list[VoiceProfile] = []
        self._remote_tts_models_loaded = False
        self._remote_tts_models: list[str] = []

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

    def _payload(
        self,
        text: str,
        *,
        format: str,
        voice: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input": str(text),
            "response_format": str(format or "wav").strip().lower() or "wav",
        }
        if self.model_id:
            payload["model"] = str(self.model_id)

        selected_voice = str(voice).strip() if isinstance(voice, str) and voice.strip() else self.voice
        if selected_voice:
            payload["voice"] = str(selected_voice)

        style = instructions if isinstance(instructions, str) and instructions.strip() else self.instructions
        if style:
            payload["instructions"] = str(style)

        if speed is not None:
            try:
                payload["speed"] = float(speed)
            except Exception:
                pass
        return payload

    def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
        return self.synthesize_to_bytes_with_voice(str(text), format=format, voice=None)

    def synthesize_to_bytes_with_voice(
        self,
        text: str,
        *,
        format: str = "wav",
        voice: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
    ) -> bytes:
        payload = self._payload(
            str(text),
            format=str(format or "wav"),
            voice=voice,
            speed=speed,
            instructions=instructions,
        )
        response = self._client.request(
            "POST",
            "/audio/speech",
            endpoint_name="audio/speech",
            json=payload,
        )
        out = extract_audio_bytes_from_response(response)
        if not out:
            raise RuntimeError("Remote TTS returned empty audio")
        return out

    def synthesize(self, text: str) -> np.ndarray:
        audio = self.synthesize_to_bytes(str(text), format="wav")
        arr, sr = decode_audio_bytes_to_array(audio)
        if sr:
            self._sample_rate = int(sr)
        return arr

    def synthesize_with_speed(self, text: str, speed: float) -> np.ndarray:
        audio = self.synthesize_to_bytes_with_voice(str(text), format="wav", speed=float(speed))
        arr, sr = decode_audio_bytes_to_array(audio)
        if sr:
            self._sample_rate = int(sr)
        return arr

    def synthesize_to_audio_chunks(self, text: str):
        audio = self.synthesize(str(text))
        yield audio, int(self.get_sample_rate())

    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        out = Path(output_path)
        fmt = str(format or out.suffix.lstrip(".") or "wav").strip().lower() or "wav"
        data = self.synthesize_to_bytes(str(text), format=fmt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(bytes(data))
        return str(out)

    def set_language(self, language: str) -> bool:
        self.language = str(language or "en").strip().lower() or "en"
        return True

    def get_supported_languages(self) -> list[str]:
        # Remote providers vary widely; treat language as a pass-through hint.
        return []

    def get_sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def is_available(self) -> bool:
        try:
            require_provider_ready(self.provider, base_url=self.base_url, api_key=self.api_key)
            return True
        except Exception:
            return False

    def get_unavailable_reason(self) -> str | None:
        try:
            require_provider_ready(self.provider, base_url=self.base_url, api_key=self.api_key)
            return None
        except Exception as e:
            if self.provider == "openai":
                return (
                    "OpenAI TTS is not configured.\n"
                    "Set OPENAI_API_KEY or pass remote_api_key=...\n"
                    f"Original error: {e}"
                )
            return (
                "OpenAI-compatible TTS is not configured.\n"
                "Set remote_base_url=... or OPENAI_BASE_URL.\n"
                f"Original error: {e}"
            )

    def get_profiles(self) -> list[VoiceProfile]:
        profiles: list[VoiceProfile] = []
        voices: list[str] = []
        if self.provider == "openai":
            voices.extend(_OPENAI_BUILTIN_VOICES)
        if self.provider == "openai":
            raw = env_first("ABSTRACTVOICE_OPENAI_TTS_VOICES")
        else:
            raw = env_first(
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICES",
                "ABSTRACTVOICE_REMOTE_TTS_VOICES",
            )
        if raw:
            voices.extend(_split_csv(raw))
        if self.voice:
            voices.append(str(self.voice))
        seen: set[str] = set()
        for voice in voices:
            key = voice.lower()
            if key in seen:
                continue
            seen.add(key)
            profiles.append(
                VoiceProfile(
                    engine_id=self.engine_id,
                    profile_id=voice,
                    label=voice,
                    params={"voice": voice},
                    tags={"provider": self.provider},
                )
            )

        for profile in self._get_remote_profiles():
            key = str(profile.profile_id or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            profiles.append(profile)
        return profiles

    def list_available_models(self, language: Optional[str] = None) -> dict[str, Any]:
        """List remote TTS voices with available/configured model ids.

        `VoiceManager.list_available_models()` historically displays local Piper
        catalogs. When this remote adapter is active, returning a remote-shaped
        catalog prevents server/web callers from falling back to the local Piper
        catalog and implying local models are installed.
        """
        _ = language
        model_ids = self._get_tts_models()
        active_model = str(self.model_id or (model_ids[0] if model_ids else "")).strip()

        catalog: dict[str, Any] = {}
        for profile in self.get_profiles():
            voice_id = str(getattr(profile, "profile_id", "") or "").strip()
            if not voice_id:
                continue
            tags = getattr(profile, "tags", None) if isinstance(getattr(profile, "tags", None), dict) else {}
            kind = str(tags.get("kind") or "voice") if isinstance(tags, dict) else "voice"
            catalog[voice_id] = {
                "name": str(getattr(profile, "label", "") or voice_id),
                "quality": "remote",
                "size_mb": 0,
                "description": str(getattr(profile, "description", "") or f"{self.engine_id} {kind}"),
                "requires_espeak": False,
                "cached": True,
                "remote": True,
                "engine": self.engine_id,
                "provider": self.provider,
                "voice": voice_id,
                "profile_id": voice_id,
                "model": active_model,
                "available_models": list(model_ids),
            }

        return {self.engine_id: catalog}

    def set_profile(self, profile_id: str) -> bool:
        voice = str(profile_id or "").strip()
        if not voice:
            return False
        self.voice = voice
        return True

    def get_default_profile_id(self, language: str | None = None) -> str | None:
        _ = language
        default = _default_voice(self.provider, None)
        return str(default).strip() if default else None

    def refresh_profiles(self) -> bool:
        self._remote_profiles_loaded = False
        self._remote_profiles = []
        self._remote_tts_models_loaded = False
        self._remote_tts_models = []
        return True

    def _get_tts_models(self) -> list[str]:
        configured = _configured_tts_models(self.provider)
        if self._remote_tts_models_loaded:
            return _dedupe(list(self._remote_tts_models) + configured)

        self._remote_tts_models_loaded = True
        found: list[str] = []
        for path in _remote_model_paths(self.provider):
            try:
                response = self._client.request(
                    "GET",
                    path,
                    endpoint_name="models",
                )
                found = _models_from_remote_payload(response_json(response))
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Remote model list failed for {path}: {e}")
                continue

        self._remote_tts_models = _dedupe(found)
        return _dedupe(list(found) + configured)

    def _get_remote_profiles(self) -> list[VoiceProfile]:
        if self._remote_profiles_loaded:
            return list(self._remote_profiles)

        self._remote_profiles_loaded = True
        found: list[VoiceProfile] = []
        for path in _remote_profile_paths(self.provider):
            try:
                response = self._client.request(
                    "GET",
                    path,
                    endpoint_name="voice profiles",
                )
                payload = response_json(response)
                found = _profiles_from_remote_payload(
                    payload,
                    engine_id=self.engine_id,
                    provider=self.provider,
                )
                # A successful endpoint, even with an empty list, is authoritative.
                break
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Remote profile list failed for {path}: {e}")
                continue

        self._remote_profiles = list(found)
        return list(found)

    def get_active_profile(self) -> VoiceProfile | None:
        if not self.voice:
            return None
        return VoiceProfile(
            engine_id=self.engine_id,
            profile_id=str(self.voice),
            label=str(self.voice),
            params={"voice": str(self.voice)},
            tags={"provider": self.provider},
        )

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "OpenAI-compatible remote TTS",
                "provider": self.provider,
                "base_url": self.base_url,
                "model_id": self.model_id,
                "voice": self.voice,
                "current_language": self.language,
            }
        )
        return info
