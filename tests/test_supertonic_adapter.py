from __future__ import annotations

import io
import importlib.util
from pathlib import Path
import wave

import numpy as np
import pytest


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


class _FakeSupertonicRuntime:
    sample_rate = 24000

    def __init__(self) -> None:
        self.loaded = False
        self.last_call: dict[str, object] = {}

    def _ensure_loaded(self) -> None:
        self.loaded = True

    def synthesize(self, text: str, **kwargs):
        self.last_call = {"text": text, **kwargs}
        return np.asarray([0.0, 0.25, -0.25], dtype=np.float32)

    def iter_audio_chunks(self, text: str, **kwargs):
        self.last_call = {"text": text, **kwargs}
        yield np.asarray([0.1, 0.2], dtype=np.float32), self.sample_rate

    def synthesize_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        return _wav_bytes(audio, self.sample_rate)


def test_supertonic_adapter_profiles_are_available_without_cached_model(tmp_path) -> None:
    from abstractvoice.adapters.tts_supertonic import SupertonicTTSAdapter

    adapter = SupertonicTTSAdapter(cache_dir=str(tmp_path), allow_downloads=False, auto_load=False)

    profiles = adapter.get_profiles()
    profile_ids = {profile.profile_id for profile in profiles}
    assert profile_ids == {"M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"}
    assert all(profile.engine_id == "supertonic" for profile in profiles)
    assert all(profile.tags.get("cached") == "false" for profile in profiles)

    assert adapter.set_profile("f3") is True
    assert adapter.get_active_profile().profile_id == "F3"
    assert adapter.set_profile("unknown") is False


def test_supertonic_adapter_uses_internal_runtime_and_voice_profile(tmp_path) -> None:
    from abstractvoice.adapters.tts_supertonic import SupertonicTTSAdapter

    runtime = _FakeSupertonicRuntime()
    adapter = SupertonicTTSAdapter(
        language="fr",
        cache_dir=str(tmp_path),
        allow_downloads=False,
        auto_load=False,
        runtime=runtime,
        voice_style="F1",
    )

    assert adapter.set_quality_preset("high") is True
    assert adapter.set_profile("M4") is True

    audio = adapter.synthesize_with_speed("Bonjour.", 1.2)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert runtime.loaded is True
    assert runtime.last_call["language"] == "fr"
    assert runtime.last_call["voice_style"] == "M4"
    assert runtime.last_call["total_steps"] == 12
    assert runtime.last_call["speed"] == 1.2

    wav_bytes = adapter.synthesize_to_bytes("Bonjour.", format="wav")
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_supertonic_registry_exposes_engine_when_onnxruntime_is_available(monkeypatch, tmp_path) -> None:
    import abstractvoice.adapters.tts_supertonic as supertonic_mod
    from abstractvoice.adapters.tts_registry import create_tts_adapter, get_supported_tts_engines

    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "onnxruntime":
            return object()
        return orig_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(supertonic_mod.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(supertonic_mod, "is_supertonic_cached", lambda *_args, **_kwargs: False)

    adapter, resolved = create_tts_adapter(
        engine="supertonic",
        language="ja",
        allow_downloads=False,
        auto_load=True,
        cache_dir=str(tmp_path),
    )

    assert "supertonic" in get_supported_tts_engines()
    assert resolved == "supertonic"
    assert adapter is not None
    assert adapter.engine_id == "supertonic"
    assert "ja" in adapter.get_supported_languages()
    assert adapter.is_available() is False


def test_supertonic_registry_reports_missing_optional_dependency(monkeypatch, tmp_path) -> None:
    import abstractvoice.adapters.tts_supertonic as supertonic_mod
    from abstractvoice.adapters.tts_registry import create_tts_adapter

    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "onnxruntime":
            return None
        return orig_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(supertonic_mod.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match=r"abstractvoice\[supertonic\]"):
        create_tts_adapter(
            engine="supertonic",
            language="en",
            allow_downloads=False,
            auto_load=True,
            cache_dir=str(tmp_path),
        )


def test_supertonic_runtime_metadata_does_not_require_external_sdk() -> None:
    from abstractvoice.supertonic.runtime import MODEL_ID, SUPERTONIC_LANGUAGES, SUPERTONIC_VOICE_STYLES

    assert MODEL_ID == "Supertone/supertonic-3"
    assert {"en", "fr", "ko", "ja"}.issubset(set(SUPERTONIC_LANGUAGES))
    assert SUPERTONIC_VOICE_STYLES == ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    assert "supertonic-py" not in pyproject
    assert "\"supertonic\"" not in pyproject
