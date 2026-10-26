"""`one_shot=True`: the whole text as ONE generation, on every surface. Weight-free."""
from __future__ import annotations

import io
import types

import numpy as np
import pytest
import soundfile as sf

from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter
from abstractvoice.qwen3_tts.runtime import (
    ONE_SHOT_MAX_CHARS,
    Qwen3TTSSettings,
    estimate_max_new_tokens,
    one_shot_max_new_tokens,
)

LONG = " ".join(f"Sentence number {i} is here." for i in range(30))  # ~800 chars, many sentences


@pytest.fixture(autouse=True)
def no_cached_qwen_weights(monkeypatch):
    monkeypatch.setattr("abstractvoice.adapters.tts_qwen3_tts.hf_cached_snapshot_dir", lambda _model_id: None)


class _Runtime:
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    is_loaded = True
    supports_native_streaming = False  # the PyTorch path segments at 200 chars by default

    def __init__(self, model_type="voice_design"):
        self.settings = Qwen3TTSSettings()
        self.calls = []
        self._type = model_type

    def model_type(self):
        return self._type

    def speaker_names(self):
        return ["ryan"]

    def _ensure_loaded(self):
        pass

    def synthesize_voice_design(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return np.full(2400, 0.1, dtype=np.float32), 24000

    synthesize_custom_voice = synthesize_voice_design

    def synthesize_clone(self, text, **kwargs):
        return self.synthesize_voice_design(text, **kwargs)


def test_one_shot_limit_and_cap():
    assert one_shot_max_new_tokens("Hello there.") == estimate_max_new_tokens("Hello there.", ceiling=4800)
    # Never cut short within the bound, even at the slowest speaking rate.
    assert one_shot_max_new_tokens("x" * ONE_SHOT_MAX_CHARS) < 4800
    with pytest.raises(ValueError, match="at most"):
        one_shot_max_new_tokens("x" * (ONE_SHOT_MAX_CHARS + 1))


@pytest.mark.parametrize("model_type", ["voice_design", "custom_voice"])
def test_adapter_one_shot_is_one_generation_with_the_raised_cap(model_type):
    runtime = _Runtime(model_type)
    adapter = Qwen3TTSAdapter(runtime=runtime)
    adapter.synthesize_to_bytes_with_voice(LONG, instructions="A calm narrator.", one_shot=True)
    assert len(runtime.calls) == 1 and runtime.calls[0][0] == LONG
    assert runtime.calls[0][1]["max_new_tokens"] == one_shot_max_new_tokens(LONG)

    runtime.calls.clear()
    adapter.synthesize_to_bytes_with_voice(LONG, instructions="A calm narrator.")
    assert len(runtime.calls) > 1  # default buffered synthesis segments long text


def test_manager_routes_one_shot_and_refuses_engines_without_it():
    from abstractvoice.vm.tts_mixin import TtsMixin

    class VM(TtsMixin):
        def __init__(self, adapter):
            self.tts_adapter = adapter
            self.language = "en"
            self.speed = 1.0
            self.metrics = None

        def _set_last_tts_metrics(self, metrics):
            self.metrics = metrics

    runtime = _Runtime("custom_voice")
    vm = VM(Qwen3TTSAdapter(runtime=runtime))
    wav = vm.speak_to_bytes(LONG, one_shot=True)
    assert sf.read(io.BytesIO(wav))[0].size and len(runtime.calls) == 1

    class _Plain:
        engine_id = "supertonic"

        def is_available(self):
            return True

        def synthesize_to_bytes(self, text, format="wav"):
            return b""

    with pytest.raises(ValueError, match="one_shot"):
        VM(_Plain()).speak_to_bytes("Hello.", one_shot=True)


def test_clone_engine_and_cloner_one_shot(tmp_path, monkeypatch):
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    runtime = _Runtime("base")
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    monkeypatch.setattr(engine, "_get_runtime", lambda: runtime)
    monkeypatch.setattr(engine, "_clone_prompt", lambda path, text: "prompt")
    ref = tmp_path / "ref.wav"
    sf.write(ref, np.zeros(24000, dtype=np.float32), 24000)

    engine.infer_to_wav_bytes(text=LONG, reference_paths=[str(ref)], reference_text="ref", one_shot=True)
    assert len(runtime.calls) == 1 and runtime.calls[0][1]["max_new_tokens"] == one_shot_max_new_tokens(LONG)

    store = VoiceCloneStore(tmp_path / "voices")
    other = store.create_voice([str(ref)], name="o", reference_text="ref", engine="omnivoice")
    cloner = VoiceCloner(store=store, allow_downloads=False)
    monkeypatch.setattr(cloner, "_get_engine_for_voice", lambda voice: types.SimpleNamespace(supports_one_shot=False))
    with pytest.raises(ValueError, match="qwen3-tts clones"):
        cloner.speak_to_bytes("Hello.", voice_id=other, one_shot=True)


def test_plugin_forwards_one_shot():
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    seen = {}

    class _VM:
        def speak_to_bytes(self, text, **kwargs):
            seen.update(kwargs)
            return b"RIFF"

    class _Owner:
        config = {"voice_manager_instance": _VM()}

    cap = _VoiceCapability(_Owner())
    cap.tts("Hello there.", one_shot=True)
    assert seen.get("one_shot") is True
    seen.clear()
    cap.tts("Hello there.")
    assert "one_shot" not in seen


def test_cli_flag_reaches_speak_to_file(tmp_path):
    from abstractvoice.examples.voice_cli import _run_one_shot_tts, parse_args

    assert parse_args([]).one_shot is False
    args = parse_args(["--one-shot", "--prompt", "Hello there.", "--output", str(tmp_path / "o.wav")])
    assert args.one_shot is True
    seen = {}

    class _VM:
        def speak_to_file(self, text, path, **kwargs):
            seen.update(kwargs)
            return path

        def cleanup(self):
            pass

    _run_one_shot_tts(args, voice_manager_factory=lambda **_k: _VM())
    assert seen.get("one_shot") is True
