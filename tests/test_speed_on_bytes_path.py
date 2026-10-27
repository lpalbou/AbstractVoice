"""Speed reaches the bytes path (speak_to_bytes, AbstractCore, /v1/audio/speech), weight-free.

Each test measures the produced audio, so it fails if speed is dropped.
"""
import io

import numpy as np
import pytest
import soundfile as sf

from abstractvoice.compatibility import build_compatibility_catalog


def _tone(seconds, sr):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _duration(wav_bytes):
    audio, sr = sf.read(io.BytesIO(wav_bytes))
    return len(audio) / sr


@pytest.mark.parametrize("speed,expected", [(1.5, 2.0), (0.75, 4.0)])
def test_qwen3_tts_bytes_stretch_after_generation(speed, expected):
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    a = object.__new__(Qwen3TTSAdapter)
    a._instructions = None
    a._active_speaker = lambda: "ryan"
    a._speaker_names = lambda: ["ryan"]
    a._synthesize_array = lambda text, speaker, instructions: _tone(3.0, 24000)
    a.get_sample_rate = lambda: 24000
    assert abs(_duration(a.synthesize_to_bytes_with_voice("hello", speed=speed)) - expected) < 0.15
    assert abs(_duration(a.synthesize_to_bytes_with_voice("hello", speed=1.0)) - 3.0) < 0.01


def test_qwen3_tts_clone_bytes_stretch_after_generation():
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    e = object.__new__(Qwen3TTSVoiceCloningEngine)
    e.infer_to_audio_chunks = lambda **kw: iter([(_tone(1.5, 24000), 24000), (_tone(1.5, 24000), 24000)])
    out = e.infer_to_wav_bytes(text="hi", reference_paths=["r.wav"], reference_text="hi", speed=1.5)
    assert abs(_duration(out) - 2.0) < 0.15


def test_supertonic_bytes_speed_is_native():
    from abstractvoice.adapters.tts_supertonic import SupertonicTTSAdapter

    a = object.__new__(SupertonicTTSAdapter)
    a._speed = 1.05
    seen = []

    class _RT:
        def synthesize_to_wav_bytes(self, audio):
            return b"wav"

    a._get_runtime = lambda: _RT()
    a._synthesize_with_speed = lambda text, speed: seen.append(speed) or _tone(1.0, 44100)
    a.set_profile = lambda v: True
    a.synthesize_to_bytes_with_voice("hello", speed=1.3)
    a.synthesize_to_bytes_with_voice("hello", speed=1.0)  # the engine's own default pace
    assert seen == [1.3, 1.05]


def test_piper_bytes_speed_is_a_time_stretch():
    from abstractvoice.adapters.tts_piper import PiperTTSAdapter

    a = object.__new__(PiperTTSAdapter)
    a._sample_rate = 22050
    a.synthesize = lambda text: _tone(3.0, 22050)
    a.set_profile = lambda v: True
    assert abs(_duration(a.synthesize_to_bytes_with_voice("hello", speed=1.5)) - 2.0) < 0.15


def test_catalog_lets_speed_through_on_the_bytes_path():
    from abstractvoice.vm.tts_mixin import TtsMixin

    class VM(TtsMixin):
        pass

    vm = VM()
    level = vm._compatibility_support_level
    assert level(kind="tts", feature="speed", provider="supertonic", surface="bytes") == "native"
    assert level(kind="tts", feature="speed", provider="piper", surface="bytes") == "emulated"
    assert level(kind="tts", feature="speed", provider="qwen3-tts",
                 model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", surface="bytes") == "emulated"
    assert level(kind="cloning", feature="speed", provider="qwen3-tts", surface="speak_bytes") == "emulated"
    assert level(kind="tts", feature="speed", provider="qwen3-tts", surface="playback") == "unsupported"


def test_model_records_inherit_their_provider_defaults():
    from abstractvoice.vm.tts_mixin import TtsMixin

    class VM(TtsMixin):
        pass

    level = VM()._compatibility_support_level
    custom = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    # Not named by the model record: inherited (was reset to unsupported, which dropped per-call voices).
    assert level(kind="tts", feature="profile", provider="qwen3-tts", model=custom, surface="bytes") == "native"
    assert level(kind="tts", feature="quality_preset", provider="qwen3-tts", model=custom, surface="bytes") == "native"
    # Named by the model record: the override wins.
    assert level(kind="tts", feature="instructions", provider="qwen3-tts",
                 model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", surface="bytes") == "unsupported"
    assert level(kind="tts", feature="profile", provider="qwen3-tts",
                 model="Qwen/Qwen3-TTS-12Hz-1.7B-Base", surface="bytes") == "unsupported"
    models = build_compatibility_catalog().to_dict()["providers"]["tts"]["qwen3-tts"]["models"]
    assert models[custom]["surfaces"]["bytes"]["quality_preset"]["support"] == "native"
