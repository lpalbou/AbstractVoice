import numpy as np

from abstractvoice.recognition import VoiceRecognizer


def test_emit_audio_level_normalizes_and_decays() -> None:
    values: list[float] = []
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    rec.audio_level_callback = lambda v: values.append(float(v))
    rec._audio_level_ema = 0.0

    rec._emit_audio_level(np.full((160, 1), 2800, dtype=np.int16))
    assert values
    first = values[-1]
    assert 0.0 <= first <= 1.0
    assert first > 0.0

    rec._emit_audio_level(np.zeros((160, 1), dtype=np.int16))
    second = values[-1]
    assert 0.0 <= second <= 1.0
    assert second < first


def test_emit_audio_level_accepts_scalar() -> None:
    values: list[float] = []
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    rec.audio_level_callback = lambda v: values.append(float(v))
    rec._audio_level_ema = 0.0

    rec._emit_audio_level(0.5)
    assert values
    assert 0.0 <= values[-1] <= 1.0
