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


def test_quiet_speech_is_visible_without_changing_recorded_samples() -> None:
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    values = []
    rec.audio_level_callback = values.append
    rec._audio_level_ema = 0.0
    # 0.002 RMS: below the old 0.004 cutoff, but ordinary quiet speech.
    chunk = np.full((480, 1), 0.002, dtype=np.float32)
    original = chunk.copy()
    for _ in range(8):
        rec._emit_audio_level(chunk)
    assert values[-1] > 0.15
    np.testing.assert_array_equal(chunk, original)
    for _ in range(16):
        rec._emit_audio_level(np.zeros_like(chunk))
    assert values[-1] < 0.001


def test_near_silent_integer_pcm_does_not_become_full_scale() -> None:
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    values = []
    rec.audio_level_callback = values.append
    rec._audio_level_ema = 0.0
    for _ in range(20):
        rec._emit_audio_level(np.ones((480, 1), dtype=np.int16))
    assert max(values) == 0
