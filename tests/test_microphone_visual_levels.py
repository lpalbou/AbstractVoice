import math

import pytest

from abstractvoice.audio.levels import VoiceLevelNormalizer


@pytest.mark.parametrize('rms', [0.0002, 0.001, 0.002])
def test_stationary_room_noise_settles_even_if_vad_calls_it_speech(rms):
    envelope = VoiceLevelNormalizer()
    for _ in range(900):
        level = envelope.push(rms, seconds=1/30, speech=True)
    assert level < 0.01
    # Syllable onset above that room floor still produces visible movement.
    assert envelope.push(rms * 5, seconds=1/30, speech=True) > 0.3


def test_quiet_syllables_and_pauses_remain_distinct():
    envelope = VoiceLevelNormalizer()
    for _ in range(90):
        envelope.push(0.0002, speech=False)
    quiet = envelope.push(0.002, speech=True)
    loud = envelope.push(0.01, speech=True)
    assert 0.2 < quiet < loud <= 1
    for _ in range(15):
        silent = envelope.push(0, speech=False)
    assert silent < 0.001


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 0])
def test_invalid_or_silent_input_never_creates_motion(value):
    envelope = VoiceLevelNormalizer()
    for _ in range(90):
        assert math.isfinite(envelope.push(value))
    assert envelope.level == 0
