import numpy as np

from abstractvoice.recognition import VoiceRecognizer


def test_full_mode_echo_gate_blocks_interrupt_on_high_correlation():
    called = {"n": 0}

    def _interrupt():
        called["n"] += 1

    vr = VoiceRecognizer(transcription_callback=lambda _t: None, stop_callback=None, debug_mode=False)
    vr.set_profile("full")
    vr.aec_enabled = False
    vr.tts_interrupt_callback = _interrupt
    vr.tts_interrupt_enabled = True

    # Feed far-end audio: a simple tone-like pattern.
    far = (np.sin(np.linspace(0, 20 * np.pi, 480)).astype(np.float32) * 0.3).astype(np.float32)
    vr.feed_far_end_audio(far, sample_rate=vr.sample_rate)

    # Near-end identical -> should be gated as echo.
    near_pcm16 = (np.clip(far, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    assert vr._is_likely_echo(near_pcm16) is True

