import time

from abstractvoice.recognition import VoiceRecognizer


def test_continuous_stop_phrase_detector_calls_stop_callback(monkeypatch):
    called = {"n": 0}

    def _stop():
        called["n"] += 1

    vr = VoiceRecognizer(transcription_callback=lambda _t: None, stop_callback=_stop, debug_mode=False)
    vr.transcriptions_paused = True

    # Force frequent checks.
    vr._stop_check_interval_s = 0.0
    vr._stop_window_s = 0.1

    # Mock STT to always "hear" stop.
    monkeypatch.setattr(vr, "_transcribe_pcm16", lambda _b, language=None, **_kwargs: "stop")

    assert vr._maybe_detect_stop_phrase_continuous(b"\x00" * 320) is True
    # Subsequent calls may also trigger, but at least once should be enough.
    assert called["n"] >= 1

