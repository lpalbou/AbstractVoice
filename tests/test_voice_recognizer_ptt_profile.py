from abstractvoice.recognition import VoiceRecognizer


def test_voice_recognizer_ptt_profile_is_more_responsive():
    vr = VoiceRecognizer(transcription_callback=lambda _t: None, stop_callback=None, chunk_duration=30)
    vr.set_profile("ptt")
    assert vr.min_speech_chunks == 1
    assert vr.silence_timeout_chunks <= int(round(1500.0 / 30.0))

