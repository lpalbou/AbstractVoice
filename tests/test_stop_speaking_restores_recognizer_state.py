from abstractvoice import VoiceManager


def test_stop_speaking_calls_on_tts_end_to_restore_recognizer_state():
    vm = VoiceManager()
    vm.set_voice_mode("stop")

    calls = {"resume_transcriptions": 0, "resume_tts_interrupt": 0}

    class FakeVoiceRecognizer:
        def resume_transcriptions(self):
            calls["resume_transcriptions"] += 1

        def resume_tts_interrupt(self):
            calls["resume_tts_interrupt"] += 1

    class FakeTtsEngine:
        def stop(self):
            return True

    vm.voice_recognizer = FakeVoiceRecognizer()
    vm.tts_engine = FakeTtsEngine()

    vm.stop_speaking()
    assert calls["resume_transcriptions"] >= 1
    assert calls["resume_tts_interrupt"] >= 1

