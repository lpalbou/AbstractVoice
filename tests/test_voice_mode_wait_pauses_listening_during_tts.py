from abstractvoice import VoiceManager


def test_wait_mode_pauses_listening_during_tts_start_and_resumes_on_end():
    vm = VoiceManager()
    vm.set_voice_mode("wait")

    calls = {"pause_listening": 0, "resume_listening": 0}

    class FakeVoiceRecognizer:
        def pause_listening(self):
            calls["pause_listening"] += 1

        def resume_listening(self):
            calls["resume_listening"] += 1

    vm.voice_recognizer = FakeVoiceRecognizer()
    vm._on_tts_start()
    vm._on_tts_end()

    assert calls["pause_listening"] == 1
    assert calls["resume_listening"] == 1

