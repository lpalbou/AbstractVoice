from abstractvoice import VoiceManager


def test_full_mode_does_not_pause_tts_interrupt_when_aec_enabled():
    vm = VoiceManager()
    vm.set_voice_mode("full")

    calls = {"pause": 0}

    class FakeVoiceRecognizer:
        aec_enabled = True

        def pause_tts_interrupt(self):
            calls["pause"] += 1

    vm.voice_recognizer = FakeVoiceRecognizer()
    vm._on_tts_start()
    assert calls["pause"] == 0


def test_full_mode_pauses_tts_interrupt_when_aec_disabled():
    vm = VoiceManager()
    vm.set_voice_mode("full")

    calls = {"pause": 0}

    class FakeVoiceRecognizer:
        aec_enabled = False

        def pause_tts_interrupt(self):
            calls["pause"] += 1

    vm.voice_recognizer = FakeVoiceRecognizer()
    vm._on_tts_start()
    assert calls["pause"] == 1

