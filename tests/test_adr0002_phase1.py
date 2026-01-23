from abstractvoice import VoiceManager


def test_default_voice_mode_is_wait():
    vm = VoiceManager()
    assert vm._voice_mode == "wait"


def test_stop_phrase_stops_speaking_but_does_not_force_stop_listening(monkeypatch):
    calls = {"stop_speaking": 0, "stop_listening": 0}

    class FakeVoiceRecognizer:
        def __init__(self, transcription_callback, stop_callback, whisper_model, debug_mode, **_kwargs):
            self.transcription_callback = transcription_callback
            self.stop_callback = stop_callback
            self.is_running = False

        def start(self, tts_interrupt_callback=None):
            self.is_running = True
            self.tts_interrupt_callback = tts_interrupt_callback
            return True

        def stop(self):
            self.is_running = False
            return True

    # Patch the resolver used inside SttMixin.listen()
    import abstractvoice.vm.stt_mixin as stt_mixin_module
    monkeypatch.setattr(stt_mixin_module, "import_voice_recognizer", lambda: FakeVoiceRecognizer)

    vm = VoiceManager()

    # Patch methods we care about.
    monkeypatch.setattr(vm, "stop_speaking", lambda: calls.__setitem__("stop_speaking", calls["stop_speaking"] + 1) or True)
    monkeypatch.setattr(vm, "stop_listening", lambda: calls.__setitem__("stop_listening", calls["stop_listening"] + 1) or True)

    vm.listen(on_transcription=lambda _t: None, on_stop=None)

    # Simulate stop phrase detection -> recognizer invokes stop_callback.
    assert vm.voice_recognizer is not None
    vm.voice_recognizer.stop_callback()

    assert calls["stop_speaking"] == 1
    assert calls["stop_listening"] == 0


def test_stop_phrase_normalization_is_conservative():
    from abstractvoice.stop_phrase import is_stop_phrase

    phrases = ["ok stop", "okay stop"]
    assert is_stop_phrase("ok stop", phrases) is True
    assert is_stop_phrase("Okay, stop!!!", phrases) is True
    assert is_stop_phrase("stop", phrases) is False

