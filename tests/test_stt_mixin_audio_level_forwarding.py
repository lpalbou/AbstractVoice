from abstractvoice.vm import stt_mixin as stt_module
from abstractvoice.vm.stt_mixin import SttMixin


class _DummyVoiceManager(SttMixin):
    def __init__(self) -> None:
        self.voice_recognizer = None
        self.whisper_model = "tiny"
        self.debug_mode = False
        self._aec_enabled = False
        self._aec_stream_delay_ms = 0
        self.language = None
        self.allow_downloads = True
        self._voice_mode = "stop"
        self._transcription_callback = None
        self._stop_callback = None

    def stop_speaking(self):
        return None


def test_listen_forwards_audio_level_callback(monkeypatch) -> None:
    captured = {}

    class _DummyRecognizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def set_profile(self, mode):
            captured["profile"] = mode

        def start(self, tts_interrupt_callback=None):
            captured["tts_interrupt_callback"] = tts_interrupt_callback
            return True

    monkeypatch.setattr(stt_module, "import_voice_recognizer", lambda: _DummyRecognizer)

    vm = _DummyVoiceManager()
    levels = []
    ok = vm.listen(
        on_transcription=lambda _text: None,
        on_stop=lambda: None,
        on_audio_level=lambda lvl: levels.append(float(lvl)),
    )
    assert ok is True
    assert callable(captured.get("audio_level_callback"))
