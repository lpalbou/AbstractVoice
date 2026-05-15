import numpy as np


def test_audio_player_resamples_on_enqueue_when_sample_rate_differs(monkeypatch):
    # This test must be stable in headless/CI environments (no real audio devices).
    # We avoid opening a PortAudio stream by setting a dummy `stream` object.
    from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer

    player = NonBlockingAudioPlayer(sample_rate=48000, debug_mode=False)
    player.stream = object()  # prevent start_stream() / device I/O
    monkeypatch.setattr(player, "_maybe_restart_stream_for_default_device_change", lambda: None)

    audio_24k = np.zeros((24000,), dtype=np.float32)  # 1 second at 24kHz
    player.play_audio(audio_24k, sample_rate=24000)

    out = player.audio_queue.get_nowait()
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert len(out) == 48000  # resampled to output rate


def test_tts_engine_stop_can_keep_stream_open():
    from abstractvoice.tts.adapter_tts_engine import AdapterTTSEngine

    class DummyAdapter:
        def is_available(self):
            return True

        def get_sample_rate(self):
            return 22050

        def synthesize(self, text: str):
            return np.zeros((22050,), dtype=np.float32)

    engine = AdapterTTSEngine(DummyAdapter(), debug_mode=False)
    dummy_stream = object()
    engine.audio_player.stream = dummy_stream
    engine.audio_player.is_playing = True

    ok = engine.stop(close_stream=False)
    assert ok is True
    assert engine.audio_player.stream is dummy_stream


def test_audio_player_prefers_device_default_rate_on_stream_open(monkeypatch):
    from abstractvoice.tts import tts_engine as tts_engine_module
    from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer

    opened = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            opened.append(kwargs)

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    class FakeSoundDevice:
        OutputStream = FakeStream

        @staticmethod
        def query_devices(device=None, kind=None):
            if device is None and kind == "output":
                return {"index": 0, "name": "Default Output", "default_samplerate": 48000, "max_output_channels": 2}
            if isinstance(device, int) and kind == "output":
                return {"index": device, "name": f"Output {device}", "default_samplerate": 44100, "max_output_channels": 2}
            return [
                {"index": 0, "name": "Default Output", "default_samplerate": 48000, "max_output_channels": 2},
                {"index": 1, "name": "Other Output", "default_samplerate": 44100, "max_output_channels": 2},
            ]

    monkeypatch.setattr(tts_engine_module, "_import_sounddevice", lambda: FakeSoundDevice)

    player = NonBlockingAudioPlayer(sample_rate=24000, debug_mode=False)
    player.start_stream()

    assert opened
    assert opened[0]["samplerate"] == 48000
    assert player.sample_rate == 48000
