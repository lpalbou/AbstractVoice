import numpy as np
from types import SimpleNamespace


def test_audio_player_resamples_on_enqueue_when_sample_rate_differs(monkeypatch):
    # This test must be stable in headless/CI environments (no real audio devices).
    # We avoid opening a PortAudio stream by setting a dummy `stream` object.
    from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer

    player = NonBlockingAudioPlayer(sample_rate=48000, debug_mode=False)
    player.stream = SimpleNamespace(active=True)  # prevent device I/O
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
                # Device 0 IS the default output above; it must not report different
                # properties when asked for by index. (It did, which is why opening the
                # default EXPLICITLY — rather than as `device=None` — surfaced here.)
                if device == 0:
                    return {"index": 0, "name": "Default Output", "default_samplerate": 48000, "max_output_channels": 2}
                return {"index": device, "name": f"Output {device}", "default_samplerate": 44100, "max_output_channels": 2}
            return [
                {"index": 0, "name": "Default Output", "default_samplerate": 48000, "max_output_channels": 2},
                {"index": 1, "name": "Other Output", "default_samplerate": 44100, "max_output_channels": 2},
            ]

    monkeypatch.setattr(tts_engine_module, "_import_sounddevice", lambda: FakeSoundDevice)
    # Since 2026-09-18 the player opens the wanted device EXPLICITLY instead of passing
    # `device=None` (PortAudio's "default" is the one from process start, which is how
    # speech kept coming out of the built-in speakers after the user switched output).
    # So the fake machine must also say which device is the default; without this the
    # resolver would answer from the real hardware this test is not running on.
    from abstractvoice.tts import audio_devices

    monkeypatch.setattr(audio_devices, "_import_sounddevice", lambda: FakeSoundDevice)
    monkeypatch.setattr(audio_devices, "_ca_output_device_ids", lambda: [])
    monkeypatch.setattr(audio_devices, "_coreaudio", lambda: (None, None))

    player = NonBlockingAudioPlayer(sample_rate=24000, debug_mode=False)
    player.start_stream()

    assert opened
    assert opened[0]["samplerate"] == 48000
    assert player.sample_rate == 48000


def test_audio_player_reopens_inactive_stream_before_queueing(monkeypatch):
    from abstractvoice.tts import tts_engine as module

    opened = []

    class Stream:
        def __init__(self, **kwargs):
            self.active = False
            self.closed = False
            opened.append(self)

        def start(self):
            self.active = True

        def close(self):
            self.closed = True

    fake_sd = SimpleNamespace(
        OutputStream=Stream,
        query_devices=lambda *args: {"default_samplerate": 48000, "max_output_channels": 2},
    )
    monkeypatch.setattr(module, "_import_sounddevice", lambda: fake_sd)
    player = module.NonBlockingAudioPlayer()
    monkeypatch.setattr(player, "_resolve_stream_targets", lambda: [(0, "XREAL", "")])
    monkeypatch.setattr(player, "_maybe_restart_stream_for_default_device_change", lambda: None)
    stale = Stream()
    player.stream = stale  # Device stopped it, but the Python object survives.
    player.play_audio(np.ones(480, dtype=np.float32), sample_rate=48000)

    assert stale.closed
    assert player.stream is not stale and player.stream.active
    assert player.audio_queue.qsize() == 1
    # A healthy stream is reused; no reopen or loss of pending audio.
    player.play_audio(np.ones(480, dtype=np.float32), sample_rate=48000)
    assert len(opened) == 2
    assert player.audio_queue.qsize() == 2
