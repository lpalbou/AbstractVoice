import numpy as np

from abstractvoice import VoiceManager


def test_open_tts_text_stream_pipes_text_deltas_to_playback_engine():
    vm = VoiceManager()

    class FakeAdapter:
        engine_id = "fake"

        def is_available(self):
            return True

        def get_sample_rate(self):
            return 24000

        def synthesize_to_audio_chunks(self, _text: str):
            yield np.zeros((240,), dtype=np.float32), 24000

    class FakeEngine:
        def __init__(self):
            self.started = 0
            self.enqueued = 0

        def begin_playback(self, callback=None, **_kwargs):
            _ = callback
            self.started += 1

        def enqueue_audio(self, _a, **_kwargs):
            self.enqueued += 1

        def stop(self, *, close_stream: bool = True):
            _ = close_stream
            return True

        def is_paused(self):
            return False

        def is_active(self):
            return False

    vm.tts_adapter = FakeAdapter()
    vm.tts_engine = FakeEngine()

    s = vm.open_tts_text_stream()
    assert s.push("Hello world.")
    s.close()
    assert s.join(timeout=2.0)
    assert vm.tts_engine.started == 1
    assert vm.tts_engine.enqueued >= 1


def test_open_tts_text_stream_is_cancelled_by_stop_speaking():
    vm = VoiceManager()

    class FakeAdapter:
        engine_id = "fake"

        def is_available(self):
            return True

        def get_sample_rate(self):
            return 24000

        def synthesize_to_audio_chunks(self, _text: str):
            yield np.zeros((240,), dtype=np.float32), 24000

    class FakeEngine:
        def begin_playback(self, callback=None, **_kwargs):
            _ = callback
            return

        def enqueue_audio(self, _a, **_kwargs):
            return

        def stop(self, *, close_stream: bool = True):
            _ = close_stream
            return True

        def is_paused(self):
            return False

        def is_active(self):
            return False

    vm.tts_adapter = FakeAdapter()
    vm.tts_engine = FakeEngine()

    s = vm.open_tts_text_stream()
    vm.stop_speaking()
    assert s.join(timeout=2.0)

