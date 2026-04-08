import time

import numpy as np

from abstractvoice import VoiceManager


def test_stop_speaking_cancels_base_streaming_worker(monkeypatch):
    vm = VoiceManager()

    produced = {"n": 0}

    class FakeAdapter:
        engine_id = "fake"

        def is_available(self):
            return True

        def get_sample_rate(self):
            return 24000

        def synthesize_to_audio_chunks(self, _text: str):
            # Yield many chunks slowly (simulates a streaming backend).
            for _ in range(50):
                produced["n"] += 1
                yield np.zeros((240,), dtype=np.float32), 24000
                time.sleep(0.01)

    class FakeEngine:
        def begin_playback(self, callback=None, **_kwargs):
            self._cb = callback

        def enqueue_audio(self, _a, **_kwargs):
            return

        def stop(self, *, close_stream: bool = True):
            _ = close_stream
            return True

        def is_paused(self):
            return False

    vm.tts_adapter = FakeAdapter()
    vm.tts_engine = FakeEngine()

    # Enable streamed delivery for base TTS.
    vm.set_tts_delivery_mode("streamed")

    vm.speak("hello")
    time.sleep(0.05)
    vm.stop_speaking()
    n_after_stop = produced["n"]
    time.sleep(0.05)

    # Best-effort: allow 1-2 more chunks due to scheduling.
    assert produced["n"] <= n_after_stop + 2

