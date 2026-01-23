import time

from abstractvoice import VoiceManager


def test_interrupting_cloned_speak_does_not_resume_old_audio(monkeypatch):
    vm = VoiceManager()

    # Fake minimal tts_engine surface needed by speak() streaming path.
    class FakeAudioPlayer:
        sample_rate = 24000

        def play_audio(self, _a):
            return

    produced = []

    class FakeEngine:
        audio_player = FakeAudioPlayer()

        def begin_playback(self, callback=None, **_kwargs):
            self._cb = callback

        def enqueue_audio(self, a):
            # mark which utterance this chunk came from
            produced.append(float(a[0]) if len(a) else -1.0)

        def stop(self):
            return True

    vm.tts_engine = FakeEngine()

    class FakeCloner:
        def speak_to_audio_chunks(self, text, *, voice_id, speed=None, max_chars=240):
            if "first" in str(text):
                for _ in range(200):
                    yield ([0.1] * 240, 24000)
                    time.sleep(0.002)
            else:
                for _ in range(10):
                    yield ([0.2] * 240, 24000)
                    time.sleep(0.002)

    monkeypatch.setattr(vm, "_get_voice_cloner", lambda: FakeCloner())

    vm.speak("first message", voice="voice_id")
    time.sleep(0.01)
    vm.speak("second message", voice="voice_id")

    def _has(v: float, *, eps: float = 1e-3) -> bool:
        return any(abs(x - v) <= eps for x in produced)

    # Wait until we see some "second" chunks.
    deadline = time.time() + 0.5
    while time.time() < deadline and not _has(0.2):
        time.sleep(0.01)

    assert _has(0.2), "Second utterance should produce audio"

    # After the first 0.2 appears, we should not get any more 0.1 chunks.
    idx = next(i for i, x in enumerate(produced) if abs(x - 0.2) <= 1e-3)
    time.sleep(0.05)
    tail = produced[idx:]
    assert not any(abs(x - 0.1) <= 1e-3 for x in tail)

