import time

from abstractvoice import VoiceManager


def test_stop_speaking_sets_cancel_token_for_cloned_tts(monkeypatch):
    vm = VoiceManager()

    # Fake minimal tts_engine surface needed by speak() streaming path.
    class FakeAudioPlayer:
        sample_rate = 24000

        def play_audio(self, _a):
            return

    class FakeEngine:
        audio_player = FakeAudioPlayer()

        def begin_playback(self, callback=None, **_kwargs):
            self._cb = callback

        def enqueue_audio(self, _a):
            return

        def stop(self):
            return True

    vm.tts_engine = FakeEngine()

    produced = {"n": 0}

    class FakeCloner:
        def speak_to_audio_chunks(self, _text, *, voice_id, speed=None, max_chars=120):
            # Yield many chunks slowly.
            for _ in range(50):
                produced["n"] += 1
                yield ([0.0] * 240, 24000)
                time.sleep(0.01)

    monkeypatch.setattr(vm, "_get_voice_cloner", lambda: FakeCloner())

    vm.speak("hello", voice="voice_id")
    time.sleep(0.05)
    vm.stop_speaking()
    n_after_stop = produced["n"]
    time.sleep(0.05)
    assert produced["n"] <= n_after_stop + 2

