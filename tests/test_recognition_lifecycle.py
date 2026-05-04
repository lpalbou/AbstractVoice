from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

import abstractvoice.recognition as recognition_module
from abstractvoice.recognition import VoiceRecognizer


def _minimal_recognizer() -> VoiceRecognizer:
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    rec.debug_mode = False
    rec.sample_rate = 16000
    rec.chunk_duration = 30
    rec.chunk_size = 4
    rec.min_speech_chunks = 1
    rec.silence_timeout_chunks = 1
    rec.is_running = True
    rec.thread = None
    rec.stream = None
    rec._startup_event = threading.Event()
    rec._thread_error = None
    rec.last_error = None
    rec.tts_interrupt_callback = None
    rec.tts_interrupt_enabled = True
    rec.listening_paused = False
    rec.audio_level_callback = None
    rec._audio_level_ema = 0.0
    rec.transcriptions_paused = False
    rec.stop_callback = None
    rec.stop_phrases = []
    rec._profile = "stop"
    rec._echo_gate_enabled = False
    rec.aec_enabled = False
    rec._aec = None
    rec.last_stt_metrics = None
    return rec


def test_callback_error_does_not_repeat_completed_utterance(monkeypatch) -> None:
    rec = _minimal_recognizer()
    speech = np.ones((rec.chunk_size, 1), dtype=np.int16)
    silence = np.zeros((rec.chunk_size, 1), dtype=np.int16)
    chunks = [speech, silence, silence]

    class FakeStream:
        def start(self):
            return None

        def read(self, _chunk_size):
            if chunks:
                return chunks.pop(0), False
            rec.is_running = False
            raise RuntimeError("done")

        def abort(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

    class FakeDetector:
        def is_speech(self, audio_data: bytes) -> bool:
            return any(audio_data)

    monkeypatch.setattr(
        recognition_module,
        "_import_audio_deps",
        lambda: SimpleNamespace(InputStream=lambda **_kwargs: FakeStream()),
    )
    rec.voice_detector = FakeDetector()

    transcriptions: list[bytes] = []

    def transcribe(audio_bytes: bytes) -> str:
        transcriptions.append(audio_bytes)
        return "hello"

    def fail_callback(_text: str) -> None:
        raise RuntimeError("callback failed")

    rec._transcribe_pcm16 = transcribe
    rec.transcription_callback = fail_callback

    rec._recognition_loop()

    assert len(transcriptions) == 1
    assert isinstance(rec.last_error, RuntimeError)


def test_stop_closes_stream_before_join() -> None:
    rec = _minimal_recognizer()
    events: list[str] = []

    class FakeStream:
        def abort(self):
            events.append("abort")

        def stop(self):
            events.append("stop")

        def close(self):
            events.append("close")

    class FakeThread:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            events.append("join")
            assert "abort" in events
            assert "close" in events
            self.alive = False

    rec.stream = FakeStream()
    rec.thread = FakeThread()

    assert rec.stop(timeout=0.01) is True
    assert events == ["abort", "stop", "close", "join"]
    assert rec.thread is None


def test_start_returns_false_when_stream_startup_fails(monkeypatch) -> None:
    rec = VoiceRecognizer.__new__(VoiceRecognizer)
    rec.debug_mode = False
    rec.is_running = False
    rec.thread = None
    rec.stream = None

    def fail_import():
        raise RuntimeError("no input device")

    monkeypatch.setattr(recognition_module, "_import_audio_deps", fail_import)

    assert rec.start() is False
    assert isinstance(rec.last_error, RuntimeError)
