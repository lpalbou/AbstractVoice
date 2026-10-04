"""Round 6 (R6.1): streamed speech starts on a short first sentence and is
synthesised AHEAD of playback, one segment at a time; Supertonic reports where
it runs (device + reason) and does not oversubscribe the CPU."""
from __future__ import annotations

import numpy as np

from abstractvoice.vm.tts_mixin import STREAM_FIRST_SEGMENT_MAX_CHARS, TtsMixin


class _CountingAdapter:
    engine_id = "fake"

    def __init__(self):
        self.calls: list[str] = []

    def is_available(self):
        return True

    def get_max_chars(self):
        return 300

    def execution_device(self):
        return {"device": "cpu", "reason": "test reason"}

    def synthesize_to_audio_chunks(self, text):
        self.calls.append(text)
        yield np.zeros(240, dtype=np.float32), 24000


class _VM(TtsMixin):
    def __init__(self, adapter):
        self.tts_adapter = adapter
        self.language = "en"
        self.speed = 1.0
        self.metrics = None

    def _set_last_tts_metrics(self, metrics):
        self.metrics = metrics


def _reply(words: int) -> str:
    sentence = "The gateway reads this reply aloud while the next sentence is synthesised."
    out, n = [], 0
    while n < words:
        out.append(sentence)
        n += len(sentence.split())
    return " ".join(out)


def test_a_2000_word_reply_yields_the_first_chunk_before_the_second_is_synthesised():
    adapter = _CountingAdapter()
    vm = _VM(adapter)
    stream = vm.speak_to_audio_chunks(_reply(2000))
    first = next(stream)
    assert first[1] == 24000
    # Exactly ONE segment was synthesised when the first audio is available.
    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) <= STREAM_FIRST_SEGMENT_MAX_CHARS
    second = next(stream)
    assert second is not None and len(adapter.calls) == 2
    rest = list(stream)
    assert len(adapter.calls) == 2 + len(rest)
    assert all(len(c) <= 240 for c in adapter.calls)
    assert vm.metrics["device"] == "cpu" and vm.metrics["device_reason"] == "test reason"
    assert vm.metrics["first_segment_max_chars"] == STREAM_FIRST_SEGMENT_MAX_CHARS


def test_first_segment_cap_is_one_short_clause():
    assert STREAM_FIRST_SEGMENT_MAX_CHARS <= 60


def test_supertonic_reports_cpu_with_reason_and_bounded_threads():
    from abstractvoice.supertonic.runtime import SupertonicRuntime

    rt = SupertonicRuntime(allow_downloads=False, providers=["CPUExecutionProvider"])
    info = rt.execution_device()
    assert info["device"] == "cpu"
    assert "CoreML" in info["reason"]
    assert 1 <= info["threads"] <= SupertonicRuntime.DEFAULT_INTRA_OP_THREADS
    assert SupertonicRuntime(allow_downloads=False, intra_op_num_threads=7)._intra_op_threads() == 7
