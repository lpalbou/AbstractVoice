"""Round 17: streamed chunks play back to back, so the silence an engine leaves around each one
adds up at every boundary (Supertonic: ~0.5 s before and ~0.6 s after each sentence, a ~1 s hole).
Each streamed chunk keeps at most 0.05 s before its first sound and 0.25 s after its last."""
from __future__ import annotations

import numpy as np

from abstractvoice.vm.tts_mixin import (
    STREAM_KEEP_LEADING_SILENCE_S,
    STREAM_KEEP_TRAILING_SILENCE_S,
    TtsMixin,
    trim_stream_silence,
)

SR = 44100


def _sentence(lead_s=0.5, voice_s=1.0, tail_s=0.6):
    voice = 0.2 * np.sin(2 * np.pi * 180 * np.arange(int(voice_s * SR)) / SR).astype(np.float32)
    return np.concatenate([np.zeros(int(lead_s * SR), np.float32), voice, np.zeros(int(tail_s * SR), np.float32)])


def test_a_chunk_keeps_a_short_lead_and_tail_and_all_its_voice():
    out = trim_stream_silence(_sentence(), SR)
    assert abs(len(out) / SR - (STREAM_KEEP_LEADING_SILENCE_S + 1.0 + STREAM_KEEP_TRAILING_SILENCE_S)) < 0.01
    assert np.abs(out).max() == np.abs(_sentence()).max()


def test_a_chunk_without_padding_is_unchanged_and_silence_is_dropped():
    tight = _sentence(lead_s=0, tail_s=0)
    assert len(trim_stream_silence(tight, SR)) == len(tight)
    assert trim_stream_silence(np.zeros(SR, np.float32), SR).size == 0


class _PaddedAdapter:
    engine_id = "fake"

    def is_available(self):
        return True

    def get_max_chars(self):
        return 300

    def synthesize_to_audio_chunks(self, text):
        yield _sentence(), SR


class _VM(TtsMixin):
    def __init__(self):
        self.tts_adapter = _PaddedAdapter()
        self.language = "en"
        self.speed = 1.0

    def _set_last_tts_metrics(self, metrics):
        self.metrics = metrics


def test_back_to_back_sentences_are_separated_by_a_short_pause_not_a_second():
    text = "The first sentence is spoken here. The second sentence follows right after it. " * 3
    chunks = [a for a, _ in _VM().speak_to_audio_chunks(text)]
    assert len(chunks) >= 2
    for a, b in zip(chunks, chunks[1:]):
        tail = len(a) - 1 - int(np.flatnonzero(np.abs(a) > 0.003)[-1])
        head = int(np.flatnonzero(np.abs(b) > 0.003)[0])
        assert (tail + head) / SR <= STREAM_KEEP_LEADING_SILENCE_S + STREAM_KEEP_TRAILING_SILENCE_S + 0.01
