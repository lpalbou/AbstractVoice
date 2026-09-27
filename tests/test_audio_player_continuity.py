"""Exercise the real output callback without opening an audio device."""

import threading

import numpy as np
import pytest

from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer


def render(player, frames, channels=1):
    out = np.full((frames, channels), np.nan, dtype=np.float32)
    player._audio_callback(out, frames, None, None)
    return out


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("sizes", [(3, 5), (1,) * 8, (0, 3, 0, 5, 0), (8,)])
def test_callback_fills_across_chunks_without_inserting_silence(channels, sizes):
    player = NonBlockingAudioPlayer()
    expected = np.arange(1, 9, dtype=np.float32) / 10
    offset = 0
    for size in sizes:
        player.audio_queue.put(expected[offset:offset + size])
        offset += size
    delivered = []
    player.on_audio_chunk = lambda audio, sr: delivered.append(audio.copy())
    out = render(player, 8, channels)
    np.testing.assert_array_equal(out, np.repeat(expected[:, None], channels, axis=1))
    np.testing.assert_array_equal(np.concatenate(delivered), expected)


def test_partial_chunk_continues_and_only_real_underflow_is_zero_filled():
    player = NonBlockingAudioPlayer()
    player.audio_queue.put(np.arange(1, 6, dtype=np.float32))
    np.testing.assert_array_equal(render(player, 3)[:, 0], [1, 2, 3])
    player.audio_queue.put(np.array([6, 7], dtype=np.float32))
    np.testing.assert_array_equal(render(player, 6)[:, 0], [4, 5, 6, 7, 0, 0])


def test_pause_preserves_pending_samples_and_resume_is_contiguous():
    player = NonBlockingAudioPlayer()
    player.audio_queue.put(np.array([1, 2, 3], dtype=np.float32))
    player.audio_queue.put(np.array([4, 5], dtype=np.float32))
    player.is_playing = True
    np.testing.assert_array_equal(render(player, 2)[:, 0], [1, 2])
    assert player.pause()
    np.testing.assert_array_equal(render(player, 4)[:, 0], [0, 0, 0, 0])
    assert player.current_position == 2
    assert player.resume()
    np.testing.assert_array_equal(render(player, 3)[:, 0], [3, 4, 5])


def test_lifecycle_does_not_end_between_queued_chunks(monkeypatch):
    # Run notification callbacks synchronously to assert their order deterministically.
    class InlineThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(threading, "Thread", InlineThread)
    events = []
    player = NonBlockingAudioPlayer()
    player.is_playing = True
    player.on_audio_start = lambda: events.append("start")
    player.on_audio_end = lambda: events.append("end")
    player.playback_complete_callback = lambda: events.append("complete")
    for size in (2, 3):
        player.audio_queue.put(np.ones(size, dtype=np.float32))
    render(player, 8)
    assert events == ["start"]  # The last output buffer has not played yet.
    render(player, 8)
    assert events == ["start", "end", "complete"]
    render(player, 8)
    assert events == ["start", "end", "complete"]
    player.audio_queue.put(np.ones(2, dtype=np.float32))
    player.is_playing = True
    render(player, 8)
    assert events == ["start", "end", "complete", "start"]
    render(player, 8)
    assert events == ["start", "end", "complete", "start", "end", "complete"]
    render(player, 8)
    assert events == ["start", "end", "complete", "start", "end", "complete"]


def test_failing_audio_hook_does_not_drop_remaining_queued_samples():
    player = NonBlockingAudioPlayer()
    def fail(*args):
        raise RuntimeError("consumer failed")
    player.on_audio_chunk = fail
    player.audio_queue.put(np.ones(3, dtype=np.float32))
    player.audio_queue.put(np.ones(5, dtype=np.float32))
    np.testing.assert_array_equal(render(player, 8)[:, 0], np.ones(8))
