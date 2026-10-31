"""Device recovery uses the real queue/callback, with no hardware or models."""

from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from abstractvoice.audio.resample import linear_resample_mono
from abstractvoice.tts import tts_engine as module


def render(player, frames, channels=1):
    out = np.full((frames, channels), np.nan, dtype=np.float32)
    player._audio_callback(out, frames, None, None)
    return out


def fake_output(monkeypatch, *, rate=48000, on_start=None, failing=False):
    streams = []

    class Stream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active = False
            self.closed = False
            streams.append(self)

        def start(self):
            if failing:
                raise RuntimeError("output disconnected")
            self.active = True
            if on_start:
                on_start()

        def stop(self):
            self.active = False

        def close(self):
            self.active = False
            self.closed = True

    def forbidden_refresh():
        raise AssertionError("recovery must not restart global PortAudio")

    sd = SimpleNamespace(
        OutputStream=Stream,
        query_devices=lambda *a: {"default_samplerate": rate, "max_output_channels": 2},
        _terminate=forbidden_refresh,
        _initialize=forbidden_refresh,
    )
    monkeypatch.setattr(module, "_import_sounddevice", lambda: sd)
    return Stream, streams


def player_with_pending(monkeypatch, *, rate=48000):
    player = module.NonBlockingAudioPlayer(sample_rate=rate)
    player.stream = SimpleNamespace(active=True, stop=lambda: None, close=lambda: None)
    monkeypatch.setattr(player, "_maybe_restart_stream_for_default_device_change", lambda: None)
    monkeypatch.setattr(player, "_resolve_stream_targets", lambda: [(0, "Test output", "")])
    first = np.arange(1, 9, dtype=np.float32) / 10
    second = np.array([.2, .3, .4, .5], dtype=np.float32)
    player.play_audio(first, sample_rate=rate)
    player.play_audio(second, sample_rate=rate)
    np.testing.assert_array_equal(render(player, 3)[:, 0], first[:3])
    return player, first, second


def test_progress_snapshot_tracks_only_submitted_audio_and_pending_frames(monkeypatch):
    player, first, second = player_with_pending(monkeypatch)
    status = player.playback_status()
    assert status == {
        "frames_written": 3, "pending_frames": len(first) + len(second) - 3,
        "is_playing": True,
        "paused": False, "stream_active": True, "sample_rate": 48000,
        "callback_error": "", "recovering_output": False,
    }
    assert player.pause()
    render(player, 5)
    assert player.playback_status()["frames_written"] == 3
    assert player.playback_status()["paused"]
    player.stop_stream()
    assert player.playback_status()["pending_frames"] == 0
    assert player.playback_status()["frames_written"] == 3  # monotonic across utterances


@pytest.mark.parametrize("channels", [1, 2])
def test_recovery_preserves_only_unsubmitted_suffix_and_queue_order(monkeypatch, channels):
    player, first, second = player_with_pending(monkeypatch)
    Stream, streams = fake_output(monkeypatch)
    old = Stream()
    player.stream = old
    assert player.recover_stream()
    assert old.closed
    assert len(streams) == 2
    expected = np.concatenate([first[3:], second])
    out = render(player, len(expected), channels)
    np.testing.assert_array_equal(out, np.repeat(expected[:, None], channels, axis=1))
    assert player.playback_status()["frames_written"] == len(first) + len(second)
    assert player.playback_status()["pending_frames"] == 0


def test_recovery_preserves_pause_and_resamples_pending_audio_exactly_once(monkeypatch):
    player, first, second = player_with_pending(monkeypatch, rate=24000)
    assert player.pause()
    fake_output(monkeypatch, rate=48000)
    assert player.recover_stream()
    expected = np.concatenate([
        linear_resample_mono(first[3:], 24000, 48000),
        linear_resample_mono(second, 24000, 48000),
    ])
    assert player.playback_status()["paused"]
    assert player.playback_status()["pending_frames"] == len(expected)
    np.testing.assert_array_equal(render(player, len(expected)), np.zeros((len(expected), 1)))
    assert player.resume()
    np.testing.assert_array_equal(render(player, len(expected))[:, 0], expected)


def test_recovery_snapshots_after_old_output_has_finished_its_last_callback(monkeypatch):
    player, first, second = player_with_pending(monkeypatch)
    fake_output(monkeypatch)
    final = []
    player.stream = SimpleNamespace(
        active=False,
        stop=lambda: final.append((player.playback_status()["recovering_output"], render(player, 2)[:, 0])),
        close=lambda: None,
    )
    assert player.recover_stream()
    assert final[0][0]  # visible before close/stop can block on device I/O
    np.testing.assert_array_equal(final[0][1], first[3:5])
    expected = np.concatenate([first[5:], second])
    np.testing.assert_array_equal(render(player, len(expected))[:, 0], expected)


def test_inactive_stream_recovery_on_next_enqueue_converts_old_pending_rate(monkeypatch):
    player, first, second = player_with_pending(monkeypatch, rate=24000)
    fake_output(monkeypatch, rate=48000)
    player.stream.active = False
    third = np.array([.6, .7], dtype=np.float32)
    player.play_audio(third, sample_rate=24000)
    expected = np.concatenate([
        linear_resample_mono(first[3:], 24000, 48000),
        linear_resample_mono(second, 24000, 48000),
        linear_resample_mono(third, 24000, 48000),
    ])
    np.testing.assert_array_equal(render(player, len(expected))[:, 0], expected)


def test_recovery_disables_refresh_even_when_host_and_stale_device_allow_it(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    fake_output(monkeypatch)
    player.allow_device_refresh = lambda: True
    checks = []

    def resolve():
        checks.append(player._may_refresh_devices())
        return [(0, "Test output", "")]

    monkeypatch.setattr(player, "_resolve_stream_targets", resolve)
    assert player.recover_stream()
    assert checks == [False]
    assert player._may_refresh_devices()  # host policy restored after recovery


def test_recovery_failure_retains_pending_audio_without_duplication(monkeypatch):
    player, first, second = player_with_pending(monkeypatch)
    fake_output(monkeypatch, failing=True)
    with pytest.raises(RuntimeError, match="output disconnected"):
        player.recover_stream()
    assert player.stream is None
    expected = np.concatenate([first[3:], second])
    np.testing.assert_array_equal(render(player, len(expected))[:, 0], expected)
    assert not player._recovering_output


def test_cancelled_recovery_does_not_reopen_or_disturb_new_playback(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    _, streams = fake_output(monkeypatch)
    before = player.playback_status()
    assert not player.recover_stream(cancelled=lambda: True)
    assert streams == []
    assert player.playback_status() == before


def test_cancel_during_reopen_cannot_resume_pending_old_generation(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    cancelled = threading.Event()
    _, streams = fake_output(monkeypatch, on_start=cancelled.set)
    assert not player.recover_stream(cancelled=cancelled.is_set)
    assert player.stream is None
    assert streams[-1].closed
    assert player.playback_status()["pending_frames"] == 0
    assert not player.is_playing


def test_stop_during_reopen_serializes_and_clears_old_generation(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    opened, release, cancelled = threading.Event(), threading.Event(), threading.Event()

    def block_open():
        opened.set()
        assert release.wait(2)

    fake_output(monkeypatch, on_start=block_open)
    recovered = []
    recovery = threading.Thread(target=lambda: recovered.append(player.recover_stream(cancelled=cancelled.is_set)))
    recovery.start()
    assert opened.wait(2)
    cancelled.set()
    stop = threading.Thread(target=player.stop_stream)
    stop.start()
    release.set()
    recovery.join(2)
    stop.join(2)
    assert not recovery.is_alive() and not stop.is_alive()
    assert recovered == [False]
    assert player.stream is None
    assert player.playback_status()["pending_frames"] == 0


def test_callback_exception_is_available_to_control_thread(monkeypatch):
    player = module.NonBlockingAudioPlayer()

    def fail(*args):
        raise ValueError("invalid output buffer")

    monkeypatch.setattr(player, "_fill_output_buffer", fail)
    np.testing.assert_array_equal(render(player, 2), np.zeros((2, 1)))
    assert player.playback_status()["callback_error"] == "ValueError: invalid output buffer"


@pytest.mark.parametrize("operation", ["enqueue", "recovery"])
def test_callback_status_hook_does_not_deadlock_concurrent_control(monkeypatch, operation):
    player, _, _ = player_with_pending(monkeypatch)
    fake_output(monkeypatch)
    hook_entered, control_entered, hook_done = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    def hook(*args):
        hook_entered.set()
        if not control_entered.wait(1):
            outcomes.append("control did not enter")
            return
        player.playback_status()
        hook_done.set()

    player.on_audio_chunk = hook
    callback = threading.Thread(target=lambda: render(player, 2), daemon=True)
    callback.start()
    assert hook_entered.wait(1)
    if operation == "recovery":
        def wait_for_callback():
            control_entered.set()
            if not hook_done.wait(1):
                outcomes.append("callback deadlocked during stop")

        player.stream = SimpleNamespace(active=False, stop=wait_for_callback, close=lambda: None)
        control = threading.Thread(target=player.recover_stream, daemon=True)
    else:
        def start_while_callback_active():
            control_entered.set()
            if not hook_done.wait(1):
                outcomes.append("callback deadlocked during enqueue")

        monkeypatch.setattr(player, "start_stream", start_while_callback_active)
        control = threading.Thread(target=lambda: player.play_audio(np.ones(2, dtype=np.float32)), daemon=True)
    control.start()
    callback.join(2)
    control.join(2)
    assert not callback.is_alive() and not control.is_alive()
    assert hook_done.is_set()
    assert outcomes == []


def test_cancel_during_rate_conversion_does_not_enqueue_old_generation(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch, rate=24000)
    fake_output(monkeypatch, rate=48000)
    cancelled = threading.Event()

    def convert_and_cancel(audio, source_rate, target_rate):
        cancelled.set()
        return linear_resample_mono(audio, source_rate, target_rate)

    monkeypatch.setattr(module, "linear_resample_mono", convert_and_cancel)
    assert not player.recover_stream(cancelled=cancelled.is_set)
    assert player.stream is None
    assert player.playback_status()["pending_frames"] == 0


@pytest.mark.parametrize("operation", ["initial-open", "implicit-recovery", "explicit-recovery"])
def test_nonblocking_stop_request_cancels_inflight_open_or_recovery(monkeypatch, operation):
    player, _, _ = player_with_pending(monkeypatch)
    if operation == "initial-open":
        player.stop_stream()
    elif operation == "implicit-recovery":
        player.stream.active = False
    entered, release = threading.Event(), threading.Event()

    def block_open():
        entered.set()
        assert release.wait(2)

    fake_output(monkeypatch, on_start=block_open)
    if operation == "explicit-recovery":
        action = player.recover_stream
    else:
        action = lambda: player.play_audio(np.array([.5, .6], dtype=np.float32), sample_rate=48000)
    worker = threading.Thread(target=action, daemon=True)
    worker.start()
    assert entered.wait(1)
    started = time.monotonic()
    assert not player.try_stop_stream()
    assert time.monotonic() - started < .1
    assert player._stop_requested.is_set()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert player.stream is None
    assert player.playback_status()["pending_frames"] == 0
    assert not player.is_playing


def test_nonblocking_stop_on_idle_control_clears_audio_and_next_play_can_reuse_player(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    fake_output(monkeypatch)
    assert player.try_stop_stream()
    assert player.stream is None
    assert player.playback_status()["pending_frames"] == 0
    assert not player.recover_stream()  # cannot resurrect a stopped utterance
    player.play_audio(np.array([.2, .3], dtype=np.float32), sample_rate=48000)
    assert not player._stop_requested.is_set()
    np.testing.assert_array_equal(render(player, 2)[:, 0], np.array([.2, .3], dtype=np.float32))


def test_stop_request_silences_callbacks_before_busy_control_can_close_output(monkeypatch):
    player, _, _ = player_with_pending(monkeypatch)
    locked, release = threading.Event(), threading.Event()

    def hold_control():
        with player._stream_lock:
            locked.set()
            assert release.wait(2)

    control = threading.Thread(target=hold_control, daemon=True)
    control.start()
    assert locked.wait(1)
    try:
        before = player.playback_status()
        assert not player.try_stop_stream()
        np.testing.assert_array_equal(render(player, 5), np.zeros((5, 1)))
        after = player.playback_status()
        assert after["frames_written"] == before["frames_written"]
        assert after["pending_frames"] == before["pending_frames"]
    finally:
        release.set()
        control.join(2)
    player.stop_stream()


@pytest.mark.parametrize("cancel_during_route_change", [False, True])
def test_play_audio_after_idle_default_route_change_preserves_only_real_stop_intent(monkeypatch, cancel_during_route_change):
    player = module.NonBlockingAudioPlayer(sample_rate=48000)
    opened, release = threading.Event(), threading.Event()

    def on_start():
        opened.set()
        if cancel_during_route_change:
            assert release.wait(2)

    Stream, streams = fake_output(monkeypatch, on_start=on_start)
    old = Stream()
    old.active = True
    player.stream = old
    player._opened_device_key = "previous-default"
    from abstractvoice.tts import audio_devices

    wanted = SimpleNamespace(key="new-default", name="New output")
    monkeypatch.setattr(audio_devices, "resolve_output_device", lambda *a: wanted)
    monkeypatch.setattr(player, "_resolve_stream_targets", lambda: [(0, "New output", "")])
    speech = np.array([.1, .2, .3], dtype=np.float32)
    worker = threading.Thread(target=lambda: player.play_audio(speech, sample_rate=48000), daemon=True)
    worker.start()
    assert opened.wait(1)
    if cancel_during_route_change:
        assert not player.try_stop_stream()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert old.closed
    if cancel_during_route_change:
        assert player._stop_requested.is_set()
        assert player.playback_status()["pending_frames"] == 0
        assert player.stream is None
    else:
        assert not player._stop_requested.is_set()
        assert player._opened_device_key == wanted.key
        assert player.playback_status()["pending_frames"] == len(speech)
        np.testing.assert_array_equal(render(player, len(speech))[:, 0], speech)
