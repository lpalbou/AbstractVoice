"""Which speaker the audio comes out of (2026-09-18).

Operator report: "it only output on the MBP but not on other output device (eg xreal one
pro or headphone). it is critical that we can properly select the audio output" — with
the system default output set to the XREAL glasses and speech coming out of the built-in
speakers anyway.

Two defects, pinned separately below.

1. `start_stream()` built a candidate list of EVERY output device and walked it in order,
   opening whichever one worked. So when the wanted device failed to open for any reason,
   speech went to the next entry in PortAudio's list — on a Mac, the built-in speakers —
   and the only trace was a `print` behind `debug_mode`.
2. Device identity was a PortAudio INDEX, and the "default output" came from
   `sd.query_devices(None, "output")`. PortAudio resolves both inside `Pa_Initialize()`,
   once: in an app that runs for hours, that is the default from launch. Switching to a
   headset an hour in changed nothing the code could observe, so the idle restart check
   could never fire.

No real audio hardware is opened anywhere in this file.
"""

from __future__ import annotations

import pytest

from abstractvoice.tts import audio_devices
from abstractvoice.tts.audio_devices import OutputDevice
from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer

GLASSES = OutputDevice(index=1, name="XREAL One Pro", uid="AppleUSBAudioEngine:XREAL:1", channels=2, is_system_default=True)
SPEAKERS = OutputDevice(index=4, name="MacBook Pro Speakers", uid="BuiltInSpeakerDevice", channels=2)
HEADSET_HOTPLUGGED = OutputDevice(index=None, name="Sony WH-1000XM5", uid="BT-SONY-XM5")


@pytest.fixture()
def devices(monkeypatch):
    """A machine whose default output is the glasses, with the speakers also present."""
    present = [GLASSES, SPEAKERS]
    monkeypatch.setattr(audio_devices, "list_output_devices", lambda: list(present))
    monkeypatch.setattr(audio_devices, "system_default_output", lambda: next(d for d in present if d.is_system_default))
    return present


def _player() -> NonBlockingAudioPlayer:
    player = NonBlockingAudioPlayer(sample_rate=48000)
    player.allow_device_refresh = lambda: False  # never re-init PortAudio in a test
    return player


# --------------------------------------------------------------------------- resolution


def test_a_spec_resolves_by_uid_then_name_then_index(devices) -> None:
    assert audio_devices.resolve_output_device("BuiltInSpeakerDevice") is SPEAKERS
    assert audio_devices.resolve_output_device("MacBook Pro Speakers") is SPEAKERS
    assert audio_devices.resolve_output_device(4) is SPEAKERS
    assert audio_devices.resolve_output_device(None) is GLASSES  # follow the system
    assert audio_devices.resolve_output_device("") is GLASSES
    assert audio_devices.resolve_output_device("Nonexistent") is None


def test_the_persisted_key_is_the_uid_because_indices_and_names_are_not_stable() -> None:
    assert SPEAKERS.key == "BuiltInSpeakerDevice"
    assert OutputDevice(index=2, name="USB Audio").key == "USB Audio"  # no UID: fall back


def test_describe_names_a_missing_device_instead_of_pretending(devices) -> None:
    assert audio_devices.describe_device_spec(None) == "System default (XREAL One Pro)"
    assert audio_devices.describe_device_spec("BuiltInSpeakerDevice") == "MacBook Pro Speakers"
    assert audio_devices.describe_device_spec("BT-SONY-XM5") == "BT-SONY-XM5 (not connected)"


# --------------------------------------------------------------------------- targets


def test_following_the_system_default_targets_that_device_explicitly(devices) -> None:
    """Explicitly, by index — not `device=None`. PortAudio's own idea of "the default"
    is frozen at initialization; the system's is current."""
    assert _player()._resolve_stream_targets() == [(1, "XREAL One Pro", "")]


def test_a_pinned_device_is_the_only_target(devices) -> None:
    player = _player()
    player.set_output_device("BuiltInSpeakerDevice")
    assert player._resolve_stream_targets() == [(4, "MacBook Pro Speakers", "")]


def test_a_pinned_device_that_is_gone_falls_back_to_the_default_and_SAYS_SO(devices) -> None:
    player = _player()
    player.set_output_device("BT-SONY-XM5")
    targets = player._resolve_stream_targets()
    assert [(index, label) for index, label, _ in targets] == [(1, "XREAL One Pro")]
    problem = targets[0][2]
    assert "BT-SONY-XM5" in problem and "XREAL One Pro" in problem


def test_speech_never_wanders_to_an_arbitrary_third_device(devices) -> None:
    """THE BUG. Whatever happens, the only devices playback may use are the one asked
    for and the system default — never 'the next one in the list that opens'."""
    extra = OutputDevice(index=7, name="Jump Desktop Audio", uid="jump-audio", channels=8)
    devices.append(extra)
    for spec in (None, "BuiltInSpeakerDevice", "BT-SONY-XM5"):
        player = _player()
        player.set_output_device(spec)
        for index, label, _ in player._resolve_stream_targets():
            assert index != 7 and label != "Jump Desktop Audio"


def test_a_device_connected_after_launch_is_offered_once_portaudio_is_refreshed(monkeypatch) -> None:
    """PortAudio cannot see it yet (index None). With refresh allowed, the player
    re-enumerates and then targets it; the refresh is the ONLY automatic one."""
    state = {"refreshed": False}
    present = [GLASSES, SPEAKERS, HEADSET_HOTPLUGGED]

    def _resolve(spec):
        if spec == "BT-SONY-XM5":
            return OutputDevice(index=9, name="Sony WH-1000XM5", uid="BT-SONY-XM5", channels=2) if state["refreshed"] else HEADSET_HOTPLUGGED
        return next((d for d in present if d.uid == spec or d.is_system_default and spec in (None, "")), None)

    monkeypatch.setattr(audio_devices, "resolve_output_device", _resolve)
    monkeypatch.setattr(audio_devices, "system_default_output", lambda: GLASSES)

    player = NonBlockingAudioPlayer(sample_rate=48000)
    player.allow_device_refresh = lambda: True
    monkeypatch.setattr(player, "refresh_device_list", lambda: state.__setitem__("refreshed", True) or True)
    player.set_output_device("BT-SONY-XM5")
    assert player._resolve_stream_targets() == [(9, "Sony WH-1000XM5", "")]


def test_the_device_list_is_never_refreshed_while_the_host_is_listening() -> None:
    """Refreshing restarts PortAudio, which invalidates EVERY open stream in the
    process — including the microphone. The host's gate must be obeyed."""
    player = NonBlockingAudioPlayer(sample_rate=48000)
    player.allow_device_refresh = lambda: False
    assert player.refresh_device_list() is False


# --------------------------------------------------------------------------- reporting


def test_a_problem_is_reported_to_the_host_once_per_distinct_problem(devices) -> None:
    player = _player()
    heard: list[str] = []
    player.on_output_device_problem = heard.append
    player._report_device_problem("Speech is playing on the built-in speakers.")
    player._report_device_problem("Speech is playing on the built-in speakers.")
    player._report_device_problem("Could not open the audio output.")
    assert heard == ["Speech is playing on the built-in speakers.", "Could not open the audio output."]


def test_choosing_a_device_clears_old_complaints(devices) -> None:
    player = _player()
    heard: list[str] = []
    player.on_output_device_problem = heard.append
    player._report_device_problem("same message")
    player.set_output_device("BuiltInSpeakerDevice")
    player._report_device_problem("same message")
    assert heard == ["same message", "same message"]


# --------------------------------------------------------------------------- moving


def test_the_stream_moves_when_the_system_default_changes(devices, monkeypatch) -> None:
    """The reported scenario: replies keep coming out of the speakers after the user
    switches the Mac's output to the glasses."""
    player = _player()
    player.stream = object()
    player._opened_device_key = SPEAKERS.key  # stream was opened on the speakers
    restarted: list[str] = []
    monkeypatch.setattr(player, "_is_idle", lambda: True)
    monkeypatch.setattr(player, "stop_stream", lambda: restarted.append("stop"))
    monkeypatch.setattr(player, "start_stream", lambda: restarted.append("start"))

    player._maybe_restart_stream_for_default_device_change()

    assert restarted == ["stop", "start"]
    assert player._opened_device_key == GLASSES.key


def test_the_stream_is_left_alone_when_the_device_did_not_change(devices, monkeypatch) -> None:
    player = _player()
    player.stream = object()
    player._opened_device_key = GLASSES.key
    monkeypatch.setattr(player, "_is_idle", lambda: True)
    monkeypatch.setattr(player, "stop_stream", lambda: pytest.fail("must not restart"))
    player._maybe_restart_stream_for_default_device_change()


def test_a_sentence_is_never_cut_in_half_to_change_device(devices, monkeypatch) -> None:
    player = _player()
    player.stream = object()
    player._opened_device_key = SPEAKERS.key
    monkeypatch.setattr(player, "_is_idle", lambda: False)  # mid-utterance
    monkeypatch.setattr(player, "stop_stream", lambda: pytest.fail("must not restart mid-sentence"))
    player._maybe_restart_stream_for_default_device_change()


# --------------------------------------------------------------------------- the holes
#
# Found by the adversarial pass on the first version of this fix.


def test_an_unseeable_default_is_reported_instead_of_played_silently(monkeypatch) -> None:
    """THE HOLE. When the live default is not in PortAudio's frozen list and a refresh is
    refused (microphone open), the only option left is `device=None` — which means
    PortAudio's STARTUP default, i.e. the very device the user is not listening to. The
    first version appended that with no problem text and a label naming the device it was
    NOT using: the original bug, wearing a correct-looking label."""
    unseeable = OutputDevice(index=None, name="XREAL One Pro", uid="XREAL-UID", is_system_default=True)
    monkeypatch.setattr(audio_devices, "resolve_output_device", lambda spec: unseeable)
    monkeypatch.setattr(audio_devices, "system_default_output", lambda: unseeable)

    player = NonBlockingAudioPlayer(sample_rate=48000)
    player.allow_device_refresh = lambda: False  # the mic is open
    (index, label, problem), = player._resolve_stream_targets()

    assert index is None  # nothing else is available
    assert "XREAL One Pro" in problem and "startup device" in problem
    assert label != "XREAL One Pro", "the label must not name the device we are not using"


def test_silence_is_kept_when_nothing_can_be_resolved_at_all(monkeypatch) -> None:
    """No CoreAudio and no PortAudio answer: there is no evidence anything is wrong, so
    let PortAudio choose without crying wolf."""
    monkeypatch.setattr(audio_devices, "resolve_output_device", lambda spec: None)
    monkeypatch.setattr(audio_devices, "system_default_output", lambda: None)
    player = NonBlockingAudioPlayer(sample_rate=48000)
    player.allow_device_refresh = lambda: False
    assert player._resolve_stream_targets() == [(None, "the system default", "")]
