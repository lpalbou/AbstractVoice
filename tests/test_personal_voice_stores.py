"""Personal voice stores and a shared library (backlog 0065); a clone's language (backlog 0063).

Weight-free: stores live in temporary folders and no engine runs.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import pytest

from abstractvoice.cloning.store import LayeredVoiceStore, LibraryVoiceError, VoiceCloneStore, open_voice_store


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes((np.sin(np.arange(24000) / 10) * 3000).astype("int16").tobytes())
    return buf.getvalue()


def _layered(tmp_path: Path, who: str) -> LayeredVoiceStore:
    library = VoiceCloneStore(tmp_path / "library", builtin_voices=False)
    return LayeredVoiceStore(VoiceCloneStore(tmp_path / who, builtin_voices=False), [library])


def test_two_personal_stores_never_see_each_other(tmp_path: Path) -> None:
    alice, bob = _layered(tmp_path, "alice"), _layered(tmp_path, "bob")
    a = alice.create_voice_from_wav_bytes(_wav(), name="alice-voice", engine="qwen3-tts")
    assert [v["voice_id"] for v in alice.list_voices()] == [a]
    assert alice.list_voices()[0]["meta"]["scope"] == "personal"
    assert bob.list_voices() == []
    with pytest.raises(KeyError):
        bob.get_voice(a)
    with pytest.raises(KeyError):
        bob.delete_voice(a)


def test_library_voices_are_listed_for_everyone_and_read_only(tmp_path: Path) -> None:
    alice, bob = _layered(tmp_path, "alice"), _layered(tmp_path, "bob")
    a = alice.create_voice_from_wav_bytes(_wav(), name="narrator", engine="qwen3-tts")
    shared = alice.deploy_to_library(a)
    assert shared != a and alice.scope_of(a) == "personal"  # the personal voice is kept
    for store in (alice, bob):
        record = store.get_voice_dict(shared)
        assert record["name"] == "narrator" and record["meta"]["scope"] == "library"
        assert store.resolve_reference_paths(shared)[0].exists()
        for change in (lambda: store.rename_voice(shared, "x"), lambda: store.delete_voice(shared),
                       lambda: store.set_reference_text(shared, "x"), lambda: store.set_language(shared, "fr", source="request")):
            with pytest.raises(LibraryVoiceError):
                change()
    with pytest.raises(LibraryVoiceError):
        bob.deploy_to_library(shared)  # a library voice is not bob's to deploy


def test_open_voice_store_defaults(tmp_path: Path) -> None:
    default = open_voice_store()
    assert isinstance(default, VoiceCloneStore)  # unconfigured: today's machine-user store
    layered = open_voice_store(tmp_path / "me")
    assert isinstance(layered, LayeredVoiceStore)
    # Without explicit libraries, the machine user's store (built-in voices included) is the library.
    assert any(v["meta"].get("builtin") and v["meta"]["scope"] == "library" for v in layered.list_voices()) or not any(
        v["meta"].get("builtin") for v in default.list_voices()
    )
    explicit = open_voice_store(tmp_path / "me", [tmp_path / "lib"])
    assert explicit.libraries[0]._base_dir == tmp_path / "lib"


def test_a_shared_manager_speaks_from_the_callers_store(tmp_path: Path) -> None:
    """Two capabilities with the same engine settings share one VoiceManager (one copy of the
    weights); the manager's lock points it at the caller's store."""
    from abstractvoice.integrations import abstractcore_plugin as plugin

    class FakeCloner:
        def __init__(self):
            self.store = VoiceCloneStore(tmp_path / "unused", builtin_voices=False)

    class FakeVM:
        _voice_cloner = FakeCloner()

    vm = FakeVM()

    def capability(who: str):
        class Owner:
            config = {"voice_store_dir": str(tmp_path / who), "voice_library_dirs": [str(tmp_path / "library")]}

        cap = plugin._VoiceCapability.__new__(plugin._VoiceCapability)
        cap._owner = Owner()
        cap._vm = None
        return cap

    alice, bob = capability("alice"), capability("bob")
    with alice._vm_lock(vm):
        a = vm._voice_cloner.store.create_voice_from_wav_bytes(_wav(), name="mine", engine="qwen3-tts")
    def personal(store):
        return [v["voice_id"] for v in store.list_voices() if v["meta"]["scope"] == "personal"]

    with bob._vm_lock(vm):
        assert personal(vm._voice_cloner.store) == []
        assert any(v["meta"].get("builtin") for v in vm._voice_cloner.store.list_voices())  # the library's built-ins
        assert vm.voice_store_dir == str(tmp_path / "bob")
    with alice._vm_lock(vm):
        assert personal(vm._voice_cloner.store) == [a]
    # Management goes to the caller's own store too.
    assert alice.get_cloned_voice(a)["name"] == "mine"
    with pytest.raises(KeyError):
        bob.get_cloned_voice(a)
    deployed = alice.deploy_voice_to_library(a)
    assert deployed["meta"]["scope"] == "library" and bob.get_cloned_voice(deployed["voice_id"])["name"] == "mine"


# ---------------------------------------------------------------------------------------
# Backlog 0063: a clone records its language
# ---------------------------------------------------------------------------------------


def test_a_stated_language_is_recorded_for_every_input_form(tmp_path: Path) -> None:
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(store=VoiceCloneStore(tmp_path / "s", builtin_voices=False))
    from_bytes = cloner.clone_voice_from_wav_bytes(_wav(), name="b", engine="qwen3-tts", language="FR")
    path = tmp_path / "ref.wav"
    path.write_bytes(_wav())
    from_path = cloner.clone_voice(str(path), name="p", engine="qwen3-tts", language="fr")
    for vid in (from_bytes, from_path):
        meta = cloner.get_cloned_voice(vid)["meta"]
        assert meta["language_code"] == "fr" and meta["language_source"] == "request"


def test_without_a_stated_language_the_transcript_decides_or_nothing_is_guessed(tmp_path: Path) -> None:
    pytest.importorskip("py3langid")
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(store=VoiceCloneStore(tmp_path / "s", builtin_voices=False))
    fr = cloner.clone_voice_from_wav_bytes(
        _wav(), engine="qwen3-tts", reference_text="Bonjour et bienvenue à tous, nous allons commencer la réunion."
    )
    assert cloner.get_cloned_voice(fr)["meta"]["language_code"] == "fr"
    assert cloner.get_cloned_voice(fr)["meta"]["language_source"] == "text"
    unknown = cloner.clone_voice_from_wav_bytes(_wav(), engine="qwen3-tts", reference_text="Hi.")
    assert "language_code" not in cloner.get_cloned_voice(unknown)["meta"]  # too short: no guess
    # A transcript set later teaches the language once.
    cloner.set_reference_text(unknown, "Guten Morgen, wir beginnen jetzt mit der Besprechung des Projekts.")
    assert cloner.get_cloned_voice(unknown)["meta"]["language_code"] == "de"


def test_the_plugin_passes_the_language_for_every_input_form() -> None:
    from abstractvoice.integrations import abstractcore_plugin as plugin

    calls = []

    class VM:
        cloning_engine = "qwen3-tts"

        def clone_voice(self, path, **kw):
            calls.append(("path", kw.get("language")))
            return "v1"

        def clone_voice_from_wav_bytes(self, data, **kw):
            calls.append(("bytes", kw.get("language")))
            return "v2"

    cap = plugin._VoiceCapability.__new__(plugin._VoiceCapability)
    cap._owner = type("O", (), {"config": {}})()
    cap._vm = None
    cap._get_vm_for_clone_request = lambda **kw: VM()
    cap.clone(b"RIFF", provider="qwen3-tts", language="fr")
    cap.clone("/tmp/ref.wav", provider="qwen3-tts", metadata={"language": "de"})
    assert calls == [("bytes", "fr"), ("path", "de")]
