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


# ---------------------------------------------------------------------------------------
# Regressions from the 0.19.0 adversarial review
# ---------------------------------------------------------------------------------------


def _isolated_default_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VoiceCloneStore:
    """The machine user's store, redirected to a temporary folder for the test."""
    from abstractvoice.cloning import store as store_module

    real = store_module.VoiceCloneStore
    default_dir = tmp_path / "machine-store"

    def factory(base_dir=None, **kwargs):
        if base_dir is None:
            return real(default_dir, builtin_voices=False)
        return real(base_dir, **kwargs)

    monkeypatch.setattr(store_module, "VoiceCloneStore", factory)
    return real(default_dir, builtin_voices=False)


def _capability(config: dict):
    from abstractvoice.integrations import abstractcore_plugin as plugin

    class Owner:
        pass

    Owner.config = dict(config)
    cap = plugin._VoiceCapability.__new__(plugin._VoiceCapability)
    cap._owner = Owner()
    cap._vm = None
    return cap


class _LazyVM:
    """Like VoiceManager: the cloner is built lazily from the manager's store settings."""

    def __init__(self):
        self._voice_cloner = None
        self.voice_store_dir = None
        self.voice_library_dirs = []

    def _cloner(self):
        if self._voice_cloner is None:
            from abstractvoice.cloning.store import open_voice_store

            self._voice_cloner = type("C", (), {})()
            self._voice_cloner.store = open_voice_store(self.voice_store_dir, self.voice_library_dirs or None)
        return self._voice_cloner

    def get_cloned_voice(self, voice_id):
        return self._cloner().store.get_voice_dict(voice_id)

    def rename_cloned_voice(self, voice_id, name):
        self._cloner().store.rename_voice(voice_id, name)

    def set_cloned_voice_reference_text(self, voice_id, text):
        self._cloner().store.set_reference_text(voice_id, text)

    def delete_cloned_voice(self, voice_id):
        self._cloner().store.delete_voice(voice_id)

    def list_cloned_voices(self):
        return self._cloner().store.list_voices()


@pytest.mark.parametrize("first", ["alice", "plain"])
def test_a_lazily_built_cloner_never_leaks_a_personal_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str) -> None:
    _isolated_default_store(tmp_path, monkeypatch)
    vm = _LazyVM()
    alice = _capability({"voice_store_dir": str(tmp_path / "alice")})
    plain = _capability({})
    for cap in (alice, plain):
        cap._vm = vm  # the shared manager
    order = [alice, plain] if first == "alice" else [plain, alice]
    voice_id = None
    for cap in order:
        with cap._vm_lock(vm):
            ids = [v["voice_id"] for v in vm.list_cloned_voices()]
            if cap is alice:
                voice_id = vm._cloner().store.create_voice_from_wav_bytes(_wav(), name="alice-private", engine="qwen3-tts")
    assert vm._voice_cloner.store is not alice._configured_voice_store()  # back on its own store at rest
    with plain._vm_lock(vm):
        assert voice_id not in [v["voice_id"] for v in vm.list_cloned_voices()]
    # Unconfigured management goes through the lock and never reaches alice's store.
    with pytest.raises(KeyError):
        plain.get_cloned_voice(voice_id)
    with pytest.raises(KeyError):
        plain.rename_cloned_voice(voice_id, "stolen")
    with pytest.raises(KeyError):
        plain.delete_cloned_voice(voice_id)
    assert alice.get_cloned_voice(voice_id)["name"] == "alice-private"


def test_management_never_sees_a_store_applied_by_another_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolated_default_store(tmp_path, monkeypatch)
    vm = _LazyVM()
    alice, plain = _capability({"voice_store_dir": str(tmp_path / "alice")}), _capability({})
    alice._vm = plain._vm = vm
    with alice._vm_lock(vm):
        vid = vm._cloner().store.create_voice_from_wav_bytes(_wav(), name="mine", engine="qwen3-tts")
        # While alice holds the manager, an unconfigured management call waits for the lock
        # rather than reading alice's store: proven by the at-rest checks below.
    with pytest.raises(KeyError):
        plain.get_cloned_voice(vid)


@pytest.mark.parametrize("engine", ["omnivoice", "chroma", None])
def test_cloning_with_a_transcript_and_no_language_never_fails(tmp_path: Path, engine) -> None:
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(store=VoiceCloneStore(tmp_path / "s", builtin_voices=False))
    vid = cloner.clone_voice_from_wav_bytes(_wav(), engine=engine, reference_text="Hello everyone, welcome to the weekly meeting today.")
    meta = cloner.get_cloned_voice(vid)["meta"]
    assert meta.get("language_code") in (None, "en")


def test_the_language_is_the_recordings_whatever_the_engine_speaks(tmp_path: Path) -> None:
    pytest.importorskip("py3langid")
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(store=VoiceCloneStore(tmp_path / "s", builtin_voices=False))
    vid = cloner.clone_voice_from_wav_bytes(
        _wav(), engine="f5_tts", reference_text="Bonjour et bienvenue à tous, nous allons commencer la réunion."
    )
    assert cloner.get_cloned_voice(vid)["meta"]["language_code"] == "fr"  # not forced into f5's en/zh


def test_a_stated_language_wins_over_meta(tmp_path: Path) -> None:
    from abstractvoice.cloning.manager import VoiceCloner

    cloner = VoiceCloner(store=VoiceCloneStore(tmp_path / "s", builtin_voices=False))
    vid = cloner.clone_voice_from_wav_bytes(_wav(), engine="qwen3-tts", meta={"language_code": "en"}, language="de")
    assert cloner.get_cloned_voice(vid)["meta"]["language_code"] == "de"


def test_a_library_voice_without_a_transcript_can_be_spoken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractvoice.cloning.manager import VoiceCloner

    library = VoiceCloneStore(tmp_path / "library", builtin_voices=False)
    lib_id = library.create_voice_from_wav_bytes(_wav(), name="shared", engine="qwen3-tts")  # no transcript
    store = LayeredVoiceStore(VoiceCloneStore(tmp_path / "me", builtin_voices=False), [library])
    cloner = VoiceCloner(store=store)

    class FakeSTT:
        def __init__(self, *a, **k):
            pass

        def is_available(self):
            return True

        def transcribe(self, path):
            return "This is the shared voice."

    import abstractvoice.adapters.stt_faster_whisper as fw

    monkeypatch.setattr(fw, "FasterWhisperAdapter", FakeSTT)
    assert cloner._ensure_reference_text(lib_id) == "This is the shared voice."
    assert cloner._ensure_reference_text(lib_id) == "This is the shared voice."  # remembered, no store write
    assert not (library.get_voice(lib_id).reference_text or "")  # the library is unchanged


def test_a_deployed_voice_names_no_path_of_the_deployer(tmp_path: Path) -> None:
    alice = _layered(tmp_path, "alice")
    ref = tmp_path / "secret" / "alice-recording.wav"
    ref.parent.mkdir()
    ref.write_bytes(_wav())
    vid = alice.create_voice([ref], name="mine", engine="qwen3-tts", meta={"source": str(ref)})
    shared = alice.deploy_to_library(vid)
    assert alice.get_voice_dict(shared)["meta"]["source"] == "library"


def test_concurrent_deploys_keep_every_voice(tmp_path: Path) -> None:
    import threading

    stores = [_layered(tmp_path, f"user{i}") for i in range(6)]
    ids = [s.create_voice_from_wav_bytes(_wav(), name=f"v{i}", engine="qwen3-tts") for i, s in enumerate(stores)]
    deployed = []
    threads = [threading.Thread(target=lambda s=s, v=v: deployed.append(s.deploy_to_library(v))) for s, v in zip(stores, ids)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    library_ids = {v["voice_id"] for v in VoiceCloneStore(tmp_path / "library", builtin_voices=False).list_voices()}
    assert set(deployed) <= library_ids and len(library_ids) == 6


def test_management_inside_a_held_manager_lock_does_not_deadlock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host that manages voices from inside its own TTS stream loop (the stream holds the
    manager's lock between chunks) re-enters the lock; only the outermost holder applies and
    restores the store, so the stream keeps its store until it ends."""
    import threading

    _isolated_default_store(tmp_path, monkeypatch)
    vm = _LazyVM()
    alice = _capability({"voice_store_dir": str(tmp_path / "alice")})
    alice._vm = vm
    outcome = {}

    def stream_loop():
        with alice._vm_lock(vm):  # the stream holds the manager
            vid = vm._cloner().store.create_voice_from_wav_bytes(_wav(), name="mine", engine="qwen3-tts")
            outcome["record"] = alice.get_cloned_voice(vid)  # management inside the loop
            outcome["still_alice"] = Path(vm._voice_cloner.store._base_dir) == tmp_path / "alice"

    t = threading.Thread(target=stream_loop, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "deadlock: management inside a held manager lock"
    assert outcome["record"]["name"] == "mine" and outcome["still_alice"] is True
    assert Path(vm._voice_cloner.store._base_dir) != tmp_path / "alice"  # restored at the outermost release


def test_a_nested_holder_gets_its_own_store_and_hands_the_manager_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    _isolated_default_store(tmp_path, monkeypatch)
    vm = _LazyVM()
    alice, plain = _capability({"voice_store_dir": str(tmp_path / "alice")}), _capability({})
    alice._vm = plain._vm = vm
    seen = {}

    def flow():
        with alice._vm_lock(vm):
            vid = vm._cloner().store.create_voice_from_wav_bytes(_wav(), name="alice-private", engine="qwen3-tts")
            with pytest.raises(KeyError):
                plain.get_cloned_voice(vid)  # nested, unconfigured: its own store, not alice's
            seen["back_to_alice"] = vm.get_cloned_voice(vid)["name"]

    t = threading.Thread(target=flow, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert seen["back_to_alice"] == "alice-private"
