"""The built-in Qwen3-TTS voice pack: integrity, store layer, engine and plugin wiring. Weight-free."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from abstractvoice.cloning.store import BUILTIN_VOICE_PACK_DIR, VoiceCloneStore, builtin_voice_pack

PACK = json.loads((BUILTIN_VOICE_PACK_DIR / "voices.json").read_text(encoding="utf-8"))
VOICES = PACK["voices"]


# --- pack integrity -----------------------------------------------------------


def test_pack_header_and_ids():
    assert PACK["format"] == "abstractvoice-qwen3-voice-pack/1"
    assert PACK["engine"] == "qwen3-tts" and PACK["model"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    ids = [v["id"] for v in VOICES]
    assert len(ids) == len(set(ids)) == 70
    for vid in ids:
        assert re.fullmatch(r"(en-us|en-gb|fr|es|de|it)-(female|male)-[a-z]+(-[a-z]+)?", vid), vid
        assert not re.fullmatch(r"[0-9a-f]{32}", vid)  # never collides with user clone ids


@pytest.mark.parametrize("voice", VOICES, ids=lambda v: v["id"])
def test_every_voice_is_complete(voice):
    assert voice["sex"] in {"female", "male"} and voice["style"] in {"standard", "bass"}
    assert voice["language_code"] in {"en", "fr", "es", "de", "it"}
    assert voice["name"] and voice["nationality"] and voice["description"].strip()
    assert len(voice["reference_text"].split()) >= 20
    assert voice["model"] == PACK["model"]
    with np.load(BUILTIN_VOICE_PACK_DIR / voice["prompt"]) as arrays:
        codes, emb = arrays["ref_codes"], arrays["speaker_embedding"]
    assert codes.dtype == np.int16 and codes.ndim == 2 and codes.shape[1] == 16 and codes.shape[0] == voice["frames"]
    assert codes.shape[0] > 100 and 0 <= int(codes.min()) and int(codes.max()) < 2048
    assert emb.dtype == np.float32 and emb.shape == (2048,) and np.isfinite(emb).all()


def test_every_nationality_has_both_sexes_and_a_bass():
    by = {}
    for v in VOICES:
        by.setdefault(v["nationality"], []).append(v)
    assert set(by) == {"American", "British", "French", "Spanish", "German", "Italian"}
    for nationality, vs in by.items():
        assert {v["sex"] for v in vs} == {"female", "male"}, nationality
        assert any(v["style"] == "bass" and v["sex"] == "male" for v in vs), nationality


def test_package_data_ships_the_prompts():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"assets/**/*.npz"' in pyproject


# --- store layer --------------------------------------------------------------


def _ref(tmp_path):
    p = tmp_path / "ref.wav"
    p.write_bytes(b"RIFF....WAVE")
    return p


def test_user_store_lists_builtins_after_own_voices(tmp_path):
    store = VoiceCloneStore(tmp_path / "voices", builtin_voices=True)
    own = store.create_voice([str(_ref(tmp_path))], name="Mine", reference_text="hello", engine="omnivoice")
    listed = [v["voice_id"] for v in store.list_voices()]
    assert listed[0] == own and len(listed) == 71
    assert listed[1:] == sorted(builtin_voice_pack())


def test_explicit_store_excludes_builtins_unless_asked(tmp_path):
    assert VoiceCloneStore(tmp_path / "a").list_voices() == []
    with pytest.raises(KeyError):
        VoiceCloneStore(tmp_path / "a").get_voice("fr-female-camille")


def test_builtin_record_and_reference(tmp_path):
    store = VoiceCloneStore(tmp_path / "voices", builtin_voices=True)
    rec = store.get_voice_dict("fr-female-camille")
    assert rec["engine"] == "qwen3-tts" and rec["name"] == "Camille"
    assert rec["meta"]["builtin"] is True and rec["meta"]["model_id"] == PACK["model"]
    assert rec["meta"]["sex"] == "female" and rec["meta"]["language_code"] == "fr"
    (path,) = store.resolve_reference_paths("fr-female-camille")
    assert path == BUILTIN_VOICE_PACK_DIR / "prompts" / "fr-female-camille.npz" and path.exists()
    assert store.is_builtin("fr-female-camille") and store.normalize_reference_audio("fr-female-camille") == 0


@pytest.mark.parametrize("action", ["rename", "delete", "reference_text", "export"])
def test_builtin_voices_are_read_only(tmp_path, action):
    store = VoiceCloneStore(tmp_path / "voices", builtin_voices=True)
    calls = {
        "rename": lambda: store.rename_voice("fr-female-camille", "X"),
        "delete": lambda: store.delete_voice("fr-female-camille"),
        "reference_text": lambda: store.set_reference_text("fr-female-camille", "x"),
        "export": lambda: store.export_voice("fr-female-camille", tmp_path / "x.zip"),
    }
    with pytest.raises(ValueError, match="built-in voice"):
        calls[action]()
    assert store.get_voice("fr-female-camille").name == "Camille"


# --- engine and plugin --------------------------------------------------------


def test_qwen_clone_engine_rebuilds_the_stored_prompt_without_audio(monkeypatch):
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    seen = {}

    class _Runtime:
        def clone_prompt_from_arrays(self, **kwargs):
            seen.update(kwargs)
            return "prompt"

        def build_clone_prompt(self, **_kwargs):
            pytest.fail("a stored prompt must not be re-encoded from audio")

    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    monkeypatch.setattr(engine, "_get_runtime", lambda: _Runtime())
    path = BUILTIN_VOICE_PACK_DIR / "prompts" / "fr-female-camille.npz"
    assert engine._clone_prompt(str(path), "Bonjour.") == "prompt"
    assert seen["ref_codes"].shape[1] == 16 and seen["speaker_embedding"].shape == (2048,)
    assert seen["ref_text"] == "Bonjour." and seen["x_vector_only"] is False


def test_plugin_lists_and_protects_builtin_voices(tmp_path, monkeypatch):
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    store = VoiceCloneStore(tmp_path / "voices", builtin_voices=True)
    monkeypatch.setattr("abstractvoice.cloning.store.VoiceCloneStore", lambda *a, **k: store)
    monkeypatch.setattr(_VoiceCapability, "_get_vm", lambda self: pytest.fail("management must not build a manager"))

    class _Owner:
        config = {}

    cap = _VoiceCapability(_Owner())
    assert cap.get_cloned_voice("en-gb-male-oliver")["meta"]["nationality"] == "British"
    with pytest.raises(ValueError, match="built-in voice"):
        cap.delete_cloned_voice("en-gb-male-oliver")


def test_plugin_catalog_entries_expose_the_profile_in_tags(tmp_path, monkeypatch):
    from abstractvoice.integrations import abstractcore_plugin as plugin

    store = VoiceCloneStore(tmp_path / "voices", builtin_voices=True)
    entry = {"voice_id": "x", **store.get_voice_dict("en-us-male-harrison")}
    tags = plugin._clone_profile_tags(entry)
    assert tags["builtin"] is True and tags["sex"] == "male" and tags["style"] == "bass"
    assert tags["nationality"] == "American" and tags["language_code"] == "en"
    assert plugin._clone_profile_tags({"meta": None}) == {}


def test_cli_voice_prefers_an_exact_clone_id_over_permissive_profiles():
    from abstractvoice.examples.voice_cli import _apply_tts_voice_profile

    class _VM:
        def get_cloned_voice(self, voice_id):
            if voice_id == "fr-female-camille":
                return {"voice_id": voice_id}
            raise KeyError(voice_id)

        def set_profile(self, name, kind="tts"):
            return name != "fr-female-camille" or pytest.fail("a clone id must not be tried as a profile")

    assert _apply_tts_voice_profile(_VM(), "fr-female-camille") == "fr-female-camille"
    assert _apply_tts_voice_profile(_VM(), "alloy") is None  # a profile name still selects the profile
