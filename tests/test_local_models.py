"""Local model presence is a filesystem question."""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractvoice import local_models
from abstractvoice.adapters.tts_piper import (
    PiperTTSAdapter,
    cached_piper_model_ids,
    cached_piper_voice_profiles,
)


def _write_snapshot(hub: Path, repo_id: str, *filenames: str) -> Path:
    snapshot = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
    return snapshot


@pytest.fixture()
def hub(tmp_path, monkeypatch):
    """An empty Hugging Face hub cache for the duration of a test."""
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    return cache


def test_hf_repo_is_cached_finds_a_downloaded_snapshot(hub):
    _write_snapshot(hub, "meituan-longcat/LongCat-AudioDiT-1B", "config.json", "model.safetensors")

    assert local_models.hf_repo_is_cached("meituan-longcat/LongCat-AudioDiT-1B") is True
    assert local_models.hf_repo_is_cached("k2-fsa/OmniVoice") is False
    assert local_models.hf_repo_is_cached("") is False


def test_hf_repo_is_cached_finds_weights_in_a_subdirectory(hub):
    _write_snapshot(hub, "k2-fsa/OmniVoice", "config.json", "audio_tokenizer/model.safetensors")

    assert local_models.hf_repo_is_cached("k2-fsa/OmniVoice") is True


def test_hf_repo_is_cached_rejects_a_metadata_only_snapshot(hub):
    # Hugging Face keeps these in every snapshot, including one whose download
    # was interrupted before any weights arrived.
    _write_snapshot(hub, "k2-fsa/OmniVoice", "README.md", ".gitattributes", "config.json")

    assert local_models.hf_repo_is_cached("k2-fsa/OmniVoice") is False


def test_hf_repo_is_cached_rejects_a_snapshot_whose_blobs_were_collected(hub):
    snapshot = _write_snapshot(hub, "k2-fsa/OmniVoice", "config.json")
    (snapshot / "model.safetensors").symlink_to(hub / "blobs" / "gone")

    assert local_models.hf_repo_is_cached("k2-fsa/OmniVoice") is False


def test_hf_repo_is_cached_accepts_a_local_checkpoint_directory(hub, tmp_path):
    checkpoint = tmp_path / "my-finetune"
    checkpoint.mkdir()
    assert local_models.hf_repo_is_cached(str(checkpoint)) is False  # no weights yet

    (checkpoint / "model.safetensors").write_bytes(b"x")
    assert local_models.hf_repo_is_cached(str(checkpoint)) is True


def test_hf_repo_is_cached_does_not_resolve_a_repo_id_against_the_cwd(hub, tmp_path, monkeypatch):
    # A directory that happens to shadow a repo id must not make it look cached.
    (tmp_path / "k2-fsa" / "OmniVoice").mkdir(parents=True)
    (tmp_path / "k2-fsa" / "OmniVoice" / "model.safetensors").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)

    assert local_models.hf_repo_is_cached("k2-fsa/OmniVoice") is False
    assert local_models.cached_tts_model_ids("omnivoice") == []


def test_cached_tts_model_ids_reports_declared_ids_only_when_downloaded(hub):
    assert local_models.cached_tts_model_ids("audiodit") == []

    _write_snapshot(hub, "meituan-longcat/LongCat-AudioDiT-1B", "model.safetensors")

    assert local_models.cached_tts_model_ids("audiodit") == ["meituan-longcat/LongCat-AudioDiT-1B"]


def test_cached_tts_model_ids_covers_omnivoice_too(hub):
    _write_snapshot(hub, "k2-fsa/OmniVoice", "model.safetensors")

    assert local_models.cached_tts_model_ids("omnivoice") == ["k2-fsa/OmniVoice"]
    assert local_models.cached_tts_model_ids("audiodit") == []


def test_cached_tts_model_ids_accepts_a_checkpoint_outside_the_packaged_catalog(hub):
    """An operator's own finetune is selectable; the packaged catalog only knows
    the default repo, so a declared-ids-only filter would erase the engine."""
    _write_snapshot(hub, "myorg/my-audiodit-finetune", "model.safetensors")

    assert local_models.cached_tts_model_ids("audiodit") == []
    assert local_models.cached_tts_model_ids(
        "audiodit", extra_candidates=["myorg/my-audiodit-finetune"]
    ) == ["myorg/my-audiodit-finetune"]


def test_cached_tts_model_ids_ignores_blank_extra_candidates(hub):
    _write_snapshot(hub, "meituan-longcat/LongCat-AudioDiT-1B", "model.safetensors")

    assert local_models.cached_tts_model_ids("audiodit", extra_candidates=["", "  "]) == [
        "meituan-longcat/LongCat-AudioDiT-1B"
    ]


def test_cached_tts_model_ids_is_empty_for_remote_and_unknown_engines(hub):
    assert local_models.cached_tts_model_ids("openai") == []
    assert local_models.cached_tts_model_ids("openai-compatible") == []
    assert local_models.cached_tts_model_ids("nope") == []
    assert local_models.cached_tts_model_ids("") == []


def test_cached_tts_model_ids_accepts_underscore_engine_ids(hub):
    _write_snapshot(hub, "meituan-longcat/LongCat-AudioDiT-1B", "model.safetensors")

    assert local_models.cached_tts_model_ids("AudioDiT") == ["meituan-longcat/LongCat-AudioDiT-1B"]


def test_cached_piper_model_ids_needs_both_the_weights_and_the_config(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    english = PiperTTSAdapter.PIPER_MODELS["en"][1]
    french = PiperTTSAdapter.PIPER_MODELS["fr"][1]

    (voices / f"{english}.onnx").write_bytes(b"x")
    (voices / f"{english}.onnx.json").write_text("{}")
    (voices / f"{french}.onnx").write_bytes(b"x")  # interrupted: no config

    assert cached_piper_model_ids(str(voices)) == [english]


def test_cached_piper_model_ids_is_empty_when_nothing_is_downloaded(tmp_path):
    assert cached_piper_model_ids(str(tmp_path / "missing")) == []


def test_cached_piper_model_ids_defaults_to_the_piper_voice_directory(monkeypatch, tmp_path):
    voices = tmp_path / ".piper" / "models"
    voices.mkdir(parents=True)
    english = PiperTTSAdapter.PIPER_MODELS["en"][1]
    (voices / f"{english}.onnx").write_bytes(b"x")
    (voices / f"{english}.onnx.json").write_text("{}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert cached_piper_model_ids() == [english]


def test_cached_piper_voice_profiles_covers_only_downloaded_voices(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    english = PiperTTSAdapter.PIPER_MODELS["en"][1]
    (voices / f"{english}.onnx").write_bytes(b"x")
    (voices / f"{english}.onnx.json").write_text("{}")

    profiles = cached_piper_voice_profiles(str(voices))

    assert [profile.profile_id for profile in profiles] == ["amy"]
    assert profiles[0].engine_id == "piper"
    assert profiles[0].params["model"] == english
    assert cached_piper_voice_profiles(str(tmp_path / "missing")) == []


def test_supertonic_models_follow_its_own_cache_probe(monkeypatch, hub):
    monkeypatch.setattr("abstractvoice.supertonic.is_supertonic_cached", lambda: True)
    assert local_models.cached_tts_model_ids("supertonic") == ["supertonic-3"]

    monkeypatch.setattr("abstractvoice.supertonic.is_supertonic_cached", lambda: False)
    assert local_models.cached_tts_model_ids("supertonic") == []
