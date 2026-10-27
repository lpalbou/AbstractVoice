"""The TTS model list includes the cloning checkpoints, marked (0.19.2).

Qwen3-TTS Base checkpoints speak cloned voices only. They are listed among the
provider's TTS models so a picker can select the checkpoint a clone needs; the
detailed listing marks them ``role: "cloning"`` ("<id> (clones)"). Preset-voice
synthesis on one is still refused with the existing sentence. Weight-free.
"""

from __future__ import annotations

import json

import pytest

BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CUSTOM = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@pytest.fixture
def qwen_cache(tmp_path, monkeypatch):
    """Two declared checkpoints on disk (CustomVoice, Base) and an operator's own Base."""
    import abstractvoice.local_models as lm

    snapshots = {}
    for name, model_type in [(CUSTOM, "custom_voice"), (BASE, "base"), ("myorg/my-qwen-base", "base")]:
        d = tmp_path / name.replace("/", "--")
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"tts_model_type": model_type}))
        snapshots[name] = d
    monkeypatch.setattr(lm, "_declared_model_ids", lambda engine_id: [CUSTOM, BASE] if engine_id == "qwen3-tts" else [])
    monkeypatch.setattr(lm, "hf_repo_is_cached", lambda model_id: str(model_id) in snapshots)
    monkeypatch.setattr(lm, "hf_cached_snapshot_dir", lambda model_id: snapshots.get(str(model_id)))
    return snapshots


def test_the_cloning_checkpoints_are_listed_only_when_asked(qwen_cache):
    import abstractvoice.local_models as lm

    assert lm.cached_tts_model_ids("qwen3-tts") == [CUSTOM], "the preset-voice list is unchanged"
    assert lm.cached_tts_model_ids("qwen3-tts", include_cloning=True) == [CUSTOM, BASE]
    assert lm.cached_tts_model_ids("qwen3-tts", extra_candidates=["myorg/my-qwen-base"], include_cloning=True) == [
        CUSTOM,
        BASE,
        "myorg/my-qwen-base",
    ]


def test_a_checkpoint_role_comes_from_the_catalog_or_its_config(qwen_cache):
    import abstractvoice.local_models as lm

    assert lm.tts_model_role("qwen3-tts", BASE) == "cloning"
    assert lm.tts_model_role("qwen3-tts", "Qwen/Qwen3-TTS-12Hz-1.7B-Base") == "cloning", "declared, even if not on disk"
    assert lm.tts_model_role("qwen3-tts", "myorg/my-qwen-base") == "cloning", "an operator's own Base, by its config"
    assert lm.tts_model_role("qwen3-tts", CUSTOM) == "tts"
    assert lm.tts_model_role("omnivoice", "k2-fsa/OmniVoice") == "tts"


def test_the_plugin_model_list_offers_base_and_the_details_mark_it(qwen_cache, monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    assert plugin._selectable_tts_model_ids_for_provider("qwen3-tts") == [CUSTOM, BASE]
    details = plugin._tts_model_details_by_provider({"qwen3-tts": [CUSTOM, BASE], "openai": ["tts-1"]})
    assert details["qwen3-tts"] == [
        {"id": CUSTOM, "role": "tts", "label": CUSTOM},
        {"id": BASE, "role": "cloning", "label": f"{BASE} (clones)"},
    ]
    assert details["openai"] == [{"id": "tts-1", "role": "tts", "label": "tts-1"}]
    # An all-provider listing still recognises the checkpoint.
    assert plugin._tts_model_detail("", BASE)["role"] == "cloning"


def test_the_light_catalog_lists_base_with_its_role(qwen_cache, monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setattr(plugin, "_engine_runtime_available", lambda engine, *_a, **_k: engine == "qwen3-tts")

    class _Owner:
        config: dict = {"tts_engine": "qwen3-tts"}

    cap = plugin._VoiceCapability(_Owner())
    catalog = cap._light_voice_catalog(provider="qwen3-tts")
    assert BASE in catalog["tts_models_by_provider"]["qwen3-tts"]
    roles = {d["id"]: d["role"] for d in catalog["tts_model_details_by_provider"]["qwen3-tts"]}
    assert roles[BASE] == "cloning" and roles[CUSTOM] == "tts"
    assert BASE in cap.list_tts_models(provider="qwen3-tts")
    detailed = {d["id"]: d for d in cap.list_models_detailed(kind="tts", provider="qwen3-tts")}
    assert detailed[BASE]["label"] == f"{BASE} (clones)"


def test_preset_voice_synthesis_on_base_is_still_refused():
    """Selecting Base as the model does not make it speak a preset voice."""
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    adapter = Qwen3TTSAdapter.__new__(Qwen3TTSAdapter)

    class _Runtime:
        def model_type(self):
            return "base"

    adapter._runtime = _Runtime()
    adapter._language = "en"
    adapter.get_max_chars = lambda: 400
    with pytest.raises(RuntimeError, match="is a cloning checkpoint; select it through voice cloning"):
        adapter._synthesize_chunk("Hello.", speaker="aiden", instructions=None)
    with pytest.raises(RuntimeError, match="is a cloning checkpoint; select it through voice cloning"):
        list(adapter._stream_chunks("Hello.", speaker="aiden", instructions=None))
