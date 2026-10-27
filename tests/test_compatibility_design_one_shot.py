"""The capability catalog declares voice design and one_shot where they exist, and nowhere else."""
from abstractvoice.compatibility import build_compatibility_catalog


def _providers(kind):
    return build_compatibility_catalog().to_dict()["providers"][kind]


def _support(entry, surface, feature):
    return entry["default_surfaces"][surface][feature]["support"]


def test_only_qwen3_tts_can_design_a_stored_voice():
    cloning = _providers("cloning")
    assert _support(cloning["qwen3-tts"], "create", "design") == "native"
    for provider, entry in cloning.items():
        if provider != "qwen3-tts":
            assert _support(entry, "create", "design") == "unsupported", provider


def test_design_languages_match_what_design_voice_accepts():
    from abstractvoice.cloning.design import DESIGN_MODEL_ID, WARMUPS

    meta = _providers("cloning")["qwen3-tts"]["default_surfaces"]["create"]["design"]["metadata"]
    assert sorted(meta["languages"]) == sorted(WARMUPS)
    assert meta["design_model"] == DESIGN_MODEL_ID


def test_one_shot_is_buffered_qwen3_tts_only():
    tts = _providers("tts")
    qwen = tts["qwen3-tts"]
    assert _support(qwen, "bytes", "one_shot") == "native"
    assert qwen["default_surfaces"]["bytes"]["one_shot"]["metadata"]["max_chars"] == 1000
    assert _support(qwen, "playback", "one_shot") == "unsupported"
    for provider, entry in tts.items():
        if provider != "qwen3-tts":
            assert _support(entry, "bytes", "one_shot") == "unsupported", provider
    assert _support(_providers("cloning")["qwen3-tts"], "speak_bytes", "one_shot") == "native"
    assert _support(_providers("cloning")["omnivoice"], "speak_bytes", "one_shot") == "unsupported"


def test_one_shot_per_checkpoint_through_the_plugin_lookup():
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    class _Owner:
        config = {"voice_allow_downloads": False}

    cap = _VoiceCapability(_Owner())
    for model, expected in (
        ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "native"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "native"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "native"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "unsupported"),
    ):
        support = cap.get_capability_support(kind="tts", feature="one_shot", provider="qwen3-tts", model=model)
        assert support["support"] == expected, model
    design = cap.find_compatible_models(kind="cloning", feature="design", surface="create")
    assert design and {m["provider"] for m in design} == {"qwen3-tts"}


def test_one_shot_limit_matches_the_runtime_limit():
    from abstractvoice.qwen3_tts.runtime import ONE_SHOT_MAX_CHARS

    assert _providers("tts")["qwen3-tts"]["default_surfaces"]["bytes"]["one_shot"]["metadata"]["max_chars"] == ONE_SHOT_MAX_CHARS
