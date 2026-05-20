from abstractvoice.compatibility import build_compatibility_catalog
from abstractvoice.vm.tts_mixin import TtsMixin


def test_compatibility_catalog_contains_current_provider_inventory() -> None:
    catalog = build_compatibility_catalog()
    data = catalog.to_dict()

    assert set(data["providers"]["tts"].keys()) >= {
        "openai",
        "openai-compatible",
        "piper",
        "supertonic",
        "omnivoice",
        "audiodit",
    }
    assert set(data["providers"]["stt"].keys()) >= {
        "openai",
        "openai-compatible",
        "faster-whisper",
        "transformers-asr",
    }
    assert set(data["providers"]["cloning"].keys()) >= {
        "f5_tts",
        "omnivoice",
        "audiodit",
        "chroma",
        "openai",
        "openai-compatible",
    }


def test_compatibility_catalog_support_queries_are_surface_aware() -> None:
    catalog = build_compatibility_catalog()

    openai_instructions = catalog.support_for(
        kind="tts",
        provider="openai",
        model="gpt-4o-mini-tts",
        surface="bytes",
        feature="instructions",
    )
    assert openai_instructions is not None
    assert openai_instructions.support == "native"

    compatible_instructions = catalog.support_for(
        kind="tts",
        provider="openai-compatible",
        model=None,
        surface="bytes",
        feature="instructions",
    )
    assert compatible_instructions is not None
    assert compatible_instructions.support == "conditional"

    piper_playback_speed = catalog.support_for(
        kind="tts",
        provider="piper",
        model="en_US-amy-medium",
        surface="playback",
        feature="speed",
    )
    assert piper_playback_speed is not None
    assert piper_playback_speed.support == "emulated"

    piper_bytes_speed = catalog.support_for(
        kind="tts",
        provider="piper",
        model="en_US-amy-medium",
        surface="bytes",
        feature="speed",
    )
    assert piper_bytes_speed is not None
    assert piper_bytes_speed.support == "unsupported"

    openai_prompt = catalog.support_for(
        kind="stt",
        provider="openai",
        model="gpt-4o-transcribe",
        surface="transcribe",
        feature="prompt",
    )
    assert openai_prompt is not None
    assert openai_prompt.support == "unsupported"

    audiodit_clone_speed = catalog.support_for(
        kind="cloning",
        provider="audiodit",
        model="meituan-longcat/LongCat-AudioDiT-1B",
        surface="speak_bytes",
        feature="speed",
    )
    assert audiodit_clone_speed is not None
    assert audiodit_clone_speed.support == "unsupported"

    chroma_clone_speed = catalog.support_for(
        kind="cloning",
        provider="chroma",
        model="FlashLabs/Chroma-4B",
        surface="speak_bytes",
        feature="speed",
    )
    assert chroma_clone_speed is not None
    assert chroma_clone_speed.support == "emulated"

    f5_clone_speed = catalog.support_for(
        kind="cloning",
        provider="f5_tts",
        model="mrfakename/OpenF5-TTS-Base",
        surface="speak_bytes",
        feature="speed",
    )
    assert f5_clone_speed is not None
    assert f5_clone_speed.support == "native"


def test_compatibility_catalog_marks_tts_1_instruction_support_as_unsupported() -> None:
    catalog = build_compatibility_catalog()

    openai_tts1 = catalog.support_for(
        kind="tts",
        provider="openai",
        model="tts-1",
        surface="bytes",
        feature="instructions",
    )
    assert openai_tts1 is not None
    assert openai_tts1.support == "unsupported"

    openai_tts1_hd = catalog.support_for(
        kind="tts",
        provider="openai",
        model="tts-1-hd",
        surface="bytes",
        feature="instructions",
    )
    assert openai_tts1_hd is not None
    assert openai_tts1_hd.support == "unsupported"


def test_compatibility_catalog_find_models_by_feature() -> None:
    catalog = build_compatibility_catalog()

    matches = catalog.find_models(
        kind="tts",
        feature="instructions",
        surface="bytes",
        support_in=("native", "conditional"),
    )

    assert any(item["provider"] == "openai" and item["model"] == "gpt-4o-mini-tts" for item in matches)
    assert any(item["provider"] == "openai-compatible" for item in matches)


def test_compatibility_catalog_attaches_active_unknown_model_to_current_provider() -> None:
    catalog = build_compatibility_catalog(
        current_tts_provider="openai",
        current_tts_model="gpt-future-tts",
    )

    support = catalog.support_for(
        kind="tts",
        provider="openai",
        model="gpt-future-tts",
        surface="bytes",
        feature="instructions",
    )

    assert support is not None
    assert support.support == "conditional"


def test_compatibility_catalog_attaches_active_stt_model_to_current_provider_without_adapter() -> None:
    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = None
            self.stt_adapter = None
            self._stt_engine_name = "openai"
            self.stt_engine = "openai"
            self.stt_model = "gpt-future-transcribe"

    catalog = _VM().get_compatibility_catalog()
    provider = catalog.get_provider(kind="stt", provider="openai")

    assert provider is not None
    assert "gpt-future-transcribe" in provider.models


def test_compatibility_catalog_attaches_remote_cloning_model_before_cloner_load() -> None:
    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = None
            self.stt_adapter = None
            self.cloning_engine = "openai-compatible"
            self.tts_model = "custom-compatible-tts"

    catalog = _VM().get_compatibility_catalog()
    provider = catalog.get_provider(kind="cloning", provider="openai-compatible")

    assert provider is not None
    assert "custom-compatible-tts" in provider.models


def test_voice_manager_query_helpers_use_central_catalog() -> None:
    class _Adapter:
        engine_id = "openai"
        model_id = "gpt-4o-mini-tts"

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()
            self.tts_model = "gpt-4o-mini-tts"
            self.stt_model = "gpt-4o-mini-transcribe"

    vm = _VM()

    support = vm.get_capability_support(
        kind="tts",
        provider="openai",
        model="gpt-4o-mini-tts",
        surface="bytes",
        feature="instructions",
    )
    assert support is not None
    assert support["support"] == "native"

    matches = vm.find_compatible_models(
        kind="cloning",
        feature="multi_reference_audio",
        surface="create",
        support_in=("native",),
    )
    providers = {item["provider"] for item in matches}
    assert "f5_tts" in providers
    assert "audiodit" in providers


def test_voice_manager_tts_capabilities_use_engine_preference_when_adapter_is_not_loaded() -> None:
    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = None
            self._tts_engine_preference = "openai-compatible"
            self.tts_model = "custom-compatible-tts"
            self.stt_model = None

    caps = _VM().get_tts_capabilities().to_dict()

    assert caps["speed"]["support"] == "conditional"
    assert caps["instructions"]["support"] == "conditional"
