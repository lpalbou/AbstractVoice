from __future__ import annotations


def test_transformers_asr_exposes_cohere_transcribe_model_id() -> None:
    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter

    ids = TransformersASRAdapter.selectable_model_ids()
    assert "CohereLabs/cohere-transcribe-03-2026" in ids
    assert "cohere-transcribe-03-2026" in ids


def test_abstractcore_plugin_lists_cohere_transcribe_for_transformers_asr() -> None:
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    class _Owner:
        logger = None

    models = _VoiceCapability(_Owner()).list_models(kind="stt", provider="transformers-asr")
    assert "CohereLabs/cohere-transcribe-03-2026" in models
    assert "cohere-transcribe-03-2026" in models

