"""Qwen3-ASR on Transformers 5.x, weight-free (backlog 0946).

The vendored Qwen3-ASR model is written against transformers 4.57. On 5.x it
failed in five places, each only when a model is CONSTRUCTED or RUN -- import
alone proves nothing:

1. ``Qwen3ASRConfig()`` raised ``AttributeError: thinker_config`` (the 5.x parent
   constructor calls ``get_text_config()`` before the sub-config was set);
2. weight init asked the rotary module for ``compute_default_rope_parameters``;
3. ``create_causal_mask`` rejected ``input_embeds=`` / ``cache_position=`` (5.9+);
4. ``generate()`` passes no ``cache_position`` any more, which the prefill/decode
   split indexed;
5. ``from_pretrained`` left the audio tower's non-persistent sinusoid position
   table uninitialized (5.x re-creates such buffers through ``_init_weights``).

A tiny random model exercises all five: build, save, reload, audio+text
forward, and greedy generation with and without the KV cache.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest


def _transformers_5_4_installed() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    from packaging.version import Version

    return Version(importlib.metadata.version("transformers")).release[:2] >= (5, 4)


# On Transformers < 5.4 (Python 3.9's stt-hf) the loader refuses on purpose;
# tests/test_qwen3_asr_transformers_floor.py covers that side.
pytestmark = pytest.mark.skipif(
    not _transformers_5_4_installed(),
    reason="Qwen3-ASR needs abstractvoice[stt-hf] with Transformers 5.4+ (torch + transformers)",
)

AUDIO_TOKEN, AUDIO_START, AUDIO_END = 60, 61, 62
MEL_BINS, FRAMES = 16, 60


def _tiny_config():
    from abstractvoice.qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig

    return Qwen3ASRConfig(
        thinker_config={
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 256,
                # The shape real checkpoints ship (Qwen/Qwen3-ASR-0.6B config.json).
                "rope_scaling": {
                    "rope_type": "default",
                    "type": "default",
                    "interleaved": True,
                    "mrope_interleaved": True,
                    "mrope_section": [2, 1, 1],
                },
            },
            "audio_config": {
                "num_mel_bins": MEL_BINS,
                "encoder_layers": 1,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 32,
                "d_model": 16,
                "output_dim": 32,
                "n_window": 4,
                "n_window_infer": 8,
                "downsample_hidden_size": 8,
                "max_source_positions": 64,
            },
            "audio_token_id": AUDIO_TOKEN,
            "audio_start_token_id": AUDIO_START,
            "audio_end_token_id": AUDIO_END,
        }
    )


def _audio_prompt():
    import torch

    from abstractvoice.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths

    n_audio = int(_get_feat_extract_output_lengths(torch.tensor(FRAMES)))
    input_ids = torch.tensor([[1, AUDIO_START] + [AUDIO_TOKEN] * n_audio + [AUDIO_END, 2, 3]])
    generator = torch.Generator().manual_seed(1)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "input_features": torch.randn(1, MEL_BINS, FRAMES, generator=generator),
        "feature_attention_mask": torch.ones(1, FRAMES, dtype=torch.long),
    }


@pytest.fixture(scope="module")
def tiny_model_pair(tmp_path_factory):
    import torch

    from abstractvoice.qwen3_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3ASRForConditionalGeneration(_tiny_config()).eval()
    path = tmp_path_factory.mktemp("qwen3_asr_tiny")
    model.save_pretrained(path)
    reloaded = Qwen3ASRForConditionalGeneration.from_pretrained(path).eval()
    return model, reloaded


def test_default_config_constructs():
    from abstractvoice.qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig

    config = Qwen3ASRConfig()
    assert config.get_text_config().hidden_size == config.thinker_config.text_config.hidden_size


def test_config_round_trips_through_dict():
    from abstractvoice.qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig

    config = _tiny_config()
    again = Qwen3ASRConfig(**config.to_dict())
    assert again.thinker_config.text_config.hidden_size == 32
    assert again.thinker_config.audio_token_id == AUDIO_TOKEN


def test_rotary_module_offers_default_rope_parameters():
    from abstractvoice.qwen3_asr.modeling_qwen3_asr import Qwen3ASRThinkerTextRotaryEmbedding

    text_config = _tiny_config().thinker_config.text_config
    inv_freq, scaling = Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters(text_config)
    assert tuple(inv_freq.shape) == (text_config.head_dim // 2,)
    assert scaling == 1.0


def test_save_reload_keeps_every_weight(tiny_model_pair):
    import torch

    model, reloaded = tiny_model_pair
    original, loaded = model.state_dict(), reloaded.state_dict()
    assert set(original) == set(loaded)
    for key, value in original.items():
        assert torch.equal(value, loaded[key]), key
    # Non-persistent buffers are not in the checkpoint: 5.x rebuilds them through
    # `_init_weights`, and the audio tower's sinusoid table stayed garbage.
    for key, value in dict(model.named_buffers()).items():
        assert torch.equal(value, dict(reloaded.named_buffers())[key]), key


def test_audio_forward_matches_after_reload(tiny_model_pair):
    import torch

    model, reloaded = tiny_model_pair
    prompt = _audio_prompt()
    with torch.no_grad():
        first = model.thinker(**prompt).logits
        second = reloaded.thinker(**prompt).logits
    assert tuple(first.shape) == (1, prompt["input_ids"].shape[1], 64)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_cached_generation_matches_uncached(tiny_model_pair):
    """Decode steps must use cache-derived positions: greedy output with the KV
    cache has to equal a full recomputation at every step."""
    import torch

    _, model = tiny_model_pair
    prompt = _audio_prompt()
    common = dict(max_new_tokens=6, do_sample=False, eos_token_id=[63], pad_token_id=0)
    with torch.no_grad():
        cached = model.thinker.generate(**prompt, use_cache=True, **common)
        model.thinker.rope_deltas = None
        uncached = model.thinker.generate(**prompt, use_cache=False, **common)
    assert cached.shape[1] > prompt["input_ids"].shape[1]
    assert torch.equal(cached, uncached)


def test_wrapper_generate_returns_new_tokens(tiny_model_pair):
    import torch

    _, model = tiny_model_pair
    prompt = _audio_prompt()
    with torch.no_grad():
        result = model.generate(**prompt, max_new_tokens=4, eos_token_id=[63], do_sample=False, pad_token_id=0)
    assert result.sequences.shape[1] > prompt["input_ids"].shape[1]


def test_adapter_loads_the_vendored_classes_by_name(monkeypatch):
    """transformers 5.17 ships its own `qwen3_asr` model type, so an AutoModel
    lookup silently returned that class (which has no `generate()`)."""
    import torch

    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter
    from abstractvoice.qwen3_asr import modeling_qwen3_asr, processing_qwen3_asr

    loaded = {}

    class _Model(torch.nn.Module):
        pass

    class _Processor:
        feature_extractor = type("FE", (), {"sampling_rate": 16000})()

    def fake_model(cls, model_id, **kwargs):
        loaded["model"] = (cls.__name__, model_id)
        return _Model()

    def fake_processor(cls, model_id, **kwargs):
        loaded["processor"] = (cls.__name__, model_id)
        return _Processor()

    monkeypatch.setattr(
        modeling_qwen3_asr.Qwen3ASRForConditionalGeneration, "from_pretrained", classmethod(fake_model)
    )
    monkeypatch.setattr(processing_qwen3_asr.Qwen3ASRProcessor, "from_pretrained", classmethod(fake_processor))

    adapter = TransformersASRAdapter(model_id="Qwen/Qwen3-ASR-0.6B", device="cpu", allow_downloads=False)
    adapter._ensure_loaded_qwen3_asr(torch_device=torch.device("cpu"), torch_dtype=None, local_only=True)
    assert loaded == {
        "model": ("Qwen3ASRForConditionalGeneration", "Qwen/Qwen3-ASR-0.6B"),
        "processor": ("Qwen3ASRProcessor", "Qwen/Qwen3-ASR-0.6B"),
    }
