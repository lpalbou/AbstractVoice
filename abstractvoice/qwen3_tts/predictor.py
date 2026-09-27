"""Package-owned, fixed-depth residual-codebook generation.

This is not a replacement for Transformers' public ``generate``. The talker
needs only codec IDs from a two-token, unpadded prefill and a fixed set of heads.
Other generation contracts retain the original path, before consuming any RNG.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Explicit neutral semantics, NOT whatever defaults a future Transformers
# version chooses. Unknown fields also fall back, even if newly defaulted.
_NEUTRAL = dict.fromkeys("""
    min_new_tokens max_time stop_strings cache_implementation cache_config min_p
    top_h bad_words_ids forced_bos_token_id forced_eos_token_id
    exponential_decay_length_penalty suppress_tokens begin_suppress_tokens
    sequence_bias watermarking_config pad_token_id bos_token_id eos_token_id
    decoder_start_token_id prompt_lookup_num_tokens max_matching_ngram_size
    assistant_early_exit compile_config continuous_batching_config penalty_alpha
    dola_layers constraints force_words_ids prefill_chunk_size
""".split(), (None,))
_NEUTRAL.update(dict.fromkeys("""
    early_stopping renormalize_logits remove_invalid_values token_healing
    output_attentions output_scores output_logits is_assistant disable_compile
    low_memory
""".split(), (None, False)))
_NEUTRAL.update(dict.fromkeys("""
    min_length no_repeat_ngram_size encoder_no_repeat_ngram_size epsilon_cutoff
    eta_cutoff diversity_penalty
""".split(), (None, 0)))
_NEUTRAL.update(dict.fromkeys("""
    num_beams num_beam_groups num_return_sequences typical_p repetition_penalty
    encoder_repetition_penalty length_penalty guidance_scale
""".split(), (None, 1)))
_NEUTRAL.update(
    use_cache=(True,), num_assistant_tokens=(None, 20),
    num_assistant_tokens_schedule=(None, "constant"),
    assistant_confidence_threshold=(None, 0.4),
    assistant_lookbehind=(None, 10), target_lookbehind=(None, 10),
)
_OVERRIDDEN = {
    "max_length", "max_new_tokens", "do_sample", "temperature", "top_k", "top_p",
    "output_hidden_states", "return_dict_in_generate",
    "_from_model_config", "transformers_version",
}


def _unsupported_reason(predictor, inputs_embeds, count, config):
    if predictor.training:
        return "training mode"
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[1] != 2:
        return "prefill must contain exactly two unpadded embeddings"
    if count != predictor.config.num_code_groups - 1 or count <= 0:
        return "nonstandard codebook count"
    if any(kind != "full_attention" for kind in predictor.config.layer_types):
        return "non-full-attention cache"
    if config.use_cache is not True:
        return "KV caching disabled"
    for key, value in config.to_dict().items():
        if key in _OVERRIDDEN:
            continue
        if key not in _NEUTRAL or value not in _NEUTRAL[key]:
            return f"generation option {key}={value!r}"
    return None


def _valid_probabilities(probabilities):
    """Validate all distributions together, without per-step GPU reductions."""
    return (
        torch.isfinite(probabilities).all()
        & (probabilities >= 0).all()
        & (probabilities.sum(dim=-1) > 0).all()
    )


def _sample_exponential_unchecked(probabilities):
    """Private draw; caller MUST validate probabilities before exposing IDs.

    Same exponential-race construction as torch.multinomial(n=1). Clamping
    exact-zero noise prevents a masked 0/0 from winning.
    """
    noise = torch.empty_like(probabilities).exponential_()
    noise.clamp_min_(torch.finfo(probabilities.dtype).tiny)
    torch.div(probabilities, noise, out=noise)
    return noise.argmax(dim=-1, keepdim=True)


@torch.no_grad()
def generate_codebooks(
    predictor, *, inputs_embeds, max_new_tokens, do_sample, top_p, top_k, temperature,
):
    """Return residual codec IDs, with call-local KV state and stock sampling.

    Resolve configuration afresh so mutations cannot leave a stale fast path.
    Fallback happens only before inference; model/sampling failures propagate
    without retrying an already consumed random sequence. The opt-in exponential
    sampler validates before returning the complete frame; a failed request can
    consume more RNG/work than stock multinomial, but never emits invalid codes.
    """
    kwargs = dict(
        inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens,
        do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
        output_hidden_states=True, return_dict_in_generate=True,
    )
    if getattr(predictor, "_abstractvoice_predictor", "auto") == "reference":
        return predictor.generate(**kwargs).sequences

    config, _ = predictor._prepare_generation_config(None, **kwargs)
    reason = _unsupported_reason(predictor, inputs_embeds, max_new_tokens, config)
    if reason is not None:
        logger.warning_once("#FALLBACK Qwen codebook generation uses Transformers: %s", reason)
        return predictor.generate(**kwargs).sequences

    processors = LogitsProcessorList()
    if config.do_sample:
        if config.temperature is not None and config.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(config.temperature))
        if config.top_k is not None and config.top_k != 0:
            processors.append(TopKLogitsWarper(config.top_k))
        if config.top_p is not None and config.top_p < 1.0:
            processors.append(TopPLogitsWarper(config.top_p))

    cache = DynamicCache(config=predictor.config)
    token = None
    tokens = []
    pending_probabilities = []
    sampler = getattr(predictor, "_abstractvoice_sampler", "multinomial")
    if sampler not in {"multinomial", "exponential"}:
        raise ValueError("Qwen codebook sampler must be multinomial or exponential")
    for step in range(max_new_tokens):
        result = predictor(
            input_ids=token, inputs_embeds=inputs_embeds if step == 0 else None,
            past_key_values=cache, generation_steps=step, use_cache=True,
            output_hidden_states=False, output_attentions=False, return_dict=True,
        )
        cache = result.past_key_values
        scores = processors(None, result.logits[:, -1, :].float())
        if config.do_sample:
            probabilities = scores.softmax(dim=-1)
            if sampler == "exponential":
                token = _sample_exponential_unchecked(probabilities)
                pending_probabilities.append(probabilities)
            else:
                token = torch.multinomial(probabilities, num_samples=1)
        else:
            token = scores.argmax(dim=-1, keepdim=True)
        tokens.append(token)
    # A 16-codebook, batch-one frame retains only 15 * 2048 * 4 = 120 KiB.
    # One batched check avoids both per-step synchronization and many tiny
    # reduction kernels. Nothing from this frame escapes before validation.
    if pending_probabilities and not bool(_valid_probabilities(torch.stack(pending_probabilities))):
        raise RuntimeError("Qwen codebook sampling failed: probability tensor contains inf, nan, "
                           "a negative value, or a zero-probability distribution")
    return torch.cat(tokens, dim=-1)
