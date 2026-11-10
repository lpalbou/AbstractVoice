"""Package-owned, fixed-depth residual-codebook generation.

This is not a replacement for Transformers' public ``generate``. The talker
needs only codec IDs from a two-token, unpadded prefill and a fixed set of heads.
Other generation contracts retain the original path, before consuming any RNG.
"""

from __future__ import annotations

import gc
import itertools
import threading
import weakref
from collections import OrderedDict

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
# Added in later Transformers 5.x releases (absent before, None by default
# since): multi-token prediction, speculative-decoding variants and static-
# cache sizing. Each is inert only at its unset value.
_NEUTRAL.update(dict.fromkeys("""
    max_cache_len assistant_ensemble_weight speculation_type
""".split(), (None,)))
_NEUTRAL.update(use_mtp=(None, False))
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


def _frame(predictor, inputs_embeds, count, processors, do_sample, sampler):
    """One fixed-depth frame: ``count`` residual IDs from a two-embedding prefill.

    Returns the IDs, the probabilities still to validate (exponential race
    only) and the raw last-position logits of every step. The eager path and
    the CUDA graph capture run this same function, so both issue the same
    kernels in the same order.
    """
    cache = DynamicCache(config=predictor.config)
    token = None
    tokens = []
    pending_probabilities = []
    step_logits = []
    for step in range(count):
        result = predictor(
            input_ids=token, inputs_embeds=inputs_embeds if step == 0 else None,
            past_key_values=cache, generation_steps=step, use_cache=True,
            output_hidden_states=False, output_attentions=False, return_dict=True,
        )
        cache = result.past_key_values
        logits = result.logits[:, -1, :]
        step_logits.append(logits)
        scores = processors(None, logits.float())
        if do_sample:
            probabilities = scores.softmax(dim=-1)
            if sampler == "exponential":
                token = _sample_exponential_unchecked(probabilities)
                pending_probabilities.append(probabilities)
            else:
                token = torch.multinomial(probabilities, num_samples=1)
        else:
            token = scores.argmax(dim=-1, keepdim=True)
        tokens.append(token)
    return torch.cat(tokens, dim=-1), pending_probabilities, step_logits


def _raise_invalid_probabilities():
    raise RuntimeError("Qwen codebook sampling failed: probability tensor contains inf, nan, "
                       "a negative value, or a zero-probability distribution")


# ------------------------------------------------------------------ CUDA graphs
#
# On CUDA the frame is launch-bound: 15 steps of a 5-layer model are hundreds
# of tiny kernels, and the GPU idles while Python issues them (0.6B, float32,
# Quadro RTX 5000: 89 ms per frame eager, 25 ms replayed). Every frame has
# the same shapes (a prefill of two embeddings, then one token at KV lengths 3
# to 16), so the whole frame is captured once per shape and sampling setting
# and replayed. A replay runs the captured kernels of the eager loop on the
# same weights: the same arithmetic, so the same logits, IDs and RNG offsets
# (CUDA graphs advance the Philox generator exactly as eager launches do).
# The draw inside the graph is the exponential race, which is torch's own
# n=1 multinomial construction minus its two host synchronizations per step;
# the probabilities are validated once per frame, before any ID escapes, as
# for the opt-in exponential sampler.

_GRAPHS = weakref.WeakKeyDictionary()  # predictor -> _Graphs
_GRAPH_FAILURES = weakref.WeakKeyDictionary()  # predictor -> reason the capture failed
_GRAPHS_PER_PREDICTOR = 4


def _weights_fingerprint(predictor):
    # A replay reads the weights at their captured addresses: a moved or
    # re-cast model must never replay an old graph.
    return tuple(t.data_ptr() for t in itertools.chain(predictor.parameters(), predictor.buffers()))


def _has_hooks(predictor):
    # Hooks are Python: a replay would skip them, so an observed model (forward
    # hooks, or an Accelerate offload hook that moves weights) stays eager.
    from torch.nn.modules import module as nn_module

    if any(getattr(nn_module, name, None) for name in ("_global_forward_hooks", "_global_forward_pre_hooks")):
        return True
    return any(m._forward_hooks or m._forward_pre_hooks or hasattr(m, "_hf_hook") for m in predictor.modules())


class _Graphs(OrderedDict):
    """A model's captured frames (key -> _CodebookGraph), all for one weights fingerprint."""

    fingerprint = None


class _CodebookGraph:
    """One captured frame for one input shape and sampling setting."""

    def __init__(self, predictor, inputs_embeds, count, processors, do_sample):
        device = inputs_embeds.device
        self.lock = threading.Lock()  # one replay at a time: the input and outputs are static
        with torch.inference_mode(False):  # writable by later calls in any grad mode
            self.embeds = inputs_embeds.detach().clone()  # the static input every replay reads
        rng = torch.cuda.get_rng_state(device)
        stream = torch.cuda.Stream(device=device)
        try:
            # One eager pass on the capture stream initializes lazy library
            # state outside the capture; its random draws are rolled back.
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                _frame(predictor, self.embeds, count, processors, do_sample, "exponential")
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.set_rng_state(rng, device)
            self.graph = torch.cuda.CUDAGraph()
            # No cyclic garbage collection while capturing: collecting another
            # model's graphs frees device memory, which invalidates a capture.
            # (torch.cuda.graph collects once before the capture begins.)
            collecting = gc.isenabled()
            gc.disable()
            try:
                with torch.cuda.graph(self.graph, stream=stream, capture_error_mode="thread_local"):
                    self.ids, pending, self.logits = _frame(
                        predictor, self.embeds, count, processors, do_sample, "exponential",
                    )
                    self.valid = _valid_probabilities(torch.stack(pending)) if pending else None
            finally:
                if collecting:
                    gc.enable()
        finally:
            # A capture launches nothing; this undoes the warm-up's draws (and
            # anything a failed capture might have consumed).
            torch.cuda.set_rng_state(rng, device)

    def __call__(self, inputs_embeds):
        with self.lock:
            self.embeds.copy_(inputs_embeds)
            self.graph.replay()
            if self.valid is not None and not bool(self.valid):
                _raise_invalid_probabilities()
            return self.ids.clone()  # the next replay overwrites the static output


def _sampling_key(config):
    if not config.do_sample:
        return (False,)
    return (True, config.temperature, config.top_k, config.top_p)


def _graphed_frame(predictor, inputs_embeds, count, processors, config):
    """The frame's IDs from a CUDA graph replay, or None to run the eager loop.

    None is returned before anything runs (RNG untouched): when graphs are off
    for this model, when something observes it through hooks, inside another
    capture, or after a capture failed once for this model (logged once).
    """
    if inputs_embeds.device.type != "cuda" or not getattr(predictor, "_abstractvoice_cuda_graphs", True):
        return None
    if predictor in _GRAPH_FAILURES or torch.cuda.is_current_stream_capturing() or _has_hooks(predictor):
        return None
    graphs = _GRAPHS.get(predictor)
    if graphs is None:
        graphs = _GRAPHS[predictor] = _Graphs()
    fingerprint = _weights_fingerprint(predictor)
    if graphs.fingerprint != fingerprint:
        graphs.clear()  # the weights moved (or this is the first frame): every graph is stale
        graphs.fingerprint = fingerprint
    key = (tuple(inputs_embeds.shape), inputs_embeds.dtype, inputs_embeds.device, count,
           predictor.config._attn_implementation, _sampling_key(config))
    graph = graphs.get(key)
    if graph is None:
        try:
            graph = _CodebookGraph(predictor, inputs_embeds, count, processors, config.do_sample)
        except Exception as error:  # capture is an optimization: fall back, never fail the request
            _GRAPH_FAILURES[predictor] = f"{type(error).__name__}: {error}"
            logger.warning_once("#FALLBACK Qwen codebook CUDA graph unavailable, using the eager loop: %s",
                                _GRAPH_FAILURES[predictor])
            return None
        graphs[key] = graph
        while len(graphs) > _GRAPHS_PER_PREDICTOR:
            graphs.popitem(last=False)
    graphs.move_to_end(key)
    return graph(inputs_embeds)


def cuda_graph_state(predictor):
    """``"captured"``, ``"failed: <reason>"`` or ``"unused"``: whether this model replays CUDA graphs."""
    if predictor in _GRAPH_FAILURES:
        return f"failed: {_GRAPH_FAILURES[predictor]}"
    return "captured" if _GRAPHS.get(predictor) else "unused"


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
    On CUDA the frame replays a captured CUDA graph of this same loop (see
    above); both samplers then draw by the exponential race and validate per
    frame, with the IDs and RNG consumption of the eager loop.
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

    sampler = getattr(predictor, "_abstractvoice_sampler", "multinomial")
    if sampler not in {"multinomial", "exponential"}:
        raise ValueError("Qwen codebook sampler must be multinomial or exponential")
    ids = _graphed_frame(predictor, inputs_embeds, max_new_tokens, processors, config)
    if ids is not None:
        return ids
    ids, pending_probabilities, _ = _frame(
        predictor, inputs_embeds, max_new_tokens, processors, config.do_sample, sampler,
    )
    # A 16-codebook, batch-one frame retains only 15 * 2048 * 4 = 120 KiB.
    # One batched check avoids both per-step synchronization and many tiny
    # reduction kernels. Nothing from this frame escapes before validation.
    if pending_probabilities and not bool(_valid_probabilities(torch.stack(pending_probabilities))):
        _raise_invalid_probabilities()
    return ids
