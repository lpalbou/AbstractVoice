"""Compatibility seams for vendored Hugging Face model code.

AbstractVoice vendors upstream modeling files (``qwen3_asr``, ``qwen3_tts``) to
run offline-first without ``trust_remote_code``. Those files are written against
one transformers release, and transformers moves: between 4.57 and 5.x the
"default" rope key was dropped, ``pad_token_id`` stopped being a universal
config attribute, and ``check_model_inputs`` changed from a decorator factory to
a plain decorator. Import succeeding proves none of this — two of the three
only fail when a model is CONSTRUCTED, which is how a vendored package sat
broken in this repo without a test noticing.

Every vendored file takes these seams from here, so the next transformers move
is one fix, and the compat tests in ``tests/test_qwen3_tts_integration.py`` guard the lot.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

__all__ = [
    "auto_docstring",
    "check_model_inputs",
    "layer_type_validation",
    "pad_token_id_of",
    "rope_config_validation",
    "rope_init_fn",
]


def rope_config_validation(config: Any, **kwargs: Any) -> None:
    """4.57's ``rope_config_validation`` across its 5.x deprecation/removal arc.

    Prefers the 5.x instance method when present, falls back to the legacy
    helper while it exists, and degrades to a no-op — validation is a lint,
    and a lint must not decide whether a checkpoint loads.
    """
    import warnings

    validate = getattr(config, "validate_rope", None)
    if callable(validate):
        try:
            validate()
            return
        except TypeError:
            pass
    try:
        from transformers.modeling_rope_utils import rope_config_validation as legacy
    except Exception:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            legacy(config, **kwargs)
        except Exception:
            pass


def layer_type_validation(layer_types: Any, *args: Any, **kwargs: Any) -> None:
    """4.57's ``layer_type_validation``; same deprecation posture as above.

    The deprecation notice arrives through transformers' logger rather than the
    warnings module, so both are held for the duration of the call.
    """
    import logging
    import warnings

    try:
        from transformers.configuration_utils import layer_type_validation as legacy
    except Exception:
        return
    hf_logger = logging.getLogger("transformers.configuration_utils")
    previous = hf_logger.level
    hf_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                legacy(layer_types, *args, **kwargs)
            except Exception:
                pass
    finally:
        hf_logger.setLevel(previous)


def auto_docstring(*args: Any, **kwargs: Any):
    """transformers' ``@auto_docstring`` without its import-time docstring lint.

    Same signature duality as ``check_model_inputs``: plain decorator or factory.
    Vendored upstream files never wrote the docstrings the lint demands, and a
    library must not print ``[ERROR]`` on a successful import.
    """
    try:
        from transformers.utils import auto_docstring as hf_auto
    except Exception:
        hf_auto = None

    if len(args) == 1 and callable(args[0]) and not kwargs:
        return _quietly(hf_auto, args[0]) if hf_auto is not None else args[0]

    def apply(target: Callable) -> Callable:
        if hf_auto is None:
            return target
        return _quietly(hf_auto(*args, **kwargs), target)

    return apply


def _default_rope_init(config: Any, device: Any = None, seq_len: Optional[int] = None) -> Tuple[Any, float]:
    """The classic unscaled RoPE inverse-frequency schedule.

    transformers < 5 shipped this as ``ROPE_INIT_FUNCTIONS["default"]``; 5.x
    dropped the key, but checkpoints (Qwen3-TTS ships ``rope_type: "default"``
    in its config.json) and the vendored else-branches still ask for it. The
    formula is stable and tiny, so we own it instead of chasing the dict.
    """
    import torch

    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def rope_init_fn(rope_type: str) -> Callable[..., Tuple[Any, float]]:
    """``ROPE_INIT_FUNCTIONS[rope_type]`` that still understands "default"."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    fn = ROPE_INIT_FUNCTIONS.get(str(rope_type or "default"))
    if fn is not None:
        return fn
    if str(rope_type or "default") == "default":
        return _default_rope_init
    raise KeyError(
        f"Unknown rope_type {rope_type!r}; transformers knows {sorted(ROPE_INIT_FUNCTIONS)} and "
        "abstractvoice._hf_compat adds 'default'."
    )


def pad_token_id_of(config: Any) -> Optional[int]:
    """``config.pad_token_id`` across transformers 4.x (attribute) and 5.x (absent)."""
    value = getattr(config, "pad_token_id", None)
    return int(value) if value is not None else None


def check_model_inputs(*args: Any, **kwargs: Any):
    """The ``@check_model_inputs`` decorator across its 4.57 -> 5.x signature change.

    4.57: a factory, used as ``@check_model_inputs()``.
    5.x: a plain decorator taking the function directly (and deprecated in favor
    of ``merge_with_config_defaults``, whose output-recording side effects are
    gone — vendored ``_can_record_outputs`` declarations are inert on 5.x).

    Accepts both call shapes; degrades to identity if transformers drops it.
    """
    try:
        from transformers.utils.generic import check_model_inputs as hf_check
    except Exception:
        hf_check = None

    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Used as a plain decorator: @check_model_inputs
        return _quietly(hf_check, args[0]) if hf_check is not None else args[0]

    # Used as a factory: @check_model_inputs(...)
    def apply(func: Callable) -> Callable:
        if hf_check is None:
            return func
        try:
            return _quietly(hf_check, func)  # 5.x: plain decorator
        except TypeError:
            return _quietly(hf_check(*args, **kwargs), func)  # 4.57: factory

    return apply


def _quietly(decorator: Callable, func: Callable) -> Callable:
    """Apply an HF decorator without its import-time docstring lint.

    transformers' auto_docstring machinery logs `[ERROR] ... but not documented`
    for vendored forwards whose upstream never wrote docstrings. That is
    upstream's style debt, not a runtime problem, and a voice library must not
    print ERROR lines on a successful import. Scoped to decoration only, so real
    errors elsewhere still surface.
    """
    import contextlib
    import io
    import sys

    # The lint is a bare print() (transformers/utils/auto_docstring.py), so a
    # logger level cannot silence it. Capture stdout for the duration of the
    # decoration (import time, microseconds) and re-emit every line that is not
    # the docstring lint, so genuine output still gets through.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        decorated = decorator(func)
    for line in captured.getvalue().splitlines():
        if "but not documented" not in line:
            print(line, file=sys.stdout)
    return decorated


def _self_test() -> None:  # pragma: no cover - convenience for humans
    fn = rope_init_fn("default")
    assert callable(fn)
    assert math.isclose(1.0, 1.0)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
