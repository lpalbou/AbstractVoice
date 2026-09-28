"""Transformers version floors for the vendored model code, checked before loading.

The vendored Qwen3-ASR and Qwen3-TTS models run on Transformers 5 only; on an
older release they fail deep inside generation (for example
``create_causal_mask() got an unexpected keyword argument 'inputs_embeds'``).
Their loaders call :func:`require_transformers` first, so the user gets one plain
sentence before any download or weight load instead.

The floors equal the install extras' floors in ``pyproject.toml`` (``stt-hf``
and ``qwen3-tts``); ``tests/test_transformers_floor.py`` fails when they drift.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "QWEN3_ASR_MIN_TRANSFORMERS",
    "QWEN3_TTS_MIN_TRANSFORMERS",
    "UnsupportedTransformersVersion",
    "require_transformers",
]

QWEN3_ASR_MIN_TRANSFORMERS = "5.4.0"  # the stt-hf extra's floor (Python 3.10+)
QWEN3_TTS_MIN_TRANSFORMERS = "5.9.0"  # the qwen3-tts extra's floor (Python 3.10+)


class UnsupportedTransformersVersion(RuntimeError):
    """The installed Transformers release cannot run the requested model.

    Raised before any weights load. Downloading a model would not fix it, so
    callers must never rewrite it into a "model not downloaded" message.
    """


def require_transformers(
    minimum: str,
    *,
    model: str,
    extra: str,
    installed: Optional[str] = None,
    hint: str = "",
) -> None:
    """Raise :class:`UnsupportedTransformersVersion` when Transformers < ``minimum``.

    ``installed`` defaults to the imported ``transformers.__version__``.
    """
    from packaging.version import Version

    if installed is None:
        import transformers

        installed = str(getattr(transformers, "__version__", "0"))
    # base_version: a 5.9.0 pre-release or dev build counts as 5.9.0.
    if Version(Version(str(installed)).base_version) >= Version(minimum):
        return
    raise UnsupportedTransformersVersion(
        f"{model} needs Transformers {minimum} or newer (Python 3.10+); "
        f"transformers {installed} is installed.\n"
        "Upgrade with:\n"
        f'  pip install -U "abstractvoice[{extra}]"   (or: pip install -U "transformers>={minimum}")\n'
        "On Python 3.9, Transformers 5 is not available: use Python 3.10 or newer"
        + (f", {hint}" if hint else "")
        + "."
    )
