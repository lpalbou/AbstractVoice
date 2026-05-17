"""Qwen3-ASR model code (Transformers compatible).

This module is derived from the official Qwen `qwen-asr` package and is shipped
here to enable offline-first inference without requiring `trust_remote_code`.

Upstream:
- Repo: https://github.com/QwenLM/Qwen3-ASR
- PyPI: https://pypi.org/project/qwen-asr/
- License: Apache-2.0
"""

from __future__ import annotations

from typing import Any


def register_transformers_qwen3_asr() -> None:
    """Register Qwen3-ASR config/model/processor with Transformers Auto* APIs.

    This is idempotent and safe to call multiple times.
    """

    try:
        from transformers import AutoConfig, AutoModel, AutoProcessor
    except Exception:
        return

    from .configuration_qwen3_asr import Qwen3ASRConfig
    from .processing_qwen3_asr import Qwen3ASRProcessor

    # Importing the model is heavier and may fail on partial Transformers installs
    # (e.g. config-only environments). Keep it lazy so callers can still parse
    # configs without torch-model availability.
    try:
        from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    except Exception:
        Qwen3ASRForConditionalGeneration = None  # type: ignore

    # AutoConfig.register raises ValueError if a key is already registered.
    try:
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    except Exception:
        pass
    if Qwen3ASRForConditionalGeneration is not None:
        try:
            AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        except Exception:
            pass
    try:
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except Exception:
        pass


__all__ = [
    "register_transformers_qwen3_asr",
]
