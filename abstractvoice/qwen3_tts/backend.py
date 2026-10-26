"""Which compute backend runs Qwen3-TTS: ``mlx`` on Apple silicon, ``torch`` everywhere else (ADR 0008).

The policy is deliberately small and framework-free (no torch, no mlx import at module level), so it
can run -- and be tested -- on any machine, including CI's Linux runners:

* ``auto`` (the default): ``mlx`` when the interpreter runs on macOS/arm64 AND ``mlx`` is importable,
  else ``torch`` (the existing PyTorch path: CUDA, MPS, CPU per ADR 0005).
* ``ABSTRACTVOICE_QWEN3_TTS_BACKEND=torch|mlx`` overrides ``auto``; asking for ``mlx`` where it cannot
  run is a plain error, never a silent fallback.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from typing import Any, Dict, Mapping, Optional

BACKEND_ENV = "ABSTRACTVOICE_QWEN3_TTS_BACKEND"
QWEN3_TTS_BACKENDS = ("auto", "torch", "mlx")

# How the MLX backend holds the talker weights: bf16 as published, or quantized on load.
MLX_QUANTIZATION_ENV = "ABSTRACTVOICE_QWEN3_TTS_MLX_QUANTIZATION"
MLX_QUANTIZATIONS = ("bf16", "8bit", "4bit")
MLX_DEFAULT_QUANTIZATION = "8bit"
_MLX_QUANTIZATION_ALIASES = {
    "": MLX_DEFAULT_QUANTIZATION,
    "auto": MLX_DEFAULT_QUANTIZATION,
    "none": "bf16",
    "bfloat16": "bf16",
    "8": "8bit",
    "int8": "8bit",
    "q8": "8bit",
    "4": "4bit",
    "int4": "4bit",
    "q4": "4bit",
}


def normalize_backend(value: Any) -> str:
    """``"auto"`` (default, also for None/empty), ``"torch"`` or ``"mlx"``; anything else raises."""
    text = str(value if value is not None else "").strip().lower() or "auto"
    if text not in QWEN3_TTS_BACKENDS:
        raise ValueError(f"Qwen3-TTS backend must be one of {', '.join(QWEN3_TTS_BACKENDS)} (got {value!r})")
    return text


def normalize_mlx_quantization(value: Any, *, env: Optional[Mapping[str, str]] = None) -> str:
    """``"bf16"`` | ``"8bit"`` | ``"4bit"``. ``auto``/None/empty read ``ABSTRACTVOICE_QWEN3_TTS_MLX_QUANTIZATION``,
    else the default (8-bit). Anything else raises."""
    text = str(value if value is not None else "").strip().lower()
    if text in ("", "auto"):
        environment = os.environ if env is None else env
        text = str(environment.get(MLX_QUANTIZATION_ENV) or "").strip().lower()
    text = _MLX_QUANTIZATION_ALIASES.get(text, text)
    if text not in MLX_QUANTIZATIONS:
        raise ValueError(f"Qwen3-TTS MLX quantization must be one of {', '.join(MLX_QUANTIZATIONS)} (got {value!r})")
    return text


def mlx_supported_platform(*, system: Optional[str] = None, machine: Optional[str] = None) -> bool:
    """MLX runs on Apple silicon macOS only."""
    system = (system if system is not None else sys.platform).lower()
    machine = (machine if machine is not None else platform.machine()).lower()
    return system.startswith("darwin") and machine in ("arm64", "aarch64")


def mlx_importable() -> bool:
    try:
        return importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx.core") is not None
    except Exception:
        return False


def mlx_unavailable_reason(*, system: Optional[str] = None, machine: Optional[str] = None, importable: Optional[bool] = None) -> Optional[str]:
    """Why the MLX backend cannot run here, or None when it can."""
    if not mlx_supported_platform(system=system, machine=machine):
        return "the MLX backend runs on Apple silicon macOS only (this is not darwin/arm64)"
    if not (mlx_importable() if importable is None else bool(importable)):
        return 'the `mlx` package is not installed (pip install "abstractvoice[qwen3-tts]" on Apple silicon installs it)'
    return None


def select_backend(
    preference: Any = None,
    *,
    env: Optional[Mapping[str, str]] = None,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    importable: Optional[bool] = None,
) -> str:
    """Resolve ``preference`` (or ``ABSTRACTVOICE_QWEN3_TTS_BACKEND``, or ``auto``) to ``"torch"`` | ``"mlx"``.

    ``env``/``system``/``machine``/``importable`` exist so the policy is testable anywhere.
    """
    environment = os.environ if env is None else env
    requested = normalize_backend(preference if preference not in (None, "", "auto") else environment.get(BACKEND_ENV))
    reason = mlx_unavailable_reason(system=system, machine=machine, importable=importable)
    if requested == "mlx":
        if reason is not None:
            raise RuntimeError(
                f"Qwen3-TTS backend 'mlx' was requested ({BACKEND_ENV} or settings) but {reason}. "
                "Use the default (auto) or 'torch'."
            )
        return "mlx"
    if requested == "torch":
        return "torch"
    return "torch" if reason is not None else "mlx"


def describe_backend_selection(preference: Any = None, *, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Diagnostic view of the policy (for ``runtime_info`` and bug reports)."""
    environment = os.environ if env is None else env
    reason = mlx_unavailable_reason()
    try:
        selected = select_backend(preference, env=environment)
        error = None
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        selected, error = None, str(e)
    return {
        "requested": normalize_backend(preference) if preference not in (None, "") else None,
        "env": environment.get(BACKEND_ENV),
        "platform": f"{sys.platform}/{platform.machine()}",
        "mlx_available": reason is None,
        "mlx_unavailable_reason": reason,
        "selected": selected,
        "error": error,
    }
