"""Qwen3-TTS (12Hz family) model code and runtime.

The modeling/configuration/processing/codec files are vendored from the
official ``qwen-tts`` package (0.1.1) to enable offline-first inference without
``trust_remote_code`` and without its pinned dependency set; each file's header
lists its local modifications exhaustively. ``runtime.py`` is AbstractVoice's
own policy layer (offline-first snapshots, ADR 0005 device/dtype, presets).

Upstream:
- Repo: https://github.com/QwenLM/Qwen3-TTS
- PyPI: https://pypi.org/project/qwen-tts/
- License: Apache-2.0

This ``__init__`` stays import-light: importing the package must not pull torch
or transformers. Import submodules explicitly (``.runtime``, ``.codec``, ...).
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MODEL_ID",
    "KNOWN_MODEL_IDS",
]

# Safe to import: runtime.py defers torch/transformers to load time.
from .runtime import DEFAULT_MODEL_ID, KNOWN_MODEL_IDS  # noqa: E402
