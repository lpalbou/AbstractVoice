"""Quality preset normalization (engine-agnostic).

AbstractVoice exposes a small set of human-friendly quality presets across
engines. Engines map these presets onto their own parameters (e.g. diffusion
steps, sampling knobs).

Canonical presets:
- low
- standard
- high

Backward-compatible aliases:
- fast     -> low
- balanced -> standard
"""

from __future__ import annotations

from typing import Literal


QualityPreset = Literal["low", "standard", "high"]


_ALIASES: dict[str, QualityPreset] = {
    # Canonical
    "low": "low",
    "standard": "standard",
    "high": "high",
    # Back-compat
    "fast": "low",
    "balanced": "standard",
    # Convenience
    "std": "standard",
    "normal": "standard",
    "default": "standard",
    "medium": "standard",
    "med": "standard",
}


def normalize_quality_preset(preset: str) -> QualityPreset:
    p = str(preset or "").strip().lower()
    p2 = _ALIASES.get(p)
    if p2 is None:
        raise ValueError("preset must be one of: low|standard|high")
    return p2

