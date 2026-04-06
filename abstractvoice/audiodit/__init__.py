"""AudioDiT (LongCat-AudioDiT) model code.

This module is derived from Meituan LongCat's `LongCat-AudioDiT` repository and
is provided here to enable offline-first inference without `trust_remote_code`.

Upstream:
- Repo: https://github.com/meituan-longcat/LongCat-AudioDiT
- License: MIT
"""

from .configuration_audiodit import AudioDiTConfig, AudioDiTVaeConfig
from .modeling_audiodit import (
    AudioDiTModel,
    AudioDiTOutput,
    AudioDiTPreTrainedModel,
    AudioDiTTransformer,
    AudioDiTVae,
)

__all__ = [
    "AudioDiTConfig",
    "AudioDiTVaeConfig",
    "AudioDiTOutput",
    "AudioDiTPreTrainedModel",
    "AudioDiTModel",
    "AudioDiTTransformer",
    "AudioDiTVae",
]

