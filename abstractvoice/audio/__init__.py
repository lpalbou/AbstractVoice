"""Audio utilities (small, dependency-light)."""

from .resample import linear_resample_mono
from .recorder import record_wav

__all__ = ["linear_resample_mono", "record_wav"]

