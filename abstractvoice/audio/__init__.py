"""Audio utilities (small, dependency-light)."""

from .resample import linear_resample_mono

__all__ = ["linear_resample_mono", "record_wav"]


def __getattr__(name: str):
    if name == "record_wav":
        from .recorder import record_wav

        return record_wav
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
