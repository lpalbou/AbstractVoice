"""Compatibility wrapper for `webrtcvad` without `pkg_resources`.

Why this exists
--------------
The upstream `webrtcvad` Python shim imports `pkg_resources` to get its version:

    import pkg_resources
    __version__ = pkg_resources.get_distribution("webrtcvad").version

Recent setuptools versions can remove `pkg_resources`, which breaks importing
`webrtcvad` even though the actual VAD implementation (`_webrtcvad` extension)
is present and functional.

This module provides a minimal drop-in surface (`Vad`, `valid_rate_and_frame_length`)
implemented directly on top of `_webrtcvad` and uses `importlib.metadata` for
version discovery (best-effort).
"""

from __future__ import annotations

from importlib import metadata as _metadata


try:
    import _webrtcvad  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing _webrtcvad extension; install webrtcvad-wheels "
        "(pip install \"abstractvoice[audio-io]\")"
    ) from e


# Distributions that ship the `_webrtcvad` extension. `webrtcvad-wheels` is the
# maintained fork with prebuilt wheels (no C compiler needed) and is what our
# extras install; the original `webrtcvad` sdist is still accepted.
VAD_DISTRIBUTIONS = ("webrtcvad-wheels", "webrtcvad")


def _resolve_version() -> str:
    for dist in VAD_DISTRIBUTIONS:
        try:
            return _metadata.version(dist)
        except Exception:
            continue
    return "unknown"


__version__ = _resolve_version()


class Vad:
    def __init__(self, mode: int | None = None):
        self._vad = _webrtcvad.create()
        _webrtcvad.init(self._vad)
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode: int) -> None:
        _webrtcvad.set_mode(self._vad, int(mode))

    def is_speech(self, buf: bytes, sample_rate: int, length: int | None = None) -> bool:
        ln = int(length) if length is not None else int(len(buf) / 2)
        if ln * 2 > len(buf):
            raise IndexError(
                f"buffer has {int(len(buf) / 2.0)} frames, but length argument was {ln}"
            )
        return bool(_webrtcvad.process(self._vad, int(sample_rate), buf, ln))


def valid_rate_and_frame_length(rate: int, frame_length: int) -> bool:
    return bool(_webrtcvad.valid_rate_and_frame_length(int(rate), int(frame_length)))

