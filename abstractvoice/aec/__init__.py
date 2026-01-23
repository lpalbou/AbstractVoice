"""Optional acoustic echo cancellation (AEC) support.

This package is intentionally behind an optional extra:
  pip install "abstractvoice[aec]"
"""

from .webrtc_apm import WebRtcAecProcessor

__all__ = ["WebRtcAecProcessor"]

