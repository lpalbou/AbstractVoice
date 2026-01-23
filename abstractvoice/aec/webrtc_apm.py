from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AecConfig:
    sample_rate: int = 16000
    channels: int = 1
    stream_delay_ms: int = 0
    enable_ns: bool = True
    enable_agc: bool = False


class WebRtcAecProcessor:
    """Thin wrapper around `aec-audio-processing` (WebRTC APM).

    Design goals:
    - Optional dependency (import only when enabled)
    - Byte-oriented processing (PCM16) to integrate with our VAD/STT pipeline
    """

    def __init__(self, cfg: AecConfig):
        self.cfg = cfg

        try:
            from aec_audio_processing import AudioProcessor  # type: ignore
        except Exception as e:
            raise ImportError(
                "AEC is optional and requires extra dependencies.\n"
                "Install with: pip install \"abstractvoice[aec]\"\n"
                f"Original error: {e}"
            ) from e

        ap = AudioProcessor(
            enable_aec=True,
            enable_ns=bool(cfg.enable_ns),
            enable_agc=bool(cfg.enable_agc),
        )
        ap.set_stream_format(int(cfg.sample_rate), int(cfg.channels))
        ap.set_reverse_stream_format(int(cfg.sample_rate), int(cfg.channels))
        try:
            ap.set_stream_delay(int(cfg.stream_delay_ms))
        except Exception:
            # Best-effort: some builds may not expose delay control.
            pass

        self._ap = ap

    def process(self, *, near_pcm16: bytes, far_pcm16: bytes) -> bytes:
        """Process one chunk: feed far-end then clean near-end."""
        # The WebRTC APM expects reverse stream first.
        if far_pcm16:
            self._ap.process_reverse_stream(far_pcm16)
        return self._ap.process_stream(near_pcm16)

