"""Adaptive visual loudness; never changes recorded audio or speech detection."""
from __future__ import annotations

import math


class VoiceLevelNormalizer:
    """MicrophoneEnvelope from the iPhone client, with elapsed-time stationarity.

    Display motion follows speech above the room floor. Sustained narrow-range
    noise becomes the baseline after 1.5 seconds, independently of raw VAD.
    """

    def __init__(self) -> None:
        self.level = 0.0
        self.noise_floor = -65.0
        self.recent_peak = -42.0
        self.stationary_low = self.stationary_high = -160.0
        self.stationary_seconds = 0.0

    def push(self, rms: float, *, peak: float | None = None,
             seconds: float = 1 / 30, speech: bool | None = None) -> float:
        target = 0.0
        if math.isfinite(rms):
            average = min(0.0, max(-160.0, 20 * math.log10(max(rms, 1e-8))))
            dt = min(0.2, max(0.0, seconds))
            if (self.stationary_seconds == 0 or average < self.stationary_high - 3
                    or average > self.stationary_low + 3):
                self.stationary_low = self.stationary_high = average
                self.stationary_seconds = dt
            else:
                self.stationary_low = min(self.stationary_low, average)
                self.stationary_high = max(self.stationary_high, average)
                self.stationary_seconds += dt
            if speech is False or self.stationary_seconds >= 1.5:
                room = min(-35.0, max(-80.0, average))
                weight = 0.2 if room < self.noise_floor else 0.12
                self.noise_floor += (room - self.noise_floor) * weight
            peak_db = 20 * math.log10(max(peak, 1e-8)) if peak is not None and math.isfinite(peak) else average
            db = min(average + 3, max(average, peak_db - 9))
            if db > self.recent_peak:
                self.recent_peak = db
            else:
                self.recent_peak += (db - self.recent_peak) * 0.016
            floor = max(-80.0, self.noise_floor + 3)
            span = max(18.0, self.recent_peak - floor)
            above_room = max(0.0, average - floor)
            normalized = max(0.0, min(1.0, min(db - floor, above_room + 3) / span))
            target = normalized ** 0.75
        self.level += (target - self.level) * (0.85 if target > self.level else 0.5)
        return self.level
