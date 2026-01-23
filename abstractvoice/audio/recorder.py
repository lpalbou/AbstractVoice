from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf


def record_wav(
    output_path: str | Path,
    *,
    seconds: float = 6.0,
    sample_rate: int = 24000,
    channels: int = 1,
) -> str:
    """Record microphone audio to a WAV file (best-effort).

    This is intended for interactive REPL usage (e.g., `/clone-my-voice`).
    """
    try:
        import sounddevice as sd
    except Exception as e:
        raise ImportError(
            "Microphone recording requires sounddevice.\n"
            "Install with: pip install abstractvoice\n"
            f"Original error: {e}"
        ) from e

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames = int(sample_rate * float(seconds))
    if frames <= 0:
        raise ValueError("seconds must be > 0")

    audio = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()

    # downmix to mono if needed
    if channels > 1:
        audio = np.mean(audio, axis=1, keepdims=True)

    sf.write(str(out), audio, sample_rate, subtype="PCM_16")
    return str(out)

