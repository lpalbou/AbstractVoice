from __future__ import annotations

import io
from typing import Iterable, Optional

import numpy as np
import soundfile as sf


class AudioDiTVoiceCloningEngine:
    """In-process AudioDiT voice cloning engine (prompt audio + prompt text).

    This engine is optional and requires:
      pip install "abstractvoice[audiodit]"
    """

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        model_id: str | None = None,
        revision: str | None = None,
        allow_downloads: bool = True,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "apg",
    ):
        self.debug = bool(debug)
        self._device_pref = str(device or "auto")
        self._allow_downloads = bool(allow_downloads)
        self._model_id = str(model_id) if model_id else None
        self._revision = str(revision) if revision else None

        self._steps = int(steps)
        self._cfg_strength = float(cfg_strength)
        self._guidance_method = str(guidance_method or "apg").strip().lower()

        self._runtime = None
        self._quality_preset = "standard"

    def runtime_info(self) -> dict:
        try:
            if self._runtime and hasattr(self._runtime, "runtime_info"):
                return dict(self._runtime.runtime_info())
        except Exception:
            pass
        return {"requested_device": self._device_pref, "quality_preset": self._quality_preset}

    def set_quality_preset(self, preset: str) -> None:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._quality_preset = str(p)
        if p == "low":
            self._steps = 8
            self._cfg_strength = 3.5
        elif p == "standard":
            self._steps = 16
            self._cfg_strength = 4.0
        else:
            self._steps = 24
            self._cfg_strength = 4.5

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        try:
            from ..audiodit.runtime import AudioDiTRuntime, AudioDiTSettings
        except Exception as e:
            raise RuntimeError(
                "AudioDiT cloning requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[audiodit]\""
            ) from e

        model_id = self._model_id or AudioDiTRuntime.DEFAULT_MODEL_ID
        self._runtime = AudioDiTRuntime(
            model_id=model_id,
            revision=self._revision,
            device=self._device_pref,
            allow_downloads=bool(self._allow_downloads),
            debug=bool(self.debug),
        )
        self._settings = AudioDiTSettings(
            steps=int(self._steps),
            cfg_strength=float(self._cfg_strength),
            guidance_method=str(self._guidance_method),
            seed=1024,
        )
        return self._runtime

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        language: str | None = None,
    ) -> bytes:
        chunks = []
        sr_out = 24000
        for chunk, sr in self.infer_to_audio_chunks(
            text=text,
            reference_paths=reference_paths,
            reference_text=reference_text,
            speed=speed,
            language=language,
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr_out = int(sr)
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def infer_to_audio_chunks(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        max_chars: int = 240,
        language: str | None = None,
    ):
        if not reference_text or not str(reference_text).strip():
            raise RuntimeError(
                "Missing reference_text for AudioDiT cloning.\n"
                "If you're using VoiceCloner/VoiceManager, reference_text should be auto-generated and cached.\n"
                "If you're calling this engine directly, provide reference_text or set it via the voice store."
            )
        runtime = self._get_runtime()

        # Use the active VoiceManager language when available; fall back to English.
        lang = str(language or "en").strip().lower() or "en"

        chunks, sr = runtime.generate_chunks(
            text=str(text),
            language=lang,
            prompt_audio_paths=list(reference_paths),
            prompt_text=str(reference_text),
            settings=getattr(self, "_settings", None),
            max_chars=int(max_chars),
        )

        for wav in chunks:
            mono = np.asarray(wav, dtype=np.float32).reshape(-1)
            yield mono, int(sr)

