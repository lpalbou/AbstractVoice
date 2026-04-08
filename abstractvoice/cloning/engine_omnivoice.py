from __future__ import annotations

import io
from typing import Iterable, Optional

import numpy as np
import soundfile as sf

def _split_text_batches(text: str, *, max_chars: int = 240) -> list[str]:
    """Split text into short batches, preferring sentence boundaries."""
    from ..tts.text_chunking import split_text_batches

    return split_text_batches(str(text or ""), max_chars=int(max_chars))


class OmniVoiceVoiceCloningEngine:
    """In-process OmniVoice cloning engine (reference audio + transcript).

    This engine is optional and requires:
      pip install "abstractvoice[omnivoice]"
    """

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        model_id: str | None = None,
        revision: str | None = None,
        allow_downloads: bool = True,
        num_step: int = 8,
        guidance_scale: float = 2.0,
    ):
        self.debug = bool(debug)
        self._device_pref = str(device or "auto")
        self._allow_downloads = bool(allow_downloads)
        self._model_id = str(model_id) if model_id else None
        self._revision = str(revision) if revision else None

        self._num_step = int(num_step)
        self._guidance_scale = float(guidance_scale)

        self._runtime = None
        self._quality_preset = "standard"

    def runtime_info(self) -> dict:
        try:
            if self._runtime and hasattr(self._runtime, "runtime_info"):
                return dict(self._runtime.runtime_info())
        except Exception:
            pass
        return {
            "requested_device": self._device_pref,
            "quality_preset": self._quality_preset,
            "num_step": int(self._num_step),
            "guidance_scale": float(self._guidance_scale),
        }

    def set_quality_preset(self, preset: str) -> None:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._quality_preset = str(p)
        if p == "low":
            self._num_step = 8
            self._guidance_scale = 2.0
        elif p == "standard":
            self._num_step = 12
            self._guidance_scale = 2.0
        else:
            self._num_step = 24
            self._guidance_scale = 2.0

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        try:
            from ..omnivoice.runtime import OmniVoiceRuntime
        except Exception as e:
            raise RuntimeError(
                "OmniVoice cloning requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[omnivoice]\""
            ) from e

        model_id = self._model_id or OmniVoiceRuntime.DEFAULT_MODEL_ID
        self._runtime = OmniVoiceRuntime(
            model_id=model_id,
            revision=self._revision,
            device=self._device_pref,
            allow_downloads=bool(self._allow_downloads),
            debug=bool(self.debug),
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
            max_chars=240,
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
                "Missing reference_text for OmniVoice cloning.\n"
                "If you're using VoiceCloner/VoiceManager, reference_text should be auto-generated and cached.\n"
                "If you're calling this engine directly, provide reference_text or set it via the voice store."
            )

        ref_paths = list(reference_paths or [])
        if len(ref_paths) != 1:
            raise RuntimeError(
                "OmniVoice cloning currently supports exactly one reference audio file.\n"
                "Provide a single WAV/FLAC/OGG file when creating the voice clone."
            )
        ref_path = str(ref_paths[0])

        runtime = self._get_runtime()
        model = runtime.get_model()

        # Create a reusable prompt once per utterance (avoid re-encoding prompt for each chunk).
        # IMPORTANT: OmniVoice upstream uses `torchaudio.load(path)`, which can
        # require `torchcodec` depending on torchaudio build/backends. To keep
        # AbstractVoice installs lightweight, we load via soundfile and pass
        # (waveform, sample_rate) directly.
        import torch

        wav, sr = sf.read(str(ref_path), dtype="float32", always_2d=True)
        mono = np.mean(np.asarray(wav, dtype=np.float32), axis=1).astype(np.float32).reshape(-1)
        waveform = torch.from_numpy(mono).unsqueeze(0)  # (1, T)

        prompt = model.create_voice_clone_prompt(
            ref_audio=(waveform, int(sr)),
            ref_text=str(reference_text),
            preprocess_prompt=True,
        )

        lang = str(language or "en").strip()

        chunks = _split_text_batches(str(text), max_chars=int(max_chars))
        if not chunks:
            return iter(())

        for chunk_text in chunks:
            audios = model.generate(
                text=str(chunk_text),
                language=str(lang) if lang else None,
                voice_clone_prompt=prompt,
                speed=float(speed) if (speed is not None) else None,
                num_step=int(self._num_step),
                guidance_scale=float(self._guidance_scale),
                # With a fixed `voice_clone_prompt`, we prefer stable decoding.
                # Empirically, leaving OmniVoice's default sampling temperatures can produce
                # noticeable run-to-run variability for identical text (including occasional
                # "early end" / cut-off sounding outputs). Greedy selection stabilizes both
                # duration and tail behavior without changing the cloned identity anchor.
                position_temperature=0.0,
                class_temperature=0.0,
                postprocess_output=True,
            )
            if not audios:
                continue
            a0 = audios[0]
            try:
                x = a0.detach().cpu().numpy()
            except Exception:
                x = np.asarray(a0)
            mono = np.asarray(x, dtype=np.float32).reshape(-1)
            yield mono, int(runtime.get_sample_rate())

