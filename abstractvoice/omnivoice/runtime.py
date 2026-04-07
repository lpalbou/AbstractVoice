"""OmniVoice runtime wrapper.

OmniVoice is an *optional* torch/transformers engine (Apache-2.0) providing:
- omnilingual TTS (600+ languages upstream)
- voice cloning (reference audio + transcript)
- voice design (speaker attributes via `instruct`)

This wrapper enforces AbstractVoice's core policies:
- offline-first when `allow_downloads=False` (no surprise network calls)
- lazy imports to keep base installs lightweight
- consistent device/dtype selection (see ADR 0005)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..compute import best_torch_device


@dataclass
class OmniVoiceSettings:
    """Engine settings that map to OmniVoice generation parameters.

    Keep this small and stable; higher-level adapters can expose engine-agnostic
    presets (low|standard|high) and map them onto these fields.
    """

    # Decoding / guidance
    # NOTE: OmniVoice can be extremely slow on MPS in practice because upstream
    # loads the audio tokenizer on CPU when the model device is MPS (see OmniVoice
    # `from_pretrained`), which can introduce heavy cross-device overhead.
    #
    # We keep a moderate default here; adapters expose higher-level quality presets.
    num_step: int = 8
    guidance_scale: float = 2.0
    t_shift: float = 0.1

    # Sampling
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    layer_penalty_factor: float = 5.0
    # Optional RNG seed for reproducible voice design / sampling.
    # Note: OmniVoice does not accept a `seed=` kwarg; we apply it by seeding RNGs
    # inside `generate_audio()` (best-effort, accelerator-dependent).
    seed: int | None = None

    # Pre/post processing
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True

    # Long-form generation
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    def to_generate_kwargs(self) -> dict[str, Any]:
        return {
            "num_step": int(self.num_step),
            "guidance_scale": float(self.guidance_scale),
            "t_shift": float(self.t_shift),
            "position_temperature": float(self.position_temperature),
            "class_temperature": float(self.class_temperature),
            "layer_penalty_factor": float(self.layer_penalty_factor),
            "denoise": bool(self.denoise),
            "preprocess_prompt": bool(self.preprocess_prompt),
            "postprocess_output": bool(self.postprocess_output),
            "audio_chunk_duration": float(self.audio_chunk_duration),
            "audio_chunk_threshold": float(self.audio_chunk_threshold),
        }


class OmniVoiceRuntime:
    """Lazy-loading wrapper around `omnivoice.OmniVoice`."""

    DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        dtype_name: str | None = None,
        allow_downloads: bool = True,
        debug: bool = False,
    ):
        self.model_id = str(model_id or self.DEFAULT_MODEL_ID)
        self.revision = str(revision) if revision else None
        self._device_pref = str(device or "auto").strip().lower() or "auto"
        self._dtype_pref = str(dtype_name).strip().lower() if dtype_name else None
        self.allow_downloads = bool(allow_downloads)
        self.debug = bool(debug)

        self._resolved_device: str | None = None
        self._resolved_dtype: str | None = None

        self._model = None

    def runtime_info(self) -> dict[str, Any]:
        return {
            "engine_id": "omnivoice",
            "model_id": str(self.model_id),
            "revision": str(self.revision) if self.revision else None,
            "requested_device": str(self._device_pref),
            "resolved_device": self._resolved_device,
            "resolved_dtype": self._resolved_dtype,
            "allow_downloads": bool(self.allow_downloads),
        }

    def _resolve_device_base(self) -> str:
        pref = str(self._device_pref or "auto").strip().lower() or "auto"
        if pref == "auto":
            # Auto means: choose the best available torch device.
            # On Apple Silicon this is typically MPS (Metal).
            return str(best_torch_device()).strip().lower() or "cpu"
        return pref

    def _resolve_device_map(self) -> str:
        # OmniVoice upstream examples use "cuda:0". Keep that convention.
        dev = self._resolve_device_base()
        if dev == "cuda":
            return "cuda:0"
        return dev

    def _ensure_local_snapshot(self) -> str:
        """Resolve a local folder path for the HF snapshot (offline-first)."""
        # Allow loading from an existing local checkpoint directory (e.g. a fine-tuned
        # OmniVoice checkpoint). This avoids forcing everything through HF Hub and
        # keeps offline-first semantics.
        try:
            from pathlib import Path

            p = Path(str(self.model_id)).expanduser()
            if p.exists() and p.is_dir():
                return str(p)
        except Exception:
            pass

        local_only = not bool(self.allow_downloads)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for OmniVoice model management") from e

        # Avoid HF hub token warnings when the caller explicitly disabled downloads.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"^Warning: You are sending unauthenticated requests to the HF Hub\\..*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"^You are sending unauthenticated requests to the HF Hub\\..*",
            )
            try:
                return str(
                    snapshot_download(
                        repo_id=str(self.model_id),
                        revision=str(self.revision) if self.revision else None,
                        local_files_only=bool(local_only),
                    )
                )
            except Exception as e:
                if local_only:
                    raise RuntimeError(
                        "OmniVoice weights are not available locally and downloads are disabled.\n"
                        "Fix options:\n"
                        "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                        "  - Or prefetch explicitly: abstractvoice-prefetch --omnivoice\n"
                        f"Model: {self.model_id}"
                    ) from e
                raise

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OmniVoice requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[omnivoice]\""
            ) from e

        # Disable transformers progress bars (e.g. "Loading weights") for cleaner REPL UX.
        # Env vars alone don't reliably work once these libs are already imported.
        try:
            from transformers.utils import logging as _tf_logging

            _tf_logging.disable_progress_bar()
        except Exception:
            pass

        try:
            from omnivoice import OmniVoice
        except Exception as e:
            msg = (
                "OmniVoice failed to import.\n"
                "This usually means a missing/broken optional dependency.\n"
                "\n"
                "Install:\n"
                "  pip install \"abstractvoice[omnivoice]\"\n"
                "\n"
                f"Import error: {e}\n"
            )

            # Common pitfall: installing OmniVoice can downgrade torch/torchaudio
            # (OmniVoice pins torch==2.8.*). If an incompatible torchvision remains
            # installed, Transformers imports can fail with errors like:\n"
            #   RuntimeError: operator torchvision::nms does not exist
            # Fix: install a matching torchvision for torch 2.8.* (0.23.*).
            lowered = str(e).lower()
            if "torchvision" in lowered or "nms does not exist" in lowered or "operator torchvision::" in lowered:
                msg += (
                    "\n"
                    "Detected a torchvision/torch mismatch.\n"
                    "Fix (recommended):\n"
                    "  python -m pip install --upgrade --force-reinstall \"torchvision==0.23.*\"\n"
                    "\n"
                    "Alternative (if you don't need torchvision):\n"
                    "  python -m pip uninstall torchvision\n"
                )
            raise RuntimeError(msg) from e

        device_base = self._resolve_device_base()
        device_map = self._resolve_device_map()
        self._resolved_device = str(device_map)

        # Resolve dtype consistently with AbstractVoice policy.
        try:
            from ..compute import resolve_torch_dtype

            dt = resolve_torch_dtype(device=str(device_base), dtype_name=self._dtype_pref)
            self._resolved_dtype = str(dt).replace("torch.", "")
        except Exception:
            dt = None
            self._resolved_dtype = None

        local_dir = self._ensure_local_snapshot()

        # Load from the local snapshot directory to prevent OmniVoice's own
        # internal `snapshot_download()` call (offline-first).
        try:
            kwargs: dict[str, Any] = {
                "device_map": str(device_map),
                "train": False,
                "load_asr": False,
            }
            if dt is not None:
                # Transformers expects `torch_dtype`; OmniVoice README uses `dtype`,
                # but passing torch_dtype is the most compatible choice.
                kwargs["torch_dtype"] = dt

            model = OmniVoice.from_pretrained(str(local_dir), **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load OmniVoice model: {e}") from e

        try:
            model.eval()
        except Exception:
            pass

        self._model = model

    def get_model(self):
        self._ensure_loaded()
        assert self._model is not None
        return self._model

    def get_sample_rate(self) -> int:
        try:
            m = self.get_model()
            sr = int(getattr(m, "sampling_rate", 24000) or 24000)
            return sr
        except Exception:
            return 24000

    def generate_audio(
        self,
        *,
        text: str,
        language: str | None,
        instruct: str | None,
        voice_clone_prompt: Any | None = None,
        duration: float | None,
        speed: float | None,
        settings: OmniVoiceSettings,
    ) -> tuple[np.ndarray, int]:
        """Generate a mono float32 waveform (voice design or clone prompt)."""
        model = self.get_model()
        kwargs = settings.to_generate_kwargs()
        if duration is not None:
            kwargs["duration"] = float(duration)
        if speed is not None:
            kwargs["speed"] = float(speed)
        if language is not None:
            kwargs["language"] = str(language)
        if instruct is not None:
            kwargs["instruct"] = str(instruct)
        if voice_clone_prompt is not None:
            kwargs["voice_clone_prompt"] = voice_clone_prompt

        seed = getattr(settings, "seed", None)
        seed_i = None
        if seed is not None:
            try:
                seed_i = int(seed)
            except Exception:
                seed_i = None

        # OmniVoice voice design/sampling uses torch RNG (e.g., gumbel sampling).
        # If a seed is provided, isolate RNG changes as best-effort so callers
        # outside AbstractVoice aren't surprised by global RNG mutation.
        if seed_i is None:
            audios = model.generate(text=str(text), **kwargs)
        else:
            py_state = None
            np_state = None
            torch_state = None
            cuda_states = None
            try:
                import random as _random
                import torch

                py_state = _random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                try:
                    if torch.cuda.is_available():
                        cuda_states = torch.cuda.get_rng_state_all()
                except Exception:
                    cuda_states = None

                _random.seed(int(seed_i))
                np.random.seed(int(seed_i))
                torch.manual_seed(int(seed_i))
                try:
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed_i))
                except Exception:
                    pass

                audios = model.generate(text=str(text), **kwargs)
            except Exception:
                # Best-effort: if anything about seeding fails, still try to synthesize.
                audios = model.generate(text=str(text), **kwargs)
            finally:
                # Best-effort: restore RNG state so callers aren't surprised.
                try:
                    import random as _random
                    import torch

                    if py_state is not None:
                        _random.setstate(py_state)
                    if np_state is not None:
                        np.random.set_state(np_state)
                    if torch_state is not None:
                        torch.set_rng_state(torch_state)
                    if cuda_states is not None:
                        try:
                            torch.cuda.set_rng_state_all(cuda_states)
                        except Exception:
                            pass
                except Exception:
                    pass
        if not audios:
            return np.zeros((0,), dtype=np.float32), int(self.get_sample_rate())

        a0 = audios[0]
        try:
            x = a0.detach().cpu().numpy()
        except Exception:
            x = np.asarray(a0)
        mono = np.asarray(x, dtype=np.float32).reshape(-1)
        return mono, int(self.get_sample_rate())


def prefetch_omnivoice(
    *,
    model_id: str = OmniVoiceRuntime.DEFAULT_MODEL_ID,
    revision: str | None = None,
    allow_downloads: bool = True,
) -> str:
    """Explicit prefetch: download weights and tokenizer into HF cache."""
    if not allow_downloads:
        raise ValueError("prefetch requires allow_downloads=True")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to prefetch OmniVoice") from e

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Warning: You are sending unauthenticated requests to the HF Hub\\..*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^You are sending unauthenticated requests to the HF Hub\\..*",
        )
        path = snapshot_download(
            repo_id=str(model_id),
            revision=str(revision) if revision else None,
        )
    return str(path)

