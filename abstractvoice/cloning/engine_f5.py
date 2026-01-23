from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf

from ..audio.resample import linear_resample_mono


def _load_as_mono_float(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # downmix
    mono = np.mean(audio, axis=1).astype(np.float32)
    return mono, int(sr)


@dataclass(frozen=True)
class OpenF5Artifacts:
    model_cfg: Path
    ckpt_file: Path
    vocab_file: Path


class F5TTSVoiceCloningEngine:
    """In-process F5-TTS voice cloning engine (optional extra).

    Why in-process
    --------------
    The CLI approach re-loads a multi-GB model on every utterance (very slow).
    Running in-process allows:
    - one-time model/vocoder load
    - per-voice reference preprocessing cache
    - much lower latency per utterance
    """

    def __init__(
        self,
        *,
        whisper_model: str = "tiny",
        debug: bool = False,
        device: str = "auto",
        nfe_step: int = 16,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        vocoder_name: str = "vocos",
        target_rms: float = 0.1,
        cross_fade_duration: float = 0.15,
    ):
        self.debug = debug
        self._whisper_model = whisper_model
        self._stt = None
        self._device_pref = device

        # Speed/quality knobs (lower nfe_step = faster, usually lower quality).
        self._nfe_step = int(nfe_step)
        self._cfg_strength = float(cfg_strength)
        self._sway_sampling_coef = float(sway_sampling_coef)
        self._vocoder_name = str(vocoder_name)
        self._target_rms = float(target_rms)
        self._cross_fade_duration = float(cross_fade_duration)

        # Lazy heavy objects (loaded on first inference).
        self._f5_model = None
        self._f5_vocoder = None
        self._f5_device = None

    def _get_stt(self):
        """Lazy-load STT to avoid surprise model downloads."""
        if self._stt is None:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            self._stt = FasterWhisperAdapter(
                model_size=self._whisper_model,
                device="cpu",
                compute_type="int8",
            )
        return self._stt

    def _ensure_f5_runtime(self) -> None:
        try:
            import importlib.util

            if importlib.util.find_spec("f5_tts") is None:
                raise ImportError("f5_tts not installed")
        except Exception as e:
            raise RuntimeError(
                "Voice cloning requires the optional dependency group.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[cloning]\"\n"
                f"Original error: {e}"
            ) from e

    def _artifact_root(self) -> Path:
        cache_dir = Path(os.path.expanduser("~/.cache/abstractvoice/openf5"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_openf5_artifacts_local(self) -> OpenF5Artifacts:
        """Resolve artifacts from the local cache directory without any network calls."""
        root = self._artifact_root()
        cfg = next(iter(root.rglob("*.yaml")), None) or next(iter(root.rglob("*.yml")), None)
        ckpt = next(iter(root.rglob("*.pt")), None)
        vocab = next(iter(root.rglob("vocab*.txt")), None) or next(iter(root.rglob("*.txt")), None)
        if not (cfg and ckpt and vocab):
            raise RuntimeError(
                "OpenF5 artifacts are not present locally.\n"
                "In the REPL run: /cloning_download\n"
                f"Looked under: {root}"
            )
        return OpenF5Artifacts(model_cfg=cfg, ckpt_file=ckpt, vocab_file=vocab)

    def ensure_openf5_artifacts_downloaded(self) -> OpenF5Artifacts:
        """Explicit prefetch entry point (REPL should call this, not speak())."""
        # Keep output quiet in interactive contexts.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        # Lazy import: keep core install light.
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to download OpenF5 artifacts.\n"
                "Install with: pip install huggingface_hub"
            ) from e

        snapshot_download(
            repo_id="mrfakename/OpenF5-TTS-Base",
            local_dir=str(self._artifact_root()),
            local_dir_use_symlinks=False,
        )
        return self._resolve_openf5_artifacts_local()

    def are_openf5_artifacts_available(self) -> bool:
        """Return True if artifacts are already present locally (no downloads)."""
        root = self._artifact_root()
        cfg = next(iter(root.rglob("*.yaml")), None) or next(iter(root.rglob("*.yml")), None)
        ckpt = next(iter(root.rglob("*.pt")), None)
        vocab = next(iter(root.rglob("vocab*.txt")), None) or next(iter(root.rglob("*.txt")), None)
        return bool(cfg and ckpt and vocab)

    def _resolve_device(self) -> str:
        if self._device_pref and self._device_pref != "auto":
            return str(self._device_pref)
        # Match f5_tts default device logic.
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Load vocoder + model once (expensive)."""
        if self._f5_model is not None and self._f5_vocoder is not None and self._f5_device is not None:
            return

        self._ensure_f5_runtime()
        artifacts = self._resolve_openf5_artifacts_local()

        # Silence HF progress bars during internal downloads (REPL UX).
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # Some f5_tts utilities print; keep it quiet unless debug.
        import contextlib
        import io

        from omegaconf import OmegaConf
        from hydra.utils import get_class

        from f5_tts.infer.utils_infer import load_model, load_vocoder

        device = self._resolve_device()

        model_cfg = OmegaConf.load(str(artifacts.model_cfg))
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        # load vocoder + model
        if self.debug:
            self._f5_vocoder = load_vocoder(vocoder_name=self._vocoder_name, device=device)
            self._f5_model = load_model(
                model_cls,
                model_arc,
                str(artifacts.ckpt_file),
                mel_spec_type=self._vocoder_name,
                vocab_file=str(artifacts.vocab_file),
                device=device,
            )
        else:
            buf_out = io.StringIO()
            buf_err = io.StringIO()
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                self._f5_vocoder = load_vocoder(vocoder_name=self._vocoder_name, device=device)
                self._f5_model = load_model(
                    model_cls,
                    model_arc,
                    str(artifacts.ckpt_file),
                    mel_spec_type=self._vocoder_name,
                    vocab_file=str(artifacts.vocab_file),
                    device=device,
                )

        self._f5_device = device

    def _prepare_reference_wav(
        self, reference_paths: Iterable[str | Path], *, target_sr: int = 24000, max_seconds: float = 15.0
    ) -> Path:
        paths = [Path(p) for p in reference_paths]
        if not paths:
            raise ValueError("reference_paths must contain at least one path")

        # Only support WAV/FLAC/OGG that soundfile can read reliably without extra system deps.
        supported = {".wav", ".flac", ".ogg"}
        for p in paths:
            if p.suffix.lower() not in supported:
                raise ValueError(
                    f"Unsupported reference audio format: {p.suffix}. "
                    f"Provide WAV/FLAC/OGG (got: {p})."
                )

        merged: List[np.ndarray] = []
        for p in paths:
            mono, sr = _load_as_mono_float(p)
            mono = linear_resample_mono(mono, sr, target_sr)
            merged.append(mono)

        audio = np.concatenate(merged) if merged else np.zeros((0,), dtype=np.float32)
        max_len = int(target_sr * max_seconds)
        if len(audio) > max_len:
            audio = audio[:max_len]

        # Write PCM16 WAV for maximum compatibility with downstream tools.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, audio, target_sr, subtype="PCM_16")
        return Path(tmp.name)

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str | Path],
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        self._ensure_model_loaded()

        ref_wav = self._prepare_reference_wav(reference_paths)
        try:
            if not reference_text:
                stt = self._get_stt()
                if not stt.is_available():
                    raise RuntimeError("STT (faster-whisper) is required to auto-transcribe reference audio.")
                reference_text = stt.transcribe(str(ref_wav))

            from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
            import contextlib
            import io
            import warnings

            # f5_tts prints a lot (progress bars, ref_text, batching info).
            # Keep default UX clean unless debug is enabled.
            out_buf = io.StringIO()
            err_buf = io.StringIO()
            stdout_cm = contextlib.nullcontext() if self.debug else contextlib.redirect_stdout(out_buf)
            stderr_cm = contextlib.nullcontext() if self.debug else contextlib.redirect_stderr(err_buf)

            with warnings.catch_warnings():
                # Torchaudio emits noisy deprecation warnings; they don't help users here.
                warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\..*")
                with stdout_cm, stderr_cm:
                    ref_audio_path, ref_text = preprocess_ref_audio_text(
                        str(ref_wav),
                        str(reference_text),
                        show_info=(print if self.debug else (lambda *_a, **_k: None)),
                    )
                    audio_segment, final_sr, _spec = infer_process(
                        ref_audio_path,
                        ref_text,
                        str(text),
                        self._f5_model,
                        self._f5_vocoder,
                        mel_spec_type=self._vocoder_name,
                        show_info=(print if self.debug else (lambda *_a, **_k: None)),
                        progress=None,  # disable tqdm
                        target_rms=self._target_rms,
                        cross_fade_duration=self._cross_fade_duration,
                        nfe_step=self._nfe_step,
                        cfg_strength=self._cfg_strength,
                        sway_sampling_coef=self._sway_sampling_coef,
                        speed=float(speed) if speed is not None else 1.0,
                        fix_duration=None,
                        device=self._f5_device,
                    )

            buf = io.BytesIO()
            sf.write(buf, audio_segment, int(final_sr), format="WAV", subtype="PCM_16")
            return buf.getvalue()
        finally:
            try:
                Path(ref_wav).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

