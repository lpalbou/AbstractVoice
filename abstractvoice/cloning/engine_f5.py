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


def _load_as_torch_channels_first(path: Path):
    """Load audio as a float32 torch Tensor shaped (channels, frames).

    We prefer `soundfile` over `torchaudio.load()` because torchaudio's I/O backend
    can vary by version (e.g. TorchCodec requirements) and may emit noisy stderr
    logs during decode that corrupt interactive CLI output.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover - torch is required by f5_tts runtime anyway
        raise RuntimeError("torch is required for F5 cloning inference") from e

    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile: (frames, channels) -> torch: (channels, frames)
    arr = np.ascontiguousarray(audio.T, dtype=np.float32)
    return torch.from_numpy(arr), int(sr)


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
        self._quality_preset = "balanced"

        # Lazy heavy objects (loaded on first inference).
        self._f5_model = None
        self._f5_vocoder = None
        self._f5_device = None

    def runtime_info(self) -> dict:
        """Return best-effort runtime info for debugging/perf validation."""
        info = {"requested_device": self._device_pref, "resolved_device": self._f5_device, "quality_preset": self._quality_preset}
        try:
            m = self._f5_model
            if m is not None and hasattr(m, "parameters"):
                p = next(iter(m.parameters()), None)
                if p is not None and hasattr(p, "device"):
                    info["model_param_device"] = str(p.device)
        except Exception:
            pass
        try:
            import torch

            info["torch_version"] = getattr(torch, "__version__", "?")
            info["cuda_available"] = bool(torch.cuda.is_available())
            try:
                info["mps_available"] = bool(torch.backends.mps.is_available())
            except Exception:
                info["mps_available"] = False
        except Exception:
            pass
        return info

    def set_quality_preset(self, preset: str) -> None:
        """Set speed/quality preset.

        Presets tune diffusion steps; lower steps are faster but can reduce quality.
        """
        p = (preset or "").strip().lower()
        if p not in ("fast", "balanced", "high"):
            raise ValueError("preset must be one of: fast|balanced|high")
        self._quality_preset = p
        if p == "fast":
            self._nfe_step = 8
            self._cfg_strength = 1.8
        elif p == "balanced":
            self._nfe_step = 16
            self._cfg_strength = 2.0
        else:
            self._nfe_step = 24
            self._cfg_strength = 2.2

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

        import warnings

        with warnings.catch_warnings():
            # huggingface_hub deprecated `local_dir_use_symlinks`; keep prefetch UX clean.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r".*local_dir_use_symlinks.*deprecated.*",
            )
            snapshot_download(
                repo_id="mrfakename/OpenF5-TTS-Base",
                local_dir=str(self._artifact_root()),
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
        from ..compute import best_torch_device

        return best_torch_device()

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
                # Deliberately do NOT auto-transcribe in the engine layer:
                # it can implicitly download STT weights and pollute interactive UX.
                raise RuntimeError(
                    "Missing reference_text for cloning.\n"
                    "Provide reference_text when cloning, or set it via the voice store."
                )

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
                # NOTE: redirecting sys.stdout affects the REPL spinner thread (global),
                # so we avoid redirects here. Instead, we call lower-level primitives
                # that don't print when properly configured.

                # Minimal normalization: f5_tts expects sentence-ending punctuation.
                ref_text = str(reference_text or "").strip()
                if ref_text and not (ref_text.endswith(". ") or ref_text.endswith("。") or ref_text.endswith(".")):
                    ref_text = ref_text + ". "
                elif ref_text.endswith("."):
                    ref_text = ref_text + " "

                # Avoid f5_tts preprocess_ref_audio_text() because it prints loudly.
                # We already clipped/resampled reference audio in _prepare_reference_wav().
                ref_audio_path = str(ref_wav)

                # Build gen_text batches with a simple chunker (no prints).
                gen_text = str(text)
                batches: List[str] = []
                max_chars = 160
                cur = ""
                for part in gen_text.replace("\n", " ").split(" "):
                    if not part:
                        continue
                    if len((cur + " " + part).strip()) <= max_chars:
                        cur = (cur + " " + part).strip()
                    else:
                        if cur:
                            batches.append(cur)
                        cur = part.strip()
                if cur:
                    batches.append(cur)

                from f5_tts.infer.utils_infer import infer_batch_process
                import numpy as _np
                audio, sr = _load_as_torch_channels_first(Path(ref_audio_path))
                # infer_batch_process returns a generator yielding final_wave at the end.
                final_wave, final_sr, _spec = next(
                    infer_batch_process(
                        (audio, sr),
                        ref_text if ref_text else " ",  # must not be empty
                        batches or [" "],
                        self._f5_model,
                        self._f5_vocoder,
                        mel_spec_type=self._vocoder_name,
                        progress=None,
                        target_rms=self._target_rms,
                        cross_fade_duration=self._cross_fade_duration,
                        nfe_step=self._nfe_step,
                        cfg_strength=self._cfg_strength,
                        sway_sampling_coef=self._sway_sampling_coef,
                        speed=float(speed) if speed is not None else 1.0,
                        fix_duration=None,
                        device=self._f5_device,
                        streaming=False,
                    )
                )

                audio_segment = _np.asarray(final_wave, dtype=_np.float32)
                final_sr = int(final_sr)

            buf = io.BytesIO()
            sf.write(buf, audio_segment, int(final_sr), format="WAV", subtype="PCM_16")
            return buf.getvalue()
        finally:
            try:
                Path(ref_wav).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

    def infer_to_audio_chunks(
        self,
        *,
        text: str,
        reference_paths: Iterable[str | Path],
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
        max_chars: int = 120,
        chunk_size: int = 2048,
    ):
        """Yield (audio_chunk, sample_rate) for progressive playback.

        Note: F5 sampling itself is not truly streaming mid-step. This yields chunks
        after each batch completes, which is still valuable for perceived latency.
        """
        self._ensure_model_loaded()

        ref_wav = self._prepare_reference_wav(reference_paths)
        try:
            if not reference_text:
                raise RuntimeError(
                    "Missing reference_text for cloning.\n"
                    "Provide reference_text when cloning, or set it via the voice store."
                )

            import warnings
            with warnings.catch_warnings():
                # Keep REPL and API logs clean.
                warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\..*")
                warnings.filterwarnings("ignore", category=UserWarning, message=r".*TorchCodec.*")

                ref_text = str(reference_text or "").strip()
                if ref_text and not (ref_text.endswith(". ") or ref_text.endswith("。") or ref_text.endswith(".")):
                    ref_text = ref_text + ". "
                elif ref_text.endswith("."):
                    ref_text = ref_text + " "

                # Prefer sentence boundaries to reduce audible "cuts".
                import re

                def _split_batches(s: str, limit: int) -> List[str]:
                    s = " ".join(str(s).replace("\n", " ").split()).strip()
                    if not s:
                        return []
                    sentences = re.split(r"(?<=[\.\!\?\。])\s+", s)
                    out: List[str] = []
                    cur_s = ""
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent:
                            continue
                        if len(sent) > limit:
                            # Fallback: word-based chunking for very long sentences.
                            words = sent.split(" ")
                            tmp = ""
                            for w in words:
                                cand = (tmp + " " + w).strip()
                                if len(cand) <= limit:
                                    tmp = cand
                                else:
                                    if tmp:
                                        out.append(tmp)
                                    tmp = w
                            if tmp:
                                out.append(tmp)
                            continue
                        cand = (cur_s + " " + sent).strip()
                        if len(cand) <= limit:
                            cur_s = cand
                        else:
                            if cur_s:
                                out.append(cur_s)
                            cur_s = sent
                    if cur_s:
                        out.append(cur_s)
                    return out

                batches = _split_batches(text, int(max_chars)) or [" "]

                from f5_tts.infer.utils_infer import infer_batch_process
                audio, sr = _load_as_torch_channels_first(Path(ref_wav))

                for chunk, sr_out in infer_batch_process(
                    (audio, sr),
                    ref_text if ref_text else " ",
                    batches,
                    self._f5_model,
                    self._f5_vocoder,
                    mel_spec_type=self._vocoder_name,
                    progress=None,
                    target_rms=self._target_rms,
                    cross_fade_duration=self._cross_fade_duration,
                    nfe_step=self._nfe_step,
                    cfg_strength=self._cfg_strength,
                    sway_sampling_coef=self._sway_sampling_coef,
                    speed=float(speed) if speed is not None else 1.0,
                    fix_duration=None,
                    device=self._f5_device,
                    streaming=True,
                    chunk_size=int(chunk_size),
                ):
                    yield np.asarray(chunk, dtype=np.float32), int(sr_out)
        finally:
            try:
                Path(ref_wav).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
