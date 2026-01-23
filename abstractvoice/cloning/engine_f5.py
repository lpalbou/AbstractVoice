from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    """Wraps `f5-tts_infer-cli` for voice cloning (optional extra).

    Uses our STT (faster-whisper) to auto-generate ref_text when not provided.
    """

    def __init__(self, *, whisper_model: str = "tiny", debug: bool = False):
        self.debug = debug
        self._whisper_model = whisper_model
        self._stt = None

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

    def _ensure_infer_runner(self) -> List[str]:
        """Return a command prefix to run the F5 inference CLI in *this* environment.

        We intentionally avoid relying on global shims (pyenv) because they can point
        to a different Python environment than the running process.
        """
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

        # Prefer the script installed alongside the current interpreter.
        venv_exe = Path(sys.executable).resolve().parent / "f5-tts_infer-cli"
        if venv_exe.exists():
            return [str(venv_exe)]

        # Fallback: run as a module (works as long as f5_tts is installed).
        # Ref: python -m f5_tts.infer.infer_cli
        return [sys.executable, "-m", "f5_tts.infer.infer_cli"]

    def _artifact_root(self) -> Path:
        cache_dir = Path(os.path.expanduser("~/.cache/abstractvoice/openf5"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_openf5_artifacts(self, *, local_files_only: bool = False) -> OpenF5Artifacts:
        # Lazy import: keep core install light.
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to download OpenF5 artifacts.\n"
                "Install with: pip install huggingface_hub"
            ) from e

        local_dir = snapshot_download(
            repo_id="mrfakename/OpenF5-TTS-Base",
            local_dir=str(self._artifact_root()),
            local_files_only=bool(local_files_only),
        )
        root = Path(local_dir)

        # Heuristic selection.
        cfg = next(iter(root.rglob("*.yaml")), None) or next(iter(root.rglob("*.yml")), None)
        ckpt = next(iter(root.rglob("*.pt")), None)
        vocab = next(iter(root.rglob("vocab*.txt")), None) or next(iter(root.rglob("*.txt")), None)

        if not (cfg and ckpt and vocab):
            raise RuntimeError(f"Could not locate OpenF5 artifacts under {root}")

        return OpenF5Artifacts(model_cfg=cfg, ckpt_file=ckpt, vocab_file=vocab)

    def ensure_openf5_artifacts_downloaded(self) -> OpenF5Artifacts:
        """Explicit prefetch entry point (REPL should call this, not speak())."""
        # Keep output quiet in interactive contexts.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        return self._resolve_openf5_artifacts(local_files_only=False)

    def are_openf5_artifacts_available(self) -> bool:
        """Return True if artifacts are already present locally (no downloads)."""
        root = self._artifact_root()
        cfg = next(iter(root.rglob("*.yaml")), None) or next(iter(root.rglob("*.yml")), None)
        ckpt = next(iter(root.rglob("*.pt")), None)
        vocab = next(iter(root.rglob("vocab*.txt")), None) or next(iter(root.rglob("*.txt")), None)
        return bool(cfg and ckpt and vocab)

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
        runner = self._ensure_infer_runner()

        # Silence HF progress bars during inference (REPL UX).
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        # If already cached, never hit the network / print download status.
        artifacts = self._resolve_openf5_artifacts(local_files_only=self.are_openf5_artifacts_available())

        ref_wav = self._prepare_reference_wav(reference_paths)
        try:
            if not reference_text:
                stt = self._get_stt()
                if not stt.is_available():
                    raise RuntimeError("STT (faster-whisper) is required to auto-transcribe reference audio.")
                reference_text = stt.transcribe(str(ref_wav))

            out_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_wav.close()

            cmd = [
                *runner,
                "-mc",
                str(artifacts.model_cfg),
                "-p",
                str(artifacts.ckpt_file),
                "-v",
                str(artifacts.vocab_file),
                "-r",
                str(ref_wav),
                "-s",
                str(reference_text),
                "-t",
                str(text),
                "-w",
                str(out_wav.name),
            ]
            if speed is not None:
                cmd.extend(["--speed", str(speed)])

            if self.debug:
                print("F5 CLI:", " ".join(cmd))

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                stderr = ""
                try:
                    stderr = (e.stderr or b"").decode("utf-8", errors="replace")
                except Exception:
                    stderr = str(e.stderr)
                raise RuntimeError(f"F5 inference failed (exit={e.returncode}).\n{stderr}") from e
            data = Path(out_wav.name).read_bytes()
            return data
        finally:
            try:
                Path(ref_wav).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

