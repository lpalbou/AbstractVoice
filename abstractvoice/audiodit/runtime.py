"""AudioDiT runtime wrapper (LongCat-AudioDiT).

This is the integration layer that maps AbstractVoice concepts (language, duration
heuristics, offline-first downloads) onto the AudioDiT HF model.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..audio.resample import linear_resample_mono
from ..compute import best_torch_device


_RE_QUOTE = re.compile(r"""["“”‘’]""")
_RE_WS = re.compile(r"\s+")
_RE_SENT_END = re.compile(r"(?<=[\.\!\?\。\！\？])\s+")


def _normalize_text(s: str) -> str:
    # Match upstream normalization for stability:
    # - lowercase
    # - normalize quotes
    # - collapse whitespace
    t = str(s or "").lower()
    t = _RE_QUOTE.sub(" ", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def _approx_duration_from_text(text: str, *, max_duration: float) -> float:
    """Approximate duration from text (upstream AudioDiT heuristic).

    Upstream uses a per-character heuristic (alpha vs CJK) that empirically
    stabilizes perceived voice/pitch across different text lengths.
    """

    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21

    t = re.sub(r"\s+", "", str(text or ""))
    num_zh = 0
    num_en = 0
    num_other = 0
    for c in t:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1

    # Assign punctuation/digits to the majority class (upstream behavior).
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other

    max_d = float(max_duration) if max_duration is not None else 30.0
    return float(min(max_d, float(num_zh) * ZH_DUR_PER_CHAR + float(num_en) * EN_DUR_PER_CHAR))


def _split_text_batches(text: str, *, max_chars: int = 240) -> list[str]:
    """Split text into short batches, preferring sentence boundaries."""
    s = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]

    # Split on common sentence terminators (keep it simple + multilingual).
    parts = re.split(_RE_SENT_END, s)
    out: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_chars:
            # Fallback: word-based chunking for very long sentences.
            words = p.split(" ")
            tmp = ""
            for w in words:
                cand = (tmp + " " + w).strip()
                if len(cand) <= max_chars:
                    tmp = cand
                else:
                    if tmp:
                        out.append(tmp)
                    tmp = w
            if tmp:
                out.append(tmp)
            cur = ""
            continue

        cand = (cur + " " + p).strip()
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def _audio_metrics(mono: np.ndarray, *, sample_rate: int) -> dict[str, float]:
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"peak": 0.0, "rms": 0.0, "env_var": 0.0}
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))

    # 20ms envelope variation (cheap proxy for "not stationary noise").
    win = max(1, int(round(float(sample_rate) * 0.02)))
    n = int(x.size // win)
    if n <= 0:
        env_var = 0.0
    else:
        frames = x[: n * win].reshape(n, win)
        env = np.sqrt(np.mean(frames * frames, axis=1))
        env_var = float(env.max() - env.min())
    return {"peak": peak, "rms": rms, "env_var": env_var}


def _is_weak_audio(mono: np.ndarray, *, sample_rate: int) -> bool:
    m = _audio_metrics(mono, sample_rate=sample_rate)
    # Empirically, some AudioDiT outputs can collapse to quiet, stationary noise for longer inputs.
    # We treat that as "weak" and fall back to smaller text chunks.
    return bool(m["peak"] < 0.08 and m["rms"] < 0.02)


def _split_words_in_half(text: str) -> tuple[str, str]:
    words = " ".join(str(text or "").split()).strip().split(" ")
    words = [w for w in words if w]
    if len(words) <= 1:
        s = " ".join(words).strip()
        return s, ""
    mid = max(1, len(words) // 2)
    left = " ".join(words[:mid]).strip()
    right = " ".join(words[mid:]).strip()
    return left, right


def _silence_s(seconds: float, *, sample_rate: int) -> np.ndarray:
    n = int(max(0.0, float(seconds)) * int(sample_rate))
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    return np.zeros((n,), dtype=np.float32)


def _load_audio_mono_24k(paths: Iterable[str], *, target_sr: int, max_seconds: float | None = None) -> np.ndarray:
    """Load one or more audio files as mono float32 at `target_sr` and concat."""
    import soundfile as sf

    merged: list[np.ndarray] = []
    for p in paths:
        audio, sr = sf.read(str(p), always_2d=True, dtype="float32")
        mono = np.mean(audio, axis=1).astype(np.float32).reshape(-1)
        if int(sr) != int(target_sr):
            mono = linear_resample_mono(mono, int(sr), int(target_sr))
        merged.append(mono)

    out = np.concatenate(merged) if merged else np.zeros((0,), dtype=np.float32)
    if max_seconds is not None and out.size:
        out = out[: int(round(float(max_seconds) * float(target_sr)))]
    return out.astype(np.float32)


@dataclass
class AudioDiTSettings:
    steps: int = 16
    cfg_strength: float = 4.0
    guidance_method: str = "apg"  # cfg|apg
    seed: int | None = 1024


class AudioDiTRuntime:
    """Lazy-loading runtime for AudioDiTModel + tokenizer."""

    DEFAULT_MODEL_ID = "meituan-longcat/LongCat-AudioDiT-1B"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        device: str = "auto",
        dtype: str | None = None,
        allow_downloads: bool = True,
        debug: bool = False,
    ):
        self.model_id = str(model_id or self.DEFAULT_MODEL_ID)
        self.revision = str(revision) if revision else None
        self._device_pref = str(device or "auto")
        self._dtype_pref = str(dtype) if dtype else None
        self.allow_downloads = bool(allow_downloads)
        self.debug = bool(debug)

        self._model = None
        self._tokenizer = None
        self._resolved_device: str | None = None

    def runtime_info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "requested_device": self._device_pref,
            "resolved_device": self._resolved_device,
            "allow_downloads": bool(self.allow_downloads),
        }

    def _resolve_device(self) -> str:
        if self._device_pref and self._device_pref != "auto":
            return str(self._device_pref)
        return best_torch_device()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._resolved_device is not None:
            return

        # Keep interactive UX quiet by default.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TQDM", "1")

        try:
            import torch
            from transformers import AutoTokenizer
            from transformers.utils import logging as _tf_logging
        except Exception as e:
            raise RuntimeError(
                "AudioDiT requires optional dependencies.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[audiodit]\""
            ) from e

        # Disable transformers progress bars (e.g. "Loading weights") for REPL UX.
        # Env vars alone don't reliably work once these libs are already imported.
        try:
            _tf_logging.disable_progress_bar()
        except Exception:
            pass

        # Import model code only when needed (avoids heavy deps in base install).
        from .modeling_audiodit import AudioDiTModel

        device = self._resolve_device()
        self._resolved_device = device

        local_only = not bool(self.allow_downloads)
        try:
            # Suppress noisy FutureWarnings during model construction (e.g. torch weight_norm deprecation).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                model = AudioDiTModel.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    local_files_only=local_only,
                )
        except Exception as e:
            if local_only:
                raise RuntimeError(
                    "AudioDiT weights are not available locally and downloads are disabled.\n"
                    "Fix options:\n"
                    "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                    "  - Or prefetch explicitly: abstractvoice-prefetch --audiodit\n"
                    f"Model: {self.model_id}"
                ) from e
            raise

        # Move to device first; keep transformer weights in their checkpoint dtype.
        try:
            model.to(device)
        except Exception:
            # Best-effort: if move fails, keep on CPU.
            model.to("cpu")
            self._resolved_device = "cpu"
            device = "cpu"

        # VAE runs in fp16 in upstream (matching original).
        try:
            if hasattr(model, "vae") and hasattr(model.vae, "to_half"):
                model.vae.to_half()
        except Exception:
            pass

        model.eval()

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model.config.text_encoder_model),
                local_files_only=local_only,
            )
        except Exception as e:
            if local_only:
                raise RuntimeError(
                    "AudioDiT tokenizer files are not available locally and downloads are disabled.\n"
                    "Fix options:\n"
                    "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                    "  - Or prefetch explicitly: abstractvoice-prefetch --audiodit\n"
                    f"Tokenizer: {getattr(model.config, 'text_encoder_model', '?')}"
                ) from e
            raise

        self._model = model
        self._tokenizer = tokenizer

    def _frames_from_seconds(self, seconds: float, *, sr: int, hop: int) -> int:
        seconds = max(0.0, float(seconds))
        hop = int(hop) if hop else 2048
        sr = int(sr) if sr else 24000
        return int(seconds * sr // hop)

    def generate_chunks(
        self,
        *,
        text: str,
        language: str,
        prompt_audio_paths: Iterable[str] | None = None,
        prompt_audio: np.ndarray | None = None,
        prompt_audio_sr: int | None = None,
        prompt_text: str | None = None,
        settings: AudioDiTSettings | None = None,
        rolling_prompt: bool = True,
        max_chars: int = 240,
        max_prompt_seconds: float = 15.0,
    ) -> tuple[list[np.ndarray], int]:
        """Generate audio in sequential chunks (one per text batch)."""
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        import torch

        model = self._model
        tokenizer = self._tokenizer

        sr = int(getattr(model.config, "sampling_rate", 24000) or 24000)
        hop = int(getattr(model.config, "latent_hop", 2048) or 2048)
        max_wav_s = float(getattr(model.config, "max_wav_duration", 30) or 30)
        max_frames = int(max_wav_s * sr // hop)

        st = settings or AudioDiTSettings()
        if st.guidance_method not in ("cfg", "apg"):
            raise ValueError("guidance_method must be cfg|apg")

        # Seed for determinism (best-effort).
        if st.seed is not None:
            try:
                torch.manual_seed(int(st.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(int(st.seed))
            except Exception:
                pass

        gen_text = _normalize_text(text)
        if not gen_text:
            return [], sr

        # Initial split: prefer sentence boundaries.
        # We only fall back to smaller chunks if the model output is detected as weak.
        batches = _split_text_batches(gen_text, max_chars=int(max_chars)) or [" "]

        # Load prompt audio (optional).
        prompt_wav = None
        prompt_frames = 0
        prompt_time_s = 0.0
        norm_prompt_text = _normalize_text(prompt_text or "")
        has_external_prompt = False

        def _encode_prompt_audio(wav_np: np.ndarray) -> tuple[torch.Tensor | None, int, float]:
            wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
            if wav_np.size == 0:
                return None, 0, 0.0
            pw = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            with torch.no_grad():
                _, frames = model.encode_prompt_audio(pw.to(model.device))
            time_s = float(frames) * float(hop) / float(sr)
            return pw, int(frames), float(time_s)

        if prompt_audio_paths:
            wav = _load_audio_mono_24k(
                prompt_audio_paths,
                target_sr=sr,
                max_seconds=float(max_prompt_seconds) if max_prompt_seconds else None,
            )
            if wav.size:
                prompt_wav, prompt_frames, prompt_time_s = _encode_prompt_audio(wav)
                has_external_prompt = bool(prompt_wav is not None)

        if (not has_external_prompt) and prompt_audio is not None:
            wav = np.asarray(prompt_audio, dtype=np.float32)
            try:
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1).astype(np.float32)
            except Exception:
                pass
            wav = wav.reshape(-1)
            sr_in = int(prompt_audio_sr) if prompt_audio_sr is not None else int(sr)
            if sr_in != int(sr):
                wav = linear_resample_mono(wav, sr_in, int(sr))
            if max_prompt_seconds:
                wav = wav[: int(round(float(max_prompt_seconds) * float(sr)))]
            if wav.size:
                prompt_wav, prompt_frames, prompt_time_s = _encode_prompt_audio(wav)
                has_external_prompt = bool(prompt_wav is not None)

        use_rolling_prompt = bool(rolling_prompt) and (not has_external_prompt) and len(batches) > 1
        rolling_wav: torch.Tensor | None = None
        rolling_frames: int = 0
        rolling_time_s: float = 0.0
        rolling_text: str = ""

        def _synth_one(
            batch_text: str,
            *,
            _prompt_wav: torch.Tensor | None,
            _prompt_frames: int,
            _prompt_time_s: float,
            _prompt_text: str,
        ) -> np.ndarray:
            batch_text = _normalize_text(batch_text)
            if not batch_text:
                return np.zeros((0,), dtype=np.float32)

            # Duration estimate for this batch.
            gen_s = _approx_duration_from_text(batch_text, max_duration=max(0.1, max_wav_s - float(_prompt_time_s)))
            if _prompt_wav is not None and _prompt_text:
                # Match pace: scale gen duration by prompt_time / expected_prompt_time.
                exp_prompt_s = _approx_duration_from_text(_prompt_text, max_duration=float(max_wav_s))
                if exp_prompt_s > 1e-3:
                    ratio = float(_prompt_time_s) / float(exp_prompt_s)
                    ratio = float(min(1.5, max(1.0, ratio)))
                    gen_s *= ratio

            gen_frames = self._frames_from_seconds(gen_s, sr=sr, hop=hop)
            duration_frames = min(max_frames, int(_prompt_frames + gen_frames))
            duration_frames = max(int(_prompt_frames), int(duration_frames))

            if _prompt_wav is not None and _prompt_text:
                full_text = f"{_prompt_text} {batch_text}".strip()
            else:
                full_text = batch_text

            inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
            out = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_audio=_prompt_wav,
                duration=int(duration_frames),
                steps=int(st.steps),
                cfg_strength=float(st.cfg_strength),
                guidance_method=str(st.guidance_method),
            )
            wav = out.waveform.squeeze().detach().float().cpu().numpy().astype(np.float32).reshape(-1)
            return wav

        def _synth_robust(
            batch_text: str,
            *,
            _prompt_wav: torch.Tensor | None,
            _prompt_frames: int,
            _prompt_time_s: float,
            _prompt_text: str,
            depth: int = 0,
            max_depth: int = 6,
        ) -> list[np.ndarray]:
            wav = _synth_one(
                batch_text,
                _prompt_wav=_prompt_wav,
                _prompt_frames=_prompt_frames,
                _prompt_time_s=_prompt_time_s,
                _prompt_text=_prompt_text,
            )
            if _prompt_wav is not None:
                return [wav]
            if not _is_weak_audio(wav, sample_rate=sr):
                return [wav]
            if depth >= max_depth:
                return [wav]
            left, right = _split_words_in_half(batch_text)
            if not right:
                return [wav]
            return _synth_robust(
                left,
                _prompt_wav=_prompt_wav,
                _prompt_frames=_prompt_frames,
                _prompt_time_s=_prompt_time_s,
                _prompt_text=_prompt_text,
                depth=depth + 1,
                max_depth=max_depth,
            ) + _synth_robust(
                right,
                _prompt_wav=_prompt_wav,
                _prompt_frames=_prompt_frames,
                _prompt_time_s=_prompt_time_s,
                _prompt_text=_prompt_text,
                depth=depth + 1,
                max_depth=max_depth,
            )

        chunks: list[np.ndarray] = []
        for batch_text in batches:
            eff_wav = prompt_wav
            eff_frames = int(prompt_frames)
            eff_time_s = float(prompt_time_s)
            eff_text = str(norm_prompt_text)
            if use_rolling_prompt and rolling_wav is not None:
                eff_wav = rolling_wav
                eff_frames = int(rolling_frames)
                eff_time_s = float(rolling_time_s)
                eff_text = str(rolling_text)

            if eff_wav is None:
                wavs = _synth_robust(
                    batch_text,
                    _prompt_wav=None,
                    _prompt_frames=0,
                    _prompt_time_s=0.0,
                    _prompt_text="",
                )
                chunks.extend(wavs)
                produced = wavs[-1] if wavs else np.zeros((0,), dtype=np.float32)
            else:
                produced = _synth_one(
                    batch_text,
                    _prompt_wav=eff_wav,
                    _prompt_frames=eff_frames,
                    _prompt_time_s=eff_time_s,
                    _prompt_text=eff_text,
                )
                chunks.append(produced)

            if use_rolling_prompt and rolling_wav is None and produced.size:
                # Use the first generated chunk as a voice anchor for subsequent chunks.
                anchor = np.asarray(produced, dtype=np.float32).reshape(-1)
                # Empirically, very long prompt excerpts can destabilize perceived pitch.
                # Keep the rolling anchor short even if callers allow a longer prompt.
                anchor_max_s = 8.0
                if max_prompt_seconds:
                    try:
                        anchor_max_s = min(float(anchor_max_s), float(max_prompt_seconds))
                    except Exception:
                        anchor_max_s = 8.0
                anchor = anchor[: int(round(float(anchor_max_s) * float(sr)))]
                rt = _normalize_text(batch_text)
                if anchor.size and rt:
                    rolling_wav, rolling_frames, rolling_time_s = _encode_prompt_audio(anchor)
                    rolling_text = rt

        # Insert small silences between chunks for intelligibility.
        if prompt_wav is None and len(chunks) > 1:
            spaced: list[np.ndarray] = []
            for i, c in enumerate(chunks):
                spaced.append(c)
                if i != len(chunks) - 1:
                    spaced.append(_silence_s(0.04, sample_rate=sr))
            chunks = spaced

        return chunks, int(sr)

    def generate(
        self,
        *,
        text: str,
        language: str,
        prompt_audio_paths: Iterable[str] | None = None,
        prompt_audio: np.ndarray | None = None,
        prompt_audio_sr: int | None = None,
        prompt_text: str | None = None,
        settings: AudioDiTSettings | None = None,
        rolling_prompt: bool = True,
        max_chars: int = 240,
        max_prompt_seconds: float = 15.0,
    ) -> tuple[np.ndarray, int]:
        """Generate a single waveform (mono float32) and return (audio, sample_rate)."""
        chunks, sr = self.generate_chunks(
            text=text,
            language=language,
            prompt_audio_paths=prompt_audio_paths,
            prompt_audio=prompt_audio,
            prompt_audio_sr=prompt_audio_sr,
            prompt_text=prompt_text,
            settings=settings,
            rolling_prompt=rolling_prompt,
            max_chars=max_chars,
            max_prompt_seconds=max_prompt_seconds,
        )
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        return audio.astype(np.float32), int(sr)


def prefetch_audiodit(*, model_id: str = AudioDiTRuntime.DEFAULT_MODEL_ID, allow_downloads: bool = True) -> str:
    """Explicit prefetch: download weights and tokenizer into HF cache."""
    if not allow_downloads:
        raise ValueError("prefetch requires allow_downloads=True")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is required to prefetch AudioDiT") from e

    # Download weights/config into HF cache.
    path = snapshot_download(str(model_id))

    # Also fetch tokenizer assets (UMT5).
    try:
        from transformers import AutoConfig, AutoTokenizer

        cfg = AutoConfig.from_pretrained(str(model_id))
        tok_id = getattr(cfg, "text_encoder_model", None) or "google/umt5-base"
        AutoTokenizer.from_pretrained(str(tok_id))
    except Exception:
        # Best-effort; weights are the critical part.
        pass

    return str(path)

