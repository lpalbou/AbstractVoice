"""Qwen3-TTS runtime wrapper (AbstractVoice-owned).

Maps AbstractVoice concepts onto the vendored Qwen3-TTS cores: offline-first
snapshot resolution, backend selection (ADR 0008: MLX on Apple silicon, PyTorch
elsewhere), the shared torch device/dtype policy (ADR 0005), model variant
gating (custom_voice / base / voice_design), sampling presets, frame-level
streaming on the MLX backend, and a serialization lock so concurrent synthesis
cannot interleave generation state.

Everything model-facing (prompt formats, generate flows) lives in
``orchestration.py`` (PyTorch, upstream-verbatim) and ``mlx/model.py`` (MLX).
Everything policy-facing lives here.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Tuple

import numpy as np

from ..audio.fade import apply_edge_fades
from ..compute import resolve_torch_runtime
from .backend import (
    BACKEND_ENV,
    MLX_QUANTIZATION_ENV,
    describe_backend_selection,
    normalize_backend,
    normalize_mlx_quantization,
    require_mlx_version,
    resolve_mlx_quantization,
    select_backend,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# The five 12Hz repos this integration understands. Others load too — the gate
# is `tts_model_type` read from the snapshot config, not this list.
KNOWN_MODEL_IDS = (
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
)

# Codec frame rate: 12.5 codes/second of audio. max_new_tokens derives from the
# text length instead of riding generation_config.json's 8192 (= ~11 minutes of
# audio per utterance -- a runaway-generation budget, not a sane cap).
_CODES_PER_SECOND = 12.5
# The SLOWEST measured speaking rate across the preset voices: dylan renders
# English at 4.8 chars/s (and CJK syllabary sits near 5 chars/s), so 4.0 with
# the 1.5x factor leaves ~1.9x headroom over the slowest real clip. EOS ends
# normal generation long before the cap; a tight "fit" here silently truncated
# half a sentence for slow presets, which is worse than a loose cap.
_MIN_CHARS_PER_SECOND = 4.0

# Utterance edges are faded over 5 ms (de-click). Streamed chunks are contiguous
# pieces of one waveform: only the first chunk's start and the last chunk's end
# are faded, never the seams.
_EDGE_FADE_MS = 5.0

# One generation per reply on the MLX backend. Texts longer than this are cut at
# sentence boundaries into independent generations (800 chars is ~55 s of speech
# at the typical 15 chars/s, ~700 codec frames); below it the whole reply is one
# sample, so the voice cannot drift between sentences.
NATIVE_STREAM_MAX_CHARS = 800


def _finish_utterance(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """De-click completed utterances, before playback or file/plugin delivery.

    Qwen can begin/end away from zero. Fade just 5 ms at each edge without
    changing duration or the interior. Do not apply this to native codec packets:
    those are contiguous pieces of one utterance, not independent waveforms.
    """
    sr = int(sample_rate)
    return apply_edge_fades(audio, sample_rate=sr, fade_ms=_EDGE_FADE_MS), sr


# One-shot synthesis (`one_shot=True` on the speak/clone surfaces): the whole text
# is ONE generation, never segmented, so the voice cannot be re-sampled between
# sentences. The bound keeps one generation inside a budget the model handles;
# the runaway cap is raised so that even the slowest preset voice (4 chars/s,
# with the usual 1.5x headroom) is never cut short within that bound.
ONE_SHOT_MAX_CHARS = 1000
ONE_SHOT_MAX_NEW_TOKENS = 4800


def one_shot_max_new_tokens(text: str) -> int:
    """The runaway cap for a one-shot generation; raises when the text is too long for one."""
    n = len(str(text or "").strip())
    if n > ONE_SHOT_MAX_CHARS:
        raise ValueError(
            f"one_shot synthesis takes at most {ONE_SHOT_MAX_CHARS} characters in a single generation "
            f"(got {n}). Split the text yourself, or omit one_shot to let it be segmented."
        )
    return estimate_max_new_tokens(text, ceiling=ONE_SHOT_MAX_NEW_TOKENS)


def estimate_max_new_tokens(text: str, *, floor: int = 96, ceiling: int = 2048) -> int:
    """RUNAWAY cap for one utterance's codec tokens, derived from its length.

    This is protection against degenerate generation, not a duration estimate:
    the model stops at its own EOS in normal operation (verified token-identical
    to upstream). Sized for the slowest preset voice so it can never cut real
    speech; one conservative rate covers Latin and CJK alike.
    """
    seconds = max(1.0, len(str(text)) / _MIN_CHARS_PER_SECOND) * 1.5
    tokens = int(seconds * _CODES_PER_SECOND) + 32
    return max(int(floor), min(int(ceiling), tokens))


# Codebook predictor and sampler choices (see Qwen3TTSSettings). Every surface
# (CLI, REPL, web example, AbstractCore plugin settings) validates through the
# two normalizers below, so a bad value fails before any weights load.
QWEN3_TTS_PREDICTORS = ("auto", "reference")
QWEN3_TTS_SAMPLERS = ("multinomial", "exponential")


def normalize_qwen3_tts_predictor(value: Any) -> str:
    """``"auto"`` (default, also for None/empty) or ``"reference"``; anything else raises."""
    return _normalize_choice(value, QWEN3_TTS_PREDICTORS, "Qwen3-TTS predictor")


def normalize_qwen3_tts_sampler(value: Any) -> str:
    """``"multinomial"`` (default, also for None/empty) or ``"exponential"``; anything else raises."""
    return _normalize_choice(value, QWEN3_TTS_SAMPLERS, "Qwen3-TTS sampler")


def _normalize_choice(value: Any, choices: tuple, label: str) -> str:
    text = str(value if value is not None else "").strip().lower() or choices[0]
    if text not in choices:
        raise ValueError(f"{label} must be one of {', '.join(choices)} (got {value!r})")
    return text


@dataclass
class Qwen3TTSSettings:
    """Sampling knobs; quality presets map onto these (both samplers together)."""

    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    subtalker_temperature: float = 0.9
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    quality_preset: str = "standard"
    # Codebook generation, read when the model loads (quality presets never
    # touch these). predictor: "auto" (guarded fixed-depth loop, see
    # predictor.py) or "reference" (Transformers generate, for comparison).
    # sampler: "multinomial" (torch stock) or "exponential" (opt-in
    # exponential race with frame-level validation); "reference" ignores it.
    # Both are PyTorch-specific: the MLX backend accepts them, runs its own
    # fixed-depth loop with categorical (multinomial) sampling, and reports
    # the requested values as not applied in ``runtime_info()``.
    predictor: str = "auto"
    sampler: str = "multinomial"
    # Compute backend (ADR 0008): "auto" picks MLX on Apple silicon when `mlx`
    # is importable (or whatever ABSTRACTVOICE_QWEN3_TTS_BACKEND says), else torch.
    backend: str = "auto"
    # MLX only: talker weights as published (bf16) or quantized on load (8bit,
    # the default, or 4bit); "auto" reads ABSTRACTVOICE_QWEN3_TTS_MLX_QUANTIZATION.
    mlx_quantization: str = "auto"
    # MLX only: frame-level streaming cadence, in 80 ms codec frames. The first
    # chunk is smaller so speech starts sooner; later chunks are larger so the
    # per-chunk decode overhead stays small.
    stream_first_chunk_frames: int = 6
    stream_chunk_frames: int = 12

    def __post_init__(self) -> None:
        self.predictor = normalize_qwen3_tts_predictor(self.predictor)
        self.sampler = normalize_qwen3_tts_sampler(self.sampler)
        self.backend = normalize_backend(self.backend)
        self.mlx_quantization = normalize_mlx_quantization(self.mlx_quantization)
        self.stream_first_chunk_frames = max(1, int(self.stream_first_chunk_frames))
        self.stream_chunk_frames = max(1, int(self.stream_chunk_frames))

    def apply_quality_preset(self, preset: str) -> None:
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self.quality_preset = p
        # Lower temperature/top_k = steadier, faster-converging output; "high"
        # keeps upstream's defaults, which their card tunes for expressiveness.
        if p == "low":
            self.temperature = 0.7
            self.top_k = 20
            self.subtalker_temperature = 0.7
            self.subtalker_top_k = 20
        elif p == "standard":
            self.temperature = 0.8
            self.top_k = 40
            self.subtalker_temperature = 0.8
            self.subtalker_top_k = 40
        else:  # high
            self.temperature = 0.9
            self.top_k = 50
            self.subtalker_temperature = 0.9
            self.subtalker_top_k = 50

    def generate_kwargs(self) -> dict:
        return dict(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            subtalker_temperature=self.subtalker_temperature,
            subtalker_top_k=self.subtalker_top_k,
            subtalker_top_p=self.subtalker_top_p,
        )


@dataclass
class _EdgeFader:
    """Fade the first chunk in and the last chunk out of a stream of contiguous chunks.

    The tail of every chunk (one fade length) is withheld and prepended to the
    next chunk, so when the stream ends the last fade length is still in hand
    and can be faded out. Adds 5 ms to each chunk's latency, nothing else.
    """

    sample_rate: int
    held: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    started: bool = False

    @property
    def fade_samples(self) -> int:
        return max(1, int(round(self.sample_rate * _EDGE_FADE_MS / 1000.0)))

    def push(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return None
        if not self.started:
            self.started = True
            mono = apply_edge_fades(mono, sample_rate=self.sample_rate, fade_ms=_EDGE_FADE_MS, fade_out=False)
        joined = np.concatenate([self.held, mono]) if self.held.size else mono
        n = self.fade_samples
        if joined.size <= n:
            self.held = joined
            return None
        self.held = joined[-n:].copy()
        out = joined[:-n]
        return out if out.size else None

    def flush(self) -> Optional[np.ndarray]:
        if not self.held.size:
            return None
        out = apply_edge_fades(self.held, sample_rate=self.sample_rate, fade_ms=_EDGE_FADE_MS, fade_in=False)
        self.held = np.zeros(0, dtype=np.float32)
        return out


# A generation that misses its end of speech (EOS) keeps emitting codec frames -- a repeated
# syllable, then near-silence with stray bursts -- until max_new_tokens, which is sized for the
# slowest voice (estimate_max_new_tokens: up to 2048 frames, 164 s). Seen live on 2026-11-08: a
# 619-character segment ran to the 163.84 s cap; its speech stopped at 25 s and the listener heard
# a syllable, then 2 minutes of silence, then the next segment. Real speech never keeps a 4 s
# window 90 % quiet: over 52 real takes of that reply (Qwen3-TTS 0.6B Base clone, MLX 8-bit) the
# quietest 4 s window was 34 % quiet; the runaway reached 90 % 15 s after its speech stopped.
# Such a window ends the generation as its EOS would have: the reply goes on with its next segment.
RUNAWAY_WINDOW_SECONDS = 4.0
RUNAWAY_QUIET_FRACTION = 0.9
_QUIET_FRAME_SECONDS = 0.05
_QUIET_FRAME_RMS = 0.01  # -40 dBFS: a 50 ms frame below this is quiet (speech frames sit near -17 dBFS)


class _QuietRunaway:
    """Finds where one generation went quiet for good (it missed its EOS).

    Fed the generation's audio in order, in chunks of any size. Counting starts at the first
    audible frame; once ``RUNAWAY_QUIET_FRACTION`` of the frames of the last
    ``RUNAWAY_WINDOW_SECONDS`` are quiet, ``push`` returns how many samples of that chunk come
    before the quiet window (keep those, end the generation) and ``quiet_from`` is where the
    window began, in samples from the start of the generation.
    """

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate) or 24000
        self.hop = max(1, int(round(self.sample_rate * _QUIET_FRAME_SECONDS)))
        self.window = max(1, int(round(RUNAWAY_WINDOW_SECONDS / _QUIET_FRAME_SECONDS)))
        self._frames: collections.deque = collections.deque()
        self._quiet = 0
        self._heard = False
        self._pending = np.zeros(0, dtype=np.float32)  # the start of a frame not yet complete
        self.seen = 0  # samples pushed so far
        self.quiet_from: Optional[int] = None

    def push(self, chunk: np.ndarray) -> Optional[int]:
        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
        chunk_start = self.seen
        self.seen += mono.size
        joined = np.concatenate([self._pending, mono]) if self._pending.size else mono
        joined_start = chunk_start - self._pending.size
        n = joined.size // self.hop
        self._pending = joined[n * self.hop:].copy()
        if not n:
            return None
        frames = joined[: n * self.hop].astype(np.float64).reshape(n, self.hop)
        quiet = np.sqrt(np.mean(frames * frames, axis=1)) < _QUIET_FRAME_RMS
        for k, is_quiet in enumerate(quiet):
            if not is_quiet:
                self._heard = True
            if not self._heard:
                continue
            self._frames.append(bool(is_quiet))
            self._quiet += int(is_quiet)
            if len(self._frames) > self.window:
                self._quiet -= int(self._frames.popleft())
            if len(self._frames) == self.window and self._quiet >= RUNAWAY_QUIET_FRACTION * self.window:
                window_end = joined_start + (k + 1) * self.hop
                self.quiet_from = window_end - self.window * self.hop
                return max(0, self.quiet_from - chunk_start)
        return None


class Qwen3TTSRuntime:
    """Offline-first loader + synthesis entry points for one Qwen3-TTS snapshot."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: Optional[str] = None,
        device: str = "auto",
        dtype_name: Optional[str] = None,
        allow_downloads: bool = True,
        debug: bool = False,
        settings: Optional[Qwen3TTSSettings] = None,
        backend: Optional[str] = None,
    ) -> None:
        self.model_id = str(model_id or DEFAULT_MODEL_ID)
        self.revision = revision
        self.allow_downloads = bool(allow_downloads)
        self.debug = bool(debug)
        # A private copy: `backend=` below must not change a settings object the caller shares.
        self.settings = copy.copy(settings) if settings is not None else Qwen3TTSSettings()
        if backend is not None:
            # Validated now (a typo fails before any weights load).
            self.settings.backend = normalize_backend(backend)

        self._device_pref = str(device or "auto")
        # The published checkpoints are BF16; fp16 overflows the codec's RVQ
        # epsilon division and the talker's logits. bf16 on accelerators,
        # fp32 on CPU, unless the caller pins something else. (ADR 0005: the
        # shared helpers resolve the device; this sets the *default* dtype.)
        self._dtype_pref = dtype_name

        self._model = None  # orchestration.Qwen3TTSModel (PyTorch backend)
        self._mlx = None  # mlx.model.Qwen3TTSMLXModel (MLX backend)
        self._backend: Optional[str] = None  # resolved at load
        self._resolved_device: Optional[str] = None
        self._resolved_dtype: Optional[str] = None
        self._resolved_dtype_reason: Optional[str] = None  # the shared policy's why (PyTorch backend)
        self._predictor_mode: Optional[str] = None
        self._sampler: Optional[str] = None
        self._lock = threading.Lock()

    # One generation owns the model at a time: the MLX codec decoder keeps its
    # streaming buffers on shared layers, so two generations must never
    # interleave, and an MLX stream holds the lock until it is exhausted or
    # closed. Waiting is therefore bounded: a call that cannot get the model in
    # that time is told why, instead of hanging behind a stream that a consumer
    # stopped reading without closing.
    LOCK_TIMEOUT_S = 600.0
    UNLOAD_LOCK_TIMEOUT_S = 10.0

    @contextlib.contextmanager
    def _owned(self, *, timeout: Optional[float] = None, action: str = "synthesis"):
        wait = self.LOCK_TIMEOUT_S if timeout is None else float(timeout)
        if not self._lock.acquire(timeout=wait):
            raise RuntimeError(
                f"Qwen3-TTS {action} timed out after {wait:.0f} s waiting for the model: another generation "
                "holds it. A streamed reply keeps the model until its chunk iterator is exhausted or closed "
                "(call .close() on an iterator you stop reading)."
            )
        try:
            yield
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ backend

    def resolve_backend(self) -> str:
        """``"mlx"`` | ``"torch"`` per the policy (ADR 0008); raises when a requested backend cannot run."""
        if self._backend is not None:
            return self._backend
        return select_backend(self.settings.backend)

    @property
    def backend(self) -> Optional[str]:
        """The backend the loaded model runs on (None before loading)."""
        return self._backend

    @property
    def supports_native_streaming(self) -> bool:
        """True when ``stream_*`` yields codec-frame chunks from ONE generation (MLX backend)."""
        return self.resolve_backend() == "mlx"

    # ------------------------------------------------------------------ loading

    def snapshot_dir(self, *, allow_downloads: Optional[bool] = None) -> str:
        """Resolve the local snapshot directory, offline-first.

        A local directory path is accepted as-is. Otherwise the Hugging Face
        cache is consulted; downloads happen only when downloads are allowed
        AND the caller is loading — introspection paths (``peek_config``,
        ``speaker_names``) always pass ``allow_downloads=False``, because
        listing voices must never fetch 2.5 GB as a side effect.
        """
        candidate = os.path.expanduser(self.model_id)
        if os.path.isdir(candidate):
            return candidate

        effective = self.allow_downloads if allow_downloads is None else bool(allow_downloads)

        try:
            from huggingface_hub import snapshot_download
        except Exception as e:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required for Qwen3-TTS model management") from e

        try:
            return str(
                snapshot_download(
                    repo_id=self.model_id,
                    revision=self.revision,
                    local_files_only=not effective,
                )
            )
        except Exception as e:
            if not effective:
                raise RuntimeError(
                    "Qwen3-TTS weights are not available locally and downloads are disabled.\n"
                    "Fix options:\n"
                    "  - Enable downloads: VoiceManager(..., allow_downloads=True)\n"
                    "  - Or prefetch explicitly: abstractvoice-prefetch --qwen3-tts\n"
                    f"Model: {self.model_id}"
                ) from e
            raise

    def _resolve_runtime(self):
        # The shared policy picks the default dtype (ADR 0005): float32 on the CPU and on
        # NVIDIA cards without native bfloat16 (before Ampere), bfloat16 on newer ones. A float16
        # answer (its MPS/XPU default, or ABSTRACTVOICE_TORCH_DTYPE) becomes bfloat16: fp16
        # overflows this model (see __init__); only an explicit dtype_name pins float16.
        allow_cpu_fallback = self._device_pref == "auto"
        runtime = resolve_torch_runtime(
            device=self._device_pref,
            dtype_name=self._dtype_pref,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        pinned = str(self._dtype_pref or "").strip().lower() not in ("", "auto")
        reason = runtime.dtype_reason
        if not pinned and runtime.resolved_dtype_name == "float16":
            runtime = resolve_torch_runtime(
                device=self._device_pref,
                dtype_name="bfloat16",
                allow_cpu_fallback=allow_cpu_fallback,
            )
            reason = f"float16 overflows Qwen3-TTS, so bfloat16 (the policy chose float16: {reason})"
        self._resolved_device = str(runtime.resolved_device)
        self._resolved_dtype = str(runtime.resolved_dtype_name)
        self._resolved_dtype_reason = reason
        return runtime

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._owned():
            if self.is_loaded:
                return
            # The policy decides once, before any import of torch or mlx.
            backend = select_backend(self.settings.backend)
            if backend == "mlx":
                self._load_mlx()
            else:
                self._load_torch()

    def _load_torch(self) -> None:
        import torch

        from ..transformers_floor import QWEN3_TTS_MIN_TRANSFORMERS, require_transformers

        # Re-validated here: the fields are plain attributes and may have
        # been reassigned after construction.
        predictor_mode = normalize_qwen3_tts_predictor(self.settings.predictor)
        sampler = normalize_qwen3_tts_sampler(self.settings.sampler)
        # Before any download or weight load: on an older Transformers the
        # vendored model fails mid-generation instead.
        require_transformers(QWEN3_TTS_MIN_TRANSFORMERS, model="Qwen3-TTS", extra="qwen3-tts")

        from .orchestration import Qwen3TTSModel
        local_dir = self.snapshot_dir()
        runtime = self._resolve_runtime()

        model = Qwen3TTSModel.from_pretrained(local_dir, dtype=runtime.torch_dtype)
        model.model.to(runtime.resolved_device)
        model.model.eval()
        model.model.talker.code_predictor._abstractvoice_predictor = predictor_mode
        model.model.talker.code_predictor._abstractvoice_sampler = sampler
        self._predictor_mode = predictor_mode
        self._sampler = sampler
        # The codec pinned fp32 at load (see codec.py); move it to the device.
        codec = model.model.speech_tokenizer
        codec.model = codec.model.to(device=runtime.resolved_device)
        codec.device = model.model.device
        model.device = model.model.device
        self._model = model
        self._backend = "torch"

    def _load_mlx(self) -> None:
        # Validated here too: the fields may have been reassigned after construction.
        normalize_qwen3_tts_predictor(self.settings.predictor)
        normalize_qwen3_tts_sampler(self.settings.sampler)
        # The environment is read here, when the MLX backend actually loads, so a
        # stray quantization variable never affects the PyTorch backend.
        quantization = resolve_mlx_quantization(self.settings.mlx_quantization)
        require_mlx_version()

        from .mlx.model import Qwen3TTSMLXModel

        local_dir = self.snapshot_dir()
        model = Qwen3TTSMLXModel.from_snapshot(local_dir, quantization=quantization)
        # Metal kernels compile on first use: pay for it in the loading phase,
        # like every other local engine's warm-up, not on the first utterance.
        model.report.prime_seconds = model.prime()
        self._mlx = model
        self._backend = "mlx"
        self._resolved_device = "metal"
        self._resolved_dtype = "bfloat16" if quantization == "bf16" else f"bfloat16+{quantization}"
        # PyTorch-only switches: recorded as requested, reported as not applied.
        self._predictor_mode = "mlx"
        self._sampler = "multinomial"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._mlx is not None

    def _model_or_raise(self):
        """The loaded PyTorch model, read under the lock the caller already holds.

        A concurrent ``unload()`` between ``_ensure_loaded`` and the lock leaves
        ``_model`` None; that is a caller-visible lifecycle conflict, not a crash.
        """
        model = self._model
        if model is None:
            raise RuntimeError("Qwen3-TTS runtime was unloaded while a synthesis was starting; retry.")
        return model

    def _mlx_or_raise(self):
        model = self._mlx
        if model is None:
            raise RuntimeError("Qwen3-TTS runtime was unloaded while a synthesis was starting; retry.")
        return model

    def unload(self) -> None:
        with self._owned(timeout=self.UNLOAD_LOCK_TIMEOUT_S, action="unload"):
            self._model = None
            mlx_model, self._mlx = self._mlx, None
            self._backend = None
        try:
            import gc

            if mlx_model is not None:
                mlx_model.unload()
            gc.collect()
            if mlx_model is None:
                import torch

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

    # ----------------------------------------------------------- introspection

    def model_type(self) -> str:
        """"custom_voice" | "base" | "voice_design", from the loaded model or its config."""
        if self._mlx is not None:
            return str(self._mlx.model_type)
        if self._model is not None:
            return str(self._model.model.tts_model_type)
        return str(self.peek_config().get("tts_model_type") or "")

    def peek_config(self) -> dict:
        """Read config.json from the LOCAL snapshot without loading weights.

        Never downloads: introspection is a discovery operation.
        """
        import json

        from ..local_models import hf_cached_snapshot_dir

        snapshot = hf_cached_snapshot_dir(self.model_id)
        if snapshot is None:
            # Fall back to the resolver for local checkpoint dirs and clearer errors.
            snapshot = self.snapshot_dir(allow_downloads=False)
        path = os.path.join(str(snapshot), "config.json")
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def speaker_names(self) -> list[str]:
        """Preset speaker names, weight-free (from config's talker spk_id).

        Loaded and peek paths normalize identically (sorted, lowercase): the
        listing must not change shape because weights happen to be resident.
        """
        if self._mlx is not None:
            names = self._mlx.supported_speakers()
        elif self._model is not None:
            names = self._model.get_supported_speakers() or []
        else:
            cfg = self.peek_config()
            talker = cfg.get("talker_config") or {}
            names = list(talker.get("spk_id") or {})
        return sorted(str(name).lower() for name in names)

    def language_names(self) -> list[str]:
        """Supported language names; same loaded/peek coherence as speakers.

        The peek path replicates the loaded model's own rule (``["auto"]`` plus
        the non-dialect ``codec_language_id`` keys) rather than dumping raw
        config keys: the raw keys include dialect ids the model itself REJECTS
        at synthesis time, and omit ``auto`` -- a listing must not change set
        membership depending on whether weights happen to be resident.
        """
        if self._mlx is not None:
            names = list(self._mlx.supported_languages())
        elif self._model is not None:
            names = list(self._model.get_supported_languages() or [])
        else:
            cfg = self.peek_config()
            talker = cfg.get("talker_config") or {}
            names = ["auto"] + [
                name for name in (talker.get("codec_language_id") or {}) if "dialect" not in str(name)
            ]
        return sorted(str(name).lower() for name in names)

    def runtime_info(self) -> dict:
        sampler = "multinomial" if self._predictor_mode == "reference" else self._sampler
        # PyTorch on CUDA: "captured" once the codebook frame replays a CUDA graph
        # (after the first generation), "failed: <reason>" when it fell back to eager.
        cuda_graphs = None
        if self._backend == "torch" and self._model is not None:
            from .predictor import cuda_graph_state

            cuda_graphs = cuda_graph_state(self._model.model.talker.code_predictor)
            if cuda_graphs == "captured" and sampler == "multinomial" and self._predictor_mode != "reference":
                # Same IDs and RNG use as torch.multinomial; validated per frame, not per step.
                sampler = "multinomial (cuda graph: exponential race, per-frame validation)"
        info = {
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "backend": self._backend,
            "backend_requested": self.settings.backend,
            "device": self._resolved_device,
            "dtype": self._resolved_dtype,
            "dtype_reason": self._resolved_dtype_reason,
            "codec_dtype": "float32",
            "model_type": self.model_type() if self.is_loaded else None,
            "quality_preset": self.settings.quality_preset,
            "predictor_mode": self._predictor_mode,
            "sampler": sampler,
            "cuda_graphs": cuda_graphs,
        }
        if self._backend is None:
            info["backend_selection"] = describe_backend_selection(self.settings.backend)
        if self._backend == "mlx":
            report = self._mlx.report if self._mlx is not None else None
            info.update(
                {
                    "mlx_quantization": report.quantization if report is not None else self.settings.mlx_quantization,
                    "mlx_weight_bytes": (
                        {
                            "talker": int(getattr(report, "talker_bytes", 0)),
                            "codec": int(getattr(report, "codec_bytes", 0)),
                            "speaker_encoder": int(getattr(report, "speaker_encoder_bytes", 0)),
                        }
                        if report is not None
                        else None
                    ),
                    "native_streaming": True,
                    "stream_first_chunk_frames": int(self.settings.stream_first_chunk_frames),
                    "stream_chunk_frames": int(self.settings.stream_chunk_frames),
                    # The PyTorch codebook switches do not exist on MLX: say so, and keep the requested values visible.
                    "pytorch_switches": {
                        "predictor": self.settings.predictor,
                        "sampler": self.settings.sampler,
                        "applied": False,
                        "note": "MLX runs its own fixed-depth codebook loop with categorical (multinomial) sampling",
                    },
                }
            )
            if report is not None:
                info["load_seconds"] = {k: round(float(v), 3) for k, v in report.seconds.items()}
                info["prime_seconds"] = None if report.prime_seconds is None else round(float(report.prime_seconds), 3)
                info["quantized_linear_layers"] = int(report.quantized_linear_layers)
        return info

    # ----------------------------------------------------------- MLX helpers

    def _sampling_params(self, overrides: dict):
        from .mlx.model import SamplingParams

        kwargs = self.settings.generate_kwargs()
        kwargs.update({k: v for k, v in overrides.items() if k != "max_new_tokens"})
        return SamplingParams.from_generate_kwargs(kwargs)

    def _mlx_stream(
        self, text: str, build_prompt, overrides: dict, report: Optional[dict] = None
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """One generation, yielded as contiguous chunks faded only at the utterance edges.

        The lock is held for the whole generation (the MLX model has one
        decoder state); a consumer that stops iterating releases it through
        the generator's ``finally``.
        """
        max_new_tokens = int(overrides.get("max_new_tokens") or estimate_max_new_tokens(text))
        sampling = self._sampling_params(overrides)
        with self._owned():
            model = self._mlx_or_raise()
            prompt = build_prompt(model)
            sr = int(model.sample_rate)
            fader = _EdgeFader(sample_rate=sr)
            watch = _QuietRunaway(sr)
            runaway = False
            delivered = 0  # samples handed on (a runaway's cut tail is not)
            frames = model.stream(
                prompt,
                sampling,
                max_new_tokens=max_new_tokens,
                chunk_frames=int(self.settings.stream_chunk_frames),
                first_chunk_frames=int(self.settings.stream_first_chunk_frames),
            )
            try:
                for chunk in frames:
                    keep = watch.push(chunk)
                    if keep is not None:
                        runaway = True
                        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)[:keep]
                    delivered += int(np.asarray(chunk).size)
                    out = fader.push(chunk)
                    if out is not None:
                        yield out, sr
                    if runaway:
                        break
            finally:
                frames.close()  # ends the generation (and frees the decoder) before the lock is released
            tail = fader.flush()
            if tail is not None:
                yield tail, sr
            end = self._generation_end(text, watch, sr, max_new_tokens, runaway=runaway, delivered=delivered)
            if report is not None:
                report.update(end)

    @staticmethod
    def _generation_end(
        text: str, watch: "_QuietRunaway", sr: int, max_new_tokens: int, *, runaway: bool, delivered: int
    ) -> dict:
        """How a completed generation ended: ``{"ended": "eos" | "quiet_runaway" | "cap", "audio_s"`` (the
        audio delivered) ``, "max_new_tokens"}``, plus ``"quiet_from_s"`` for a runaway ended at its quiet window."""
        audio_s = delivered / float(sr)
        end = {"ended": "eos", "audio_s": round(audio_s, 2), "max_new_tokens": int(max_new_tokens)}
        if runaway:
            end["ended"] = "quiet_runaway"
            end["quiet_from_s"] = round(float(watch.quiet_from or 0) / float(sr), 2)
            logger.warning(
                "Qwen3-TTS: a generation of %d characters missed its end of speech and went quiet at %.1f s; "
                "ended there instead of at its %.1f s cap, the reply goes on",
                len(str(text)), end["quiet_from_s"], max_new_tokens / _CODES_PER_SECOND,
            )
        elif watch.seen >= int(max_new_tokens * sr / _CODES_PER_SECOND):
            end["ended"] = "cap"
            logger.warning(
                "Qwen3-TTS: a generation of %d characters missed its end of speech and ran to its cap (%.1f s)",
                len(str(text)), audio_s,
            )
        return end

    def _finish_generation(
        self, audio: np.ndarray, sample_rate: int, text: str, max_new_tokens: Any, report: Optional[dict] = None
    ) -> tuple[np.ndarray, int]:
        """A whole PyTorch generation, cut where it went quiet for good (``_QuietRunaway``), then de-clicked."""
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        watch = _QuietRunaway(int(sample_rate))
        keep = watch.push(mono)
        if keep is not None:
            mono = mono[:keep]
        cap = int(max_new_tokens or estimate_max_new_tokens(text))
        end = self._generation_end(text, watch, int(sample_rate), cap, runaway=keep is not None, delivered=mono.size)
        if report is not None:
            report.update(end)
        return _finish_utterance(mono, sample_rate)

    @staticmethod
    def _collect(chunks: Iterator[Tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
        parts = []
        sr = 24000
        for chunk, rate in chunks:
            parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr = int(rate)
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return audio, sr

    # -------------------------------------------------------------- synthesis

    def synthesize_custom_voice(
        self,
        text: str,
        *,
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance with a preset speaker (CustomVoice models)."""
        self._ensure_loaded()
        if self._backend == "mlx":
            # The stream already fades the utterance edges; one generation either way.
            return self._collect(self.stream_custom_voice(text, speaker=speaker, language=language, instruct=instruct, **overrides))
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._owned():
            wavs, sr = self._model_or_raise().generate_custom_voice(
                text=str(text),
                speaker=str(speaker),
                language=language or "Auto",
                instruct=instruct or None,
                **kwargs,
            )
        return self._finish_generation(wavs[0], sr, text, kwargs.get("max_new_tokens"))

    def stream_custom_voice(
        self,
        text: str,
        *,
        speaker: str,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        **overrides: Any,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Contiguous audio chunks of ONE generation (MLX: codec-frame cadence; torch: the whole utterance once)."""
        self._ensure_loaded()
        if self._backend != "mlx":
            yield self.synthesize_custom_voice(text, speaker=speaker, language=language, instruct=instruct, **overrides)
            return
        yield from self._mlx_stream(
            text,
            lambda model: model.custom_voice_prompt(
                str(text), speaker=str(speaker), language=(language or "auto"), instruct=instruct or None
            ),
            overrides,
        )

    def synthesize_voice_design(
        self,
        text: str,
        *,
        instruct: str,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance with a designed voice (VoiceDesign models).

        ``instruct`` is required: an empty description yields an arbitrary voice,
        which is a silent quality cliff rather than an error (ADR 0007 demands
        explicit degradation, so we refuse instead).
        """
        if not str(instruct or "").strip():
            raise ValueError(
                "Qwen3-TTS VoiceDesign requires a non-empty voice description "
                "(the `instructions` selector). Without one the voice is arbitrary."
            )
        self._ensure_loaded()
        if self._backend == "mlx":
            return self._collect(self.stream_voice_design(text, instruct=instruct, language=language, **overrides))
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._owned():
            wavs, sr = self._model_or_raise().generate_voice_design(
                text=str(text),
                instruct=str(instruct),
                language=language or "Auto",
                **kwargs,
            )
        return self._finish_generation(wavs[0], sr, text, kwargs.get("max_new_tokens"))

    def stream_voice_design(
        self,
        text: str,
        *,
        instruct: str,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        if not str(instruct or "").strip():
            raise ValueError(
                "Qwen3-TTS VoiceDesign requires a non-empty voice description "
                "(the `instructions` selector). Without one the voice is arbitrary."
            )
        self._ensure_loaded()
        if self._backend != "mlx":
            yield self.synthesize_voice_design(text, instruct=instruct, language=language, **overrides)
            return
        yield from self._mlx_stream(
            text,
            lambda model: model.voice_design_prompt(str(text), instruct=str(instruct), language=(language or "auto")),
            overrides,
        )

    def build_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: Optional[str],
        x_vector_only: bool,
    ):
        """Voice-clone prompt items from reference audio (Base models).

        The MODE IS THE CALLER'S DECISION (ADR 0003): ICL needs ``ref_text``,
        x-vector-only ignores it. This method never infers one from the other's
        absence — the cloning manager owns transcripts and their fallbacks.

        Returns a backend-specific opaque prompt; pass it back to
        ``synthesize_clone`` / ``stream_clone`` on the same runtime.
        """
        self._ensure_loaded()
        with self._owned():
            if self._backend == "mlx":
                audio, sr = _load_reference_audio(ref_audio)
                return self._mlx_or_raise().build_clone_prompt(audio, sr, ref_text=ref_text, x_vector_only=bool(x_vector_only))
            return self._model_or_raise().create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
            )

    # ------------------------------------------------- portable clone prompts
    #
    # A voice-clone prompt is four things on both backends: the reference
    # transcript, its 12Hz codec codes (frames x 16 codebooks), the speaker
    # embedding of the Base checkpoint that built it, and the mode. Exported as
    # plain arrays it is a few kilobytes and can be stored instead of reference
    # audio (the built-in voice pack does this), then rebuilt on either backend.

    def export_clone_prompt(self, clone_prompt: Any) -> dict:
        """Plain arrays for a prompt built by :meth:`build_clone_prompt` on this runtime.

        Returns ``{"ref_codes": int16 (T, 16) | None, "speaker_embedding": float32 (D,),
        "x_vector_only": bool, "ref_text": str | None}``.
        """
        item = clone_prompt[0] if isinstance(clone_prompt, (list, tuple)) else clone_prompt
        if hasattr(item, "ref_spk_embedding"):  # PyTorch VoiceClonePromptItem
            codes, emb = item.ref_code, item.ref_spk_embedding
            x_only = bool(item.x_vector_only_mode)
            to_np = lambda t: t.detach().to("cpu").float().numpy() if hasattr(t, "detach") else np.asarray(t)  # noqa: E731
        else:  # MLX MLXClonePrompt
            codes, emb = item.ref_codes, item.speaker_embedding
            x_only = bool(item.x_vector_only_mode)
            to_np = lambda t: np.array(t)  # noqa: E731
        ref_codes = None if (x_only or codes is None) else to_np(codes).astype(np.int16)
        if ref_codes is not None and (ref_codes.ndim != 2 or ref_codes.shape[1] != 16):
            raise ValueError(f"unexpected reference code shape {ref_codes.shape}; expected (frames, 16)")
        return {
            "ref_codes": ref_codes,
            "speaker_embedding": to_np(emb).astype(np.float32).reshape(-1),
            "x_vector_only": x_only,
            "ref_text": item.ref_text,
        }

    def speaker_similarity(self, audio: np.ndarray, *, clone_prompt: Any, sample_rate: int = 24000) -> float:
        """Cosine similarity between the speaker of ``audio`` and the voice ``clone_prompt`` was built for.

        Both embeddings come from the SAME encoder: the Base checkpoint's speaker
        encoder that built the prompt's speaker embedding. Used to check that a
        cloned generation actually carries the cloned voice (the 0.6B Base model
        occasionally samples a whole generation in another, stock-sounding voice
        from the same prompt).

        Deliberately lock-free: the speaker encoder is stateless and separate from
        the decoder state the lock protects, and it is called while a clone stream
        (which holds the lock) is suspended between chunks.
        """
        ref = np.asarray(self.export_clone_prompt(clone_prompt)["speaker_embedding"], dtype=np.float32).reshape(-1)
        wav = np.asarray(audio, dtype=np.float32).reshape(-1)
        if int(sample_rate) != 24000:
            from ..audio.resample import sinc_resample_mono

            wav = sinc_resample_mono(wav, int(sample_rate), 24000).astype(np.float32)
        if self._backend == "mlx":
            emb = np.array(self._mlx_or_raise().speaker_embedding(wav), dtype=np.float32).reshape(-1)
        else:
            model = self._model_or_raise()
            emb = model.model.extract_speaker_embedding(audio=wav, sr=24000).detach().to("cpu").float().numpy().reshape(-1)
        if emb.shape != ref.shape:
            raise ValueError(f"speaker embedding sizes differ ({emb.shape[0]} vs {ref.shape[0]}): the prompt was built by another checkpoint")
        denom = float(np.linalg.norm(emb) * np.linalg.norm(ref))
        return float(emb @ ref) / denom if denom > 0 else 0.0

    def clone_prompt_from_arrays(
        self,
        *,
        ref_codes: Optional[np.ndarray],
        speaker_embedding: np.ndarray,
        ref_text: Optional[str],
        x_vector_only: bool = False,
    ):
        """Rebuild a voice-clone prompt for THIS runtime's backend from :meth:`export_clone_prompt` arrays.

        The speaker embedding must come from the same Base checkpoint as this runtime
        (its size is checked against the loaded model).
        """
        if not x_vector_only and (ref_codes is None or not str(ref_text or "").strip()):
            raise ValueError("an ICL clone prompt needs both reference codes and the reference transcript")
        self._ensure_loaded()
        emb = np.asarray(speaker_embedding, dtype=np.float32).reshape(-1)
        codes = None if x_vector_only else np.asarray(ref_codes).astype(np.int32)
        if codes is not None and (codes.ndim != 2 or codes.shape[1] != 16):
            raise ValueError(f"reference codes must be (frames, 16); got {codes.shape}")
        if self._backend == "mlx":
            import mlx.core as mx

            from .mlx.model import MLXClonePrompt

            expected = int(self._mlx_or_raise().config.talker_config.hidden_size)
            if emb.shape[0] != expected:
                raise ValueError(
                    f"speaker embedding has {emb.shape[0]} values but {self.model_id} expects {expected}: "
                    "the prompt was built by another Base checkpoint"
                )
            return MLXClonePrompt(
                ref_codes=None if codes is None else mx.array(codes),
                speaker_embedding=mx.array(emb),
                x_vector_only_mode=bool(x_vector_only),
                icl_mode=not bool(x_vector_only),
                ref_text=ref_text,
            )
        import torch

        from .orchestration import VoiceClonePromptItem

        model = self._model_or_raise()
        expected = int(model.model.config.talker_config.hidden_size)
        if emb.shape[0] != expected:
            raise ValueError(
                f"speaker embedding has {emb.shape[0]} values but {self.model_id} expects {expected}: "
                "the prompt was built by another Base checkpoint"
            )
        device = model.model.device
        return [
            VoiceClonePromptItem(
                ref_code=None if codes is None else torch.as_tensor(codes, dtype=torch.long, device=device),
                ref_spk_embedding=torch.as_tensor(emb, device=device).to(model.model.dtype),
                x_vector_only_mode=bool(x_vector_only),
                icl_mode=not bool(x_vector_only),
                ref_text=ref_text,
            )
        ]

    def synthesize_clone(
        self,
        text: str,
        *,
        clone_prompt: Any,
        language: Optional[str] = None,
        report: Optional[dict] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance in a cloned voice, from a prebuilt prompt (Base models).

        ``report``, a dict, is filled with how the generation ended (``_generation_end``).
        """
        self._ensure_loaded()
        if self._backend == "mlx":
            return self._collect(
                self.stream_clone(text, clone_prompt=clone_prompt, language=language, report=report, **overrides)
            )
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._owned():
            wavs, sr = self._model_or_raise().generate_voice_clone(
                text=str(text),
                language=language or "Auto",
                voice_clone_prompt=clone_prompt,
                **kwargs,
            )
        return self._finish_generation(wavs[0], sr, text, kwargs.get("max_new_tokens"), report=report)

    def stream_clone(
        self,
        text: str,
        *,
        clone_prompt: Any,
        language: Optional[str] = None,
        report: Optional[dict] = None,
        **overrides: Any,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """``synthesize_clone`` as contiguous chunks (one chunk off the MLX backend); ``report`` likewise."""
        self._ensure_loaded()
        if self._backend != "mlx":
            yield self.synthesize_clone(text, clone_prompt=clone_prompt, language=language, report=report, **overrides)
            return
        from .mlx.model import MLXClonePrompt

        if not isinstance(clone_prompt, MLXClonePrompt):
            raise TypeError(
                "Qwen3-TTS clone prompt was built by another backend; rebuild it with build_clone_prompt() on this runtime."
            )
        yield from self._mlx_stream(
            text,
            lambda model: model.clone_prompt(str(text), clone=clone_prompt, language=(language or "auto")),
            overrides,
            report=report,
        )


def _load_reference_audio(ref_audio: Any) -> tuple[np.ndarray, int]:
    """Reference audio as (mono float32, sample rate): a local file path or an (array, sr) pair."""
    if isinstance(ref_audio, (list, tuple)) and len(ref_audio) == 1:
        ref_audio = ref_audio[0]
    if isinstance(ref_audio, str):
        import soundfile as sf

        audio, sr = sf.read(ref_audio, dtype="float32", always_2d=False)
    elif isinstance(ref_audio, tuple) and len(ref_audio) == 2 and isinstance(ref_audio[0], np.ndarray):
        audio, sr = ref_audio
    elif isinstance(ref_audio, np.ndarray):
        raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
    else:
        raise TypeError(f"Unsupported audio input type: {type(ref_audio)}")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio.astype(np.float32), int(sr)


def prefetch_qwen3_tts(*, model_id: str = DEFAULT_MODEL_ID, allow_downloads: bool = True) -> str:
    """Explicitly download a Qwen3-TTS snapshot (never called implicitly)."""
    if not allow_downloads:
        raise RuntimeError("prefetch_qwen3_tts requires allow_downloads=True")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Qwen3-TTS prefetch requires optional dependencies.\n"
            "Install with:\n"
            '  pip install "abstractvoice[qwen3-tts]"'
        ) from e
    return str(snapshot_download(str(model_id)))


__all__ = [
    "BACKEND_ENV",
    "DEFAULT_BASE_MODEL_ID",
    "DEFAULT_MODEL_ID",
    "KNOWN_MODEL_IDS",
    "MLX_QUANTIZATION_ENV",
    "NATIVE_STREAM_MAX_CHARS",
    "ONE_SHOT_MAX_CHARS",
    "one_shot_max_new_tokens",
    "QWEN3_TTS_PREDICTORS",
    "QWEN3_TTS_SAMPLERS",
    "Qwen3TTSRuntime",
    "Qwen3TTSSettings",
    "estimate_max_new_tokens",
    "normalize_qwen3_tts_predictor",
    "normalize_qwen3_tts_sampler",
    "prefetch_qwen3_tts",
]
