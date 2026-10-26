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
    select_backend,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
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
        self.settings = settings or Qwen3TTSSettings()
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
        self._predictor_mode: Optional[str] = None
        self._sampler: Optional[str] = None
        self._lock = threading.Lock()

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
        default_dtype = self._dtype_pref
        if not default_dtype:
            probe = resolve_torch_runtime(
                device=self._device_pref,
                dtype_name=None,
                allow_cpu_fallback=self._device_pref == "auto",
            )
            default_dtype = "float32" if str(probe.resolved_device) == "cpu" else "bfloat16"
        runtime = resolve_torch_runtime(
            device=self._device_pref,
            dtype_name=default_dtype,
            allow_cpu_fallback=self._device_pref == "auto",
        )
        self._resolved_device = str(runtime.resolved_device)
        self._resolved_dtype = str(runtime.resolved_dtype_name)
        return runtime

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return
        with self._lock:
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
        quantization = normalize_mlx_quantization(self.settings.mlx_quantization)

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
        with self._lock:
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
        info = {
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "backend": self._backend,
            "backend_requested": self.settings.backend,
            "device": self._resolved_device,
            "dtype": self._resolved_dtype,
            "codec_dtype": "float32",
            "model_type": self.model_type() if self.is_loaded else None,
            "quality_preset": self.settings.quality_preset,
            "predictor_mode": self._predictor_mode,
            "sampler": "multinomial" if self._predictor_mode == "reference" else self._sampler,
        }
        if self._backend is None:
            info["backend_selection"] = describe_backend_selection(self.settings.backend)
        if self._backend == "mlx":
            report = self._mlx.report if self._mlx is not None else None
            info.update(
                {
                    "mlx_quantization": report.quantization if report is not None else normalize_mlx_quantization(self.settings.mlx_quantization),
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

    def _mlx_stream(self, text: str, build_prompt, overrides: dict) -> Iterator[Tuple[np.ndarray, int]]:
        """One generation, yielded as contiguous chunks faded only at the utterance edges.

        The lock is held for the whole generation (the MLX model has one
        decoder state); a consumer that stops iterating releases it through
        the generator's ``finally``.
        """
        max_new_tokens = int(overrides.get("max_new_tokens") or estimate_max_new_tokens(text))
        sampling = self._sampling_params(overrides)
        with self._lock:
            model = self._mlx_or_raise()
            prompt = build_prompt(model)
            fader = _EdgeFader(sample_rate=int(model.sample_rate))
            for chunk in model.stream(
                prompt,
                sampling,
                max_new_tokens=max_new_tokens,
                chunk_frames=int(self.settings.stream_chunk_frames),
                first_chunk_frames=int(self.settings.stream_first_chunk_frames),
            ):
                out = fader.push(chunk)
                if out is not None:
                    yield out, int(model.sample_rate)
            tail = fader.flush()
            if tail is not None:
                yield tail, int(model.sample_rate)

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
        with self._lock:
            wavs, sr = self._model_or_raise().generate_custom_voice(
                text=str(text),
                speaker=str(speaker),
                language=language or "Auto",
                instruct=instruct or None,
                **kwargs,
            )
        return _finish_utterance(wavs[0], sr)

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
        with self._lock:
            wavs, sr = self._model_or_raise().generate_voice_design(
                text=str(text),
                instruct=str(instruct),
                language=language or "Auto",
                **kwargs,
            )
        return _finish_utterance(wavs[0], sr)

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
        with self._lock:
            if self._backend == "mlx":
                audio, sr = _load_reference_audio(ref_audio)
                return self._mlx_or_raise().build_clone_prompt(audio, sr, ref_text=ref_text, x_vector_only=bool(x_vector_only))
            return self._model_or_raise().create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=bool(x_vector_only),
            )

    def synthesize_clone(
        self,
        text: str,
        *,
        clone_prompt: Any,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> tuple[np.ndarray, int]:
        """One utterance in a cloned voice, from a prebuilt prompt (Base models)."""
        self._ensure_loaded()
        if self._backend == "mlx":
            return self._collect(self.stream_clone(text, clone_prompt=clone_prompt, language=language, **overrides))
        kwargs = self.settings.generate_kwargs()
        kwargs["max_new_tokens"] = estimate_max_new_tokens(text)
        kwargs.update(overrides)
        with self._lock:
            wavs, sr = self._model_or_raise().generate_voice_clone(
                text=str(text),
                language=language or "Auto",
                voice_clone_prompt=clone_prompt,
                **kwargs,
            )
        return _finish_utterance(wavs[0], sr)

    def stream_clone(
        self,
        text: str,
        *,
        clone_prompt: Any,
        language: Optional[str] = None,
        **overrides: Any,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        self._ensure_loaded()
        if self._backend != "mlx":
            yield self.synthesize_clone(text, clone_prompt=clone_prompt, language=language, **overrides)
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
    "QWEN3_TTS_PREDICTORS",
    "QWEN3_TTS_SAMPLERS",
    "Qwen3TTSRuntime",
    "Qwen3TTSSettings",
    "estimate_max_new_tokens",
    "normalize_qwen3_tts_predictor",
    "normalize_qwen3_tts_sampler",
    "prefetch_qwen3_tts",
]
