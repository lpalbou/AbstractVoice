"""mlx-whisper STT adapter: Whisper on the Apple GPU (MLX, Metal).

faster-whisper runs on CTranslate2, which has no Apple GPU backend (its devices
are ``cpu``/``cuda``; the macOS build links Accelerate only), so on Apple
Silicon Whisper ran on the processor: large-v3 took ~20 s for a 17 s clip on an
M5 Max. ``mlx-whisper`` (ml-explore) runs the same OpenAI Whisper weights,
converted to MLX by mlx-community, on the GPU: ~1.4 s for that clip with
large-v3, ~0.25 s with large-v3-turbo (measured 2026-10-08, round 16).

The model ids are faster-whisper's (``tiny`` ... ``large-v3``, plus
``large-v3-turbo``), mapped to mlx-community repos by :attr:`MODEL_REPOS`, so a
route can move between the two engines without changing its model. A Hugging
Face repo id or a local MLX checkpoint folder is also accepted as is.

THREADING (MLX constraint, measured): MLX binds its GPU stream to the thread
that created it; calling the model from another thread fails with
``RuntimeError: There is no Stream(gpu, N) in current thread``. Every MLX call
(load and transcribe) therefore runs on ONE dedicated worker thread shared by
all instances of this adapter (``_mlx_call``), whatever thread the caller is on.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .base import STTAdapter

logger = logging.getLogger(__name__)

_TARGET_SR = 16000
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()


def _mlx_call(fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on the single MLX worker thread and return its result (see module doc)."""

    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="abstractvoice-mlx-whisper")
        executor = _EXECUTOR
    return executor.submit(fn).result()


def apple_gpu_available() -> bool:
    """True on Apple Silicon macOS, the only place MLX runs on a GPU (no import)."""

    return sys.platform == "darwin" and os.uname().machine == "arm64"


def _mono_16k(audio: Any, sample_rate: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.reshape(-1)
    if int(sample_rate) != _TARGET_SR:
        from ..audio.resample import linear_resample_mono

        x = linear_resample_mono(x, int(sample_rate), _TARGET_SR)
    return x.astype(np.float32, copy=False)


def _decode_with_av(source: Any) -> np.ndarray:
    """Decode any container/codec PyAV reads (webm/opus, m4a, mp3, ...) to 16 kHz mono float32."""

    import av  # type: ignore

    chunks = []
    with av.open(source) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=_TARGET_SR)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _decode(source: Any) -> np.ndarray:
    """A path or a bytes buffer -> 16 kHz mono float32 (soundfile first, then PyAV)."""

    try:
        import soundfile as sf

        data, sr = sf.read(source, always_2d=True, dtype="float32")
        return _mono_16k(data, int(sr))
    except Exception:
        if isinstance(source, io.BytesIO):
            source.seek(0)
        return _decode_with_av(source)


class MLXWhisperAdapter(STTAdapter):
    """Whisper on the Apple GPU through ``mlx-whisper``."""

    ENGINE_ID = "mlx-whisper"
    engine_id = ENGINE_ID
    provider = ENGINE_ID

    # Engine model id -> mlx-community repo (fp16 MLX conversions of OpenAI Whisper).
    # Sizes (Hugging Face model API, 2026-10-08): large-v3 3.08 GB, large-v3-turbo 1.61 GB.
    MODEL_REPOS: Dict[str, str] = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    _MODEL_ALIASES: Dict[str, str] = {"large": "large-v3", "turbo": "large-v3-turbo"}
    DEFAULT_MODEL = "large-v3"

    LANGUAGES = ["en", "fr", "de", "es", "ru", "zh", "it", "pt", "ja", "ko", "ar", "hi"]

    @classmethod
    def selectable_model_ids(cls) -> list[str]:
        return list(dict.fromkeys([*cls.MODEL_REPOS.keys(), *cls._MODEL_ALIASES.keys()]))

    @classmethod
    def resolve_repo(cls, model_id: Any) -> str:
        """``large-v3`` -> ``mlx-community/whisper-large-v3-mlx``; a repo id or a path stays as given."""

        raw = str(model_id or "").strip() or cls.DEFAULT_MODEL
        key = cls._MODEL_ALIASES.get(raw.lower(), raw.lower())
        return cls.MODEL_REPOS.get(key, raw)

    def __init__(self, model_size: str = DEFAULT_MODEL, *, allow_downloads: bool = True, language: Optional[str] = None):
        self.engine_id = self.ENGINE_ID
        self.provider = self.ENGINE_ID
        self.model_id = str(model_size or self.DEFAULT_MODEL).strip() or self.DEFAULT_MODEL
        self._allow_downloads = bool(allow_downloads)
        self._current_language = language
        self._loaded_ref: Optional[str] = None  # model_id the loaded weights belong to
        self._loaded_path: Optional[str] = None
        self._unavailable_reason: Optional[str] = None
        self._mlx_whisper_available = False
        if not apple_gpu_available():
            self._unavailable_reason = "mlx-whisper runs on Apple Silicon only (MLX); use faster-whisper here"
            return
        try:
            import importlib.util

            if importlib.util.find_spec("mlx_whisper") is None:
                raise ImportError("mlx_whisper")
            self._mlx_whisper_available = True
        except Exception:
            self._unavailable_reason = (
                'mlx-whisper is not installed. Install it with: pip install "abstractvoice[stt-mlx]"'
            )
            return
        self._ensure_loaded()

    # -- loading -----------------------------------------------------------

    def _local_path(self, ref: str) -> str:
        repo = self.resolve_repo(ref)
        if Path(os.path.expanduser(repo)).is_dir():
            return str(Path(os.path.expanduser(repo)))
        from huggingface_hub import snapshot_download

        return str(snapshot_download(repo, local_files_only=not self._allow_downloads))

    def _ensure_loaded(self) -> bool:
        """Load the weights of ``self.model_id`` on the MLX thread (no-op when already loaded)."""

        if not self._mlx_whisper_available:
            return False
        ref = self.model_id
        if self._loaded_ref == ref and self._loaded_path:
            return True
        try:
            path = self._local_path(ref)

            def _load() -> None:
                import mlx.core as mx
                from mlx_whisper.transcribe import ModelHolder

                ModelHolder.get_model(path, mx.float16)

            _mlx_call(_load)
        except Exception as exc:
            if self._allow_downloads:
                logger.error("mlx-whisper could not load %s: %s", ref, exc)
                self._unavailable_reason = f"mlx-whisper could not load {ref!r}: {exc}"
            else:
                self._unavailable_reason = (
                    f"mlx-whisper model {ref!r} ({self.resolve_repo(ref)}) is not downloaded"
                )
            self._loaded_ref = self._loaded_path = None
            return False
        self._loaded_ref, self._loaded_path = ref, path
        self._unavailable_reason = None
        return True

    def unload(self) -> None:
        def _drop() -> None:
            from mlx_whisper.transcribe import ModelHolder

            ModelHolder.model = None
            ModelHolder.model_path = None
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass

        if self._loaded_path:
            try:
                _mlx_call(_drop)
            except Exception:
                pass
        self._loaded_ref = self._loaded_path = None

    # -- transcription -----------------------------------------------------

    def _transcribe_16k(
        self,
        audio: np.ndarray,
        language: Optional[str],
        *,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        if not self._ensure_loaded():
            raise RuntimeError(self._unavailable_reason or "mlx-whisper is not available")
        path = str(self._loaded_path)
        lang = language or self._current_language

        def _run() -> str:
            import mlx_whisper

            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=path,
                language=lang,
                initial_prompt=initial_prompt,
                condition_on_previous_text=bool(condition_on_previous_text),
                verbose=None,
            )
            return str(result.get("text") or "").strip()

        try:
            return _mlx_call(_run)
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        _ = hotwords  # faster-whisper only
        return self._transcribe_16k(
            _decode(str(audio_path)),
            language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        _ = hotwords
        return self._transcribe_16k(
            _decode(io.BytesIO(bytes(audio_bytes))),
            language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

    def transcribe_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
        beam_size: int | None = None,
        best_of: int | None = None,
    ) -> str:
        _ = hotwords, beam_size, best_of  # mlx-whisper decodes greedily (no beam search)
        return self._transcribe_16k(
            _mono_16k(audio_array, int(sample_rate)),
            language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

    # -- metadata ----------------------------------------------------------

    def set_language(self, language: str) -> bool:
        text = str(language or "").strip().lower()
        if not text:
            return False
        self._current_language = text
        return True

    def get_supported_languages(self) -> list[str]:
        return list(self.LANGUAGES)

    def is_available(self) -> bool:
        return self._mlx_whisper_available and self._loaded_path is not None

    def get_unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def change_model(self, model_size: str) -> bool:
        self.model_id = str(model_size or self.DEFAULT_MODEL).strip() or self.DEFAULT_MODEL
        return self._ensure_loaded()

    def execution_device(self) -> Dict[str, Any]:
        return {"device": "metal", "reason": "Apple GPU (MLX)"}

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "mlx-whisper",
                "engine_id": self.ENGINE_ID,
                "provider": self.ENGINE_ID,
                "model_id": self.model_id,
                "model_repo": self.resolve_repo(self.model_id),
                "device": "metal",
                "device_reason": "Apple GPU (MLX)",
                "compute_type": "float16",
                "current_language": self._current_language,
            }
        )
        if self._unavailable_reason:
            info["unavailable_reason"] = self._unavailable_reason
        return info
