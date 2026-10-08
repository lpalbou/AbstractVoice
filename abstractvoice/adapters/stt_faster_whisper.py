"""Faster-Whisper STT Adapter - High-performance speech recognition.

Faster-Whisper is a Whisper-family implementation using CTranslate2:
- Fast local inference
- 60% lower memory usage with INT8 quantization
- Whisper-family model accuracy
- Better CPU performance
- Supports GPU acceleration (CUDA) if available
"""

from __future__ import annotations

import sys
import os
import io
import logging
import numpy as np
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Dict, Any
import wave

from .base import STTAdapter

logger = logging.getLogger(__name__)


class FasterWhisperAdapter(STTAdapter):
    """Faster-Whisper STT adapter using faster-whisper package.
    
    This adapter provides local speech-to-text with CTranslate2-backed
    inference and low-memory quantized options.
    """

    ENGINE_ID = "faster-whisper"
    engine_id = ENGINE_ID
    provider = ENGINE_ID
    
    # Supported models (size -> (parameters, speed, accuracy))
    MODELS = {
        'tiny': {'params': '39M', 'speed': 'very_fast', 'accuracy': 'low'},
        'base': {'params': '74M', 'speed': 'fast', 'accuracy': 'good'},  # Default
        'small': {'params': '244M', 'speed': 'medium', 'accuracy': 'better'},
        'medium': {'params': '769M', 'speed': 'slow', 'accuracy': 'high'},
        'large-v2': {'params': '1550M', 'speed': 'very_slow', 'accuracy': 'best'},
        'large-v3': {'params': '1550M', 'speed': 'very_slow', 'accuracy': 'best'},
        # faster-whisper's own table maps it to mobiuslabsgmbh/faster-whisper-large-v3-turbo
        # (1.62 GB). Offered, never the default.
        'large-v3-turbo': {'params': '809M', 'speed': 'medium', 'accuracy': 'high'},
    }

    # Friendly aliases (kept intentionally small).
    # Many users expect Whisper-style "large" to work; in faster-whisper this
    # is typically best mapped to the latest `large-v3`.
    _MODEL_ALIASES = {
        "large": "large-v3",
        "turbo": "large-v3-turbo",
    }

    @classmethod
    def selectable_model_ids(cls) -> list[str]:
        """Return the model ids we want to expose in help/catalog surfaces."""
        return list(dict.fromkeys([*cls.MODELS.keys(), *cls._MODEL_ALIASES.keys()]))
    
    # Supported languages
    LANGUAGES = [
        'en', 'fr', 'de', 'es', 'ru', 'zh',  # Required 6
        'it', 'pt', 'ja', 'ko', 'ar', 'hi',  # Additional common languages
    ]
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        *,
        allow_downloads: bool = True,
    ):
        """Initialize Faster-Whisper STT adapter.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            device: Device to run on ('cpu', 'cuda', 'auto'). 'auto' (and an explicit 'cuda')
                go through `resolve_faster_whisper_device`: CUDA when it really works here,
                else the CPU with the reason recorded (`execution_device()`, `get_info()`).
            compute_type: Computation type ('int8', 'float16', 'float32', ...), or 'auto' for
                the best type the chosen device supports (int8_float16 on most NVIDIA GPUs,
                int8 on the CPU).
        """
        self.engine_id = self.ENGINE_ID
        self.provider = self.ENGINE_ID
        self.model_id = str(model_size or "base").strip() or "base"

        self._faster_whisper_available = False
        self._model = None
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._device_reason: Optional[str] = None
        self._device_refused = False
        self._current_language = None
        self._allow_downloads = bool(allow_downloads)
        
        # Try to import faster-whisper
        try:
            # Windows/Linux: make CUDA 12 cuBLAS from the NVIDIA wheels findable (no-op elsewhere).
            from ..compute.windows_cuda import prepare_windows_cuda_dlls

            prepare_windows_cuda_dlls()
            from faster_whisper import WhisperModel
            self._WhisperModel = WhisperModel
            self._faster_whisper_available = True
            logger.info("✅ Faster-Whisper initialized successfully")
            
            # Load model (best-effort). When allow_downloads=False we force offline mode
            # so we never trigger downloads in interactive contexts (e.g. REPL).
            self._load_model(model_size, device, compute_type)
            
        except ImportError as e:
            logger.warning(f"⚠️  Faster-Whisper not available: {e}")
            logger.info(
                "To install Faster-Whisper:\n"
                "  pip install faster-whisper>=0.10.0\n"
                "This will enable 4x faster STT with same accuracy."
            )
    
    def _load_model(self, model_size: str, device: str = "auto", compute_type: str = "int8") -> bool:
        """Load Faster-Whisper model.
        
        Args:
            model_size: Model size
            device: Device ('cpu', 'cuda', 'auto')
            compute_type: Computation type ('int8', 'float16', 'float32')
            
        Returns:
            True if successful, False otherwise
        """
        if not self._faster_whisper_available:
            return False

        # We accept:
        # - canonical short names in `MODELS` (tiny/base/small/medium/large-v2/large-v3)
        # - a small set of aliases (e.g. "large" -> "large-v3")
        # - arbitrary Hugging Face model ids (e.g. distil / turbo CT2 repos)
        # For unknown ids we skip metadata, but still try to load.
        raw = str(model_size or "").strip()
        if not raw:
            raw = "base"
        key = raw.strip().lower()
        if key in self._MODEL_ALIASES:
            model_size = str(self._MODEL_ALIASES[key])
        elif key in self.MODELS:
            model_size = key
        else:
            # Keep user-provided id as-is (could be HF repo id / local path).
            model_size = raw
        
        try:
            device, compute_type = self._resolve_device(device, compute_type)

            meta = self.MODELS.get(str(model_size).strip().lower())
            params = meta.get("params") if isinstance(meta, dict) else None
            params_txt = f" ({params})" if params else ""
            logger.info(f"⬇️  Loading Faster-Whisper model: {model_size}{params_txt} on {device}")

            # Load model (may auto-download if not cached).
            # When downloads are not allowed, force HF offline mode so we never pull
            # bytes from the network implicitly.
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
            old_disable_pb = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            if not self._allow_downloads:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            try:
                with warnings.catch_warnings():
                    # Avoid noisy HF Hub token warnings in offline-first flows.
                    warnings.filterwarnings(
                        "ignore",
                        message=r"^Warning: You are sending unauthenticated requests to the HF Hub\\..*",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=r"^You are sending unauthenticated requests to the HF Hub\\..*",
                    )
                    try:
                        self._model = self._construct(model_size, device, compute_type)
                    except Exception as exc:
                        if device != "cuda":
                            raise
                        # CUDA failed to load (a missing/mismatched CUDA library, out of GPU
                        # memory): run on the CPU and record why, never fail silently.
                        device, compute_type = self._fall_back_to_cpu(f"loading on CUDA failed ({exc})")
                        self._model = self._construct(model_size, device, compute_type)
            finally:
                if not self._allow_downloads:
                    if old_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = old_offline
                    if old_tf_offline is None:
                        os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    else:
                        os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
                    if old_disable_pb is None:
                        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                    else:
                        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = old_disable_pb
            
            self._model_size = model_size
            try:
                self.model_id = str(model_size)
            except Exception:
                pass
            self._device = device
            self._compute_type = compute_type
            
            logger.info(f"✅ Loaded Faster-Whisper model: {model_size}")
            return True
            
        except Exception as e:
            if self._allow_downloads:
                logger.error(f"❌ Failed to load Faster-Whisper model: {e}")
            else:
                # Offline mode: model might simply not be cached locally.
                logger.info(f"ℹ️ Faster-Whisper model '{model_size}' not available locally (offline mode).")
            return False

    def _construct(self, model_size: str, device: str, compute_type: str):
        return self._WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=None,  # Use default cache (~/.cache/huggingface)
            local_files_only=bool(not self._allow_downloads),
            use_auth_token=False if not self._allow_downloads else None,
        )

    def _resolve_device(self, device: str, compute_type: str) -> tuple[str, str]:
        """('auto'|'cuda'|'cpu', type|'auto') -> the device and compute type to load with.

        Records `_device_reason` (why not a GPU) and `_device_refused` (an explicit 'cuda'
        that cannot run here) from `resolve_faster_whisper_device`."""

        from ..compute.device import resolve_faster_whisper_device

        want = str(device or "auto").strip().lower() or "auto"
        if want == "cpu":
            choice = resolve_faster_whisper_device("cpu")
            self._device_reason = (
                "CTranslate2 (faster-whisper) has no Apple GPU backend, so Whisper runs on the processor; "
                "mlx-whisper runs it on the Apple GPU"
                if sys.platform == "darwin"
                else "the processor was requested"
            )
            self._device_refused = False
        else:
            choice = resolve_faster_whisper_device(None if want == "auto" else want)
            self._device_reason = choice.reason
            self._device_refused = bool(choice.refused)
        chosen_type = str(compute_type or "auto").strip().lower() or "auto"
        if chosen_type == "auto" or (choice.device != want and want != "auto"):
            # 'auto', or the requested device was refused: its compute type may not exist here.
            chosen_type = choice.compute_type
        elif choice.device == "cpu" and chosen_type in {"float16", "int8_float16", "bfloat16", "int8_bfloat16"}:
            chosen_type = choice.compute_type
        return choice.device, chosen_type

    def _fall_back_to_cpu(self, why: str) -> tuple[str, str]:
        from ..compute.device import resolve_faster_whisper_device

        cpu = resolve_faster_whisper_device("cpu")
        self._device_reason = f"{why}; running on the processor"
        logger.warning("faster-whisper: %s", self._device_reason)
        self._device, self._compute_type = cpu.device, cpu.compute_type
        return cpu.device, cpu.compute_type

    def _run(self, audio: Any, **kwargs: Any):
        """`WhisperModel.transcribe` with the segments materialised (decoding happens while
        they are iterated). A CUDA failure at run time (cuDNN/cuBLAS that loads lazily, GPU
        out of memory) reloads the model on the CPU, records why, and retries once."""

        try:
            segments, info = self._model.transcribe(audio, **kwargs)
            return list(segments), info
        except Exception as exc:
            if str(self._device) != "cuda":
                raise
            device, compute_type = self._fall_back_to_cpu(f"transcribing on CUDA failed ({exc})")
            self._model = self._construct(str(self._model_size), device, compute_type)
            if hasattr(audio, "seek"):
                audio.seek(0)
            segments, info = self._model.transcribe(audio, **kwargs)
            return list(segments), info

    def execution_device(self) -> Dict[str, Any]:
        """`{device, compute_type, reason, refused}`: where Whisper runs and why not on a GPU."""

        return {
            "device": self._device,
            "compute_type": self._compute_type,
            "reason": self._device_reason,
            "refused": bool(self._device_refused),
        }

    def unload(self) -> None:
        """Best-effort release of the loaded model to free memory."""
        try:
            self._model = None
        except Exception:
            pass
        try:
            import gc

            gc.collect()
        except Exception:
            pass
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        if not self.is_available():
            raise RuntimeError(
                "Faster-Whisper is not available. Install with: pip install faster-whisper>=0.10.0"
            )
        
        try:
            with warnings.catch_warnings():
                # faster-whisper can emit benign FP runtime warnings during mel extraction.
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")

                # Transcribe with faster-whisper
                segments, info = self._run(
                    audio_path,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    vad_filter=True,  # Use Voice Activity Detection
                    vad_parameters=dict(min_silence_duration_ms=500),
                    hotwords=hotwords,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=bool(condition_on_previous_text),
                )
            
            # Combine all segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            if language is None:
                logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ Faster-Whisper transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        *,
        hotwords: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        condition_on_previous_text: bool = True,
    ) -> str:
        """Transcribe audio from bytes (network use case).
        
        Args:
            audio_bytes: Audio data as bytes (WAV format)
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        # Save bytes to temporary file and transcribe
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            return self.transcribe(
                tmp_path,
                language=language,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
                condition_on_previous_text=bool(condition_on_previous_text),
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    
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
        """Transcribe audio from numpy array.
        
        Args:
            audio_array: Audio data as numpy array (float32, range -1.0 to 1.0)
            sample_rate: Sample rate of the audio in Hz
            language: Target language (optional, auto-detect if not provided)
            
        Returns:
            Transcribed text
        """
        # Fast path: pass float32 mono directly to faster-whisper (no temp files).
        # NOTE: faster-whisper expects 16kHz audio when passing an array. We resample
        # lightweightly if needed.
        if not self.is_available():
            raise RuntimeError(
                "Faster-Whisper is not available. Install with: pip install faster-whisper>=0.10.0"
            )
        try:
            import numpy as _np

            x = _np.asarray(audio_array, dtype=_np.float32).reshape(-1)
            sr = int(sample_rate)
            if sr != 16000:
                from ..audio.resample import linear_resample_mono

                x = linear_resample_mono(x, sr, 16000)
                sr = 16000

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul.*")
                # Mic / PTT transcription quality is sensitive to decoding settings.
                # Default to higher-quality beam search here, while allowing callers
                # (e.g. stop-phrase detection) to override for speed.
                beam = int(beam_size) if beam_size is not None else 5
                best = int(best_of) if best_of is not None else 5
                segments, info = self._run(
                    x,
                    language=language,
                    beam_size=beam,
                    best_of=best,
                    temperature=0.0,
                    vad_filter=False,
                    hotwords=hotwords,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=bool(condition_on_previous_text),
                    without_timestamps=True,
                )
            text = " ".join([segment.text.strip() for segment in segments])
            if language is None:
                logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            return text.strip()
        except Exception:
            # Fallback to file-based path for maximum compatibility.
            audio_bytes = self._array_to_wav_bytes(audio_array, sample_rate)
            return self.transcribe_from_bytes(
                audio_bytes,
                language=language,
                hotwords=hotwords,
                initial_prompt=initial_prompt,
                condition_on_previous_text=bool(condition_on_previous_text),
            )
    
    def _array_to_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes.
        
        Args:
            audio_array: Audio as float32 array [-1.0, 1.0]
            sample_rate: Sample rate in Hz
            
        Returns:
            WAV file as bytes
        """
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def set_language(self, language: str) -> bool:
        """Set the default language for transcription.
        
        Args:
            language: ISO 639-1 language code
            
        Returns:
            True if successful, False otherwise
        """
        if language not in self.LANGUAGES:
            logger.warning(f"⚠️  Language {language} may not be well-supported")
            return False
        
        self._current_language = language
        return True
    
    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return self.LANGUAGES.copy()
    
    def is_available(self) -> bool:
        """Check if Faster-Whisper is available and functional.
        
        Returns:
            True if the engine can be used, False otherwise
        """
        return self._faster_whisper_available and self._model is not None
    
    def change_model(self, model_size: str) -> bool:
        """Change the Whisper model size.
        
        Args:
            model_size: New model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            
        Returns:
            True if successful, False otherwise
        """
        if model_size == self._model_size:
            logger.debug(f"Model {model_size} already loaded")
            return True
        
        ok = self._load_model(model_size, self._device, self._compute_type)
        if ok:
            try:
                self.model_id = str(model_size)
            except Exception:
                pass
        return ok
    
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about Faster-Whisper engine.
        
        Returns:
            Dictionary with engine information
        """
        info = super().get_info()
        info.update({
            'engine': 'Faster-Whisper',
            'version': '1.2.0+',
            'model_size': self._model_size,
            'model_params': self.MODELS.get(self._model_size, {}).get('params', 'unknown'),
            'device': self._device,
            'compute_type': self._compute_type,
            'current_language': self._current_language,
            'performance': f"{self.MODELS.get(self._model_size, {}).get('speed', 'unknown')} speed, "
                          f"{self.MODELS.get(self._model_size, {}).get('accuracy', 'unknown')} accuracy",
            'memory_optimization': 'INT8 quantization' if self._compute_type == 'int8' else None
        })
        if self._device_reason:
            # Why Whisper is not on a GPU (no CUDA, a missing CUDA library, a refused request, a
            # CUDA failure that fell back to the CPU; on macOS: CTranslate2 has no Apple GPU
            # backend -- mlx-whisper is the Apple GPU engine).
            info['device_reason'] = self._device_reason
        if self._device_refused:
            info['device_refused'] = True
        return info
