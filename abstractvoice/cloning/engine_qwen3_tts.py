"""Qwen3-TTS voice cloning engine (12Hz Base checkpoints).

Managed path (VoiceCloner) is ICL cloning: reference audio + reference text.
The manager owns the transcript per ADR 0003 — user-provided text wins, a
missing one is auto-filled once via STT and persisted, and this engine fails
clearly rather than silently degrading when it is absent. The lower-quality
x-vector-only mode exists on the runtime API for callers who explicitly choose
it; it is never inferred from a missing transcript.
"""

from __future__ import annotations

import io
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf

from ..qwen3_tts.runtime import DEFAULT_BASE_MODEL_ID

class Qwen3TTSVoiceCloningEngine:
    """Cloning over :class:`abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime`."""

    engine_id = "qwen3-tts"

    DEFAULT_BASE_MODEL_ID = DEFAULT_BASE_MODEL_ID

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        allow_downloads: bool = True,
        model_id: str | None = None,
        predictor: str | None = None,
        sampler: str | None = None,
        backend: str | None = None,
        mlx_quantization: str | None = None,
    ) -> None:
        from ..qwen3_tts.runtime import Qwen3TTSSettings

        self.debug = bool(debug)
        # Validated now, so a typo fails before any weights load.
        self._settings = Qwen3TTSSettings(
            predictor=predictor, sampler=sampler, backend=backend, mlx_quantization=mlx_quantization
        )
        self._device = str(device or "auto")
        self._allow_downloads = bool(allow_downloads)
        self._model_id = str(model_id or self.DEFAULT_BASE_MODEL_ID)
        self._runtime = None
        # One prompt per (reference audio, reference text): reference encoding
        # costs a codec encode + speaker-embedding pass, and re-paying it per
        # text chunk would dominate chunked synthesis.
        self._prompt_cache_key: Optional[tuple] = None
        self._prompt_cache_value: Any = None

    # -------------------------------------------------------------- lifecycle

    def _get_runtime(self):
        if self._runtime is None:
            from ..qwen3_tts.runtime import Qwen3TTSRuntime

            self._runtime = Qwen3TTSRuntime(
                model_id=self._model_id,
                device=self._device,
                allow_downloads=self._allow_downloads,
                debug=self.debug,
                settings=self._settings,
            )
        return self._runtime

    def preload(self) -> dict:
        runtime = self._get_runtime()
        runtime._ensure_loaded()
        return dict(runtime.runtime_info())

    def unload(self) -> bool:
        if self._runtime is not None:
            self._runtime.unload()
        self._prompt_cache_key = None
        self._prompt_cache_value = None
        return True

    def is_loaded(self) -> bool:
        return bool(self._runtime is not None and self._runtime.is_loaded)

    def runtime_info(self) -> dict:
        try:
            return dict(self._get_runtime().runtime_info())
        except Exception:
            return {"model_id": self._model_id, "loaded": False}

    def set_quality_preset(self, preset: str) -> None:
        try:
            self._get_runtime().settings.apply_quality_preset(str(preset))
        except Exception:
            pass

    # -------------------------------------------------------------- inference

    def _clone_prompt(self, ref_path: str, reference_text: str):
        import os

        try:
            mtime = os.path.getmtime(ref_path)
        except OSError:
            mtime = None
        key = (str(ref_path), str(reference_text), mtime)
        if self._prompt_cache_key == key and self._prompt_cache_value is not None:
            return self._prompt_cache_value
        runtime = self._get_runtime()
        if str(ref_path).lower().endswith(".npz"):
            # A stored clone prompt (the built-in voice pack): codes + speaker embedding,
            # rebuilt for this runtime's backend instead of re-encoding reference audio.
            with np.load(str(ref_path)) as arrays:
                prompt = runtime.clone_prompt_from_arrays(
                    ref_codes=arrays["ref_codes"],
                    speaker_embedding=arrays["speaker_embedding"],
                    ref_text=str(reference_text),
                    x_vector_only=False,
                )
        else:
            prompt = runtime.build_clone_prompt(
                ref_audio=str(ref_path),
                ref_text=str(reference_text),
                x_vector_only=False,  # ICL; the manager guarantees the transcript.
            )
        self._prompt_cache_key = key
        self._prompt_cache_value = prompt
        return prompt

    def _one_shot(self, *, text: str, reference_paths: Iterable[str], reference_text: str, language: str | None):
        """The whole text as ONE clone generation (no segmentation, no truncation)."""
        from ..adapters.tts_qwen3_tts import _qwen_language
        from ..qwen3_tts.runtime import one_shot_max_new_tokens

        max_new_tokens = one_shot_max_new_tokens(str(text))
        if not reference_text or not str(reference_text).strip():
            raise RuntimeError("Missing reference_text for Qwen3-TTS cloning.")
        ref_paths = list(reference_paths or [])
        if len(ref_paths) != 1:
            raise RuntimeError("Qwen3-TTS cloning currently supports exactly one reference audio file.")
        runtime = self._get_runtime()
        prompt = self._clone_prompt(str(ref_paths[0]), str(reference_text))
        qwen_language = _qwen_language(language)
        audio, sr = runtime.synthesize_clone(
            str(text),
            clone_prompt=prompt,
            language=qwen_language if qwen_language != "Auto" else None,
            max_new_tokens=max_new_tokens,
        )
        yield np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)

    def infer_to_audio_chunks(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        max_chars: int = 200,
        language: str | None = None,
    ):
        """Chunked synthesis: early playback + per-utterance cancellation (ADR 0004)."""
        _ = speed  # Reported unsupported in the capability catalog.
        if not reference_text or not str(reference_text).strip():
            raise RuntimeError(
                "Missing reference_text for Qwen3-TTS cloning.\n"
                "If you're using VoiceCloner/VoiceManager, reference_text should be auto-generated and cached.\n"
                "If you're calling this engine directly, provide reference_text or set it via the voice store."
            )

        ref_paths = list(reference_paths or [])
        if len(ref_paths) != 1:
            raise RuntimeError(
                "Qwen3-TTS cloning currently supports exactly one reference audio file.\n"
                "Provide a single WAV/FLAC/OGG file when creating the voice clone."
            )

        from ..adapters.tts_qwen3_tts import _qwen_language
        from ..tts.text_chunking import split_text_batches

        runtime = self._get_runtime()
        prompt = self._clone_prompt(str(ref_paths[0]), str(reference_text))
        qwen_language = _qwen_language(language)
        language_arg = qwen_language if qwen_language != "Auto" else None

        # Native frame streaming (MLX backend, ADR 0008): first audio no longer
        # depends on the segment length, so the reply is ONE generation up to the
        # native bound and the cloned voice cannot drift between sentences.
        native = bool(getattr(runtime, "supports_native_streaming", False))
        if native:
            from ..qwen3_tts.runtime import NATIVE_STREAM_MAX_CHARS

            bound = max(int(max_chars), int(NATIVE_STREAM_MAX_CHARS))
        else:
            bound = int(max_chars)

        for chunk in split_text_batches(str(text), max_chars=bound):
            if not chunk.strip():
                continue
            if native:
                for audio, sr in runtime.stream_clone(chunk, clone_prompt=prompt, language=language_arg):
                    yield np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
                continue
            audio, sr = runtime.synthesize_clone(
                chunk,
                clone_prompt=prompt,
                language=language_arg,
            )
            yield np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)

    # `one_shot=True` on infer_to_wav_bytes: the whole text as ONE generation.
    supports_one_shot = True

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        language: str | None = None,
        one_shot: bool = False,
    ) -> bytes:
        chunks = []
        sr_out = 24000
        if one_shot:
            source = self._one_shot(
                text=text, reference_paths=reference_paths, reference_text=reference_text, language=language
            )
        else:
            source = self.infer_to_audio_chunks(
                text=text,
                reference_paths=reference_paths,
                reference_text=reference_text,
                speed=speed,
                language=language,
            )
        for chunk, sr in source:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr_out = int(sr)
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()
