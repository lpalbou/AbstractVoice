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


class Qwen3TTSVoiceCloningEngine:
    """Cloning over :class:`abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime`."""

    engine_id = "qwen3-tts"

    DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        allow_downloads: bool = True,
        model_id: str | None = None,
    ) -> None:
        self.debug = bool(debug)
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
        prompt = runtime.build_clone_prompt(
            ref_audio=str(ref_path),
            ref_text=str(reference_text),
            x_vector_only=False,  # ICL; the manager guarantees the transcript.
        )
        self._prompt_cache_key = key
        self._prompt_cache_value = prompt
        return prompt

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

        for chunk in split_text_batches(str(text), max_chars=int(max_chars)):
            if not chunk.strip():
                continue
            audio, sr = runtime.synthesize_clone(
                chunk,
                clone_prompt=prompt,
                language=qwen_language if qwen_language != "Auto" else None,
            )
            yield np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        language: str | None = None,
    ) -> bytes:
        chunks = []
        sr_out = 24000
        for chunk, sr in self.infer_to_audio_chunks(
            text=text,
            reference_paths=reference_paths,
            reference_text=reference_text,
            speed=speed,
            language=language,
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr_out = int(sr)
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()
