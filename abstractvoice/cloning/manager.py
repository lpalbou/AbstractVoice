from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .engine_f5 import F5TTSVoiceCloningEngine
from .store import VoiceCloneStore


class VoiceCloner:
    """High-level voice cloning manager (optional).

    Stores reference bundles locally and uses an engine to synthesize speech.
    """

    def __init__(
        self,
        *,
        store: Optional[VoiceCloneStore] = None,
        debug: bool = False,
        whisper_model: str = "tiny",
        reference_text_whisper_model: str = "small",
        allow_downloads: bool = True,
        default_engine: str = "f5_tts",
    ):
        self.store = store or VoiceCloneStore()
        self.debug = debug
        self._whisper_model = whisper_model
        self._reference_text_whisper_model = reference_text_whisper_model
        self._allow_downloads = bool(allow_downloads)
        self._default_engine = str(default_engine or "f5_tts").strip().lower()
        self._engines: Dict[str, Any] = {}

    def _get_engine(self, engine: str) -> Any:
        name = str(engine or "").strip().lower()
        if not name:
            raise ValueError("engine must be a non-empty string")
        if name in self._engines:
            return self._engines[name]

        # Lazy-load engines to avoid surprise model downloads during list/store operations.
        if name == "f5_tts":
            inst = F5TTSVoiceCloningEngine(whisper_model=self._whisper_model, debug=self.debug)
        elif name == "chroma":
            from .engine_chroma import ChromaVoiceCloningEngine

            inst = ChromaVoiceCloningEngine(debug=self.debug, device="auto")
        else:
            raise ValueError(f"Unknown cloning engine: {name}")

        self._engines[name] = inst
        return inst

    def set_quality_preset(self, preset: str) -> None:
        # Best-effort across loaded engines (new engines are lazy-instantiated).
        for eng in list(self._engines.values()):
            try:
                eng.set_quality_preset(preset)
            except Exception:
                pass

    def get_runtime_info(self) -> Dict[str, Any]:
        # Keep backward compatibility: return a single flat dict.
        # Prefer F5 when available, otherwise return any loaded engine info.
        if "f5_tts" in self._engines:
            try:
                return dict(self._engines["f5_tts"].runtime_info())
            except Exception:
                return {}
        for eng in self._engines.values():
            try:
                return dict(eng.runtime_info())
            except Exception:
                continue
        return {}

    def clone_voice(
        self,
        reference_audio_path: str,
        name: str | None = None,
        *,
        reference_text: str | None = None,
        engine: str | None = None,
    ) -> str:
        """Create a new cloned voice from a file or directory.

        If a directory is provided, all WAV/FLAC/OGG files inside are used.
        """
        p = Path(reference_audio_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        supported = {".wav", ".flac", ".ogg"}

        engine_name = str(engine or self._default_engine).strip().lower()
        if engine_name not in ("f5_tts", "chroma"):
            raise ValueError("engine must be one of: f5_tts|chroma")

        if p.is_dir():
            refs = sorted([x for x in p.glob("*") if x.suffix.lower() in supported])
            if not refs:
                raise ValueError(f"No supported reference audio files found in: {p}")
        else:
            if p.suffix.lower() not in supported:
                raise ValueError(
                    f"Unsupported reference audio format: {p.suffix}. "
                    f"Provide WAV/FLAC/OGG (got: {p})."
                )
            refs = [p]

        if engine_name == "chroma" and len(refs) != 1:
            raise ValueError(
                "Chroma cloning currently supports exactly one reference audio file.\n"
                "Provide a single WAV/FLAC/OGG file (not a directory with multiple files)."
            )

        voice_id = self.store.create_voice(
            refs,
            name=name,
            reference_text=reference_text,
            engine=engine_name,
            meta={"source": str(p)},
        )
        return voice_id

    def list_cloned_voices(self) -> List[Dict[str, Any]]:
        return self.store.list_voices()

    def get_cloned_voice(self, voice_id: str) -> Dict[str, Any]:
        return self.store.get_voice_dict(voice_id)

    def export_voice(self, voice_id: str, path: str) -> str:
        return self.store.export_voice(voice_id, path)

    def import_voice(self, path: str) -> str:
        return self.store.import_voice(path)

    def rename_cloned_voice(self, voice_id: str, new_name: str) -> None:
        self.store.rename_voice(voice_id, new_name)

    def delete_cloned_voice(self, voice_id: str) -> None:
        self.store.delete_voice(voice_id)

    def set_reference_text(self, voice_id: str, reference_text: str) -> None:
        self.store.set_reference_text(voice_id, reference_text, source="manual")

    def _ensure_reference_text(self, voice_id: str) -> str:
        voice = self.store.get_voice(voice_id)
        if (voice.reference_text or "").strip():
            return str(voice.reference_text).strip()

        # One-time fallback: transcribe reference audio and persist.
        ref_paths = self.store.resolve_reference_paths(voice_id)

        # Use a slightly larger model by default for better transcript quality.
        from ..adapters.stt_faster_whisper import FasterWhisperAdapter
        import numpy as np
        import soundfile as sf

        # Build a short mono float32 clip (<= 15s) at 16k for STT.
        max_seconds = 15.0
        target_sr = 16000
        merged = []
        for p in ref_paths:
            audio, sr = sf.read(str(p), always_2d=True, dtype="float32")
            mono = np.mean(audio, axis=1).astype(np.float32)
            # simple linear resample (avoid extra deps)
            from ..audio.resample import linear_resample_mono

            mono = linear_resample_mono(mono, int(sr), target_sr)
            merged.append(mono)
        clip = np.concatenate(merged) if merged else np.zeros((0,), dtype=np.float32)
        clip = clip[: int(target_sr * max_seconds)]

        stt = FasterWhisperAdapter(
            model_size=self._reference_text_whisper_model,
            device="cpu",
            compute_type="int8",
            allow_downloads=bool(self._allow_downloads),
        )
        if not stt.is_available():
            raise RuntimeError(
                "This cloned voice has no stored reference_text.\n"
                "Auto-fallback requires a cached STT model, but downloads are disabled.\n"
                "Fix options:\n"
                "  - Prefetch outside the REPL: abstractvoice-prefetch --stt small\n"
                "  - Or set it manually: /clone_set_ref_text <id> \"...\""
            )
        text = (stt.transcribe_from_array(clip, sample_rate=target_sr) or "").strip()

        # Conservative normalization: keep it short, end with punctuation.
        text = " ".join(text.split())
        if text and not (text.endswith(".") or text.endswith("!") or text.endswith("?") or text.endswith("。")):
            text = text + "."

        # Persist so we never re-transcribe for this voice.
        self.store.set_reference_text(voice_id, text, source="asr")
        return text

    def speak_to_bytes(self, text: str, *, voice_id: str, format: str = "wav", speed: Optional[float] = None) -> bytes:
        if format.lower() != "wav":
            raise ValueError("Voice cloning currently supports WAV output only.")

        voice = self.store.get_voice(voice_id)
        # Best-effort: normalize stored references (e.g. MP3-in-WAV) to avoid noisy
        # native decoder stderr output during synthesis.
        try:
            self.store.normalize_reference_audio(voice_id)
        except Exception:
            pass
        ref_paths = self.store.resolve_reference_paths(voice_id)
        ref_text = self._ensure_reference_text(voice_id)
        eng = self._get_engine(getattr(voice, "engine", None) or "f5_tts")
        return eng.infer_to_wav_bytes(text=text, reference_paths=ref_paths, reference_text=ref_text, speed=speed)

    def speak_to_audio_chunks(
        self,
        text: str,
        *,
        voice_id: str,
        speed: Optional[float] = None,
        max_chars: int = 120,
    ):
        voice = self.store.get_voice(voice_id)
        try:
            self.store.normalize_reference_audio(voice_id)
        except Exception:
            pass
        ref_paths = self.store.resolve_reference_paths(voice_id)
        ref_text = self._ensure_reference_text(voice_id)
        eng = self._get_engine(getattr(voice, "engine", None) or "f5_tts")
        return eng.infer_to_audio_chunks(
            text=text, reference_paths=ref_paths, reference_text=ref_text, speed=speed, max_chars=int(max_chars)
        )
