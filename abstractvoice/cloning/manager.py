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

    def unload_engine(self, engine: str) -> bool:
        """Best-effort unload a loaded engine to free memory.

        This does NOT delete any cloned voices on disk; it only releases runtime
        model weights/processors kept in memory.
        """
        name = str(engine or "").strip().lower()
        if not name:
            return False
        inst = self._engines.pop(name, None)
        if inst is None:
            return False
        try:
            if hasattr(inst, "unload"):
                inst.unload()
        except Exception:
            pass
        return True

    def unload_engines_except(self, keep_engine: str | None = None) -> int:
        """Unload all loaded engines except `keep_engine` (if provided)."""
        keep = str(keep_engine or "").strip().lower() or None
        removed = 0
        for name in list(self._engines.keys()):
            if keep and name == keep:
                continue
            if self.unload_engine(name):
                removed += 1
        return int(removed)

    def unload_all_engines(self) -> int:
        """Unload all loaded engines."""
        return self.unload_engines_except(None)

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
        # 3-pass ASR consensus: reduces occasional non-determinism / decoding instability.
        def _normalize_ref_text(s: str) -> str:
            s = " ".join(str(s or "").strip().split())
            if s and not (s.endswith(".") or s.endswith("!") or s.endswith("?") or s.endswith("。")):
                s = s + "."
            return s

        def _edit_distance(a: str, b: str) -> int:
            # Levenshtein distance (iterative DP, O(len(a)*len(b))).
            a = str(a or "")
            b = str(b or "")
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            # Ensure `b` is the longer string to keep the inner list small.
            if len(a) > len(b):
                a, b = b, a
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, start=1):
                cur = [i]
                for j, cb in enumerate(b, start=1):
                    ins = cur[j - 1] + 1
                    dele = prev[j] + 1
                    sub = prev[j - 1] + (0 if ca == cb else 1)
                    cur.append(min(ins, dele, sub))
                prev = cur
            return int(prev[-1])

        candidates: List[str] = []
        for _ in range(3):
            t = (stt.transcribe_from_array(clip, sample_rate=target_sr) or "").strip()
            candidates.append(_normalize_ref_text(t))

        # Majority vote on normalized candidates.
        counts: Dict[str, int] = {}
        for c in candidates:
            counts[c] = counts.get(c, 0) + 1
        best = ""
        best_n = -1
        for c, n in counts.items():
            if n > best_n:
                best = c
                best_n = int(n)

        # No majority: choose the closest candidate (consensus by edit distance).
        if best_n <= 1 and candidates:
            best_sum = None
            best_c = ""
            for i, c in enumerate(candidates):
                s = 0
                for j, other in enumerate(candidates):
                    if j == i:
                        continue
                    s += _edit_distance(c, other)
                if best_sum is None or s < best_sum:
                    best_sum = int(s)
                    best_c = c
            best = best_c

        best = _normalize_ref_text(best)
        if not best.strip():
            raise RuntimeError(
                "Failed to auto-generate reference_text from the reference audio.\n"
                "Fix options:\n"
                "  - Provide a clearer 6–10s reference sample\n"
                "  - Or set it manually: /clone_set_ref_text <id> \"...\""
            )

        # Persist so we never re-transcribe for this voice.
        self.store.set_reference_text(voice_id, best, source="asr")
        return best

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
