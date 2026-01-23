from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .engine_f5 import F5TTSVoiceCloningEngine
from .store import VoiceCloneStore


class VoiceCloner:
    """High-level voice cloning manager (optional).

    Stores reference bundles locally and uses an engine to synthesize speech.
    """

    def __init__(self, *, store: Optional[VoiceCloneStore] = None, debug: bool = False, whisper_model: str = "tiny"):
        self.store = store or VoiceCloneStore()
        self.debug = debug
        self._whisper_model = whisper_model
        self._engine: Optional[F5TTSVoiceCloningEngine] = None

    def _get_engine(self) -> F5TTSVoiceCloningEngine:
        # Lazy-load engine to avoid surprise model downloads during list/store operations.
        if self._engine is None:
            self._engine = F5TTSVoiceCloningEngine(whisper_model=self._whisper_model, debug=self.debug)
        return self._engine

    def clone_voice(self, reference_audio_path: str, name: str | None = None, *, reference_text: str | None = None) -> str:
        """Create a new cloned voice from a file or directory.

        If a directory is provided, all WAV/FLAC/OGG files inside are used.
        """
        p = Path(reference_audio_path)
        if not p.exists():
            raise FileNotFoundError(str(p))

        if p.is_dir():
            refs = sorted([x for x in p.glob("*") if x.suffix.lower() in {".wav", ".flac", ".ogg"}])
            if not refs:
                raise ValueError(f"No supported reference audio files found in: {p}")
        else:
            refs = [p]

        voice_id = self.store.create_voice(
            refs,
            name=name,
            reference_text=reference_text,
            engine="f5_tts",
            meta={"source": str(p)},
        )
        return voice_id

    def list_cloned_voices(self) -> List[Dict[str, Any]]:
        return self.store.list_voices()

    def export_voice(self, voice_id: str, path: str) -> str:
        return self.store.export_voice(voice_id, path)

    def import_voice(self, path: str) -> str:
        return self.store.import_voice(path)

    def speak_to_bytes(self, text: str, *, voice_id: str, format: str = "wav", speed: Optional[float] = None) -> bytes:
        if format.lower() != "wav":
            raise ValueError("Voice cloning currently supports WAV output only.")

        voice = self.store.get_voice(voice_id)
        ref_paths = self.store.resolve_reference_paths(voice_id)

        return self._get_engine().infer_to_wav_bytes(
            text=text,
            reference_paths=ref_paths,
            reference_text=voice.reference_text,
            speed=speed,
        )

