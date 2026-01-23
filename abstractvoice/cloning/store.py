from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import appdirs


@dataclass(frozen=True)
class ClonedVoice:
    voice_id: str
    name: str
    created_at: float
    reference_files: List[str]  # relative to voice directory
    reference_text: Optional[str] = None
    engine: str = "f5_tts"
    meta: Dict[str, Any] = None


class VoiceCloneStore:
    """Stores cloned-voice metadata + reference audio bundles locally.

    Design principles:
    - Keep the storage format portable and engine-agnostic.
    - Avoid embedding binary blobs in JSON; store files on disk.
    """

    def __init__(self, base_dir: Optional[str | Path] = None):
        if base_dir is None:
            root = Path(appdirs.user_data_dir("abstractvoice"))
            self._base_dir = root / "cloned_voices"
        else:
            self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._base_dir / "index.json"
        if not self._index_path.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Any]:
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_index(self, data: Dict[str, Any]) -> None:
        self._index_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _voice_dir(self, voice_id: str) -> Path:
        return self._base_dir / voice_id

    def resolve_reference_paths(self, voice_id: str) -> List[Path]:
        voice = self.get_voice(voice_id)
        vdir = self._voice_dir(voice.voice_id)
        return [vdir / rel for rel in voice.reference_files]

    def create_voice(
        self,
        reference_paths: Iterable[str | Path],
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        engine: str = "f5_tts",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        paths = [Path(p) for p in reference_paths]
        if not paths:
            raise ValueError("reference_paths must contain at least one file")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(str(p))
            if p.is_dir():
                raise ValueError(f"Reference path must be a file, got directory: {p}")

        voice_id = uuid.uuid4().hex
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)

        copied: List[str] = []
        for i, p in enumerate(paths):
            dest = vdir / f"ref_{i}{p.suffix.lower()}"
            shutil.copy2(p, dest)
            copied.append(dest.name)

        record = ClonedVoice(
            voice_id=voice_id,
            name=name or f"voice_{voice_id[:8]}",
            created_at=time.time(),
            reference_files=copied,
            reference_text=reference_text,
            engine=engine,
            meta=meta or {},
        )

        index = self._read_index()
        index[voice_id] = asdict(record)
        self._write_index(index)
        return voice_id

    def get_voice(self, voice_id: str) -> ClonedVoice:
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        data = index[voice_id]
        return ClonedVoice(**data)

    def get_voice_dict(self, voice_id: str) -> Dict[str, Any]:
        """Return the stored voice record as a JSON-serializable dict."""
        v = self.get_voice(voice_id)
        return {"voice_id": voice_id, **asdict(v)}

    def list_voices(self) -> List[Dict[str, Any]]:
        index = self._read_index()
        out: List[Dict[str, Any]] = []
        for voice_id, data in index.items():
            out.append({"voice_id": voice_id, **data})
        # newest first
        out.sort(key=lambda d: float(d.get("created_at", 0)), reverse=True)
        return out

    def set_reference_text(self, voice_id: str, reference_text: str, *, source: str | None = None) -> None:
        """Set (or replace) the stored reference text for a cloned voice.

        This matters a lot for cloning quality: if reference_text is garbled,
        the model often produces artifacts (wrong words bleeding into output).
        """
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        data = dict(index[voice_id])
        data["reference_text"] = str(reference_text or "")
        if source:
            meta = dict(data.get("meta") or {})
            meta["reference_text_source"] = str(source)
            data["meta"] = meta
        index[voice_id] = data
        self._write_index(index)

    def export_voice(self, voice_id: str, path: str | Path) -> str:
        """Export a voice bundle as a zip archive."""
        import zipfile

        voice = self.get_voice(voice_id)
        vdir = self._voice_dir(voice_id)
        if not vdir.exists():
            raise FileNotFoundError(str(vdir))

        out_path = Path(path)
        if out_path.suffix.lower() != ".zip":
            out_path = out_path.with_suffix(".zip")

        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("voice.json", json.dumps(asdict(voice), indent=2, sort_keys=True))
            for rel in voice.reference_files:
                fp = vdir / rel
                z.write(fp, arcname=f"refs/{rel}")

        return str(out_path)

    def import_voice(self, path: str | Path) -> str:
        """Import a voice bundle zip archive into the local store."""
        import zipfile

        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(str(src))

        with zipfile.ZipFile(src, "r") as z:
            voice_data = json.loads(z.read("voice.json").decode("utf-8"))

            # New id on import to avoid collisions.
            new_id = uuid.uuid4().hex
            vdir = self._voice_dir(new_id)
            vdir.mkdir(parents=True, exist_ok=True)

            refs = []
            for name in z.namelist():
                if not name.startswith("refs/"):
                    continue
                rel = Path(name).name
                dest = vdir / rel
                with z.open(name) as src_fp, open(dest, "wb") as out_fp:
                    shutil.copyfileobj(src_fp, out_fp)
                refs.append(rel)

            voice_data["voice_id"] = new_id
            voice_data["reference_files"] = refs

            index = self._read_index()
            index[new_id] = voice_data
            self._write_index(index)

        return new_id

    def rename_voice(self, voice_id: str, new_name: str) -> None:
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        data = dict(index[voice_id])
        data["name"] = str(new_name or "").strip() or data.get("name") or f"voice_{voice_id[:8]}"
        index[voice_id] = data
        self._write_index(index)

    def delete_voice(self, voice_id: str) -> None:
        """Delete a voice entry and its reference files from disk."""
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")

        vdir = self._voice_dir(voice_id)
        try:
            if vdir.exists():
                shutil.rmtree(vdir)
        except Exception:
            # If deletion fails, do not leave index in an inconsistent state.
            raise

        del index[voice_id]
        self._write_index(index)

