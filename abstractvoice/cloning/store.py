from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import threading
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import appdirs

_STDERR_FD_LOCK = threading.Lock()


class _SilenceStderrFD:
    """Temporarily redirect OS-level stderr (fd=2) to /dev/null.

    Some native decoders (e.g. mpg123 via libsndfile) write directly to fd=2,
    bypassing Python's sys.stderr. We use this to keep interactive CLI output
    clean when decoding odd inputs like MP3-in-WAV.
    """

    def __enter__(self):
        self._lock = _STDERR_FD_LOCK
        self._lock.acquire()
        self._devnull_fd = None
        self._saved_stderr_fd = None
        try:
            self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
            self._saved_stderr_fd = os.dup(2)
            os.dup2(self._devnull_fd, 2)
        except Exception:
            self.__exit__(None, None, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._saved_stderr_fd is not None:
                try:
                    os.dup2(self._saved_stderr_fd, 2)
                except Exception:
                    pass
        finally:
            try:
                if self._saved_stderr_fd is not None:
                    os.close(self._saved_stderr_fd)
            except Exception:
                pass
            try:
                if self._devnull_fd is not None:
                    os.close(self._devnull_fd)
            except Exception:
                pass
            try:
                self._lock.release()
            except Exception:
                pass
        return False


# Built-in voices: a read-only voice pack shipped inside the package (see
# docs/voice-pack.md). Each voice is a Qwen3-TTS clone prompt (reference codec
# codes + speaker embedding of the 1.7B Base checkpoint) and its profile.
BUILTIN_VOICE_PACK_DIR = Path(__file__).resolve().parents[1] / "assets" / "voices" / "qwen3_tts"
_BUILTIN_CACHE: Dict[str, Any] = {}


def builtin_voice_pack() -> Dict[str, Dict[str, Any]]:
    """The built-in voices by id (read once). Empty when the pack is absent."""
    if "voices" not in _BUILTIN_CACHE:
        index_path = BUILTIN_VOICE_PACK_DIR / "voices.json"
        voices: Dict[str, Dict[str, Any]] = {}
        if index_path.exists():
            pack = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in pack.get("voices") or []:
                voices[str(entry["id"])] = dict(entry, engine=pack.get("engine", "qwen3-tts"))
        _BUILTIN_CACHE["voices"] = voices
    return _BUILTIN_CACHE["voices"]


def _builtin_record(entry: Dict[str, Any]) -> "ClonedVoice":
    return ClonedVoice(
        voice_id=str(entry["id"]),
        name=str(entry["name"]),
        created_at=0.0,
        reference_files=[str(entry["prompt"])],
        reference_text=str(entry["reference_text"]),
        engine=str(entry.get("engine") or "qwen3-tts"),
        meta={
            "builtin": True,
            "model_id": str(entry["model"]),
            "nationality": entry.get("nationality"),
            "language": entry.get("language"),
            "language_code": entry.get("language_code"),
            "sex": entry.get("sex"),
            "style": entry.get("style"),
            "description": entry.get("description"),
            "reference_text_source": "builtin",
        },
    )


_INDEX_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_INDEX_THREAD_LOCKS_GUARD = threading.Lock()
_INDEX_DEPTH = threading.local()


class _IndexLock:
    """A per-folder lock: a thread lock in this process, plus an OS file lock where available."""

    def __init__(self, base_dir: Path):
        key = str(Path(base_dir).resolve())
        self._key = key
        with _INDEX_THREAD_LOCKS_GUARD:
            self._thread_lock = _INDEX_THREAD_LOCKS.setdefault(key, threading.RLock())
        self._path = Path(base_dir) / ".index.lock"
        self._fh = None

    def __enter__(self) -> "_IndexLock":
        self._thread_lock.acquire()
        depth = _INDEX_DEPTH.__dict__.setdefault(self._key, 0)
        _INDEX_DEPTH.__dict__[self._key] = depth + 1
        if depth:
            return self  # nested in this thread: the outer holder has the file lock
        try:
            import fcntl  # POSIX only

            self._fh = open(self._path, "a+")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except ImportError:
            self._fh = None
        except Exception:
            if self._fh is not None:
                self._fh.close()
            self._fh = None
        return self

    def __exit__(self, *exc: Any) -> None:
        _INDEX_DEPTH.__dict__[self._key] -= 1
        try:
            if self._fh is not None:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                self._fh.close()
        finally:
            self._fh = None
            self._thread_lock.release()


def _locked_index(method):
    """Run a store method that changes the index under the folder's index lock."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._index_mutation():
            return method(self, *args, **kwargs)

    return wrapper

@dataclass(frozen=True)
class ClonedVoice:
    voice_id: str
    name: str
    created_at: float
    reference_files: List[str]  # relative to voice directory
    reference_text: Optional[str] = None
    engine: str = "omnivoice"
    meta: Dict[str, Any] = None


class VoiceCloneStore:
    """Stores cloned-voice metadata + reference audio bundles locally.

    Design principles:
    - Keep the storage format portable and engine-agnostic.
    - Avoid embedding binary blobs in JSON; store files on disk.
    """

    def __init__(self, base_dir: Optional[str | Path] = None, *, builtin_voices: Optional[bool] = None):
        # The user's own store (no base_dir) also lists the built-in voice pack;
        # a store at an explicit directory does not, unless asked.
        self._builtin = (base_dir is None) if builtin_voices is None else bool(builtin_voices)
        if base_dir is None:
            root = Path(appdirs.user_data_dir("abstractvoice"))
            self._base_dir = root / "cloned_voices"
        else:
            self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._base_dir / "index.json"
        if not self._index_path.exists():
            with _IndexLock(self._base_dir):
                if not self._index_path.exists():  # another process may have created it meanwhile
                    self._write_index({})

    def _read_index(self) -> Dict[str, Any]:
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_index(self, data: Dict[str, Any]) -> None:
        # Written beside and moved in whole: a reader never sees a half-written index.
        tmp = self._index_path.with_name(self._index_path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._index_path)

    def _index_mutation(self):
        """Serialise one read-modify-write of the index across threads and processes (a shared
        library is written by several hosts' deploys)."""
        return _IndexLock(self._base_dir)

    def _voice_dir(self, voice_id: str) -> Path:
        if self._builtin and voice_id in builtin_voice_pack() and voice_id not in self._read_index():
            return BUILTIN_VOICE_PACK_DIR
        return self._base_dir / voice_id

    def is_builtin(self, voice_id: str) -> bool:
        """True for a voice of the built-in pack (read-only)."""
        return self._builtin and voice_id in builtin_voice_pack() and voice_id not in self._read_index()

    def _refuse_builtin(self, voice_id: str, action: str) -> None:
        if self.is_builtin(voice_id):
            raise ValueError(
                f"{voice_id!r} is a built-in voice and cannot be {action}. "
                "Clone its reference into your own voice to customize it."
            )

    def resolve_reference_paths(self, voice_id: str) -> List[Path]:
        voice = self.get_voice(voice_id)
        vdir = self._voice_dir(voice.voice_id)
        return [vdir / rel for rel in voice.reference_files]

    def normalize_reference_audio(self, voice_id: str) -> int:
        """Best-effort normalize stored references for a voice.

        Currently converts WAV files that are actually MPEG-compressed (MP3-in-WAV)
        into standard PCM16 WAVs. This prevents native decoder warnings like:
          "Illegal Audio-MPEG-Header ... Trying to resync ..."
        from polluting interactive CLI output during cloning synthesis.
        """
        try:
            voice = self.get_voice(voice_id)
        except Exception:
            return 0

        vdir = self._voice_dir(voice.voice_id)
        converted = 0
        for rel in (voice.reference_files or []):
            p = vdir / str(rel)
            try:
                if self._normalize_wav_mpeg_to_pcm_inplace(p):
                    converted += 1
            except Exception:
                # Normalization is best-effort; inference can still attempt decode.
                continue
        return converted

    def _normalize_wav_mpeg_to_pcm_inplace(self, path: Path) -> bool:
        if path.suffix.lower() != ".wav":
            return False
        if not path.exists():
            return False
        try:
            import soundfile as sf
        except Exception:
            return False

        try:
            info = sf.info(str(path))
        except Exception:
            return False

        # Example: format=WAV subtype=MPEG_LAYER_III
        fmt = str(getattr(info, "format", "") or "").strip().upper()
        subtype = str(getattr(info, "subtype", "") or "").strip().upper()
        if fmt != "WAV" or not subtype.startswith("MPEG"):
            return False

        # Decode once (silencing native decoder stderr), then rewrite as PCM16 WAV.
        with _SilenceStderrFD():
            audio, sr = sf.read(str(path), always_2d=True, dtype="float32")

        tmp = tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            sf.write(str(tmp_path), audio, int(sr), format="WAV", subtype="PCM_16")
            tmp_path.replace(path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
        return True

    @_locked_index
    def create_voice(
        self,
        reference_paths: Iterable[str | Path],
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        engine: str = "omnivoice",
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
            if p.suffix.lower() == ".wav":
                # If the WAV container is actually MPEG-compressed, normalize to PCM16 WAV
                # to avoid noisy mpg123 "resync" messages later during synthesis.
                try:
                    import soundfile as sf

                    info = sf.info(str(p))
                    fmt = str(getattr(info, "format", "") or "").strip().upper()
                    subtype = str(getattr(info, "subtype", "") or "").strip().upper()
                    if fmt == "WAV" and subtype.startswith("MPEG"):
                        with _SilenceStderrFD():
                            audio, sr = sf.read(str(p), always_2d=True, dtype="float32")
                        sf.write(str(dest), audio, int(sr), format="WAV", subtype="PCM_16")
                        copied.append(dest.name)
                        continue
                except Exception:
                    # Fall back to raw copy; synthesis may still attempt decode.
                    pass

            shutil.copy2(p, dest)
            copied.append(dest.name)

        meta_out = dict(meta or {})
        if (reference_text or "").strip() and not meta_out.get("reference_text_source"):
            # Keep metadata consistent with `set_reference_text(..., source="manual")`.
            meta_out["reference_text_source"] = "manual"

        record = ClonedVoice(
            voice_id=voice_id,
            name=name or f"voice_{voice_id[:8]}",
            created_at=time.time(),
            reference_files=copied,
            reference_text=reference_text,
            engine=engine,
            meta=meta_out,
        )

        index = self._read_index()
        index[voice_id] = asdict(record)
        self._write_index(index)
        return voice_id

    @_locked_index
    def create_voice_from_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        engine: str = "omnivoice",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a cloned voice from an in-memory WAV payload.

        This is primarily intended for client/server integrations where reference
        audio arrives as uploaded bytes instead of an existing local file path.
        """
        if not wav_bytes:
            raise ValueError("wav_bytes must be non-empty")

        voice_id = uuid.uuid4().hex
        vdir = self._voice_dir(voice_id)
        vdir.mkdir(parents=True, exist_ok=True)

        dest = vdir / "ref_0.wav"
        dest.write_bytes(wav_bytes)

        # Best-effort normalization: if the WAV container is actually MPEG-compressed,
        # rewrite it as PCM16 WAV to avoid noisy native decoder stderr later.
        try:
            self._normalize_wav_mpeg_to_pcm_inplace(dest)
        except Exception:
            pass

        meta_out = dict(meta or {})
        if (reference_text or "").strip() and not meta_out.get("reference_text_source"):
            # Keep metadata consistent with `set_reference_text(..., source="manual")`.
            meta_out["reference_text_source"] = "manual"

        record = ClonedVoice(
            voice_id=voice_id,
            name=name or f"voice_{voice_id[:8]}",
            created_at=time.time(),
            reference_files=[dest.name],
            reference_text=reference_text,
            engine=str(engine or "omnivoice"),
            meta=meta_out,
        )

        index = self._read_index()
        index[voice_id] = asdict(record)
        self._write_index(index)
        return voice_id

    def get_voice(self, voice_id: str) -> ClonedVoice:
        index = self._read_index()
        if voice_id not in index:
            if self._builtin and voice_id in builtin_voice_pack():
                return _builtin_record(builtin_voice_pack()[voice_id])
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
        # newest first, then the built-in pack by id
        out.sort(key=lambda d: float(d.get("created_at", 0)), reverse=True)
        if self._builtin:
            for vid in sorted(builtin_voice_pack()):
                if vid not in index:
                    out.append(self.get_voice_dict(vid))
        return out

    @_locked_index
    def set_reference_text(self, voice_id: str, reference_text: str, *, source: str | None = None) -> None:
        """Set (or replace) the stored reference text for a cloned voice.

        This matters a lot for cloning quality: if reference_text is garbled,
        the model often produces artifacts (wrong words bleeding into output).
        """
        self._refuse_builtin(voice_id, "edited")
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

    @_locked_index
    def set_language(self, voice_id: str, language_code: str, *, source: str) -> None:
        """Record the language a voice speaks natively (`meta.language_code`) and how it is known
        (`meta.language_source`: `request` when stated, `text` when identified from its transcript)."""
        self._refuse_builtin(voice_id, "edited")
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        data = dict(index[voice_id])
        meta = dict(data.get("meta") or {})
        meta["language_code"] = str(language_code).strip().lower()
        meta["language_source"] = str(source)
        data["meta"] = meta
        index[voice_id] = data
        self._write_index(index)

    def export_voice(self, voice_id: str, path: str | Path) -> str:
        """Export a voice bundle as a zip archive."""
        import zipfile

        self._refuse_builtin(voice_id, "exported")

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

    @_locked_index
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

    @_locked_index
    def rename_voice(self, voice_id: str, new_name: str) -> None:
        self._refuse_builtin(voice_id, "renamed")
        index = self._read_index()
        if voice_id not in index:
            raise KeyError(f"Unknown voice_id: {voice_id}")
        data = dict(index[voice_id])
        data["name"] = str(new_name or "").strip() or data.get("name") or f"voice_{voice_id[:8]}"
        index[voice_id] = data
        self._write_index(index)

    @_locked_index
    def delete_voice(self, voice_id: str) -> None:
        """Delete a voice entry and its reference files from disk."""
        self._refuse_builtin(voice_id, "deleted")
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


class LibraryVoiceError(ValueError):
    """A change was asked of a voice that belongs to a read-only library."""


class LayeredVoiceStore:
    """A personal voice store (writable) in front of library stores (read-only).

    The voices a person clones or designs go to their personal store; the libraries hold the
    voices everyone may use (the built-in pack, and the voices an administrator deploys). Lookups
    search the personal store first, then each library; changes reach the personal store only.
    Every listed record carries ``meta.scope``: ``personal`` or ``library``.

    It answers the same calls as :class:`VoiceCloneStore`, so the cloner and the engines use it
    unchanged (AbstractVoice backlog 0065).
    """

    def __init__(self, personal: VoiceCloneStore, libraries: Iterable[VoiceCloneStore] = ()):
        self.personal = personal
        self.libraries = list(libraries)
        self._base_dir = personal._base_dir

    # -- lookup -----------------------------------------------------------------------------

    def _layers(self) -> List[tuple]:
        return [(self.personal, "personal")] + [(lib, "library") for lib in self.libraries]

    def _owner(self, voice_id: str) -> tuple:
        for store, scope in self._layers():
            try:
                store.get_voice(voice_id)
            except KeyError:
                continue
            return store, scope
        raise KeyError(f"Unknown voice_id: {voice_id}")

    def scope_of(self, voice_id: str) -> str:
        return self._owner(voice_id)[1]

    def _writable(self, voice_id: str, action: str) -> VoiceCloneStore:
        store, scope = self._owner(voice_id)
        if scope != "personal":
            raise LibraryVoiceError(
                f"{voice_id!r} is a library voice (shared with everyone) and cannot be {action} here. "
                "Clone its reference into your own voice to customize it."
            )
        return store

    def get_voice(self, voice_id: str) -> "ClonedVoice":
        return self._owner(voice_id)[0].get_voice(voice_id)

    def get_voice_dict(self, voice_id: str) -> Dict[str, Any]:
        store, scope = self._owner(voice_id)
        out = store.get_voice_dict(voice_id)
        out["meta"] = dict(out.get("meta") or {}, scope=scope)
        return out

    def list_voices(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for store, scope in self._layers():
            for item in store.list_voices():
                vid = str(item.get("voice_id") or "")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                out.append(dict(item, meta=dict(item.get("meta") or {}, scope=scope)))
        return out

    def is_builtin(self, voice_id: str) -> bool:
        try:
            store, _scope = self._owner(voice_id)
        except KeyError:
            return False
        return store.is_builtin(voice_id)

    def resolve_reference_paths(self, voice_id: str) -> List[Path]:
        return self._owner(voice_id)[0].resolve_reference_paths(voice_id)

    def normalize_reference_audio(self, voice_id: str) -> int:
        store, scope = self._owner(voice_id)
        # A library is read-only from here: its references are used as they are.
        return store.normalize_reference_audio(voice_id) if scope == "personal" else 0

    # -- changes (personal store only) --------------------------------------------------------

    def create_voice(self, *args: Any, **kwargs: Any) -> str:
        return self.personal.create_voice(*args, **kwargs)

    def create_voice_from_wav_bytes(self, *args: Any, **kwargs: Any) -> str:
        return self.personal.create_voice_from_wav_bytes(*args, **kwargs)

    def import_voice(self, path: str | Path) -> str:
        return self.personal.import_voice(path)

    def export_voice(self, voice_id: str, path: str | Path) -> str:
        return self._owner(voice_id)[0].export_voice(voice_id, path)

    def set_reference_text(self, voice_id: str, reference_text: str, *, source: str | None = None) -> None:
        self._writable(voice_id, "edited").set_reference_text(voice_id, reference_text, source=source)

    def set_language(self, voice_id: str, language_code: str, *, source: str) -> None:
        self._writable(voice_id, "edited").set_language(voice_id, language_code, source=source)

    def rename_voice(self, voice_id: str, new_name: str) -> None:
        self._writable(voice_id, "renamed").rename_voice(voice_id, new_name)

    def delete_voice(self, voice_id: str) -> None:
        self._writable(voice_id, "deleted").delete_voice(voice_id)

    # -- library deployment (an administrator's act; the host decides who may) ----------------

    def deploy_to_library(self, voice_id: str, library: Optional[VoiceCloneStore] = None) -> str:
        """Copy one of the personal voices into a library (the first by default); returns the
        library voice's id. The personal voice is kept."""
        target = library if library is not None else (self.libraries[0] if self.libraries else None)
        if target is None:
            raise ValueError("No voice library is configured to deploy to.")
        self._writable(voice_id, "deployed")
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self.personal.export_voice(voice_id, Path(tmp) / "voice.zip")
            library_id = target.import_voice(bundle)
        # The library record names no path of the deployer's machine or account.
        with target._index_mutation():
            index = target._read_index()
            data = dict(index.get(library_id) or {})
            meta = {k: v for k, v in dict(data.get("meta") or {}).items() if k != "source"}
            meta["source"] = "library"
            data["meta"] = meta
            index[library_id] = data
            target._write_index(index)
        return library_id


def open_voice_store(
    store_dir: Optional[str | Path] = None,
    library_dirs: Optional[Iterable[str | Path]] = None,
) -> "VoiceCloneStore | LayeredVoiceStore":
    """The voice store a configuration names.

    - nothing configured: the machine user's store (with the built-in voices), as before;
    - ``store_dir``: that folder is the personal store, in front of the libraries:
      ``library_dirs`` when given, else the machine user's store (whose voices, built-in ones
      included, everyone keeps seeing and using, read-only).
    """
    if not store_dir:
        return VoiceCloneStore()
    personal = VoiceCloneStore(store_dir, builtin_voices=False)
    dirs = [d for d in (library_dirs or []) if str(d or "").strip()]
    libraries = [VoiceCloneStore(d, builtin_voices=(i == 0)) for i, d in enumerate(dirs)] if dirs else [VoiceCloneStore()]
    return LayeredVoiceStore(personal, libraries)
