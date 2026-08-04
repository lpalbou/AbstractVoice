"""Which local voice models are on this machine.

Presence is a filesystem question and this module keeps it one. Asking a local
engine through its adapter what it could run costs seconds -- constructing the
AudioDiT adapter imports ``abstractvoice.audiodit``, which pulls in torch and
transformers -- and the adapter then reports nothing the model cache did not
already know. Nothing reached from here loads a machine-learning framework or
constructs an engine.

Two storage shapes cover every local voice engine:

* Hugging Face repos in the shared HF cache, where the selectable model id *is*
  the repo id (``audiodit``, ``omnivoice``).
* An engine-owned cache directory whose layout only the engine knows
  (``piper``, ``supertonic``); those engines expose their own filesystem probe
  and this module simply calls it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

__all__ = [
    "cached_tts_model_ids",
    "hf_cached_snapshot_dir",
    "hf_repo_is_cached",
]


# Local TTS engines whose weights are Hugging Face repos.
_HF_REPO_TTS_ENGINES = frozenset({"audiodit", "omnivoice", "qwen3-tts"})

# A cached snapshot only counts when it holds weights. Hugging Face keeps
# ``README.md`` and ``.gitattributes`` in every snapshot, so "the directory
# exists" would call an interrupted or metadata-only download ready to speak.
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".onnx", ".ckpt", ".gguf"})


def _normalize_engine(engine: object) -> str:
    return str(engine or "").strip().lower().replace("_", "-")


def cached_tts_model_ids(engine: object, *, extra_candidates: Iterable[str] = ()) -> list[str]:
    """TTS model ids for ``engine`` whose weights are on this machine.

    For Hugging Face engines the candidates are the ids this package declares,
    plus ``extra_candidates`` -- an operator who points an engine at their own
    checkpoint must be able to select it, and the packaged catalog only knows the
    defaults. Every candidate is still filtered by what is on disk. ``piper`` and
    ``supertonic`` enumerate their own cache directories instead, so there is
    nothing for a candidate list to add.

    Empty for remote engines and for local engines with nothing downloaded yet:
    the answer is "what can this machine speak with right now", not "what could
    it fetch".
    """

    engine_id = _normalize_engine(engine)
    extra = [str(value).strip() for value in extra_candidates if str(value).strip()]

    if engine_id == "piper":
        from .adapters.tts_piper import cached_piper_model_ids

        return cached_piper_model_ids()

    if engine_id == "supertonic":
        from .supertonic import is_supertonic_cached

        return _declared_model_ids(engine_id) if is_supertonic_cached() else []

    if engine_id in _HF_REPO_TTS_ENGINES:
        declared = _declared_model_ids(engine_id)
        candidates = declared + [item for item in extra if item not in declared]
        return [model_id for model_id in candidates if hf_repo_is_cached(model_id)]

    return []


def hf_repo_is_cached(model_id: object) -> bool:
    """True when ``model_id`` is downloaded, or is a local checkpoint directory.

    Reads the documented Hugging Face cache layout
    (``<cache>/models--org--name/snapshots/<revision>/``) instead of calling
    ``snapshot_download(local_files_only=True)``: no repo-specific filenames, no
    revision resolution, and no way to reach the network by accident. Any cached
    revision counts, which matches the engines -- none of them pins one.
    """

    repo_id = str(model_id or "").strip()
    if not repo_id:
        return False

    # Engines accept a path to a local checkpoint in place of a repo id. Only
    # something written as a path is treated as one: "k2-fsa/OmniVoice" is a repo
    # id, and resolving it against the process cwd would make the answer depend on
    # where the caller happened to be started.
    if repo_id.startswith(("~", ".", os.sep)) or Path(repo_id).is_absolute():
        checkpoint = Path(repo_id).expanduser()
        return checkpoint.is_dir() and _holds_weights(checkpoint)

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        # Without huggingface_hub nothing could have been downloaded from the Hub.
        return False

    snapshots = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    try:
        revisions = [path for path in snapshots.iterdir() if path.is_dir()]
    except OSError:
        return False
    return any(_holds_weights(revision) for revision in revisions)


def hf_cached_snapshot_dir(model_id: object) -> Optional[Path]:
    """The local snapshot directory holding ``model_id``'s weights, or None.

    Same layout read as :func:`hf_repo_is_cached` — never touches the network,
    so it is safe from any discovery path. When several revisions are cached,
    the one with weights wins (most recently modified first).
    """
    repo_id = str(model_id or "").strip()
    if not repo_id:
        return None

    if repo_id.startswith(("~", ".", os.sep)) or Path(repo_id).is_absolute():
        checkpoint = Path(repo_id).expanduser()
        return checkpoint if checkpoint.is_dir() and _holds_weights(checkpoint) else None

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:
        return None

    snapshots = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    try:
        revisions = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    for revision in revisions:
        if _holds_weights(revision):
            return revision
    return None


def _holds_weights(snapshot_dir: Path) -> bool:
    """True when ``snapshot_dir`` contains at least one weights file.

    Recursive because repos put weights in subdirectories, and ``is_file()``
    resolves symlinks so a snapshot whose blobs were garbage-collected reads as
    absent rather than present.
    """

    try:
        for entry in snapshot_dir.rglob("*"):
            if entry.suffix.lower() in _WEIGHT_SUFFIXES and entry.is_file():
                return True
    except OSError:
        return False
    return False


def _declared_model_ids(engine_id: str) -> list[str]:
    """Model ids the packaged capability catalog declares for ``engine_id``."""

    from .compatibility import _asset_model_ids

    return _asset_model_ids("tts", engine_id)
