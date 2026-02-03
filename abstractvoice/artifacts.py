from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union


_ARTIFACT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def compute_artifact_id(content: bytes) -> str:
    return sha256_hex(content)[:32]


def is_artifact_ref(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("$artifact"), str) and bool(value.get("$artifact"))


def get_artifact_id(ref: Dict[str, Any]) -> str:
    return str(ref["$artifact"])


def make_media_ref(
    artifact_id: str,
    *,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    sha256: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"$artifact": str(artifact_id)}
    if content_type:
        out["content_type"] = str(content_type)
    if filename:
        out["filename"] = str(filename)
    if sha256:
        out["sha256"] = str(sha256)
    if size_bytes is not None:
        out["size_bytes"] = int(size_bytes)
    if isinstance(metadata, dict) and metadata:
        out["metadata"] = metadata
    return out


class MediaStore(Protocol):
    def store_bytes(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def load_bytes(self, artifact_id: str) -> bytes: ...

    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]: ...


class RuntimeArtifactStoreAdapter:
    """Duck-typed adapter for AbstractRuntime's ArtifactStore (no hard dependency)."""

    def __init__(self, artifact_store: Any):
        self._store = artifact_store

    def store_bytes(
        self,
        content: bytes,
        *,
        content_type: str,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        run_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        store_fn = getattr(self._store, "store", None)
        if not callable(store_fn):
            raise TypeError("Provided artifact_store does not have a callable .store(...)")

        content_b = bytes(content)
        content_type = str(content_type or "application/octet-stream")
        sha = sha256_hex(content_b)

        merged_tags: Dict[str, str] = {}
        if isinstance(tags, dict):
            merged_tags.update({str(k): str(v) for k, v in tags.items()})
        if filename and "filename" not in merged_tags:
            merged_tags["filename"] = str(filename)
        if sha and "sha256" not in merged_tags:
            merged_tags["sha256"] = sha

        try:
            meta = store_fn(
                content_b,
                content_type=content_type,
                run_id=str(run_id) if run_id else None,
                tags=merged_tags or None,
                artifact_id=str(artifact_id) if artifact_id else None,
            )
        except TypeError:
            meta = store_fn(
                content_b,
                content_type=content_type,
                run_id=str(run_id) if run_id else None,
                tags=merged_tags or None,
            )

        artifact_id_out = None
        if isinstance(meta, dict):
            artifact_id_out = meta.get("artifact_id")
        elif hasattr(meta, "artifact_id"):
            artifact_id_out = getattr(meta, "artifact_id", None)
        if not isinstance(artifact_id_out, str) or not artifact_id_out.strip():
            raise TypeError("artifact_store.store(...) did not return a usable artifact_id")

        return make_media_ref(
            str(artifact_id_out),
            content_type=content_type,
            filename=str(filename) if filename else None,
            sha256=sha,
            size_bytes=len(content_b),
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def load_bytes(self, artifact_id: str) -> bytes:
        load_fn = getattr(self._store, "load", None)
        if not callable(load_fn):
            raise TypeError("Provided artifact_store does not have a callable .load(...)")
        artifact = load_fn(str(artifact_id))
        if artifact is None:
            raise FileNotFoundError(f"Artifact not found: {artifact_id}")
        if isinstance(artifact, (bytes, bytearray)):
            return bytes(artifact)
        if hasattr(artifact, "content"):
            return bytes(getattr(artifact, "content"))
        raise TypeError("artifact_store.load(...) returned an unsupported value")

    def get_metadata(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        meta_fn = getattr(self._store, "get_metadata", None)
        if not callable(meta_fn):
            return None
        meta = meta_fn(str(artifact_id))
        if meta is None:
            return None
        if isinstance(meta, dict):
            return meta
        to_dict = getattr(meta, "to_dict", None)
        if callable(to_dict):
            out = to_dict()
            return out if isinstance(out, dict) else None
        if is_dataclass(meta):
            out = asdict(meta)
            return out if isinstance(out, dict) else None
        return None

