"""OmniVoice persistent prompt cache (tokenized reference audio).

OmniVoice's `voice_clone_prompt` conditioning is the most reliable primitive for
"persistent voices" because it conditions generation on reference *audio tokens*
instead of re-sampling a voice from `instruct` on every utterance.

This module provides a small on-disk cache for those tokenized prompts:
- One-time cost to build the prompt (generate a short reference + tokenize).
- Subsequent synthesis can reuse cached `ref_audio_tokens` immediately.

Design constraints:
- Safe to import on minimal installs (no `torch`, no `omnivoice` imports here).
- Cache is local, user-scoped, and JSON/NPZ-based for portability.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import appdirs
import numpy as np


_CACHE_VERSION = 1


def _slug(s: str) -> str:
    """Return a filesystem-safe slug (best-effort, deterministic)."""
    raw = str(s or "").strip()
    if not raw:
        return "x"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("._-") or "x"
    # Collapse repeated underscores (avoid pathological paths).
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _slug_with_hash(s: str, *, max_len: int = 60) -> str:
    raw = str(s or "")
    base = _slug(raw)[: int(max_len)]
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{base}__{h}"


def _canonical_json(obj: Any) -> str:
    """Stable string representation for spec comparisons."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        # Best-effort: as a last resort, stringify.
        return json.dumps(str(obj), ensure_ascii=False)


def get_omnivoice_prompt_cache_root() -> Path:
    return Path(appdirs.user_data_dir("abstractvoice")) / "omnivoice_prompt_cache"


def get_omnivoice_prompt_cache_dir(
    *,
    model_id: str,
    revision: str | None,
    language: str | None,
    profile_id: str,
) -> Path:
    # Include model and revision to avoid reusing prompts across incompatible tokenizers.
    root = get_omnivoice_prompt_cache_root()
    mid = _slug_with_hash(str(model_id or ""))
    rev = _slug_with_hash(str(revision or "default"))
    lang = _slug_with_hash(str(language or "any"))
    pid = _slug_with_hash(str(profile_id or ""))
    return root / mid / rev / lang / pid


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
            tmp = f.name
            f.write(bytes(data))
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp), str(path))
    finally:
        if tmp:
            try:
                os.unlink(str(tmp))
            except Exception:
                pass


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class CachedOmniVoicePrompt:
    """A cached prompt ready to be converted into a `voice_clone_prompt` object."""

    ref_audio_tokens: np.ndarray  # expected shape: (C, T)
    ref_text: str
    ref_rms: float
    meta: Dict[str, Any]


def load_cached_omnivoice_prompt(
    cache_dir: Path,
    *,
    expected_prompt_spec: Optional[Dict[str, Any]] = None,
) -> CachedOmniVoicePrompt | None:
    """Load cached prompt tokens + metadata if present and compatible."""
    meta_path = cache_dir / "prompt_meta.json"
    tokens_path = cache_dir / "prompt_tokens.npz"
    if not meta_path.exists() or not tokens_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    try:
        if int(meta.get("cache_version", 0) or 0) != int(_CACHE_VERSION):
            return None
    except Exception:
        return None

    if expected_prompt_spec is not None:
        try:
            got = meta.get("prompt_spec")
            if _canonical_json(got) != _canonical_json(expected_prompt_spec):
                return None
        except Exception:
            return None

    try:
        npz = np.load(str(tokens_path))
        tokens = np.asarray(npz["ref_audio_tokens"])
    except Exception:
        return None

    try:
        ref_text = str(meta.get("ref_text") or "").strip()
        ref_rms = float(meta.get("ref_rms") or 0.0)
    except Exception:
        return None

    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    if tokens.ndim != 2 or tokens.size <= 0:
        return None

    # Ensure an integer dtype for token ids.
    if not np.issubdtype(tokens.dtype, np.integer):
        try:
            tokens = tokens.astype(np.int32, copy=False)
        except Exception:
            return None

    return CachedOmniVoicePrompt(
        ref_audio_tokens=tokens,
        ref_text=ref_text,
        ref_rms=ref_rms,
        meta=dict(meta) if isinstance(meta, dict) else {},
    )


def save_cached_omnivoice_prompt(
    cache_dir: Path,
    *,
    ref_audio_tokens: np.ndarray,
    ref_text: str,
    ref_rms: float,
    prompt_spec: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist prompt tokens + metadata (atomic best-effort)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "prompt_meta.json"
    tokens_path = cache_dir / "prompt_tokens.npz"

    tokens = np.asarray(ref_audio_tokens)
    if tokens.ndim == 1:
        tokens = tokens.reshape(1, -1)
    if tokens.ndim != 2:
        raise ValueError("ref_audio_tokens must be a 2D array (C, T)")
    if not np.issubdtype(tokens.dtype, np.integer):
        tokens = tokens.astype(np.int32)

    meta: Dict[str, Any] = {
        "cache_version": int(_CACHE_VERSION),
        "created_at": float(time.time()),
        "ref_text": str(ref_text),
        "ref_rms": float(ref_rms),
        "tokens_shape": [int(tokens.shape[0]), int(tokens.shape[1])],
        "tokens_dtype": str(tokens.dtype),
        "prompt_spec": dict(prompt_spec or {}),
    }
    if isinstance(extra_meta, dict) and extra_meta:
        # Keep extra metadata namespaced to avoid collisions with core fields.
        meta["extra"] = dict(extra_meta)

    # Write tokens first (so we don't leave a "valid" meta pointing to missing tokens).
    tmp_npz = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(cache_dir), suffix=".npz", delete=False) as f:
            tmp_npz = f.name
        np.savez_compressed(str(tmp_npz), ref_audio_tokens=tokens)
        os.replace(str(tmp_npz), str(tokens_path))
    finally:
        if tmp_npz:
            try:
                os.unlink(str(tmp_npz))
            except Exception:
                pass

    _atomic_write_json(meta_path, meta)


def analyze_prompt_audio_mono(mono: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Lightweight audio heuristics for prompt selection (best-effort)."""
    x = np.asarray(mono, dtype=np.float32).reshape(-1)
    sr = int(sample_rate or 0)
    if x.size <= 1 or sr <= 0:
        return {"rms": 0.0, "peak": 0.0, "p99_diff": 0.0, "hf_ratio_6k": 0.0}
    if not np.isfinite(x).all():
        return {"rms": 0.0, "peak": 0.0, "p99_diff": float("inf"), "hf_ratio_6k": float("inf")}

    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    peak = float(np.max(np.abs(x)))
    diffs = np.abs(np.diff(x))
    p99_diff = float(np.percentile(diffs, 99.0)) if diffs.size else 0.0

    # High-frequency energy ratio (simple, windowless rFFT).
    try:
        spec = np.fft.rfft(x.astype(np.float64, copy=False))
        mag2 = np.abs(spec) ** 2
        freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
        total = float(np.sum(mag2))
        if total <= 0:
            hf_ratio = 0.0
        else:
            hf = float(np.sum(mag2[freqs >= 6000.0]))
            hf_ratio = float(hf / total)
    except Exception:
        hf_ratio = 0.0

    return {"rms": rms, "peak": peak, "p99_diff": p99_diff, "hf_ratio_6k": hf_ratio}

