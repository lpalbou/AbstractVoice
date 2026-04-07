"""Voice profile abstraction (cross-engine).

This module defines a small, provider-agnostic representation for "voice profiles"
and a minimal built-in loader for curated preset packs (shipped as JSON assets).

Design goals
------------
- Keep the concept engine-agnostic: profiles are selected *after* choosing a TTS engine.
- Keep built-in presets lightweight: no binary assets; JSON-only.
- Avoid hard dependencies: this module is safe to import on minimal installs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def _norm_engine_id(engine_id: str | None) -> str:
    return str(engine_id or "").strip().lower()


def _norm_profile_id(profile_id: str | None) -> str:
    # Profile ids may be provider-defined (e.g. "voice_..."), so we preserve case.
    # Matching is typically done case-insensitively by the caller.
    return str(profile_id or "").strip()


@dataclass(frozen=True)
class VoiceProfile:
    """A provider-agnostic profile describing a "voice" for an engine.

    Notes:
    - `engine_id` is always normalized to lowercase.
    - `profile_id` is treated as engine-local and stable (e.g. `female_01`, `alloy`).
    """

    engine_id: str
    profile_id: str
    label: str
    description: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Optional[Dict[str, str]] = None
    provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_id", _norm_engine_id(self.engine_id))
        object.__setattr__(self, "profile_id", _norm_profile_id(self.profile_id))
        lbl = str(self.label or "").strip()
        object.__setattr__(self, "label", lbl if lbl else self.profile_id)

        # Ensure container types are dicts when present (defensive).
        try:
            if not isinstance(self.params, dict):
                object.__setattr__(self, "params", dict(self.params or {}))
        except Exception:
            object.__setattr__(self, "params", {})

        try:
            if self.tags is not None and not isinstance(self.tags, dict):
                object.__setattr__(self, "tags", dict(self.tags or {}))
        except Exception:
            object.__setattr__(self, "tags", None)

        try:
            if self.provenance is not None and not isinstance(self.provenance, dict):
                object.__setattr__(self, "provenance", dict(self.provenance or {}))
        except Exception:
            object.__setattr__(self, "provenance", None)

    @property
    def qualified_id(self) -> str:
        return f"{self.engine_id}:{self.profile_id}"


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except Exception:
        return {}


def _iter_profile_items(payload: Any) -> Iterable[Dict[str, Any]]:
    # Accept either a list of profiles or a dict like {"profiles": [...]}.
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        items = payload.get("profiles")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield item
        return


def voice_profile_from_dict(d: Dict[str, Any], *, engine_id: str) -> VoiceProfile:
    """Parse a `VoiceProfile` from a dict (best-effort, but validated)."""
    dd = _as_dict(d)
    eng = _norm_engine_id(dd.get("engine_id") or dd.get("engine") or engine_id)
    pid = _norm_profile_id(dd.get("profile_id") or dd.get("id") or dd.get("name"))
    if not eng:
        raise ValueError("VoiceProfile.engine_id is required")
    if not pid:
        raise ValueError("VoiceProfile.profile_id is required")
    label = str(dd.get("label") or dd.get("name") or pid).strip()
    desc = dd.get("description")
    desc_s = str(desc).strip() if isinstance(desc, str) and str(desc).strip() else None
    params = _as_dict(dd.get("params"))
    tags = dd.get("tags")
    tags_d: Optional[Dict[str, str]] = None
    if isinstance(tags, dict):
        tags_d = {str(k): str(v) for k, v in tags.items()}
    prov = dd.get("provenance")
    prov_d: Optional[Dict[str, Any]] = _as_dict(prov) if isinstance(prov, dict) else None
    return VoiceProfile(
        engine_id=eng,
        profile_id=pid,
        label=label,
        description=desc_s,
        params=params,
        tags=tags_d,
        provenance=prov_d,
    )


_BUILTIN_CACHE_LOCK = threading.Lock()
_BUILTIN_CACHE: Dict[str, List[VoiceProfile]] = {}


def get_builtin_voice_profiles(engine_id: str) -> List[VoiceProfile]:
    """Return built-in profiles for an engine (JSON asset), if present.

    File convention:
    - `abstractvoice/assets/voice_profiles/<engine_id>_profiles.json`
    """
    eng = _norm_engine_id(engine_id)
    if not eng:
        return []

    with _BUILTIN_CACHE_LOCK:
        cached = _BUILTIN_CACHE.get(eng)
        if cached is not None:
            return list(cached)

    # Lazy import to keep the module import-light.
    try:
        import importlib.resources as ir

        p = ir.files("abstractvoice").joinpath("assets", "voice_profiles", f"{eng}_profiles.json")
        raw = p.read_text(encoding="utf-8")
    except Exception:
        with _BUILTIN_CACHE_LOCK:
            _BUILTIN_CACHE[eng] = []
        return []

    profiles: List[VoiceProfile] = []
    try:
        payload = json.loads(raw)
        for item in _iter_profile_items(payload):
            try:
                profiles.append(voice_profile_from_dict(item, engine_id=eng))
            except Exception:
                continue
    except Exception:
        profiles = []

    with _BUILTIN_CACHE_LOCK:
        _BUILTIN_CACHE[eng] = list(profiles)
    return list(profiles)


def clear_builtin_voice_profiles_cache(engine_id: str | None = None) -> None:
    """Clear the in-process cache for built-in voice profiles.

    Useful for REPL/dev workflows where JSON assets may change during a session.
    """
    with _BUILTIN_CACHE_LOCK:
        if engine_id is None:
            _BUILTIN_CACHE.clear()
            return
        eng = _norm_engine_id(engine_id)
        if eng:
            _BUILTIN_CACHE.pop(eng, None)


def find_voice_profile(profiles: Iterable[VoiceProfile], profile_id: str) -> Optional[VoiceProfile]:
    """Case-insensitive lookup by `profile_id` (engine-local)."""
    key = str(profile_id or "").strip()
    if not key:
        return None
    key_l = key.lower()
    for p in profiles:
        try:
            if str(getattr(p, "profile_id", "") or "").strip().lower() == key_l:
                return p
        except Exception:
            continue
    return None

