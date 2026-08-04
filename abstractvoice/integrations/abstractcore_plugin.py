from __future__ import annotations

import io
import importlib.util
import os
import threading
import time
import wave
import weakref
from functools import partial
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Union

from ..artifacts import RuntimeArtifactStoreAdapter, is_artifact_ref, get_artifact_id


# Default discovery budget for remote engines. Listing a catalog is a UI-blocking
# round trip, so it gets its own budget instead of the operator's synthesis timeout
# (60s by default). It bounds both the socket and the whole parallel batch.
# Read through `_remote_discovery_timeout_s()`, never as a default argument value:
# a default is bound at def time and would silently ignore both the environment
# override and any test that patches this.
_REMOTE_DISCOVERY_TIMEOUT_S = 5.0

# Probe key for the active VoiceManager's own catalog: an engine id cannot collide
# with it, and the active manager is reached directly rather than by engine id.
_ACTIVE_PROBE = object()

# Local TTS providers the catalog can describe from disk. This is "which providers
# are local", a fixed fact, and is what decides whether a provider-filtered listing
# takes the light path. Whether one is currently INSTALLED and cached is a different
# question -- `_catalog_safe_local_tts_engines` -- and must not decide the route: a
# filter for an uninstalled local provider that fell through to the full catalog
# would build the ACTIVE engine to answer, which is both slow and, when that engine's
# extra is missing, a crash.
_CATALOG_LOCAL_TTS_PROVIDER_IDS = frozenset({"piper", "supertonic", "audiodit", "omnivoice", "qwen3-tts"})


# THE RULE FOR EVERY FIELD IN A DISCOVERY PAYLOAD, not just the ones below:
# no field may report the absence of DATA as the absence of the THING. If we did not
# get an answer, omit the field or flag it -- never publish the empty default as fact.
# "The host was unreachable" and "the host has no models" are different facts, and so
# are "we never asked" and "there is no active profile". Every regression in this area
# has been one field that was not asked this question.


class _TTSDiscovery:
    """What one provider told us about its TTS catalog.

    Republished whole each time a fetch lands. A discovery probe makes several calls
    on ONE HTTP adapter, so they must run sequentially, and an abandoned probe has to
    keep whatever it already paid for -- a provider that answered in 100ms must never
    be reported as unreachable because an optional follow-up fetch overran.

    Read `state` ONCE and unpack it. Rebinding a single attribute is atomic, so a
    reader sees every field from before a step or every field from after it, never a
    mix; two separate reads could see a landed catalog with `None` beside it.

    `catalog is None` means the catalog fetch never landed. That is the only fetch
    which speaks for the provider's catalog, so it is the single test for "did this
    provider answer" -- "we could not reach it" must never be published as "it has
    nothing".
    """

    __slots__ = ("state",)

    def __init__(self) -> None:
        # (catalog, model ids, profiles, active profile)
        self.state: tuple = (None, [], [], None)

_VM_CACHE_LOCK = threading.Lock()
_VM_CACHE: dict[tuple, Any] = {}
_VM_LOCKS: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()
_TRUE_BOOL_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_BOOL_VALUES = {"0", "false", "no", "n", "off"}
_CLONE_RESIDENCY_PROVIDER_ALIASES = {"cloned", "clone", "cloning"}
_OMNIVOICE_FALLBACK_LANGUAGES = ["en", "fr", "de", "es", "ru", "zh", "ja", "ko"]


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.environ.get(str(key), None)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def _env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    for key in keys:
        value = _env(str(key))
        if value is not None:
            return value
    return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in _TRUE_BOOL_VALUES:
        return True
    if text in _FALSE_BOOL_VALUES:
        return False
    return bool(default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env(str(key))
    return _coerce_bool(raw, default)


def _env_float(*keys: str) -> Optional[float]:
    raw = _env_first(*keys)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _configured_remote_timeout_s(cfg: Any) -> Optional[float]:
    """The remote timeout an operator configured, if any: config wins over env."""
    if isinstance(cfg, dict) and cfg.get("voice_remote_timeout_s") not in (None, ""):
        try:
            return float(cfg["voice_remote_timeout_s"])
        except (TypeError, ValueError):
            return None
    return _env_float(
        "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
        "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_TIMEOUT_S",
    )


def _remote_discovery_timeout_s() -> float:
    """The wall-clock budget one discovery listing gets for its remote probes."""
    override = _env_float("ABSTRACTVOICE_DISCOVERY_TIMEOUT_S")
    if override is not None and override > 0:
        return float(override)
    return float(_REMOTE_DISCOVERY_TIMEOUT_S)


def _probe_in_parallel(
    probes: Mapping[Hashable, Callable[[], Any]],
    *,
    budget_s: Optional[float] = None,
) -> Dict[Any, Any]:
    """Run every probe at once and return the ones that answered within the budget.

    A key is present only if that probe answered. Callers must treat an absent key
    as "no information", never as "this provider has nothing" -- failing to reach a
    provider and a provider replying with an empty catalog are different facts, and
    every structure derived from this result has to keep them apart.

    A probe that finishes just after the deadline may still land in the result:
    late data is real data, and dropping it would buy nothing.

    One task per remote provider, never two per provider: the calls inside a task
    share one HTTP adapter whose "already fetched" flags are not synchronised.
    Serially, an unreachable provider costs the whole budget and the next one
    starts from zero; concurrently the round trips overlap, so the batch costs the
    slowest provider plus the VoiceManager constructions, which still serialise on
    the process-wide `_VM_CACHE_LOCK` (~0.2s each for a remote adapter).

    The threads are daemons so an abandoned probe cannot hold the process open at
    exit -- a pool of ordinary workers is joined by the interpreter on the way
    out, which turned a bounded call into an unbounded, uninterruptible wait.
    """
    if not probes:
        return {}

    results: Dict[Any, Any] = {}
    results_lock = threading.Lock()
    finished = threading.Semaphore(0)

    def run(key: Hashable, probe: Callable[[], Any]) -> None:
        try:
            value = probe()
            with results_lock:
                results[key] = value
        except Exception:
            pass
        finally:
            finished.release()

    started = 0
    for key, probe in probes.items():
        thread = threading.Thread(
            target=run,
            args=(key, probe),
            name=f"abstractvoice-discovery-{key}",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            # Out of threads: the probes that did start still get their budget.
            break
        started += 1

    budget = _remote_discovery_timeout_s() if budget_s is None else float(budget_s)
    deadline = time.monotonic() + max(0.0, budget)
    for _ in range(started):
        if not finished.acquire(timeout=max(0.0, deadline - time.monotonic())):
            break  # Budget spent; whatever is still in flight is abandoned.
    with results_lock:
        return dict(results)


def _norm_residency_task(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"speech", "speech_synthesis", "text_to_speech"}:
        return "tts"
    if text in {"transcribe", "transcription", "speech_to_text", "audio_transcription"}:
        return "stt"
    return text


def _norm_residency_provider(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in _CLONE_RESIDENCY_PROVIDER_ALIASES:
        return "cloned"
    return _norm_engine_id(text)


def _clone_engine_is_local(value: Any) -> bool:
    return _norm_engine_id(value) not in {"openai", "openai-compatible", "remote"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _audio_chunk_to_wav_segment_bytes(audio_chunk: Any, sample_rate: Any) -> bytes:
    """Encode one mono float/int chunk as a standalone WAV segment."""
    import numpy as np

    sr = int(sample_rate or 0)
    if sr <= 0:
        raise ValueError("TTS stream chunk has invalid sample_rate")
    arr = np.asarray(audio_chunk).reshape(-1)
    if arr.size <= 0:
        return b""
    if arr.dtype.kind == "f":
        pcm = (np.clip(arr.astype(np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    else:
        pcm = arr.astype("<i2", copy=False)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _voice_profile_to_dict(profile: Any) -> Dict[str, Any]:
    if profile is None:
        return {}
    if isinstance(profile, dict):
        out = dict(_json_safe(profile) or {})
        engine_id = _norm_engine_id(out.get("provider_id") or out.get("provider") or out.get("engine_id") or out.get("engine"))
        if engine_id:
            out.setdefault("engine_id", engine_id)
            out.setdefault("engine", engine_id)
            out.setdefault("provider_id", engine_id)
            out.setdefault("provider", engine_id)
        return out

    engine_id = _norm_engine_id(getattr(profile, "engine_id", None))
    return {
        "engine_id": _json_safe(engine_id),
        "engine": _json_safe(engine_id),
        "provider_id": _json_safe(engine_id),
        "provider": _json_safe(engine_id),
        "profile_id": _json_safe(getattr(profile, "profile_id", None)),
        "label": _json_safe(getattr(profile, "label", None)),
        "description": _json_safe(getattr(profile, "description", None)),
        "params": _json_safe(getattr(profile, "params", None) or {}),
        "tags": _json_safe(getattr(profile, "tags", None)),
        "provenance": _json_safe(getattr(profile, "provenance", None)),
    }


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _dedupe_provider_ids(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        norm = _norm_engine_id(text) or text.lower()
        key = norm.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(norm or text)
    return out


def _norm_engine_id(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"remote", "compatible", "proxy"}:
        return "openai-compatible"
    if text in {"f5-tts", "f5tts", "openf5", "open-f5"}:
        return "f5_tts"
    if text == "faster-whisper":
        return "faster-whisper"
    return text


def _engine_aliases(value: Any) -> set[str]:
    engine = _norm_engine_id(value)
    if not engine:
        return set()
    aliases = {engine, engine.replace("-", "_")}
    if engine == "f5_tts":
        aliases.update({"f5-tts", "f5tts", "openf5", "open-f5"})
    if engine == "openai-compatible":
        aliases.update({"remote", "compatible", "proxy", "openai_compatible"})
    if engine in {"faster-whisper", "faster_whisper"}:
        aliases.update({"faster-whisper", "faster_whisper", "whisper", "local"})
    return aliases


def _local_tts_engines() -> list[str]:
    try:
        from ..adapters.tts_registry import get_supported_tts_engines

        engines = get_supported_tts_engines()
    except Exception:
        engines = []
    return [
        engine
        for engine in _dedupe_strings(engines)
        if _norm_engine_id(engine) not in {"", "auto", "openai", "openai-compatible"}
    ]


def _ordered_provider_ids(values: Any, preferred: list[str]) -> list[str]:
    normalized = _dedupe_provider_ids(values)
    order = {str(provider): index for index, provider in enumerate(preferred)}
    return sorted(normalized, key=lambda item: (order.get(item, len(order)), item))


def _known_tts_provider_ids() -> list[str]:
    try:
        from ..adapters.tts_registry import get_supported_tts_engines

        raw = get_supported_tts_engines()
    except Exception:
        raw = []
    providers = [
        _norm_engine_id(item)
        for item in list(raw or [])
        if _norm_engine_id(item) != "auto"
    ]
    # OpenAI is built into the lightweight install and must remain visible even
    # when no API key is configured or the active provider is local.
    providers.extend(["openai", "openai-compatible"])
    return _ordered_provider_ids(
        providers,
        ["openai", "openai-compatible", "supertonic", "piper", "audiodit", "omnivoice"],
    )


def _known_stt_provider_ids() -> list[str]:
    return _ordered_provider_ids(
        ["openai", "openai-compatible", "faster-whisper", "transformers-asr"],
        ["openai", "openai-compatible", "faster-whisper", "transformers-asr"],
    )


def _selectable_stt_model_ids(adapter_cls: Any, *, fallback: list[str]) -> list[str]:
    getter = getattr(adapter_cls, "selectable_model_ids", None)
    if callable(getter):
        try:
            values = getter()
            if isinstance(values, (list, tuple, set)):
                return _dedupe_strings(values)
        except Exception:
            pass

    for attr in ("KNOWN_MODELS", "MODELS"):
        catalog = getattr(adapter_cls, attr, None)
        if isinstance(catalog, dict):
            return _dedupe_strings(list(catalog.keys()))

    return _dedupe_strings(fallback)


def _stt_model_ids_for_provider(provider: Any) -> list[str]:
    normalized = _norm_engine_id(provider)
    if normalized in {"faster-whisper", "faster_whisper", "whisper", "local"}:
        try:
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            return _selectable_stt_model_ids(
                FasterWhisperAdapter,
                fallback=["tiny", "base", "small", "medium", "large-v2", "large-v3", "large"],
            )
        except Exception:
            return _dedupe_strings(["tiny", "base", "small", "medium", "large-v2", "large-v3", "large"])

    if normalized in {"transformers-asr", "transformers_asr", "hf", "hf-asr"}:
        try:
            from ..adapters.stt_transformers_asr import TransformersASRAdapter

            return _selectable_stt_model_ids(
                TransformersASRAdapter,
                fallback=[
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v3-turbo",
                    "Qwen/Qwen3-ASR-1.7B",
                    "whisper-large-v3",
                    "whisper-large-v3-turbo",
                    "qwen3-asr-1.7b",
                ],
            )
        except Exception:
            return _dedupe_strings(
                [
                    "openai/whisper-large-v3",
                    "openai/whisper-large-v3-turbo",
                    "Qwen/Qwen3-ASR-1.7B",
                    "whisper-large-v3",
                    "whisper-large-v3-turbo",
                    "qwen3-asr-1.7b",
                ]
            )

    return []


def _tts_provider_uses_language_models(provider: Any) -> bool:
    return False


def _tts_provider_accepts_language_model_selector(provider: Any) -> bool:
    """Return true for legacy TTS model values that are actually language hints."""

    return _norm_engine_id(provider) == "omnivoice"


def _omnivoice_language_ids() -> list[str]:
    """Return OmniVoice language selectors without loading model weights."""

    def order(values: list[str]) -> list[str]:
        preferred = {item: index for index, item in enumerate(_OMNIVOICE_FALLBACK_LANGUAGES)}
        return sorted(_dedupe_strings(values), key=lambda item: (preferred.get(item.lower(), len(preferred)), item.lower()))

    try:
        from omnivoice.utils.lang_map import LANG_IDS  # type: ignore

        return order([str(item).strip().lower() for item in LANG_IDS if str(item).strip()])
    except Exception:
        return list(_OMNIVOICE_FALLBACK_LANGUAGES)


def _selectable_tts_model_ids_for_provider(provider: Any, extra_candidates: Any = ()) -> list[str]:
    """Return public TTS model-selector values for local providers.

    The model selector must expose actual model/checkpoint ids, and only ones
    this machine can already speak with. Language and profile/voice choices are
    separate concerns; older saved OmniVoice defaults that used a language code
    as `model` are still accepted by `_tts_model_language_selector`.

    `extra_candidates` carries checkpoint ids an owner configured for THIS provider.
    Owners resolve them through `_configured_local_tts_model_ids`, which is the only
    place that decides which engine a configured checkpoint belongs to -- resolving
    it twice, once per source, drops an engine named in config whose model is named
    in the environment.
    """

    try:
        from ..local_models import cached_tts_model_ids

        return _dedupe_strings(
            cached_tts_model_ids(
                _norm_engine_id(provider),
                extra_candidates=[str(item) for item in extra_candidates or ()],
            )
        )
    except Exception:
        return []


def _local_tts_voice_profiles(engine: Any) -> list[Dict[str, Any]]:
    """Voice records for a local engine, without loading it.

    Most local engines ship a packaged profile asset. Piper does not, because its
    voices *are* its downloaded files, so its profiles come from the same disk
    probe that reports its model ids.
    """
    records: list[Dict[str, Any]] = []
    try:
        from ..voice_profiles import get_builtin_voice_profiles

        records.extend(_voice_profile_to_dict(profile) for profile in get_builtin_voice_profiles(engine))
    except Exception:
        pass
    if _norm_engine_id(engine) == "piper":
        try:
            from ..adapters.tts_piper import cached_piper_voice_profiles

            records.extend(_voice_profile_to_dict(profile) for profile in cached_piper_voice_profiles())
        except Exception:
            pass
    if _norm_engine_id(engine) == "qwen3-tts":
        # Same shape as piper: the voices live in the downloaded snapshots'
        # configs, not in a packaged asset, so read them from disk.
        try:
            from ..adapters.tts_qwen3_tts import cached_qwen3_tts_voice_profiles

            records.extend(_voice_profile_to_dict(profile) for profile in cached_qwen3_tts_voice_profiles())
        except Exception:
            pass
    return records


def _tts_model_language_selector(provider: Any, model: Any) -> Optional[str]:
    if not _tts_provider_accepts_language_model_selector(provider):
        return None
    text = str(model or "").strip().lower()
    if not text or text == "default":
        return None
    known_languages = {item.lower() for item in _omnivoice_language_ids()}
    return text if text in known_languages else None


def _resolve_tts_provider_request(provider: Any, model: Any = None) -> tuple[str, Optional[str]]:
    provider_text = str(provider or "").strip()
    model_text = str(model).strip() if isinstance(model, str) and model.strip() else None
    if provider_text:
        raw_provider, sep, raw_model = provider_text.partition(":")
        normalized_provider = _norm_engine_id(raw_provider if sep else provider_text)
        if sep and normalized_provider in _known_tts_provider_ids():
            provider_text = normalized_provider
            if model_text is None:
                candidate = str(raw_model or "").strip()
                if candidate:
                    model_text = candidate
        else:
            provider_text = normalized_provider
    return _norm_engine_id(provider_text), model_text


def _resolve_stt_provider_request(provider: Any, model: Any = None) -> tuple[str, Optional[str]]:
    provider_text = str(provider or "").strip()
    model_text = str(model).strip() if isinstance(model, str) and model.strip() else None
    if provider_text:
        raw_provider, sep, raw_model = provider_text.partition(":")
        normalized_provider = _norm_engine_id(raw_provider if sep else provider_text)
        if sep and normalized_provider in _known_stt_provider_ids():
            provider_text = normalized_provider
            if model_text is None:
                candidate = str(raw_model or "").strip()
                if candidate:
                    model_text = candidate
        else:
            provider_text = normalized_provider
    return _norm_engine_id(provider_text), model_text


def _resolve_cloning_provider_request(provider: Any, model: Any = None) -> tuple[str, Optional[str]]:
    provider_text = str(provider or "").strip()
    model_text = str(model).strip() if isinstance(model, str) and model.strip() else None
    if provider_text:
        raw_provider, sep, raw_model = provider_text.partition(":")
        normalized_provider = _norm_engine_id(raw_provider if sep else provider_text)
        if sep and normalized_provider in _known_cloning_provider_ids():
            provider_text = normalized_provider
            if model_text is None:
                candidate = str(raw_model or "").strip()
                if candidate:
                    model_text = candidate
        else:
            provider_text = normalized_provider
    return _norm_engine_id(provider_text), model_text


def _normalize_optional_model_id(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower() == "default":
        return None
    return text


def _norm_compat_provider_id(kind: Any, provider: Any) -> str:
    normalized_kind = str(kind or "").strip().lower()
    normalized_provider = _norm_engine_id(provider)
    if normalized_kind == "stt":
        if normalized_provider in {"whisper", "local"}:
            return "faster-whisper"
        if normalized_provider in {"transformers", "hf", "hf-asr"}:
            return "transformers-asr"
    return normalized_provider


def _normalize_support_levels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = str(value).strip()
        return (text,) if text else ()
    return tuple(str(item) for item in list(value or ()) if str(item).strip())


def _known_cloning_provider_ids() -> list[str]:
    return _ordered_provider_ids(
        ["omnivoice", "f5_tts", "chroma", "audiodit", "qwen3-tts", "openai", "openai-compatible"],
        ["omnivoice", "f5_tts", "chroma", "audiodit", "qwen3-tts", "openai", "openai-compatible"],
    )


def _runtime_installed(kind: str, provider: Any) -> bool | None:
    engine = _norm_engine_id(provider)
    if not engine:
        return None
    if engine in {"openai", "openai-compatible"}:
        return True
    if kind == "stt" and engine == "faster-whisper":
        return importlib.util.find_spec("faster_whisper") is not None
    if kind == "stt" and engine == "transformers-asr":
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("soundfile") is not None
        )
    if engine == "piper":
        return (
            importlib.util.find_spec("piper") is not None
            or importlib.util.find_spec("piper_phonemize") is not None
        )
    if engine == "supertonic":
        return importlib.util.find_spec("onnxruntime") is not None
    if engine == "omnivoice":
        return importlib.util.find_spec("omnivoice") is not None
    if engine == "f5_tts":
        return importlib.util.find_spec("f5_tts") is not None
    if engine in {"audiodit", "chroma", "qwen3-tts"}:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )
    return None


def _local_tts_engine_available(engine: Any, extra_candidates: Any = ()) -> bool:
    """True when a local TTS engine is installed and has weights on this machine.

    Both halves are cheap by construction: the runtime check is `find_spec`, and
    presence is a filesystem lookup (see `abstractvoice.local_models`). Neither
    imports an engine or builds an adapter -- the AudioDiT adapter alone drags in
    torch and transformers, and only to report a catalog the cache already knows.

    Availability *is* "has a selectable model", asked through the same call, so an
    engine can never be listed with nothing to select or hidden while selectable.
    """
    normalized = _norm_engine_id(engine)
    if not normalized or normalized in {"openai", "openai-compatible"}:
        return False
    if not _engine_runtime_available(normalized):
        return False
    return bool(_selectable_tts_model_ids_for_provider(normalized, extra_candidates))


def _local_stt_engine_available(engine: Any) -> bool:
    normalized = _norm_engine_id(engine)
    if normalized == "faster-whisper":
        return bool(_runtime_installed("stt", normalized))
    if normalized == "transformers-asr":
        return bool(_runtime_installed("stt", normalized))
    return False


def _local_cloning_engine_available(engine: Any) -> bool:
    normalized = _norm_engine_id(engine)
    if normalized in {"omnivoice", "f5_tts", "chroma", "audiodit", "qwen3-tts"}:
        return bool(_runtime_installed("cloning", normalized))
    return False


def _provider_details(kind: str, providers: Any) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for provider in _dedupe_provider_ids(providers):
        remote = provider in {"openai", "openai-compatible"}
        details[provider] = {
            "id": provider,
            "provider": provider,
            "kind": str(kind),
            "remote": remote,
            "local": not remote,
            "installed": _runtime_installed(str(kind), provider),
        }
    return details


def _catalog_safe_local_tts_engines(candidates_for: Any = None) -> list[str]:
    """Local engines installed enough to expose catalog metadata.

    `candidates_for(engine)` supplies owner-configured checkpoint ids for one
    engine. It is asked per engine, not pooled: a checkpoint belongs to the single
    engine it was configured for.
    """

    return [
        engine
        for engine in _local_tts_engines()
        if _norm_engine_id(engine) in _CATALOG_LOCAL_TTS_PROVIDER_IDS
        and _local_tts_engine_available(engine, candidates_for(engine) if candidates_for else ())
    ]


def _engine_runtime_available(engine: Any, configured_providers: Any = ()) -> bool:
    normalized = _norm_engine_id(engine)
    if not normalized:
        return False
    configured_aliases: set[str] = set()
    for provider in list(configured_providers or []):
        configured_aliases.update(_engine_aliases(provider))
    if _engine_aliases(normalized) & configured_aliases:
        return True
    if normalized in {"openai", "openai-compatible"}:
        return False
    if normalized == "piper":
        return importlib.util.find_spec("piper") is not None or importlib.util.find_spec("piper_phonemize") is not None
    if normalized == "supertonic":
        return importlib.util.find_spec("onnxruntime") is not None
    if normalized == "omnivoice":
        return importlib.util.find_spec("omnivoice") is not None
    if normalized == "f5_tts":
        return importlib.util.find_spec("f5_tts") is not None
    if normalized == "audiodit":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    if normalized == "qwen3-tts":
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    return normalized in {_norm_engine_id(provider) for provider in _local_tts_engines()}


def _extract_tts_model_ids(catalog: Any) -> list[str]:
    model_ids: list[str] = []

    def add_many(values: Any) -> None:
        if isinstance(values, str):
            model_ids.append(values)
        elif isinstance(values, (list, tuple, set)):
            for value in values:
                if isinstance(value, str):
                    model_ids.append(value)
                elif isinstance(value, dict):
                    for key in ("id", "model", "model_id", "name"):
                        item = value.get(key)
                        if isinstance(item, str):
                            model_ids.append(item)
                            break

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            usable = node.get("cached") is not False or node.get("remote") is True
            if usable:
                for key in ("available_models", "tts_models", "speech_models", "audio_speech_models"):
                    add_many(node.get(key))
                for key in ("model", "model_id", "model_filename"):
                    value = node.get(key)
                    if isinstance(value, str):
                        model_ids.append(value)
            for value in node.values():
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item)

    visit(catalog)
    return _dedupe_strings(model_ids)


def _extract_provider_ids(*values: Any) -> list[str]:
    providers: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            providers.append(value.strip())

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            tags = node.get("tags")
            if isinstance(tags, dict):
                add(tags.get("provider"))
                add(tags.get("engine_id"))
            for key in ("provider", "provider_id", "engine", "engine_id", "backend", "backend_id"):
                add(node.get(key))
            for value in node.values():
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item)

    for value in values:
        visit(value)
    return _dedupe_strings(providers)


def _extract_tts_provider_ids(vm: Any, catalog: Any, profiles: Any) -> list[str]:
    adapter = getattr(vm, "tts_adapter", None)
    return _extract_provider_ids(
        {
            "engine_id": getattr(adapter, "engine_id", None),
            "provider": getattr(adapter, "provider", None),
            "backend_id": getattr(adapter, "backend_id", None),
        },
        {
            "engine_id": getattr(vm, "tts_engine", None),
            "provider_id": getattr(vm, "_abstractvoice_tts_engine", None),
            "provider": getattr(vm, "_tts_engine_name", None),
            "backend_id": getattr(vm, "tts_backend_id", None),
        },
        profiles,
        catalog,
    )


def _extract_stt_provider_ids(vm: Any) -> list[str]:
    adapter = getattr(vm, "stt_adapter", None)
    return _extract_provider_ids(
        {
            "engine_id": getattr(adapter, "engine_id", None),
            "provider": getattr(adapter, "provider", None),
            "backend_id": getattr(adapter, "backend_id", None),
        },
        {
            "engine_id": getattr(vm, "stt_engine", None),
            "provider_id": getattr(vm, "_abstractvoice_stt_engine", None),
            "provider": getattr(vm, "_stt_engine_name", None),
            "backend_id": getattr(vm, "stt_backend_id", None),
        },
    )


def _extract_stt_model_ids(vm: Any) -> list[str]:
    model_ids: list[str] = []
    adapter = getattr(vm, "stt_adapter", None)
    for key in (
        "ABSTRACTVOICE_STT_MODEL",
        "ABSTRACTVOICE_OPENAI_STT_MODEL",
        "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
        "ABSTRACTVOICE_REMOTE_STT_MODEL",
    ):
        value = _env(key)
        if isinstance(value, str) and value.strip():
            model_ids.extend(value.split(","))

    for target in (vm, adapter):
        if target is None:
            continue
        for attr in ("stt_model", "model_id", "model", "model_size", "_model_size"):
            value = getattr(target, attr, None)
            if isinstance(value, str) and value.strip():
                model_ids.append(value.strip())

    live_local_provider = _norm_engine_id(
        getattr(adapter, "engine_id", None)
        or getattr(adapter, "provider", None)
        or getattr(adapter, "backend_id", None)
    )
    engine = _norm_engine_id(
        getattr(vm, "_abstractvoice_stt_engine", None)
        or getattr(vm, "_stt_engine_name", None)
        or getattr(vm, "_stt_engine_preference", None)
        or getattr(vm, "stt_engine", None)
        or _env("ABSTRACTVOICE_STT_ENGINE", "openai")
        or "openai"
    )
    if engine in {"openai", "openai-compatible", "remote"}:
        model_ids.extend(["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
    if engine in {"faster_whisper", "faster-whisper", "whisper", "local"} and (
        live_local_provider == "faster-whisper" or _local_stt_engine_available("faster-whisper")
    ):
        model_ids.extend(_stt_model_ids_for_provider("faster-whisper"))
    if engine in {"transformers-asr", "transformers_asr", "hf", "hf-asr"} and (
        live_local_provider == "transformers-asr" or _local_stt_engine_available("transformers-asr")
    ):
        model_ids.extend(_stt_model_ids_for_provider("transformers-asr"))
    return _dedupe_strings(model_ids)


def _active_tts_model(vm: Any, catalog: Any, model_ids: list[str]) -> Optional[str]:
    adapter = getattr(vm, "tts_adapter", None)
    provider = _norm_engine_id(
        getattr(adapter, "engine_id", None)
        or getattr(adapter, "provider", None)
        or getattr(adapter, "backend_id", None)
        or getattr(vm, "_abstractvoice_tts_engine", None)
        or getattr(vm, "_tts_engine_name", None)
        or getattr(vm, "_tts_engine_preference", None)
    )
    if _tts_provider_uses_language_models(provider):
        language = getattr(adapter, "_language", None) or getattr(adapter, "language", None) or getattr(vm, "language", None)
        if isinstance(language, str) and language.strip():
            return language.strip().lower()

    model = getattr(adapter, "model_id", None)
    if isinstance(model, str) and model.strip():
        return model.strip()

    if isinstance(catalog, dict):
        for engine_catalog in catalog.values():
            if not isinstance(engine_catalog, dict):
                continue
            for item in engine_catalog.values():
                if isinstance(item, dict):
                    model = item.get("model") or item.get("model_id")
                    if isinstance(model, str) and model.strip():
                        return model.strip()
    return model_ids[0] if model_ids else None


def _current_tts_model_ids(vm: Any) -> list[str]:
    model_ids: list[str] = []
    adapter = getattr(vm, "tts_adapter", None)
    for target in (vm, adapter):
        if target is None:
            continue
        for attr in ("tts_model", "model_id", "model"):
            value = getattr(target, attr, None)
            if isinstance(value, str) and value.strip():
                model_ids.append(value.strip())
    return _dedupe_strings(model_ids)


def _current_stt_model_ids(vm: Any) -> list[str]:
    model_ids: list[str] = []
    adapter = getattr(vm, "stt_adapter", None)
    for target in (vm, adapter):
        if target is None:
            continue
        for attr in ("stt_model", "model_id", "model", "model_size", "_model_size"):
            value = getattr(target, attr, None)
            if isinstance(value, str) and value.strip():
                model_ids.append(value.strip())
    if _norm_engine_id(
        getattr(vm, "_abstractvoice_stt_engine", None)
        or getattr(vm, "_stt_engine_name", None)
        or getattr(vm, "_stt_engine_preference", None)
        or getattr(vm, "stt_engine", None)
    ) in {"faster-whisper", "faster_whisper", "whisper", "local"}:
        whisper_model = getattr(vm, "whisper_model", None)
        if isinstance(whisper_model, str) and whisper_model.strip():
            model_ids.append(whisper_model.strip())
    return _dedupe_strings(model_ids)


def _root_tts_catalog_has_provider_sections(catalog: Any) -> bool:
    if not isinstance(catalog, dict):
        return False
    known = set(_known_tts_provider_ids())
    return any(_norm_engine_id(key) in known for key in catalog.keys())


def _catalog_tts_group_provider(group_key: Any, default_provider: Any, *, provider_keyed_root: bool) -> str:
    provider_id = _norm_engine_id(group_key)
    if provider_id in _known_tts_provider_ids():
        return provider_id
    if provider_keyed_root:
        return ""
    return _norm_engine_id(default_provider)


def _extract_tts_models_by_provider(catalog: Any, *, default_provider: Any = None) -> dict[str, list[str]]:
    if not isinstance(catalog, dict):
        return {}

    models_by_provider: dict[str, list[str]] = {}
    provider_keyed_root = _root_tts_catalog_has_provider_sections(catalog)

    def visit(node: Any, provider: str) -> None:
        if isinstance(node, dict):
            usable = node.get("cached") is not False or node.get("remote") is True
            if provider and usable:
                for key in ("available_models", "tts_models", "speech_models", "audio_speech_models"):
                    values = node.get(key)
                    if isinstance(values, str):
                        _add_provider_value(models_by_provider, provider, values)
                    elif isinstance(values, (list, tuple, set)):
                        for value in values:
                            if isinstance(value, str) and value.strip():
                                _add_provider_value(models_by_provider, provider, value)
                            elif isinstance(value, dict):
                                for item_key in ("id", "model", "model_id", "name"):
                                    item = value.get(item_key)
                                    if isinstance(item, str) and item.strip():
                                        _add_provider_value(models_by_provider, provider, item)
                                        break
                for key in ("model", "model_id", "model_filename"):
                    value = node.get(key)
                    if isinstance(value, str) and value.strip():
                        _add_provider_value(models_by_provider, provider, value)
            for value in node.values():
                if isinstance(value, (dict, list, tuple, set)):
                    visit(value, provider)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                visit(item, provider)

    for group_key, group_value in catalog.items():
        provider = _catalog_tts_group_provider(group_key, default_provider, provider_keyed_root=provider_keyed_root)
        explicit_provider = _extract_provider_ids(group_value)
        if explicit_provider:
            provider = _norm_engine_id(explicit_provider[0]) or provider
        visit(group_value, provider)

    return models_by_provider


def _profiles_from_tts_catalog(catalog: Any, *, engine_id: str) -> list[Dict[str, Any]]:
    """Build profile-like records from adapter catalogs that expose cached voices."""
    engine = _norm_engine_id(engine_id)
    if not engine or not isinstance(catalog, dict):
        return []

    profiles: list[Dict[str, Any]] = []
    provider_keyed_root = _root_tts_catalog_has_provider_sections(catalog)

    for group_key, group_value in catalog.items():
        if not isinstance(group_value, dict):
            continue
        group_provider = _catalog_tts_group_provider(group_key, engine, provider_keyed_root=provider_keyed_root)
        for voice_key, raw in group_value.items():
            if not isinstance(raw, dict):
                continue
            if raw.get("cached") is False and raw.get("remote") is not True:
                continue
            entry_provider = _norm_engine_id(
                raw.get("provider")
                or raw.get("provider_id")
                or raw.get("engine")
                or raw.get("engine_id")
                or group_provider
            )
            if not entry_provider:
                continue
            model_id = raw.get("model") or raw.get("model_id") or raw.get("model_filename")
            profile_id = str(raw.get("voice") or raw.get("voice_id") or voice_key or model_id or "").strip()
            if not profile_id:
                continue
            label = str(raw.get("name") or raw.get("label") or profile_id).strip() or profile_id
            params: Dict[str, Any] = {
                "provider": entry_provider,
                "voice": str(raw.get("voice") or voice_key).strip(),
            }
            if not provider_keyed_root:
                params["language"] = str(group_key).strip()
            if isinstance(model_id, str) and model_id.strip():
                params["model"] = model_id.strip()
            profiles.append(
                {
                    "engine_id": entry_provider,
                    "engine": entry_provider,
                    "provider_id": entry_provider,
                    "provider": entry_provider,
                    "profile_id": profile_id,
                    "id": profile_id,
                    "label": label,
                    "description": _json_safe(raw.get("description")),
                    "params": params,
                    "tags": {
                        "provider": entry_provider,
                        "engine_id": entry_provider,
                        "provider_id": entry_provider,
                        "cached": "true",
                    },
                    "provenance": "abstractvoice.adapter_catalog",
                }
            )

    return profiles


def _piper_language_for_model(model_id: str) -> Optional[str]:
    requested = str(model_id or "").strip().lower()
    if not requested:
        return None
    try:
        from ..adapters.tts_piper import PiperTTSAdapter
    except Exception:
        return None

    for language, raw in getattr(PiperTTSAdapter, "PIPER_MODELS", {}).items():
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            continue
        hf_path, model_filename = str(raw[0] or ""), str(raw[1] or "")
        parts = hf_path.split("/")
        voice_id = parts[2] if len(parts) >= 3 else ""
        candidates = {
            str(language).lower(),
            model_filename.lower(),
            f"{language}:{model_filename}".lower(),
            f"{language}/{model_filename}".lower(),
        }
        if voice_id:
            candidates.update(
                {
                    voice_id.lower(),
                    f"{language}.{voice_id}".lower(),
                    f"{language}:{voice_id}".lower(),
                }
            )
        if requested in candidates:
            return str(language)
    return None


def _tts_formats_for_provider(provider: Any) -> list[str]:
    engine = _norm_engine_id(provider)
    if engine == "piper":
        return ["wav"]
    if engine in {"openai", "openai-compatible", "remote"}:
        return ["mp3", "opus", "aac", "flac", "wav", "pcm"]
    return ["wav"]


def _stt_formats_for_provider(provider: Any) -> list[str]:
    engine = _norm_engine_id(provider)
    if engine in {"openai", "openai-compatible", "remote"}:
        return ["json", "text", "verbose_json", "srt", "vtt"]
    return ["json", "text"]


def _profile_provider_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    tags = profile.get("tags")
    params = profile.get("params")
    if not isinstance(tags, dict):
        tags = {}
    if not isinstance(params, dict):
        params = {}
    return _norm_engine_id(
        profile.get("provider")
        or profile.get("engine_id")
        or tags.get("provider")
        or tags.get("engine_id")
        or params.get("provider")
        or params.get("engine_id")
    )


def _profile_model_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    params = profile.get("params")
    if not isinstance(params, dict):
        params = {}
    for value in (
        params.get("model"),
        params.get("model_id"),
        params.get("model_filename"),
        profile.get("model"),
        profile.get("model_id"),
        profile.get("language"),
        params.get("language"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profile_voice_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    params = profile.get("params")
    if not isinstance(params, dict):
        params = {}
    for value in (
        profile.get("voice_id"),
        params.get("voice"),
        profile.get("voice"),
        profile.get("profile_id"),
        profile.get("id"),
        profile.get("name"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _profile_id(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    for value in (profile.get("profile_id"), profile.get("id"), profile.get("name"), _profile_voice_id(profile)):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _add_provider_value(mapping: dict[str, list[str]], provider: Any, value: Any) -> None:
    provider_id = _norm_engine_id(provider)
    text = str(value or "").strip()
    if not provider_id or not text:
        return
    values = mapping.setdefault(provider_id, [])
    if text.lower() not in {item.lower() for item in values}:
        values.append(text)


def _order_like(values: Any, preferred: Any) -> list[str]:
    deduped = _dedupe_strings(values)
    preferred_keys = [str(item).strip().lower() for item in list(preferred or []) if str(item or "").strip()]
    order = {item: index for index, item in enumerate(preferred_keys)}
    return sorted(deduped, key=lambda item: (order.get(item.lower(), len(order)), item.lower()))


def _provider_variants(provider: Any, values: Any) -> list[str]:
    provider_id = _norm_engine_id(provider)
    if not provider_id:
        return []
    out = [provider_id]
    for value in list(values or []):
        text = str(value or "").strip()
        if text:
            out.append(f"{provider_id}:{text}")
    return _dedupe_strings(out)


def _voice_matches_tts_selection(voice: Any, *, provider: Any = None, model: Any = None) -> bool:
    if not isinstance(voice, dict):
        return False
    provider_id = _norm_engine_id(provider)
    voice_provider = _profile_provider_id(voice)
    if provider_id and not (_engine_aliases(provider_id) & _engine_aliases(voice_provider)):
        return False

    model_text = str(model or "").strip()
    if not model_text:
        return True
    voice_model = _profile_model_id(voice)
    if not voice_model:
        return True
    return voice_model.strip().lower() == model_text.lower()


def _dedupe_voice_records(values: Any) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for voice in list(values or []):
        if not isinstance(voice, dict):
            continue
        provider = _profile_provider_id(voice).lower()
        kind = str(voice.get("kind") or "profile").strip().lower() or "profile"
        voice_id = (_profile_id(voice) or _profile_voice_id(voice)).lower()
        if voice_id:
            key = (provider, kind, voice_id)
            if key in seen:
                continue
            seen.add(key)
        out.append(dict(voice))
    return out


class _BaseVoice:
    def __init__(self, owner: Any):
        self._owner = owner
        self._vm = None

    def _vm_lock(self, vm: Any) -> threading.Lock:
        """Per-VoiceManager lock for synthesis/metrics consistency."""
        with _VM_CACHE_LOCK:
            lk = _VM_LOCKS.get(vm)
            if lk is None:
                lk = threading.Lock()
                _VM_LOCKS[vm] = lk
            return lk

    def _get_vm(self):
        if self._vm is not None:
            return self._vm

        # Injection hook (tests / advanced embedding).
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                inst = cfg.get("voice_manager_instance")
                if inst is not None:
                    self._vm = inst
                    return self._vm
                factory = cfg.get("voice_manager_factory")
                if callable(factory):
                    self._vm = factory(self._owner)
                    return self._vm
        except Exception:
            pass

        # Lazy import (keeps plugin import-light).
        from ..voice_manager import VoiceManager

        # Best-effort config overrides (optional). Hosted integrations and the
        # public VoiceManager default both select OpenAI remote engines.
        language = _env("ABSTRACTVOICE_LANGUAGE", "en") or "en"
        allow_downloads = _env_bool("ABSTRACTVOICE_ALLOW_DOWNLOADS", True)
        tts_engine = _env("ABSTRACTVOICE_TTS_ENGINE", "openai") or "openai"
        stt_engine = _env("ABSTRACTVOICE_STT_ENGINE", "openai") or "openai"
        whisper_model = _env("ABSTRACTVOICE_WHISPER_MODEL", "base") or "base"
        stt_model = _env_first(
            "ABSTRACTVOICE_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
            "ABSTRACTVOICE_REMOTE_STT_MODEL",
        )
        tts_model = _env_first(
            "ABSTRACTVOICE_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL",
        )
        cloning_engine = _env("ABSTRACTVOICE_CLONING_ENGINE", "omnivoice") or "omnivoice"
        cloned_tts_streaming = _env_bool("ABSTRACTVOICE_CLONED_TTS_STREAMING", True)
        tts_delivery_mode = _env("ABSTRACTVOICE_TTS_DELIVERY_MODE")
        remote_base_url = _env("OPENAI_BASE_URL")
        remote_api_key = None
        remote_timeout_s = _env_float(
            "ABSTRACTVOICE_REMOTE_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_TIMEOUT_S",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TIMEOUT_S",
        )
        debug_mode = _env_bool("ABSTRACTVOICE_DEBUG", False)
        try:
            cfg = getattr(self._owner, "config", None)
            if isinstance(cfg, dict):
                if isinstance(cfg.get("voice_language"), str) and cfg["voice_language"].strip():
                    language = str(cfg["voice_language"]).strip().lower()
                if "voice_allow_downloads" in cfg:
                    allow_downloads = _coerce_bool(cfg.get("voice_allow_downloads"), allow_downloads)
                if isinstance(cfg.get("voice_tts_engine"), str) and str(cfg["voice_tts_engine"]).strip():
                    tts_engine = str(cfg["voice_tts_engine"]).strip().lower()
                if isinstance(cfg.get("voice_stt_engine"), str) and str(cfg["voice_stt_engine"]).strip():
                    stt_engine = str(cfg["voice_stt_engine"]).strip().lower()
                if isinstance(cfg.get("voice_whisper_model"), str) and str(cfg["voice_whisper_model"]).strip():
                    whisper_model = str(cfg["voice_whisper_model"]).strip()
                if isinstance(cfg.get("voice_tts_model"), str) and str(cfg["voice_tts_model"]).strip():
                    tts_model = str(cfg["voice_tts_model"]).strip()
                if isinstance(cfg.get("voice_stt_model"), str) and str(cfg["voice_stt_model"]).strip():
                    stt_model = str(cfg["voice_stt_model"]).strip()
                if isinstance(cfg.get("voice_cloning_engine"), str) and str(cfg["voice_cloning_engine"]).strip():
                    cloning_engine = str(cfg["voice_cloning_engine"]).strip().lower()
                if isinstance(cfg.get("voice_remote_base_url"), str) and str(cfg["voice_remote_base_url"]).strip():
                    remote_base_url = str(cfg["voice_remote_base_url"]).strip()
                if isinstance(cfg.get("voice_remote_api_key"), str) and str(cfg["voice_remote_api_key"]).strip():
                    remote_api_key = str(cfg["voice_remote_api_key"]).strip()
                if cfg.get("voice_remote_timeout_s") not in (None, ""):
                    try:
                        remote_timeout_s = float(cfg.get("voice_remote_timeout_s"))
                    except Exception:
                        remote_timeout_s = None
                if "voice_cloned_tts_streaming" in cfg:
                    cloned_tts_streaming = _coerce_bool(
                        cfg.get("voice_cloned_tts_streaming"),
                        cloned_tts_streaming,
                    )
                # Unified override for delivery mode (applies to base + clone).
                # Accept either a mode string (buffered|streamed) or a bool-ish flag.
                if "voice_tts_delivery_mode" in cfg:
                    raw = cfg.get("voice_tts_delivery_mode")
                    if raw is not None and str(raw).strip():
                        try:
                            from ..tts.delivery_mode import normalize_audio_delivery_mode

                            tts_delivery_mode = normalize_audio_delivery_mode(str(raw))
                        except Exception:
                            tts_delivery_mode = None
                elif "voice_tts_streaming" in cfg:
                    raw = cfg.get("voice_tts_streaming")
                    try:
                        from ..tts.delivery_mode import normalize_audio_delivery_mode

                        tts_delivery_mode = normalize_audio_delivery_mode(raw)
                    except Exception:
                        tts_delivery_mode = None
                if "voice_debug_mode" in cfg:
                    debug_mode = _coerce_bool(cfg.get("voice_debug_mode"), debug_mode)
        except Exception:
            pass

        key = (
            str(language),
            bool(allow_downloads),
            str(tts_engine),
            str(stt_engine),
            str(whisper_model),
            str(tts_model or ""),
            str(stt_model or ""),
            str(cloning_engine),
            bool(cloned_tts_streaming),
            str(tts_delivery_mode) if tts_delivery_mode else "",
            str(remote_base_url or ""),
            str(remote_api_key or ""),
            str(remote_timeout_s or ""),
            bool(debug_mode),
        )

        with _VM_CACHE_LOCK:
            cached = _VM_CACHE.get(key)
            if cached is None:
                cached = VoiceManager(
                    language=language,
                    allow_downloads=allow_downloads,
                    debug_mode=bool(debug_mode),
                    tts_engine=str(tts_engine),
                    stt_engine=str(stt_engine),
                    whisper_model=str(whisper_model),
                    tts_model=str(tts_model) if tts_model else None,
                    stt_model=str(stt_model) if stt_model else None,
                    cloning_engine=str(cloning_engine),
                    cloned_tts_streaming=bool(cloned_tts_streaming),
                    tts_delivery_mode=str(tts_delivery_mode) if tts_delivery_mode else None,
                    remote_base_url=str(remote_base_url) if remote_base_url else None,
                    remote_api_key=str(remote_api_key) if remote_api_key else None,
                    remote_timeout_s=remote_timeout_s,
                )
                _VM_CACHE[key] = cached
                _VM_LOCKS[cached] = threading.Lock()
            try:
                setattr(cached, "_abstractvoice_tts_engine", str(tts_engine))
                setattr(cached, "_abstractvoice_stt_engine", str(stt_engine))
            except Exception:
                pass
            self._vm = cached
            return self._vm

    def _iter_known_vms(self):
        seen: set[int] = set()
        try:
            current = self._vm
        except Exception:
            current = None
        if current is None:
            try:
                cfg = getattr(self._owner, "config", None)
                if isinstance(cfg, dict):
                    inst = cfg.get("voice_manager_instance")
                    if inst is not None:
                        current = inst
            except Exception:
                current = None
        if current is not None:
            key = id(current)
            if key not in seen:
                seen.add(key)
                yield None, current
        with _VM_CACHE_LOCK:
            cached_items = list(_VM_CACHE.items())
        for cache_key, vm in cached_items:
            if vm is None:
                continue
            key = id(vm)
            if key in seen:
                continue
            seen.add(key)
            yield cache_key, vm

    def _vm_engine_values(self, vm: Any, *, kind: str) -> set[str]:
        adapter = getattr(vm, "tts_adapter", None) if kind == "tts" else getattr(vm, "stt_adapter", None)
        values = {
            getattr(adapter, "engine_id", None),
            getattr(adapter, "provider", None),
        }
        if kind == "tts":
            values.update(
                {
                    getattr(vm, "_abstractvoice_tts_engine", None),
                    getattr(vm, "_tts_engine_name", None),
                    getattr(vm, "_tts_engine_preference", None),
                    getattr(vm, "tts_engine", None),
                }
            )
        else:
            values.update(
                {
                    getattr(vm, "_abstractvoice_stt_engine", None),
                    getattr(vm, "_stt_engine_name", None),
                    getattr(vm, "_stt_engine_preference", None),
                    getattr(vm, "stt_engine", None),
                }
            )
        out: set[str] = set()
        for value in values:
            out.update(_engine_aliases(value))
        return out

    def _get_vm_for_cloning_engine(self, cloning_engine: Optional[str] = None):
        engine = _norm_engine_id(cloning_engine)
        if not engine:
            return self._get_vm()

        current = None
        try:
            current = self._get_vm()
        except Exception:
            current = None
        if current is not None:
            current_engine = _norm_engine_id(getattr(current, "cloning_engine", None))
            if current_engine and current_engine == engine:
                return current

        cfg = getattr(self._owner, "config", None)
        override_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        override_cfg.pop("voice_manager_instance", None)
        override_cfg.pop("voice_manager_factory", None)
        override_cfg["voice_cloning_engine"] = engine

        owner = type("_AbstractVoiceCloningEngineOverride", (), {"config": override_cfg})()
        cap = self.__class__(owner)
        return cap._get_vm()

    def _get_vm_for_clone_request(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        cloning_engine: Optional[str] = None,
    ):
        provider_id, requested_model = _resolve_cloning_provider_request(provider, model)
        requested_model = _normalize_optional_model_id(requested_model)
        engine = _norm_engine_id(cloning_engine or provider_id)
        remote_model_capable = engine in {"openai", "openai-compatible"}
        if not engine:
            return self._get_vm()

        current = None
        try:
            current = self._get_vm()
        except Exception:
            current = None
        if current is not None:
            current_engine = _norm_engine_id(getattr(current, "cloning_engine", None))
            current_tts_models = {item.lower() for item in _current_tts_model_ids(current)}
            model_ok = (not remote_model_capable) or requested_model is None or requested_model.lower() in current_tts_models
            if current_engine == engine and model_ok:
                return current

        cfg = getattr(self._owner, "config", None)
        override_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        override_cfg.pop("voice_manager_instance", None)
        override_cfg.pop("voice_manager_factory", None)
        override_cfg["voice_cloning_engine"] = engine
        if remote_model_capable:
            override_cfg["voice_tts_engine"] = engine
        if remote_model_capable and requested_model:
            override_cfg["voice_tts_model"] = requested_model

        owner = type("_AbstractVoiceCloneRequestOverride", (), {"config": override_cfg})()
        cap = self.__class__(owner)
        return cap._get_vm()

    def _residency_error(
        self,
        *,
        task: str,
        provider: str | None,
        model: str | None,
        code: str,
        message: str,
        state: str = "failed",
        local: bool | None = None,
        unloadable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "task": str(task or ""),
            "provider": provider,
            "model": model,
            "state": str(state),
            "loaded": False,
            "resident": False,
            "local": _coerce_bool(local, False) if local is not None else False,
            "unloadable": bool(unloadable),
            "details": {
                "backend_id": getattr(self, "backend_id", None),
                **(dict(details) if isinstance(details, dict) else {}),
            },
            "error": {"code": str(code), "message": str(message)},
        }

    def _parse_resident_model_request(self, request: Any) -> Dict[str, Any]:
        data = dict(request) if isinstance(request, dict) else {}
        options = data.get("options")
        options_d = dict(options) if isinstance(options, dict) else {}
        task = _norm_residency_task(data.get("task") or "tts") or "tts"
        provider = _norm_residency_provider(data.get("provider"))
        model_raw = data.get("model")
        model_text = str(model_raw).strip() if isinstance(model_raw, str) else ""
        model = model_text if model_text and model_text.lower() != "default" else ""
        voice = None
        for key in ("voice", "voice_id"):
            value = options_d.get(key)
            if isinstance(value, str) and value.strip():
                voice = value.strip()
                break
        return {
            "task": task,
            "provider": provider,
            "model": model or None,
            "voice": voice,
            "options": options_d,
        }

    def _resolve_clone_residency_engine(self, *, model: Optional[str], voice: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        voice_id = str(voice or "").strip() or None
        engine = _norm_engine_id(model)
        if voice_id and not engine:
            for _cache_key, vm in self._iter_known_vms():
                try:
                    info = vm.get_cloned_voice(voice_id) if hasattr(vm, "get_cloned_voice") else None
                except Exception:
                    info = None
                if isinstance(info, dict):
                    engine = _norm_engine_id(info.get("engine"))
                    if engine:
                        break
            if not engine:
                try:
                    from ..cloning.store import VoiceCloneStore

                    info = VoiceCloneStore().get_voice_dict(voice_id)
                    if isinstance(info, dict):
                        engine = _norm_engine_id(info.get("engine"))
                except Exception:
                    engine = engine or ""
        return (engine or None), voice_id

    def _clone_residency_entry(
        self,
        *,
        engine: Optional[str],
        voice: Optional[str],
        state: str,
        resident: bool,
        local: bool,
        unloadable: bool,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        unloaded: Optional[bool] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "task": "tts",
            "provider": "cloned",
            "model": engine,
            "state": str(state),
            "loaded": bool(resident),
            "resident": bool(resident),
            "local": bool(local),
            "unloadable": bool(unloadable),
            "details": {
                "component": "cloning_engine",
                "backend_id": getattr(self, "backend_id", None),
                **(dict(details) if isinstance(details, dict) else {}),
            },
            "error": dict(error) if isinstance(error, dict) else None,
        }
        if voice:
            entry["options"] = {"voice": str(voice)}
        if unloaded is not None:
            entry["unloaded"] = bool(unloaded)
        return entry

    def _engine_residency_entry(
        self,
        *,
        task: str,
        provider: str | None,
        model: str | None,
        component: str,
        state: str,
        loaded: bool,
        local: bool,
        unloadable: bool,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        unloaded: Optional[bool] = None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "task": str(task or ""),
            "provider": provider,
            "model": model,
            "state": str(state),
            "loaded": bool(loaded),
            "resident": bool(loaded),  # backward compat
            "local": bool(local),
            "unloadable": bool(unloadable),
            "details": {
                "component": str(component or ""),
                "backend_id": getattr(self, "backend_id", None),
                **(dict(details) if isinstance(details, dict) else {}),
            },
            "error": dict(error) if isinstance(error, dict) else None,
        }
        if unloaded is not None:
            entry["unloaded"] = bool(unloaded)
        return entry

    def load_resident_model(self, request: Any) -> Dict[str, Any]:
        parsed = self._parse_resident_model_request(request)
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]
        voice = parsed["voice"]

        if task != "tts":
            return self._residency_error(
                task=task,
                provider=provider or None,
                model=model,
                code="not_implemented_yet",
                message="Residency warmup is implemented only for TTS on the voice backend.",
                state="not_implemented",
                details={"supported": {"task": "tts", "backend": getattr(self, "backend_id", None)}},
            )

        if provider != "cloned":
            provider_id = _norm_engine_id(provider)
            requested_model = _normalize_optional_model_id(model)
            if not provider_id:
                return self._residency_error(
                    task=task,
                    provider=None,
                    model=requested_model,
                    code="invalid_request",
                    message="Provide a local TTS provider id (for example piper, supertonic, omnivoice, audiodit).",
                    state="failed",
                    details={"supported": {"task": "tts", "provider": "local_tts"}},
                )
            if provider_id in {"openai", "openai-compatible"}:
                return self._residency_error(
                    task=task,
                    provider=provider_id,
                    model=requested_model,
                    code="not_supported",
                    message="Residency warmup is supported only for local TTS engines (remote providers are excluded).",
                    state="not_implemented",
                    local=False,
                    unloadable=False,
                    details={"supported": {"task": "tts", "provider": "local_tts"}},
                )

            if not _local_tts_engine_available(provider_id, self._configured_local_tts_model_ids(provider_id)):
                return self._residency_error(
                    task="tts",
                    provider=provider_id,
                    model=requested_model,
                    code="not_implemented_yet",
                    message=(
                        "Local base TTS residency requires the local engine runtime + cached model weights. "
                        "Install the provider extra (for example abstractvoice[piper] / abstractvoice[supertonic] / "
                        "abstractvoice[omnivoice] / abstractvoice[audiodit]) and prefetch models before warming."
                    ),
                    state="not_implemented",
                    local=True,
                    unloadable=False,
                    details={"supported": {"task": "tts", "provider": "local_tts"}},
                )

            opts = parsed.get("options") or {}
            language = opts.get("language")
            lang = str(language).strip().lower() if isinstance(language, str) and language.strip() else None
            if not lang:
                lang = _tts_model_language_selector(provider_id, requested_model)
            if provider_id == "piper" and not lang and requested_model:
                try:
                    lang = _piper_language_for_model(str(requested_model)) or None
                except Exception:
                    lang = None

            warmup = _coerce_bool(opts.get("warmup"), True)
            warmup_text = str(opts.get("warmup_text")).strip() if isinstance(opts.get("warmup_text"), str) else None
            warmup_format = str(opts.get("warmup_format") or "wav").strip().lower() or "wav"

            try:
                vm = self._get_vm_for_provider(tts_provider=provider_id, tts_model=requested_model)
                lk = self._vm_lock(vm)
                with lk:
                    if lang and hasattr(vm, "set_language"):
                        try:
                            vm.set_language(str(lang))
                        except Exception:
                            pass
                    if provider_id == "piper" and requested_model and hasattr(vm, "set_profile"):
                        try:
                            vm.set_profile(str(requested_model), kind="tts")
                        except Exception:
                            pass
                    preload = getattr(vm, "preload_tts_engine", None)
                    if not callable(preload):
                        return self._residency_error(
                            task=task,
                            provider=provider_id,
                            model=requested_model,
                            code="not_implemented_yet",
                            message="Base TTS preload is not available on this VoiceManager build.",
                            state="not_implemented",
                            local=True,
                            unloadable=False,
                        )
                    result = preload(
                        warmup=bool(warmup),
                        warmup_text=warmup_text,
                        warmup_format=warmup_format,
                    )
            except Exception as e:
                return self._engine_residency_entry(
                    task="tts",
                    provider=provider_id,
                    model=requested_model,
                    component="tts_engine",
                    state="failed",
                    loaded=False,
                    local=True,
                    unloadable=True,
                    error={"code": "load_failed", "message": str(e)},
                )

            details = {
                "engine_cached": bool(result.get("engine_cached", False)),
                "engine_cached_before": bool(result.get("engine_cached_before", False)),
                "engine_cached_after": bool(result.get("engine_cached_after", False)),
                "warmed": bool(result.get("warmed", False)),
                "warm_error": result.get("warm_error"),
                "warmed_via": result.get("warmed_via"),
                "runtime_info": _json_safe(result.get("runtime_info") or {}),
            }
            loaded = bool(result.get("resident", False) or result.get("engine_cached_after", False))
            return self._engine_residency_entry(
                task="tts",
                provider=provider_id,
                model=requested_model,
                component="tts_engine",
                state=str(result.get("state") or ("resident" if loaded else "configured")),
                loaded=bool(loaded),
                local=True,
                unloadable=True,
                details=details,
            )

        # Cloned TTS residency (existing behavior).
        try:
            engine, voice_id = self._resolve_clone_residency_engine(model=model, voice=voice)
            vm = self._get_vm_for_cloning_engine(engine)
            lk = self._vm_lock(vm)
            with lk:
                result = vm.preload_cloning_engine(
                    engine=engine,
                    voice=voice_id,
                    language=parsed["options"].get("language"),
                    speed=parsed["options"].get("speed"),
                )
        except Exception as e:
            return self._clone_residency_entry(
                engine=model,
                voice=voice,
                state="failed",
                resident=False,
                local=_clone_engine_is_local(model),
                unloadable=True,
                error={"code": "load_failed", "message": str(e)},
            )

        details = {
            "engine_cached": bool(result.get("engine_cached", False)),
            "warmed_via": result.get("warmed_via"),
            "runtime_info": _json_safe(result.get("runtime_info") or {}),
        }
        if "engine_cached_before" in result:
            details["engine_cached_before"] = bool(result.get("engine_cached_before"))
        if "engine_cached_after" in result:
            details["engine_cached_after"] = bool(result.get("engine_cached_after"))
        if "voice_prepared" in result:
            details["voice_prepared"] = bool(result.get("voice_prepared"))
        if result.get("voice_prepare_error"):
            details["voice_prepare_error"] = str(result.get("voice_prepare_error"))
        if "voice_warmed" in result:
            details["voice_warmed"] = bool(result.get("voice_warmed"))
        if result.get("voice_warm_error"):
            details["voice_warm_error"] = str(result.get("voice_warm_error"))
        return self._clone_residency_entry(
            engine=_norm_engine_id(result.get("engine")) or engine,
            voice=str(result.get("voice_id") or voice_id or "") or None,
            state=str(result.get("state") or "resident"),
            resident=bool(result.get("resident", False)),
            local=bool(result.get("local", True)),
            unloadable=bool(result.get("unloadable", True)),
            details=details,
        )

    def list_resident_models(self, filters: Any | None = None) -> list[Dict[str, Any]]:
        parsed = self._parse_resident_model_request(filters or {})
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]
        if task and task != "tts":
            return []
        provider_id = _norm_engine_id(provider) if provider else ""

        out: list[Dict[str, Any]] = []
        for cache_key, vm in self._iter_known_vms():
            try:
                lk = self._vm_lock(vm)
                with lk:
                    components = vm.list_resident_components() if hasattr(vm, "list_resident_components") else []
            except Exception:
                continue
            for component in list(components or []):
                if not isinstance(component, dict):
                    continue
                component_kind = str(component.get("component") or "").strip().lower()

                if component_kind == "cloning_engine":
                    if provider_id and provider_id != "cloned":
                        continue
                    engine = _norm_engine_id(component.get("engine") or component.get("model"))
                    if model and engine != _norm_engine_id(model):
                        continue
                    details = {
                        "engine_cached": bool(component.get("engine_cached", False)),
                        "runtime_info": _json_safe(component.get("runtime_info") or {}),
                    }
                    if cache_key is not None:
                        details["cache_key"] = _json_safe(list(cache_key))
                    out.append(
                        self._clone_residency_entry(
                            engine=engine or None,
                            voice=None,
                            state=str(component.get("state") or ("resident" if component.get("resident") else "configured")),
                            resident=bool(component.get("resident", False)),
                            local=bool(component.get("local", True)),
                            unloadable=bool(component.get("unloadable", True)),
                            details=details,
                        )
                    )
                    continue

                if component_kind == "tts_engine":
                    if provider_id == "cloned":
                        continue
                    engine = _norm_engine_id(component.get("engine"))
                    component_model = component.get("model")
                    component_model_s = str(component_model).strip() if isinstance(component_model, str) and component_model.strip() else None
                    if provider_id and engine != provider_id:
                        continue
                    if model and component_model_s and str(model).strip() and str(model).strip().lower() != "default":
                        if component_model_s.strip().lower() != str(model).strip().lower():
                            continue

                    details = {
                        "engine_cached": True,
                        "engine_cached_before": True,
                        "engine_cached_after": True,
                        "runtime_info": _json_safe(component.get("runtime_info") or {}),
                    }
                    if cache_key is not None:
                        details["cache_key"] = _json_safe(list(cache_key))
                    out.append(
                        self._engine_residency_entry(
                            task="tts",
                            provider=engine or None,
                            model=component_model_s,
                            component="tts_engine",
                            state=str(component.get("state") or ("resident" if component.get("resident") else "configured")),
                            loaded=bool(component.get("resident", False)),
                            local=bool(component.get("local", True)),
                            unloadable=bool(component.get("unloadable", True)),
                            details=details,
                        )
                    )
                    continue

        out.sort(key=lambda item: (str(item.get("provider") or ""), str(item.get("model") or ""), str(item.get("state") or "")))
        return out

    def unload_resident_model(self, request: Any) -> Dict[str, Any]:
        parsed = self._parse_resident_model_request(request)
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]
        voice = parsed["voice"]

        if task != "tts":
            return self._residency_error(
                task=task,
                provider=provider or None,
                model=model,
                code="not_implemented_yet",
                message="Residency unload is implemented only for TTS on the voice backend.",
                state="not_implemented",
                details={"supported": {"task": "tts", "backend": getattr(self, "backend_id", None)}},
            )

        if provider != "cloned":
            provider_id = _norm_engine_id(provider)
            requested_model = _normalize_optional_model_id(model)
            if not provider_id:
                return self._residency_error(
                    task=task,
                    provider=None,
                    model=requested_model,
                    code="invalid_request",
                    message="Provide a local TTS provider id to unload.",
                    state="failed",
                    details={"supported": {"task": "tts", "provider": "local_tts"}},
                )
            if provider_id in {"openai", "openai-compatible"}:
                return self._residency_error(
                    task=task,
                    provider=provider_id,
                    model=requested_model,
                    code="not_supported",
                    message="Residency unload is supported only for local TTS engines (remote providers are excluded).",
                    state="not_implemented",
                    local=False,
                    unloadable=False,
                    details={"supported": {"task": "tts", "provider": "local_tts"}},
                )

            unloaded_count = 0
            last_error = None
            for _cache_key, vm in self._iter_known_vms():
                try:
                    lk = self._vm_lock(vm)
                    with lk:
                        engines = self._vm_engine_values(vm, kind="tts")
                        if not (_engine_aliases(provider_id) & engines):
                            continue
                        if requested_model and hasattr(vm, "list_resident_components"):
                            comps = list(vm.list_resident_components() or [])
                            match = False
                            for comp in comps:
                                if not isinstance(comp, dict):
                                    continue
                                if str(comp.get("component") or "").strip().lower() != "tts_engine":
                                    continue
                                comp_model = comp.get("model")
                                comp_model_s = str(comp_model).strip() if isinstance(comp_model, str) and comp_model.strip() else None
                                if comp_model_s and comp_model_s.strip().lower() == str(requested_model).strip().lower():
                                    match = True
                                    break
                            if not match:
                                continue
                        unload = getattr(vm, "unload_tts_engine", None)
                        if callable(unload):
                            res = unload()
                            if bool(res.get("unloaded")):
                                unloaded_count += 1
                        else:
                            last_error = "unload_tts_engine_not_available"
                except Exception as e:
                    last_error = str(e)
                    continue

            details = {"unloaded_count": int(unloaded_count)}
            if last_error and unloaded_count == 0:
                return self._engine_residency_entry(
                    task="tts",
                    provider=provider_id,
                    model=requested_model,
                    component="tts_engine",
                    state="failed",
                    loaded=False,
                    local=True,
                    unloadable=True,
                    details=details,
                    error={"code": "unload_failed", "message": str(last_error)},
                    unloaded=False,
                )
            return self._engine_residency_entry(
                task="tts",
                provider=provider_id,
                model=requested_model,
                component="tts_engine",
                state="unloaded" if unloaded_count > 0 else "not_loaded",
                loaded=False,
                local=True,
                unloadable=True,
                details=details,
                unloaded=bool(unloaded_count > 0),
            )

        try:
            engine, voice_id = self._resolve_clone_residency_engine(model=model, voice=voice)
        except Exception as e:
            return self._clone_residency_entry(
                engine=model,
                voice=voice,
                state="failed",
                resident=False,
                local=_clone_engine_is_local(model),
                unloadable=True,
                error={"code": "resolve_failed", "message": str(e)},
            )
        if not engine:
            return self._clone_residency_entry(
                engine=None,
                voice=voice_id,
                state="failed",
                resident=False,
                local=True,
                unloadable=True,
                error={
                    "code": "invalid_request",
                    "message": "Provide a cloned engine model, or pass options.voice for a stored cloned voice.",
                },
            )

        unloaded_count = 0
        for _cache_key, vm in self._iter_known_vms():
            try:
                lk = self._vm_lock(vm)
                with lk:
                    result = vm.unload_cloning_engine(engine=engine)
            except Exception:
                continue
            if bool(result.get("unloaded")):
                unloaded_count += 1

        return self._clone_residency_entry(
            engine=engine,
            voice=voice_id,
            state="unloaded" if unloaded_count > 0 else "not_loaded",
            resident=False,
            local=_clone_engine_is_local(engine),
            unloadable=True,
            details={"unloaded_count": int(unloaded_count)},
            unloaded=bool(unloaded_count > 0),
        )

    def _config_text(self, *keys: str) -> Optional[str]:
        cfg = getattr(self._owner, "config", None)
        if isinstance(cfg, dict):
            for key in keys:
                value = cfg.get(str(key))
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _configured_remote_tts_engines(self) -> list[str]:
        engines: list[str] = []

        def add(value: Any) -> None:
            engine = _norm_engine_id(value)
            if engine in {"openai", "openai-compatible"}:
                engines.append(engine)

        requested = _norm_engine_id(
            self._config_text("voice_tts_engine")
            or _env_first("ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTGATEWAY_VOICE_TTS_ENGINE")
        )
        add(requested)

        remote_base_url = self._config_text("voice_remote_base_url") or _env("OPENAI_BASE_URL")
        remote_api_key = self._config_text("voice_remote_api_key") or _env("OPENAI_API_KEY")
        openai_specific = _env_first(
            "OPENAI_API_KEY",
            "ABSTRACTVOICE_OPENAI_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_TTS_MODELS",
            "ABSTRACTVOICE_OPENAI_TTS_VOICE",
            "ABSTRACTVOICE_OPENAI_TTS_VOICES",
        )
        compatible_specific = _env_first(
            "OPENAI_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
            "ABSTRACTVOICE_REMOTE_TTS_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODELS",
            "ABSTRACTVOICE_REMOTE_TTS_MODELS",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICE",
            "ABSTRACTVOICE_REMOTE_TTS_VOICE",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_VOICES",
            "ABSTRACTVOICE_REMOTE_TTS_VOICES",
        )

        base_url_s = str(remote_base_url or "").strip().lower()
        if requested == "openai" or openai_specific or (remote_api_key and ("api.openai.com" in base_url_s or not base_url_s)):
            add("openai")
        if requested == "openai-compatible" or compatible_specific or (remote_base_url and "api.openai.com" not in base_url_s):
            add("openai-compatible")

        return _dedupe_strings(engines)

    def _configured_tts_model_ids(self, engine: str) -> list[str]:
        provider = _norm_engine_id(engine)
        values: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.extend(value.split(","))

        add(self._config_text("voice_tts_model"))
        if provider == "openai":
            for key in (
                "ABSTRACTVOICE_OPENAI_TTS_MODEL",
                "ABSTRACTVOICE_OPENAI_TTS_MODELS",
                "ABSTRACTVOICE_TTS_MODEL",
                "ABSTRACTVOICE_REMOTE_TTS_MODEL",
            ):
                add(_env(key))
            values.extend(["gpt-4o-mini-tts", "tts-1", "tts-1-hd"])
        elif provider == "openai-compatible":
            for key in (
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODEL",
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_TTS_MODELS",
                "ABSTRACTVOICE_REMOTE_TTS_MODEL",
                "ABSTRACTVOICE_REMOTE_TTS_MODELS",
                "ABSTRACTVOICE_TTS_MODEL",
            ):
                add(_env(key))
        return _dedupe_strings(values)

    def _configured_remote_stt_engines(self) -> list[str]:
        engines: list[str] = []

        def add(value: Any) -> None:
            engine = _norm_engine_id(value)
            if engine in {"openai", "openai-compatible"}:
                engines.append(engine)

        requested = _norm_engine_id(
            self._config_text("voice_stt_engine")
            or _env_first("ABSTRACTVOICE_STT_ENGINE", "ABSTRACTGATEWAY_VOICE_STT_ENGINE")
        )
        add(requested)

        remote_base_url = self._config_text("voice_remote_base_url") or _env("OPENAI_BASE_URL")
        remote_api_key = self._config_text("voice_remote_api_key") or _env("OPENAI_API_KEY")
        openai_specific = _env_first(
            "OPENAI_API_KEY",
            "ABSTRACTVOICE_OPENAI_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_STT_MODELS",
        )
        compatible_specific = _env_first(
            "OPENAI_BASE_URL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
            "ABSTRACTVOICE_REMOTE_STT_MODEL",
            "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODELS",
            "ABSTRACTVOICE_REMOTE_STT_MODELS",
        )

        base_url_s = str(remote_base_url or "").strip().lower()
        if requested == "openai" or openai_specific or (remote_api_key and ("api.openai.com" in base_url_s or not base_url_s)):
            add("openai")
        if requested == "openai-compatible" or compatible_specific or (remote_base_url and "api.openai.com" not in base_url_s):
            add("openai-compatible")

        return _dedupe_strings(engines)

    def _configured_stt_model_ids(self, engine: str) -> list[str]:
        provider = _norm_engine_id(engine)
        values: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                values.extend(value.split(","))

        add(self._config_text("voice_stt_model"))
        if provider == "openai":
            for key in (
                "ABSTRACTVOICE_OPENAI_STT_MODEL",
                "ABSTRACTVOICE_OPENAI_STT_MODELS",
                "ABSTRACTVOICE_STT_MODEL",
                "ABSTRACTVOICE_REMOTE_STT_MODEL",
            ):
                add(_env(key))
            values.extend(["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"])
        elif provider == "openai-compatible":
            for key in (
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODEL",
                "ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODELS",
                "ABSTRACTVOICE_REMOTE_STT_MODEL",
                "ABSTRACTVOICE_REMOTE_STT_MODELS",
                "ABSTRACTVOICE_STT_MODEL",
            ):
                add(_env(key))
        return _dedupe_strings(values)

    def _configured_provider_id(self, *, kind: str) -> str:
        k = str(kind or "").strip().lower()
        vm = self._vm
        try:
            cfg = getattr(self._owner, "config", None)
            if vm is None and isinstance(cfg, dict):
                candidate = cfg.get("voice_manager_instance")
                if candidate is not None:
                    vm = candidate
        except Exception:
            vm = self._vm
        if vm is not None and k in {"tts", "stt"}:
            adapter = (
                getattr(vm, "tts_adapter", None)
                if k == "tts"
                else getattr(vm, "stt_adapter", None)
            )
            for value in (
                getattr(adapter, "engine_id", None),
                getattr(adapter, "provider", None),
                getattr(vm, f"_abstractvoice_{k}_engine", None),
                getattr(vm, f"_{k}_engine_name", None),
                getattr(vm, f"_{k}_engine_preference", None),
                getattr(vm, f"{k}_engine", None),
            ):
                provider = _norm_engine_id(value)
                if provider and provider != "auto":
                    return provider
        if k == "tts":
            provider = _norm_engine_id(
                self._config_text("voice_tts_engine")
                or _env_first("ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTGATEWAY_VOICE_TTS_ENGINE")
                or "openai"
            )
        elif k == "stt":
            provider = _norm_engine_id(
                self._config_text("voice_stt_engine")
                or _env_first("ABSTRACTVOICE_STT_ENGINE", "ABSTRACTGATEWAY_VOICE_STT_ENGINE")
                or "openai"
            )
        elif k == "cloning":
            provider = _norm_engine_id(
                self._config_text("voice_cloning_engine")
                or _env("ABSTRACTVOICE_CLONING_ENGINE", "omnivoice")
                or "omnivoice"
            )
        else:
            provider = ""
        if provider == "auto":
            return "openai"
        return provider

    def _remote_base_url(self) -> str:
        return str(
            self._config_text("voice_remote_base_url")
            or _env("OPENAI_BASE_URL")
            or ""
        ).strip()

    def _configured_remote_api_key(self) -> str:
        return str(self._config_text("voice_remote_api_key") or _env("OPENAI_API_KEY") or "").strip()

    def _openai_provider_available(self) -> bool:
        if str(_env("OPENAI_API_KEY") or "").strip():
            return True
        base_url = self._remote_base_url().lower()
        return bool(self._config_text("voice_remote_api_key") and (not base_url or "api.openai.com" in base_url))

    def _openai_compatible_provider_available(self) -> bool:
        base_url = self._remote_base_url()
        if not base_url:
            return False
        return "api.openai.com" not in base_url.lower()

    def _active_vm_provider_id(self, *, kind: str) -> str:
        k = str(kind or "").strip().lower()
        vm = self._vm
        try:
            cfg = getattr(self._owner, "config", None)
            if vm is None and isinstance(cfg, dict):
                vm = cfg.get("voice_manager_instance")
        except Exception:
            vm = self._vm
        if vm is None or k not in {"tts", "stt"}:
            return ""
        adapter = getattr(vm, "tts_adapter", None) if k == "tts" else getattr(vm, "stt_adapter", None)
        for value in (
            getattr(adapter, "engine_id", None),
            getattr(adapter, "provider", None),
            getattr(vm, f"_abstractvoice_{k}_engine", None),
            getattr(vm, f"_{k}_engine_name", None),
            getattr(vm, f"{k}_engine", None),
        ):
            provider = _norm_engine_id(value)
            if provider and provider != "auto":
                return provider
        return ""

    def _active_local_stt_provider_is_live(self, provider: Any) -> bool:
        provider_id = _norm_engine_id(provider)
        if provider_id not in {"faster-whisper", "transformers-asr"}:
            return False
        vm = self._vm
        try:
            cfg = getattr(self._owner, "config", None)
            if vm is None and isinstance(cfg, dict):
                vm = cfg.get("voice_manager_instance")
        except Exception:
            vm = self._vm
        adapter = getattr(vm, "stt_adapter", None) if vm is not None else None
        if adapter is None:
            return False
        adapter_provider = _norm_engine_id(
            getattr(adapter, "engine_id", None)
            or getattr(adapter, "provider", None)
            or getattr(adapter, "backend_id", None)
        )
        return adapter_provider == provider_id

    def _available_tts_provider_ids(self) -> list[str]:
        providers: list[str] = []
        active_provider = self._active_vm_provider_id(kind="tts")
        if active_provider in _known_tts_provider_ids():
            providers.append(active_provider)
        if self._openai_provider_available():
            providers.append("openai")
        if self._openai_compatible_provider_available():
            providers.append("openai-compatible")
        for engine in _local_tts_engines():
            normalized = _norm_engine_id(engine)
            if _local_tts_engine_available(normalized, self._configured_local_tts_model_ids(normalized)):
                providers.append(normalized)
        return _ordered_provider_ids(
            providers,
            ["openai", "openai-compatible", "supertonic", "piper", "audiodit", "omnivoice"],
        )

    def _available_stt_provider_ids(self) -> list[str]:
        providers: list[str] = []
        active_provider = self._active_vm_provider_id(kind="stt")
        if active_provider in {"openai", "openai-compatible"}:
            providers.append(active_provider)
        elif active_provider in {"faster-whisper", "transformers-asr"}:
            if _local_stt_engine_available(active_provider) or self._active_local_stt_provider_is_live(active_provider):
                providers.append(active_provider)
        elif active_provider in _known_stt_provider_ids():
            providers.append(active_provider)
        if self._openai_provider_available():
            providers.append("openai")
        if self._openai_compatible_provider_available():
            providers.append("openai-compatible")
        if _local_stt_engine_available("faster-whisper"):
            providers.append("faster-whisper")
        if _local_stt_engine_available("transformers-asr"):
            providers.append("transformers-asr")
        return _ordered_provider_ids(providers, ["openai", "openai-compatible", "faster-whisper", "transformers-asr"])

    def _available_cloning_provider_ids(self) -> list[str]:
        providers: list[str] = []
        for engine in ("omnivoice", "f5_tts", "chroma", "audiodit", "qwen3-tts"):
            if _local_cloning_engine_available(engine):
                providers.append(engine)
        clone_path = str(_env("ABSTRACTVOICE_OPENAI_VOICE_CREATE_PATH") or "").strip()
        consent_id = str(_env("ABSTRACTVOICE_OPENAI_VOICE_CONSENT_ID") or "").strip()
        if self._openai_provider_available() and (clone_path or consent_id):
            providers.append("openai")
        if self._openai_compatible_provider_available():
            providers.append("openai-compatible")
        return _ordered_provider_ids(
            providers,
            ["omnivoice", "f5_tts", "chroma", "audiodit", "qwen3-tts", "openai", "openai-compatible"],
        )

    def _configured_local_tts_model_ids(self, engine: Any) -> list[str]:
        """Checkpoint ids this owner points `engine` at.

        Integrators pass a config dict, CLI users set the environment, and both are
        resolved against ONE engine id: `_configured_provider_id` already walks
        config, then environment, then the default. Resolving each source against its
        own view of the engine would drop the mixed case -- engine named in config,
        model named in the environment -- which is ordinary in a container.

        Scoped to that engine because a checkpoint belongs to the single engine it was
        configured for; offering it to the siblings would advertise a model they
        cannot run.
        """
        if _norm_engine_id(engine) != self._configured_provider_id(kind="tts"):
            return []
        sources = (
            self._config_text("voice_tts_model") or "",
            _env_first("ABSTRACTVOICE_TTS_MODEL", default="") or "",
        )
        return _dedupe_strings(
            [item.strip() for value in sources for item in value.split(",") if item.strip()]
        )

    def _catalog_safe_local_engines(self) -> list[str]:
        """`_catalog_safe_local_tts_engines`, aware of this owner's checkpoints."""
        return _catalog_safe_local_tts_engines(self._configured_local_tts_model_ids)

    def _selectable_local_tts_models(self, provider: Any) -> list[str]:
        """`_selectable_tts_model_ids_for_provider`, aware of this owner's checkpoints."""
        return _selectable_tts_model_ids_for_provider(provider, self._configured_local_tts_model_ids(provider))

    def _local_provider_answerable_from_disk(self, provider_id: str) -> bool:
        """True when a provider-filtered listing should use the light catalog.

        A local provider is described from the cache when the cache has something to
        say about it, and also when the only alternative would be BUILDING the active
        engine — that loads one engine's weights to answer about a different provider,
        and raises outright when the active engine's extra is not installed.

        An active manager that is already built or injected is consulted instead: it
        may carry catalog entries the cache cannot know.
        """
        if _norm_engine_id(provider_id) not in _CATALOG_LOCAL_TTS_PROVIDER_IDS:
            return False
        if _norm_engine_id(provider_id) in {_norm_engine_id(item) for item in self._catalog_safe_local_engines()}:
            return True
        return self._active_vm_for_discovery() is None

    def _active_vm_for_discovery(self):
        """The active VoiceManager, but only when reading it cannot load a model.

        A manager that is already built, or one the integrator injected, is free to
        consult. Otherwise it is worth building only for a remote provider:
        building it for a local one imports the engine AND loads its weights
        (`VoiceManager` passes `auto_load=True`), which is the entire cost
        discovery must not pay -- and it buys nothing, because a local engine's
        catalog comes from the filesystem.
        """
        if self._vm is not None:
            return self._vm
        cfg = getattr(self._owner, "config", None)
        injected = isinstance(cfg, dict) and (
            cfg.get("voice_manager_instance") is not None or callable(cfg.get("voice_manager_factory"))
        )
        if not injected and self._configured_provider_id(kind="tts") not in {"openai", "openai-compatible"}:
            return None
        try:
            return self._get_vm()
        except Exception:
            return None

    def _remote_discovery_vm(self, engine: str):
        """A VoiceManager for `engine` bound to the discovery budget, not the
        operator's synthesis timeout."""
        return self._get_vm_for_provider(tts_provider=engine, remote_timeout_s=_remote_discovery_timeout_s())

    def _fill_tts_discovery(self, slot: "_TTSDiscovery", vm: Any) -> None:
        """Fetch one provider's TTS discovery into `slot`. The only probe shape.

        Sequential because the calls share one HTTP adapter whose "already fetched"
        flags are not synchronised, and each result is published the moment it lands
        so an abandoned probe keeps whatever it already paid for.

        Profiles go first deliberately. On the remote adapter `list_available_models`
        fetches the models endpoint AND the profiles endpoint, so asking for profiles
        first splits those two round trips across two publish points -- a hang in
        either one preserves the other -- and warms the adapter's profile cache so
        the catalog call behind it only pays for models. Catalog-first would collapse
        both into one all-or-nothing step.
        """
        profiles: list[Dict[str, Any]] = []
        if hasattr(vm, "get_profiles"):
            try:
                profiles = [_voice_profile_to_dict(p) for p in list(vm.get_profiles(kind="tts") or [])]
            except Exception:
                profiles = []
            slot.state = (None, [], profiles, None)

        # Before the catalog: the remote adapter reads its active profile off its own
        # `voice` attribute, so sequencing this free call behind the one that can hang
        # would publish "no voice is selected" whenever the catalog overran.
        active_profile = None
        if hasattr(vm, "get_active_profile"):
            try:
                active_profile = vm.get_active_profile(kind="tts")
            except Exception:
                active_profile = None
            slot.state = (None, [], profiles, active_profile)

        catalog = vm.list_available_models() if hasattr(vm, "list_available_models") else {}
        slot.state = (catalog, _extract_tts_model_ids(catalog), profiles, active_profile)

    def _fill_remote_tts_discovery(self, engine: str, slot: "_TTSDiscovery") -> None:
        """`_fill_tts_discovery` against a remote engine on the discovery budget."""
        self._fill_tts_discovery(slot, self._remote_discovery_vm(engine))

    def _get_vm_for_provider(
        self,
        *,
        tts_provider: Optional[str] = None,
        stt_provider: Optional[str] = None,
        tts_model: Optional[str] = None,
        stt_model: Optional[str] = None,
        remote_timeout_s: Optional[float] = None,
    ):
        tts_engine, tts_model = _resolve_tts_provider_request(tts_provider, tts_model)
        stt_engine, stt_model = _resolve_stt_provider_request(stt_provider, stt_model)
        if tts_engine == "cloned":
            tts_engine = ""
        if stt_engine == "cloned":
            stt_engine = ""
        tts_language = _tts_model_language_selector(tts_engine, tts_model)
        if tts_language:
            tts_model = None

        if not tts_engine and not stt_engine:
            return self._get_vm()

        current = None
        # A shorter budget was asked for, so the active manager -- built with the
        # operator's full synthesis timeout -- is not an acceptable substitute. Don't
        # even reach for it: building it only to reject it is the exact cost discovery
        # exists to avoid, and for a local engine it loads weights.
        if remote_timeout_s is None:
            try:
                current = self._get_vm()
            except Exception:
                current = None

        if current is not None:
            tts_ok = not tts_engine or bool(_engine_aliases(tts_engine) & self._vm_engine_values(current, kind="tts"))
            stt_ok = not stt_engine or bool(_engine_aliases(stt_engine) & self._vm_engine_values(current, kind="stt"))
            if tts_ok and tts_language:
                adapter = getattr(current, "tts_adapter", None)
                current_language = (
                    getattr(adapter, "_language", None)
                    or getattr(adapter, "language", None)
                    or getattr(current, "language", None)
                )
                tts_ok = str(current_language or "").strip().lower() == str(tts_language).strip().lower()
            if tts_ok and isinstance(tts_model, str) and tts_model.strip():
                current_tts_models = {item.lower() for item in _current_tts_model_ids(current)}
                tts_ok = tts_model.strip().lower() in current_tts_models
            if stt_ok and isinstance(stt_model, str) and stt_model.strip():
                current_stt_models = {item.lower() for item in _current_stt_model_ids(current)}
                stt_ok = stt_model.strip().lower() in current_stt_models
            if tts_ok and stt_ok:
                return current

        cfg = getattr(self._owner, "config", None)
        override_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        # Provider-specific routing must not be short-circuited by injected active
        # VoiceManager instances. The module-level VoiceManager cache still keeps
        # repeated provider/model requests cheap.
        override_cfg.pop("voice_manager_instance", None)
        override_cfg.pop("voice_manager_factory", None)

        if tts_engine:
            override_cfg["voice_tts_engine"] = tts_engine
        if stt_engine:
            override_cfg["voice_stt_engine"] = stt_engine
        if tts_language:
            override_cfg["voice_language"] = tts_language
        if isinstance(tts_model, str) and tts_model.strip():
            override_cfg["voice_tts_model"] = tts_model.strip()
        if isinstance(stt_model, str) and stt_model.strip():
            override_cfg["voice_stt_model"] = stt_model.strip()
            if stt_engine in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                override_cfg["voice_whisper_model"] = stt_model.strip()
        if remote_timeout_s is not None:
            # Shorten only: an operator who configured a tighter budget meant it.
            configured = _configured_remote_timeout_s(override_cfg)
            override_cfg["voice_remote_timeout_s"] = (
                min(float(remote_timeout_s), configured) if configured is not None else float(remote_timeout_s)
            )

        owner = type("_AbstractVoiceProviderOverride", (), {"config": override_cfg})()
        cap = self.__class__(owner)
        return cap._get_vm()

    def _maybe_store_audio(
        self,
        audio_bytes: bytes,
        *,
        artifact_store: Any,
        fmt: str,
        run_id: Optional[str],
        tags: Optional[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
    ):
        if artifact_store is None:
            return bytes(audio_bytes)
        store = RuntimeArtifactStoreAdapter(artifact_store)
        merged_tags: Dict[str, str] = {"kind": "generated_media", "modality": "audio", "task": "tts"}
        if isinstance(tags, dict):
            merged_tags.update({str(k): str(v) for k, v in tags.items()})
        return store.store_bytes(
            bytes(audio_bytes),
            content_type=f"audio/{str(fmt).lower()}",
            filename=f"tts.{str(fmt).lower()}",
            run_id=str(run_id) if run_id else None,
            tags=merged_tags,
            metadata=metadata if isinstance(metadata, dict) else None,
        )

    def _resolve_audio_bytes(self, audio: Union[bytes, Dict[str, Any], str], *, artifact_store: Any) -> bytes:
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        if isinstance(audio, dict):
            if is_artifact_ref(audio):
                if artifact_store is None:
                    raise ValueError("artifact_store is required to resolve artifact refs to bytes")
                store = RuntimeArtifactStoreAdapter(artifact_store)
                return store.load_bytes(get_artifact_id(audio))
            for key in ("content", "bytes", "data"):
                raw = audio.get(key)
                if isinstance(raw, (bytes, bytearray)):
                    return bytes(raw)
            raise ValueError("Expected an artifact ref dict or an in-memory audio payload dict with bytes")
        if isinstance(audio, str):
            from pathlib import Path

            p = Path(audio).expanduser()
            if p.exists() and p.is_file():
                return p.read_bytes()
            raise FileNotFoundError(f"File not found: {audio}")
        raise TypeError("Unsupported input type; expected bytes, artifact-ref dict, or file path")

    def _suffix_for_audio_ref(self, audio: Dict[str, Any], *, artifact_store: Any) -> str:
        """Pick a best-effort file suffix for an audio payload/artifact dict."""
        import mimetypes
        from pathlib import Path

        # Prefer explicit filename when provided (most clients include it).
        try:
            filename = audio.get("filename")
            if isinstance(filename, str) and filename.strip():
                suf = Path(filename.strip()).suffix
                if isinstance(suf, str) and suf and len(suf) <= 10:
                    return suf
        except Exception:
            pass

        # Next: content_type from ref (or artifact metadata when available).
        content_type: Optional[str] = None
        try:
            ct = audio.get("content_type")
            if isinstance(ct, str) and ct.strip():
                content_type = ct.strip()
        except Exception:
            content_type = None

        if content_type is None and artifact_store is not None:
            try:
                store = RuntimeArtifactStoreAdapter(artifact_store)
                meta = store.get_metadata(get_artifact_id(audio))
                if isinstance(meta, dict):
                    ct2 = meta.get("content_type")
                    if isinstance(ct2, str) and ct2.strip():
                        content_type = ct2.strip()
                    fn2 = meta.get("filename")
                    if isinstance(fn2, str) and fn2.strip():
                        suf = Path(fn2.strip()).suffix
                        if isinstance(suf, str) and suf and len(suf) <= 10:
                            return suf
            except Exception:
                pass

        if isinstance(content_type, str) and content_type.strip():
            # Drop charset/params (e.g. "audio/wav; codecs=...").
            base = content_type.split(";", 1)[0].strip().lower()
            ext = mimetypes.guess_extension(base) or ""
            if ext:
                return ext

        return ".bin"


class _VoiceCapability(_BaseVoice):
    backend_id = "abstractvoice:default"

    def available_providers(self) -> Dict[str, Any]:
        """Return selectable provider ids without constructing heavy runtimes."""
        tts = self._available_tts_provider_ids()
        stt = self._available_stt_provider_ids()
        cloning = self._available_cloning_provider_ids()
        combined = _dedupe_provider_ids([*tts, *stt, *cloning])
        return {
            "tts": tts,
            "stt": stt,
            "cloning": cloning,
            "providers": combined,
            "tts_providers": tts,
            "stt_providers": stt,
            "cloning_providers": cloning,
            "known_tts_providers": _known_tts_provider_ids(),
            "known_stt_providers": _known_stt_provider_ids(),
            "known_cloning_providers": _known_cloning_provider_ids(),
            "active_tts_provider": self._configured_provider_id(kind="tts"),
            "active_stt_provider": self._configured_provider_id(kind="stt"),
            "active_cloning_provider": self._configured_provider_id(kind="cloning"),
            "details": {
                "tts": _provider_details("tts", tts),
                "stt": _provider_details("stt", stt),
                "cloning": _provider_details("cloning", cloning),
            },
        }

    def _light_voice_catalog(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
    ) -> Dict[str, Any]:
        """Return provider-filtered TTS catalog metadata without building VoiceManager."""
        requested_provider = _norm_engine_id(provider)
        requested_model = str(model or "").strip().lower()
        available_providers = self.available_providers()
        tts_providers = _dedupe_provider_ids(available_providers.get("tts") or available_providers.get("tts_providers") or [])
        if requested_provider:
            tts_providers = [item for item in tts_providers if _norm_engine_id(item) == requested_provider]
            if not tts_providers and _local_tts_engine_available(requested_provider, self._configured_local_tts_model_ids(requested_provider)):
                tts_providers = [requested_provider]
        profiles: list[Dict[str, Any]] = []
        cloned_voices: list[Dict[str, Any]] = []

        if not providers_only:
            for engine in list(tts_providers):
                profiles.extend(_local_tts_voice_profiles(engine))

            try:
                from ..cloning.store import VoiceCloneStore

                for item in VoiceCloneStore().list_voices():
                    if not isinstance(item, dict):
                        continue
                    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                    clone_id = str(item.get("voice_id") or item.get("id") or item.get("name") or "").strip()
                    if not clone_id:
                        continue
                    clone_engine = _norm_engine_id(
                        item.get("engine")
                        or item.get("engine_id")
                        or meta.get("engine")
                        or meta.get("tts_engine")
                        or requested_provider
                        or "omnivoice"
                    )
                    if requested_provider and clone_engine != requested_provider:
                        continue
                    if clone_engine and clone_engine not in tts_providers and _engine_runtime_available(clone_engine, tts_providers):
                        tts_providers.append(clone_engine)
                    if clone_engine and clone_engine not in tts_providers:
                        continue
                    label = str(item.get("name") or item.get("label") or clone_id).strip() or clone_id
                    cloned_voices.append(
                        {
                            "id": clone_id,
                            "voice_id": clone_id,
                            "profile_id": clone_id,
                            "label": label,
                            "display_name": label,
                            "name": label,
                            "kind": "clone",
                            "provider": clone_engine,
                            "engine_id": clone_engine,
                            "engine": clone_engine,
                            "tags": {
                                "provider": clone_engine,
                                "engine_id": clone_engine,
                                "engine": clone_engine,
                                "source": "abstractvoice_clone_store",
                            },
                            "params": _json_safe(item),
                        }
                    )
            except Exception:
                cloned_voices = []

        profiles = _dedupe_voice_records(_json_safe(profiles))
        cloned_voices = _dedupe_voice_records(_json_safe(cloned_voices))
        if requested_model:
            profiles = [item for item in profiles if not _profile_model_id(item) or _profile_model_id(item).lower() == requested_model]
            cloned_voices = [
                item for item in cloned_voices if not _profile_model_id(item) or _profile_model_id(item).lower() == requested_model
            ]

        tts_providers = _dedupe_provider_ids(tts_providers)
        tts_models_by_provider: dict[str, list[str]] = {provider_id: [] for provider_id in tts_providers}
        tts_voices_by_provider: dict[str, list[str]] = {provider_id: [] for provider_id in tts_providers}
        tts_profiles_by_provider: dict[str, list[str]] = {provider_id: [] for provider_id in tts_providers}
        for profile in profiles:
            provider_id = _profile_provider_id(profile)
            _add_provider_value(tts_models_by_provider, provider_id, _profile_model_id(profile))
            _add_provider_value(tts_voices_by_provider, provider_id, _profile_voice_id(profile))
            _add_provider_value(tts_profiles_by_provider, provider_id, _profile_id(profile))
        for clone in cloned_voices:
            provider_id = _profile_provider_id(clone)
            _add_provider_value(tts_voices_by_provider, provider_id, _profile_voice_id(clone))
            _add_provider_value(tts_profiles_by_provider, provider_id, _profile_id(clone))

        for engine in self._configured_remote_tts_engines():
            provider_id = _norm_engine_id(engine)
            if requested_provider and provider_id != requested_provider:
                continue
            if provider_id not in tts_providers:
                tts_providers.append(provider_id)
                tts_models_by_provider.setdefault(provider_id, [])
                tts_voices_by_provider.setdefault(provider_id, [])
                tts_profiles_by_provider.setdefault(provider_id, [])
            for model_id in self._configured_tts_model_ids(engine):
                _add_provider_value(tts_models_by_provider, provider_id, model_id)

        for provider_id in list(tts_providers):
            for model_id in self._selectable_local_tts_models(provider_id):
                _add_provider_value(tts_models_by_provider, provider_id, model_id)

        tts_models = _dedupe_strings([model_id for values in tts_models_by_provider.values() for model_id in values])
        all_tts_voices = _dedupe_voice_records(list(profiles) + list(cloned_voices))
        tts_details = available_providers.get("details", {}).get("tts", {})
        tts_catalog_by_provider: dict[str, Dict[str, Any]] = {}
        for provider_id in tts_providers:
            provider_key = _norm_engine_id(provider_id)
            provider_voices = _dedupe_voice_records(
                [
                    _json_safe(voice)
                    for voice in all_tts_voices
                    if _voice_matches_tts_selection(voice, provider=provider_key, model=model)
                ]
            )
            tts_catalog_by_provider[provider_key] = {
                "provider": provider_key,
                "provider_id": provider_key,
                "remote": bool(tts_details.get(provider_key, {}).get("remote")),
                "local": bool(tts_details.get(provider_key, {}).get("local")),
                "details": _json_safe(tts_details.get(provider_key) or _provider_details("tts", [provider_key]).get(provider_key) or {}),
                "models": list(tts_models_by_provider.get(provider_key, []) or []),
                "model_variants": _provider_variants(provider_key, tts_models_by_provider.get(provider_key, [])),
                "voices": provider_voices,
                "profiles": [
                    voice for voice in provider_voices if str(voice.get("kind") or "").strip().lower() != "clone"
                ],
                "cloned_voices": [
                    voice for voice in provider_voices if str(voice.get("kind") or "").strip().lower() == "clone"
                ],
                "voices_by_model": {},
                "formats": _tts_formats_for_provider(provider_key),
            }

        controls: Dict[str, Any] = {
            "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
            "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
            "instructions": {"supported": True},
            "profile": {"supported": True},
            "voice_clone": {"supported": True},
        }
        active_provider = tts_providers[0] if tts_providers else None
        active_model = (
            _tts_model_language_selector(active_provider, requested_model)
            or (str(_env("ABSTRACTVOICE_LANGUAGE", "en") or "en").strip().lower() if _tts_provider_uses_language_models(active_provider) else None)
            or (tts_models[0] if tts_models else None)
        )
        return {
            "kind": "tts",
            "engine_id": active_provider,
            "provider_id": active_provider,
            "active_profile": None,
            "active_model": active_model,
            "active_tts_provider": active_provider,
            "active_stt_provider": (available_providers.get("stt") or [None])[0],
            "profiles": [] if providers_only else profiles,
            "voices": [] if providers_only else profiles + cloned_voices,
            "cloned_voices": [] if providers_only else cloned_voices,
            "tts_providers": tts_providers,
            "stt_providers": _dedupe_provider_ids(available_providers.get("stt") or []),
            "available_providers": available_providers,
            "available_tts_providers": available_providers.get("tts") or [],
            "available_stt_providers": available_providers.get("stt") or [],
            "available_cloning_providers": available_providers.get("cloning") or [],
            "tts_models": tts_models,
            "stt_models": [],
            "tts_models_by_provider": tts_models_by_provider,
            "stt_models_by_provider": {},
            "tts_model_roles_by_provider": {
                provider_id: "language" if _tts_provider_uses_language_models(provider_id) else "model"
                for provider_id in tts_providers
            },
            "tts_model_variants": {provider_id: _provider_variants(provider_id, values) for provider_id, values in tts_models_by_provider.items()},
            "stt_engine_variants": {},
            "tts_voices_by_provider": tts_voices_by_provider,
            "tts_profiles_by_provider": tts_profiles_by_provider,
            "tts_catalog_by_provider": tts_catalog_by_provider,
            # No `unreachable_tts_providers` here: this path contacts nobody, and an
            # empty list would claim every provider is reachable -- including the
            # remote ones it never asked. Absent means "not checked"; see THE RULE.
            "stt_catalog_by_provider": {},
            "tts_formats_by_provider": {provider_id: _tts_formats_for_provider(provider_id) for provider_id in tts_providers},
            "stt_formats_by_provider": {},
            "controls": controls,
            "tts_capabilities": {},
            "speech_request_contract": "speech_request_v1",
            "compatibility_catalog": {},
            "catalog": {},
            "catalogs": {},
            "source": "abstractvoice.light_catalog",
        }

    # Alias for callers that use list_* naming conventions.
    def list_available_providers(self) -> Dict[str, Any]:
        return self.available_providers()

    def compatibility_catalog(self) -> Dict[str, Any]:
        """Which features each provider/model supports.

        Packaged data, hinted with the current selection. A VoiceManager supplies
        only those hints -- it reads them off its adapters with `getattr` -- so it is
        consulted when one is free and the catalog is built from configuration
        otherwise. Building an engine to answer this cost 49s with a local engine
        active, and `list_cloning_models` goes through here.
        """
        vm = self._active_vm_for_discovery()
        try:
            if hasattr(vm, "get_compatibility_catalog"):
                catalog = vm.get_compatibility_catalog()
                if hasattr(catalog, "to_dict"):
                    return dict(catalog.to_dict())
                if isinstance(catalog, dict):
                    return dict(catalog)
        except Exception:
            pass

        try:
            from ..compatibility import build_compatibility_catalog

            return dict(
                build_compatibility_catalog(
                    current_tts_provider=self._configured_provider_id(kind="tts") or None,
                    current_tts_model=self._config_text("voice_tts_model") or _env_first("ABSTRACTVOICE_TTS_MODEL"),
                    current_stt_provider=self._configured_provider_id(kind="stt") or None,
                    current_stt_model=self._config_text("voice_stt_model") or _env_first("ABSTRACTVOICE_STT_MODEL"),
                    current_cloning_provider=self._configured_provider_id(kind="cloning") or None,
                ).to_dict()
            )
        except Exception:
            return {}

    def list_models(self, *, kind: str = "tts", provider: Optional[str] = None) -> list[str]:
        """List provider-filtered models for TTS/STT/cloning discovery."""
        normalized_kind = str(kind or "tts").strip().lower()
        if normalized_kind == "tts":
            return self.list_tts_models(provider=provider)
        if normalized_kind == "stt":
            return self.list_stt_models(provider=provider)
        if normalized_kind == "cloning":
            return self.list_cloning_models(provider=provider)
        raise ValueError("kind must be 'tts', 'stt', or 'cloning'")

    def list_profiles(self, *, kind: str = "tts") -> list[Dict[str, Any]]:
        """List active-engine voice profiles through the plugin boundary.

        Asks the active engine, so with a local one configured this builds it. That is
        the method's whole job -- what the engine itself reports, which an adapter may
        derive from more than the packaged assets. For engine-free voice discovery use
        `list_tts_voices(provider=...)` or `voice_catalog(provider=...)`.
        """
        vm = self._get_vm()
        profiles = []
        if hasattr(vm, "get_profiles"):
            profiles = list(vm.get_profiles(kind=str(kind or "tts")) or [])
        return [_voice_profile_to_dict(profile) for profile in profiles]

    def list_tts_models(self, provider: Optional[str] = None) -> list[str]:
        """List deduplicated TTS model ids from serveable AbstractVoice engines.

        A `list[str]` cannot say "unknown", so an unreachable remote provider looks
        the same here as one with no models. When that difference matters, read
        `voice_catalog()["unreachable_tts_providers"]`, or the provider's
        `unreachable` flag in `tts_catalog_by_provider`.
        """
        provider_id, requested_model = _resolve_tts_provider_request(provider)
        if requested_model:
            return [requested_model]
        if provider_id:
            if provider_id in {"openai", "openai-compatible"}:
                # The active manager answers for its own provider when consulting it
                # is free -- already built, or injected by an integrator. Otherwise
                # probe the remote provider directly: reaching the unfiltered catalog
                # would build the ACTIVE engine, and for a local one that loads its
                # weights -- 17.8s of AudioDiT to answer a question about OpenAI.
                active_vm = self._active_vm_for_discovery()
                if active_vm is not None and _engine_aliases(provider_id) & self._vm_engine_values(active_vm, kind="tts"):
                    entry = self.voice_catalog(provider=provider_id).get("tts_catalog_by_provider", {}).get(provider_id, {})
                    return list(entry.get("models") or [])
                slot = _TTSDiscovery()
                _probe_in_parallel({provider_id: partial(self._fill_remote_tts_discovery, provider_id, slot)})
                _catalog, models, _profiles, _active = slot.state
                return _dedupe_strings([*self._configured_tts_model_ids(provider_id), *models])
            if self._local_provider_answerable_from_disk(provider_id):
                catalog = self._light_voice_catalog(provider=provider_id)
            else:
                catalog = self.voice_catalog(provider=provider_id)
            entry = catalog.get("tts_catalog_by_provider", {}).get(provider_id, {})
            return list(entry.get("models") or [])

        active_vm = self._active_vm_for_discovery()
        active_engines = self._vm_engine_values(active_vm, kind="tts") if active_vm is not None else set()

        remote_engines = [
            engine
            for engine in self._configured_remote_tts_engines()
            if not (_engine_aliases(engine) & active_engines)
        ]

        # Every remote catalog fetch here shares one budget, the active manager's
        # included: it talks to a server too.
        slots = {engine: _TTSDiscovery() for engine in remote_engines}
        probes: Dict[Hashable, Callable[[], None]] = {}
        if active_vm is not None:
            slots[_ACTIVE_PROBE] = _TTSDiscovery()
            probes[_ACTIVE_PROBE] = partial(self._fill_tts_discovery, slots[_ACTIVE_PROBE], active_vm)
        for engine in remote_engines:
            probes[engine] = partial(self._fill_remote_tts_discovery, engine, slots[engine])
        _probe_in_parallel(probes)

        # Deterministic order: active engine, then configured ids, then fetched
        # ids, then local engines -- concurrency cannot reorder the result.
        model_ids: list[str] = []
        if _ACTIVE_PROBE in slots:
            model_ids.extend(slots[_ACTIVE_PROBE].state[1])
        for engine in self._configured_remote_tts_engines():
            model_ids.extend(self._configured_tts_model_ids(engine))
        for engine in remote_engines:
            model_ids.extend(slots[engine].state[1])
        # Local engines answer from disk, the active one included: a stat() is
        # cheaper than the skip logic that would exclude it.
        for engine in self._catalog_safe_local_engines():
            model_ids.extend(self._selectable_local_tts_models(engine))
        return _dedupe_strings(model_ids)

    def list_stt_models(self, provider: Optional[str] = None) -> list[str]:
        """List deduplicated STT model ids from serveable AbstractVoice engines."""
        provider_id, requested_model = _resolve_stt_provider_request(provider)
        if requested_model:
            return [requested_model]
        if provider_id:
            # Provider-scoped model discovery should stay lightweight and must not
            # require optional runtime dependencies to be installed.
            if provider_id in {"openai", "openai-compatible"}:
                model_ids: list[str] = []
                try:
                    # Never build the manager just to read attributes off it: with a
                    # local TTS engine configured that loads TTS weights to list STT
                    # models.
                    vm = self._active_vm_for_discovery()
                    if vm is not None and _engine_aliases(provider_id) & self._vm_engine_values(vm, kind="stt"):
                        model_ids.extend(_extract_stt_model_ids(vm))
                except Exception:
                    pass
                model_ids.extend(self._configured_stt_model_ids(provider_id))
                return _dedupe_strings(model_ids)
            if provider_id in {"faster-whisper", "transformers-asr"}:
                model_ids: list[str] = []
                model_ids.extend(self._configured_stt_model_ids(provider_id))
                model_ids.extend(_stt_model_ids_for_provider(provider_id))
                return _dedupe_strings(model_ids)

            catalog = self.voice_catalog()
            entry = catalog.get("stt_catalog_by_provider", {}).get(provider_id, {})
            return list(entry.get("models") or [])

        model_ids: list[str] = []
        try:
            # Attributes only; never worth building a manager (and with a local TTS
            # engine configured, loading TTS weights) to read them.
            vm = self._active_vm_for_discovery()
            model_ids.extend(_extract_stt_model_ids(vm))
        except Exception:
            pass
        for engine in self._configured_remote_stt_engines():
            model_ids.extend(self._configured_stt_model_ids(engine))
        if _local_stt_engine_available("faster-whisper") or self._active_local_stt_provider_is_live("faster-whisper"):
            model_ids.extend(_stt_model_ids_for_provider("faster-whisper"))
        if _local_stt_engine_available("transformers-asr") or self._active_local_stt_provider_is_live("transformers-asr"):
            model_ids.extend(_stt_model_ids_for_provider("transformers-asr"))
        return _dedupe_strings(model_ids)

    def list_cloning_models(self, provider: Optional[str] = None) -> list[str]:
        """List deduplicated cloning model ids from the central compatibility catalog."""
        provider_id, requested_model = _resolve_cloning_provider_request(provider)
        if requested_model:
            return [requested_model]

        catalog = self.compatibility_catalog()
        cloning_providers = catalog.get("providers", {}).get("cloning", {})
        if provider_id:
            entry = cloning_providers.get(provider_id, {})
            models = [
                str(model_name)
                for model_name in dict(entry.get("models") or {}).keys()
                if str(model_name).strip() and str(model_name).strip() != "*"
            ]
            return _dedupe_strings(models)

        model_ids: list[str] = []
        for entry in dict(cloning_providers or {}).values():
            for model_name in dict(entry.get("models") or {}).keys():
                text = str(model_name or "").strip()
                if text and text != "*":
                    model_ids.append(text)
        return _dedupe_strings(model_ids)

    def list_tts_voices(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        include_clones: bool = True,
    ) -> list[Dict[str, Any]]:
        """List provider/model-filtered TTS voices and cloned voices.

        As with `list_tts_models`, a list cannot say "unknown": an unreachable remote
        provider reads as one with no voices. `voice_catalog()` carries the
        distinction in `unreachable_tts_providers`.
        """
        provider_id, model_name = _resolve_tts_provider_request(provider, model)
        # Forward the filter: `voice_catalog` routes a local provider to the light
        # path, and asking for the unfiltered catalog instead would build the active
        # engine -- 8s and a torch import to produce the identical voice list.
        catalog = self.voice_catalog(provider=provider_id or None)
        if provider_id:
            entry = catalog.get("tts_catalog_by_provider", {}).get(provider_id, {})
            voices = list(entry.get("voices") or [])
        else:
            voices = list(catalog.get("voices") or [])
        out = [voice for voice in voices if _voice_matches_tts_selection(voice, provider=provider_id, model=model_name)]
        if not include_clones:
            out = [voice for voice in out if str(voice.get("kind") or "").strip().lower() != "clone"]
        return out

    def list_cloned_voices(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """List provider/model-filtered cloned TTS voices.

        Goes through the unfiltered catalog, so with a local engine configured this
        builds it. That is not free and it is not ideal, but the two catalog paths
        disagree about where clones come from -- the light one reads the clone store,
        this one also accepts clones the active manager reports -- and switching paths
        to save the time would silently drop the latter. Reconciling the two sources
        is its own change; trading data for latency here would undo the point of this
        one. Pass a provider filter for the cheap path.
        """
        return [
            voice
            for voice in self.list_tts_voices(provider=provider, model=model, include_clones=True)
            if str(voice.get("kind") or "").strip().lower() == "clone"
        ]

    def list_voices(
        self,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        include_clones: bool = True,
    ) -> list[Dict[str, Any]]:
        """Alias for provider/model-filtered TTS voice discovery."""
        return self.list_tts_voices(provider=provider, model=model, include_clones=include_clones)

    def get_capability_support(
        self,
        *,
        kind: str,
        feature: str,
        provider: str,
        model: Optional[str] = None,
        surface: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Return support metadata for one feature/provider/model/surface selection."""
        from ..compatibility import resolve_surface_name

        surface = resolve_surface_name(kind, surface)
        vm = self._active_vm_for_discovery()
        try:
            if hasattr(vm, "get_capability_support"):
                support = vm.get_capability_support(
                    kind=str(kind),
                    feature=str(feature),
                    provider=str(provider),
                    model=model,
                    surface=str(surface),
                )
                if isinstance(support, dict):
                    return dict(support)
        except Exception:
            pass

        catalog = self.compatibility_catalog()
        provider_record = (
            catalog.get("providers", {})
            .get(str(kind or "").strip().lower(), {})
            .get(_norm_compat_provider_id(kind, provider), {})
        )
        if not isinstance(provider_record, dict):
            return None
        model_key = str(model).strip() if isinstance(model, str) and model.strip() else "*"
        model_record = dict(provider_record.get("models") or {}).get(model_key)
        if not isinstance(model_record, dict) and model_key != "*":
            model_record = dict(provider_record.get("models") or {}).get("*")
        if isinstance(model_record, dict):
            features = dict(dict(model_record.get("surfaces") or {}).get(str(surface or "default"), {}) or {})
            support = features.get(str(feature or "").strip())
            if isinstance(support, dict):
                return dict(support)
        features = dict(dict(provider_record.get("default_surfaces") or {}).get(str(surface or "default"), {}) or {})
        support = features.get(str(feature or "").strip())
        if isinstance(support, dict):
            return dict(support)
        return None

    def find_compatible_models(
        self,
        *,
        kind: str,
        feature: str,
        surface: str = "default",
        support_in: Any = ("native", "emulated", "conditional"),
    ) -> list[Dict[str, Any]]:
        """Find provider/model pairs that support a feature on the requested surface."""
        from ..compatibility import resolve_surface_name

        surface = resolve_surface_name(kind, surface)
        vm = self._active_vm_for_discovery()
        support_levels = _normalize_support_levels(support_in)
        try:
            if hasattr(vm, "find_compatible_models"):
                matches = vm.find_compatible_models(
                    kind=str(kind),
                    feature=str(feature),
                    surface=str(surface),
                    support_in=support_levels,
                )
                if isinstance(matches, list):
                    return [dict(item) for item in matches if isinstance(item, dict)]
        except Exception:
            pass

        normalized_kind = str(kind or "").strip().lower()
        wanted = set(support_levels)
        out: list[Dict[str, Any]] = []
        providers = self.compatibility_catalog().get("providers", {}).get(normalized_kind, {})
        for provider_name, provider_record in dict(providers or {}).items():
            models = dict(dict(provider_record).get("models") or {})
            matched_model = False
            for model_name in list(models.keys()):
                if str(model_name).strip() == "*":
                    continue
                support = self.get_capability_support(
                    kind=normalized_kind,
                    provider=str(provider_name),
                    model=str(model_name),
                    surface=str(surface),
                    feature=str(feature),
                )
                if not isinstance(support, dict) or str(support.get("support") or "") not in wanted:
                    continue
                matched_model = True
                out.append(
                    {
                        "kind": normalized_kind,
                        "provider": str(provider_name),
                        "model": str(model_name),
                        "surface": str(surface),
                        "feature": str(feature),
                        "support": dict(support),
                    }
                )
            if matched_model:
                continue
            support = self.get_capability_support(
                kind=normalized_kind,
                provider=str(provider_name),
                model=None,
                surface=str(surface),
                feature=str(feature),
            )
            if not isinstance(support, dict) or str(support.get("support") or "") not in wanted:
                continue
            out.append(
                {
                    "kind": normalized_kind,
                    "provider": str(provider_name),
                    "model": None,
                    "surface": str(surface),
                    "feature": str(feature),
                    "support": dict(support),
                }
            )
        return out

    def clone(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Create a cloned voice through the plugin boundary."""
        provider_id, requested_model = _resolve_cloning_provider_request(provider, model)
        requested_model = _normalize_optional_model_id(requested_model)
        requested_engine = _norm_engine_id(kwargs.get("cloning_engine") or kwargs.get("engine") or provider_id)
        vm = self._get_vm_for_clone_request(
            provider=provider_id,
            model=requested_model,
            cloning_engine=requested_engine,
        )
        lk = self._vm_lock(vm)
        with lk:
            engine_name = _norm_engine_id(requested_engine or getattr(vm, "cloning_engine", None)) or None
            clone_name = str(name) if name is not None else None
            ref_text = str(reference_text) if reference_text is not None else None
            clone_meta = dict(metadata) if isinstance(metadata, dict) else None

            if isinstance(audio, str):
                return vm.clone_voice(
                    str(audio),
                    name=clone_name,
                    reference_text=ref_text,
                    engine=engine_name,
                )

            if isinstance(audio, dict):
                import os
                import tempfile

                audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                suffix = self._suffix_for_audio_ref(audio, artifact_store=artifact_store) or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    tmp_file.write(bytes(audio_bytes))
                    tmp_path = tmp_file.name
                try:
                    return vm.clone_voice(
                        tmp_path,
                        name=clone_name,
                        reference_text=ref_text,
                        engine=engine_name,
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            return vm.clone_voice_from_wav_bytes(
                bytes(audio),
                name=clone_name,
                reference_text=ref_text,
                engine=engine_name,
                meta=clone_meta,
            )

    def clone_voice(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        name: Optional[str] = None,
        reference_text: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Alias for clone() for compatibility with older duck-typed callers."""
        return self.clone(
            audio,
            name=name,
            reference_text=reference_text,
            model=model,
            provider=provider,
            artifact_store=artifact_store,
            metadata=metadata,
            **kwargs,
        )

    def voice_catalog(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        providers_only: bool = False,
    ) -> Dict[str, Any]:
        """Return JSON-safe profile/model discovery data for Core/Gateway.

        `providers_only` and a local provider filter take the light path and touch no
        engine. Everything else reports the ACTIVE engine's live state -- its profiles,
        its cloned voices, its STT side -- and so builds the active VoiceManager, which
        for a local engine loads its weights. That is deliberate rather than
        unavoidable: the light path reads cloned voices from the clone store while this
        one also accepts clones the manager reports, and switching would silently drop
        the latter. Callers that only need discovery should use `list_tts_models`,
        `list_stt_models`, `list_cloning_models`, `available_providers`, or pass
        `providers_only`/a local provider here -- all of those are engine-free.

        A provider whose probe did not answer inside the discovery budget is marked
        `tts_catalog_by_provider[provider]["unreachable"]` and listed in
        `unreachable_tts_providers`, and publishes no `catalogs` entry: its empty
        models and voices are our gap, not its catalog.
        """
        provider_id = _norm_engine_id(provider)
        if providers_only or (provider_id and self._local_provider_answerable_from_disk(provider_id)):
            return self._light_voice_catalog(provider=provider_id, model=model, providers_only=providers_only)

        vm = self._get_vm()
        available_providers = self.available_providers()
        active_tts_engines = self._vm_engine_values(vm, kind="tts")

        remote_engines = [
            engine
            for engine in self._configured_remote_tts_engines()
            if not (_engine_aliases(engine) & active_tts_engines)
        ]

        # Every remote fetch in this listing shares one budget, and every provider --
        # the active manager included -- reports through the same slot, so there is
        # one place that decides whether a provider answered.
        slots = {engine: _TTSDiscovery() for engine in remote_engines}
        slots[_ACTIVE_PROBE] = _TTSDiscovery()
        probes: Dict[Hashable, Callable[[], None]] = {
            _ACTIVE_PROBE: partial(self._fill_tts_discovery, slots[_ACTIVE_PROBE], vm)
        }
        for engine in remote_engines:
            probes[engine] = partial(self._fill_remote_tts_discovery, engine, slots[engine])
        _probe_in_parallel(probes)

        # One read of the slot: a live probe may still be publishing into it, and two
        # reads could pair a landed catalog with the fields from before it landed.
        active_catalog, active_models, active_profiles, active_profile = slots[_ACTIVE_PROBE].state
        active_answered = active_catalog is not None
        catalog = active_catalog or {}
        profiles = list(active_profiles)
        tts_models = list(active_models)
        stt_models = _extract_stt_model_ids(vm)
        tts_providers = _extract_tts_provider_ids(vm, catalog, profiles)
        stt_providers = _extract_stt_provider_ids(vm)
        live_local_stt_provider = _norm_engine_id(
            getattr(getattr(vm, "stt_adapter", None), "engine_id", None)
            or getattr(getattr(vm, "stt_adapter", None), "provider", None)
            or getattr(getattr(vm, "stt_adapter", None), "backend_id", None)
        )
        catalogs: Dict[str, Any] = {}
        active_engine = _norm_engine_id(tts_providers[0] if tts_providers else getattr(vm, "_abstractvoice_tts_engine", None))
        # Only publish a per-engine catalog for a probe that ANSWERED. Writing the
        # empty default would say "this provider has no models" about a provider we
        # merely failed to reach in time -- a live host 6s away would be rendered
        # as an empty, unusable selector.
        if active_engine and active_answered:
            catalogs[active_engine] = _json_safe(catalog)
            profiles.extend(_profiles_from_tts_catalog(catalog, engine_id=active_engine))

        # Which providers we could not reach, so the per-provider catalog below can
        # say that instead of publishing our gap as their empty catalog. Decided by
        # the one test that means "answered": did this provider's catalog land.
        unreachable_providers: set = set()
        if active_engine and not active_answered:
            unreachable_providers.add(active_engine)

        # Local engines answer from disk: `_catalog_safe_local_tts_engines` already
        # means "installed, with weights on this machine", their selectable model
        # ids come from the same filesystem probe further down, and their voices
        # are packaged profiles. Nothing here needs an engine loaded, so nothing
        # here loads one.
        for engine in self._catalog_safe_local_engines():
            tts_providers.append(engine)
            profiles.extend(_local_tts_voice_profiles(engine))

        for engine in self._configured_remote_tts_engines():
            tts_providers.append(engine)
            tts_models.extend(self._configured_tts_model_ids(engine))

        for engine in remote_engines:
            engine_catalog, engine_models, engine_profiles, _ = slots[engine].state
            # Everything that landed is kept. The catalog decides only whether this
            # provider's CATALOG is authoritative -- profiles we already paid for are
            # not withheld because a later fetch overran.
            tts_models.extend(engine_models)
            profiles.extend(engine_profiles)
            if engine_catalog is None:
                unreachable_providers.add(_norm_engine_id(engine))
                continue
            profiles.extend(_profiles_from_tts_catalog(engine_catalog, engine_id=engine))
            catalogs[_norm_engine_id(engine)] = _json_safe(engine_catalog)

        profiles = _dedupe_voice_records(_json_safe(profiles))

        available_stt_provider_ids = set(_dedupe_provider_ids(available_providers.get("stt") or []))
        available_cloning_provider_ids = set(_dedupe_provider_ids(available_providers.get("cloning") or []))
        stt_providers = [
            provider
            for provider in stt_providers
            if _norm_engine_id(provider) not in {"faster-whisper", "transformers-asr"}
            or _norm_engine_id(provider) in available_stt_provider_ids
            or _norm_engine_id(provider) == live_local_stt_provider
        ]
        for local_stt_provider in ("faster-whisper", "transformers-asr"):
            if local_stt_provider not in available_stt_provider_ids:
                continue
            stt_providers.append(local_stt_provider)
            stt_models.extend(_stt_model_ids_for_provider(local_stt_provider))

        for engine in self._configured_remote_stt_engines():
            stt_providers.append(engine)
            stt_models.extend(self._configured_stt_model_ids(engine))

        for provider_id in list(tts_providers):
            tts_models.extend(self._selectable_local_tts_models(provider_id))

        tts_models = _dedupe_strings(tts_models)
        stt_models = _dedupe_strings(stt_models)
        tts_providers = _dedupe_provider_ids(tts_providers)
        stt_providers = _dedupe_provider_ids(stt_providers)
        adapter = getattr(vm, "tts_adapter", None)
        cloned_voices: list[Dict[str, Any]] = []
        if hasattr(vm, "list_cloned_voices"):
            try:
                raw_clones = vm.list_cloned_voices()
            except Exception:
                raw_clones = []
            if isinstance(raw_clones, list):
                for item in raw_clones:
                    if isinstance(item, dict):
                        clone_id = item.get("voice_id") or item.get("id") or item.get("name")
                        clone_name = item.get("name") or item.get("label") or clone_id
                        if isinstance(clone_id, str) and clone_id.strip():
                            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
                            clone_engine = _norm_engine_id(
                                item.get("engine")
                                or item.get("engine_id")
                                or meta.get("engine")
                                or meta.get("tts_engine")
                                or getattr(vm, "_abstractvoice_tts_engine", None)
                                or getattr(vm, "tts_engine", None)
                                or active_engine
                            )
                            clone_engine_available = bool(
                                clone_engine
                                and (
                                    clone_engine in {"openai", "openai-compatible"}
                                    and (
                                        clone_engine in available_cloning_provider_ids
                                        or clone_engine in _dedupe_provider_ids(tts_providers)
                                    )
                                )
                            )
                            if clone_engine and not clone_engine_available and not _engine_runtime_available(clone_engine, tts_providers):
                                continue
                            if clone_engine:
                                tts_providers.append(clone_engine)
                            label = str(clone_name or clone_id).strip() or clone_id.strip()
                            cloned_voices.append(
                                {
                                    "id": clone_id.strip(),
                                    "voice_id": clone_id.strip(),
                                    "profile_id": clone_id.strip(),
                                    "label": label,
                                    "display_name": label,
                                    "name": label,
                                    "kind": "clone",
                                    "provider": clone_engine,
                                    "engine_id": clone_engine,
                                    "engine": clone_engine,
                                    "tags": {
                                        "provider": clone_engine,
                                        "engine_id": clone_engine,
                                        "engine": clone_engine,
                                        "source": "abstractvoice_clone_store",
                                    },
                                    "params": _json_safe(item),
                                }
                            )
                    elif isinstance(item, str) and item.strip():
                        clone_engine = _norm_engine_id(
                            getattr(vm, "_abstractvoice_tts_engine", None)
                            or getattr(vm, "tts_engine", None)
                            or active_engine
                        )
                        clone_engine_available = bool(
                            clone_engine
                            and (
                                clone_engine in {"openai", "openai-compatible"}
                                and (
                                    clone_engine in available_cloning_provider_ids
                                    or clone_engine in _dedupe_provider_ids(tts_providers)
                                )
                            )
                        )
                        if clone_engine and not clone_engine_available and not _engine_runtime_available(clone_engine, tts_providers):
                            continue
                        if clone_engine:
                            tts_providers.append(clone_engine)
                        cloned_voices.append(
                            {
                                "id": item.strip(),
                                "voice_id": item.strip(),
                                "profile_id": item.strip(),
                                "label": item.strip(),
                                "display_name": item.strip(),
                                "name": item.strip(),
                                "kind": "clone",
                                "provider": clone_engine,
                                "engine_id": clone_engine,
                                "engine": clone_engine,
                            }
                        )
        cloned_voices = _dedupe_voice_records(_json_safe(cloned_voices))
        if cloned_voices:
            tts_providers = _dedupe_provider_ids(tts_providers)

        tts_models_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        tts_voices_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        tts_profiles_by_provider: dict[str, list[str]] = {provider: [] for provider in tts_providers}
        for provider in tts_providers:
            for model_id in self._selectable_local_tts_models(provider):
                _add_provider_value(tts_models_by_provider, provider, model_id)
        for profile in profiles:
            provider = _profile_provider_id(profile)
            _add_provider_value(tts_models_by_provider, provider, _profile_model_id(profile))
            _add_provider_value(tts_voices_by_provider, provider, _profile_voice_id(profile))
            _add_provider_value(tts_profiles_by_provider, provider, _profile_id(profile))

        for provider, model_ids in _extract_tts_models_by_provider(catalog, default_provider=active_engine).items():
            for model_id in model_ids:
                _add_provider_value(tts_models_by_provider, provider, model_id)

        for engine, engine_catalog in catalogs.items():
            for provider, model_ids in _extract_tts_models_by_provider(engine_catalog, default_provider=engine).items():
                for model_id in model_ids:
                    _add_provider_value(tts_models_by_provider, provider, model_id)

        for engine in self._configured_remote_tts_engines():
            provider = _norm_engine_id(engine)
            for model_id in self._configured_tts_model_ids(engine):
                _add_provider_value(tts_models_by_provider, provider, model_id)

        for model_id in tts_models:
            if len(tts_providers) == 1:
                _add_provider_value(tts_models_by_provider, tts_providers[0], model_id)

        for clone in cloned_voices:
            clone_provider = _profile_provider_id(clone) or active_engine
            _add_provider_value(tts_voices_by_provider, clone_provider, _profile_voice_id(clone))
            _add_provider_value(tts_profiles_by_provider, clone_provider, _profile_id(clone))

        stt_models_by_provider: dict[str, list[str]] = {provider: [] for provider in stt_providers}
        active_stt_provider = self._active_vm_provider_id(kind="stt")
        if _norm_engine_id(active_stt_provider) not in stt_models_by_provider:
            active_stt_provider = stt_providers[0] if stt_providers else ""
        for model_id in _extract_stt_model_ids(vm):
            _add_provider_value(stt_models_by_provider, active_stt_provider, model_id)
        for engine in self._configured_remote_stt_engines():
            provider = _norm_engine_id(engine)
            for model_id in self._configured_stt_model_ids(engine):
                _add_provider_value(stt_models_by_provider, provider, model_id)
        if "faster-whisper" in stt_providers:
            for model_id in _stt_model_ids_for_provider("faster-whisper"):
                _add_provider_value(stt_models_by_provider, "faster-whisper", model_id)
        if "transformers-asr" in stt_providers:
            for model_id in _stt_model_ids_for_provider("transformers-asr"):
                _add_provider_value(stt_models_by_provider, "transformers-asr", model_id)

        for provider, model_ids in list(tts_models_by_provider.items()):
            tts_models_by_provider[provider] = _order_like(model_ids, tts_models)
        for provider, model_ids in list(stt_models_by_provider.items()):
            stt_models_by_provider[provider] = _order_like(model_ids, stt_models)

        stt_engine_variants = {
            provider: _provider_variants(provider, models)
            for provider, models in stt_models_by_provider.items()
        }
        tts_model_variants = {
            provider: _provider_variants(provider, models)
            for provider, models in tts_models_by_provider.items()
        }

        all_tts_voices = _dedupe_voice_records(list(profiles) + list(cloned_voices))
        tts_catalog_by_provider: dict[str, Dict[str, Any]] = {}
        tts_details = available_providers.get("details", {}).get("tts", {})
        for provider in tts_providers:
            provider_id = _norm_engine_id(provider)
            provider_voices = _dedupe_voice_records(
                [
                    _json_safe(voice)
                    for voice in all_tts_voices
                    if _voice_matches_tts_selection(voice, provider=provider_id)
                ]
            )
            provider_profiles = [
                voice for voice in provider_voices if str(voice.get("kind") or "").strip().lower() != "clone"
            ]
            provider_clones = [
                voice for voice in provider_voices if str(voice.get("kind") or "").strip().lower() == "clone"
            ]
            provider_models = list(tts_models_by_provider.get(provider_id, []) or [])
            voices_by_model = {
                model_id: [
                    _json_safe(voice)
                    for voice in provider_voices
                    if _voice_matches_tts_selection(voice, provider=provider_id, model=model_id)
                ]
                for model_id in provider_models
            }
            tts_catalog_by_provider[provider_id] = {
                "provider": provider_id,
                "provider_id": provider_id,
                "remote": bool(tts_details.get(provider_id, {}).get("remote")),
                "local": bool(tts_details.get(provider_id, {}).get("local")),
                "details": _json_safe(tts_details.get(provider_id) or _provider_details("tts", [provider_id]).get(provider_id) or {}),
                "models": provider_models,
                "model_variants": tts_model_variants.get(provider_id, [provider_id]),
                "voices": provider_voices,
                "profiles": provider_profiles,
                "cloned_voices": provider_clones,
                "voices_by_model": voices_by_model,
                "formats": _tts_formats_for_provider(provider_id),
            }
            if provider_id in unreachable_providers:
                # The empty voices/models below are OUR gap, not this provider's
                # catalog: we did not reach it inside the discovery budget. Say so,
                # so a slow-but-live host is not rendered as having nothing. Only
                # ever set when we actually tried and failed -- absent means no claim.
                tts_catalog_by_provider[provider_id]["unreachable"] = True

        stt_catalog_by_provider: dict[str, Dict[str, Any]] = {}
        stt_details = available_providers.get("details", {}).get("stt", {})
        for provider in stt_providers:
            provider_id = _norm_engine_id(provider)
            provider_models = list(stt_models_by_provider.get(provider_id, []) or [])
            stt_catalog_by_provider[provider_id] = {
                "provider": provider_id,
                "provider_id": provider_id,
                "remote": bool(stt_details.get(provider_id, {}).get("remote")),
                "local": bool(stt_details.get(provider_id, {}).get("local")),
                "details": _json_safe(stt_details.get(provider_id) or _provider_details("stt", [provider_id]).get(provider_id) or {}),
                "models": provider_models,
                "model_variants": stt_engine_variants.get(provider_id, [provider_id]),
                "formats": _stt_formats_for_provider(provider_id),
            }

        controls: Dict[str, Any] = {
            "speed": {"supported": True, "min": 0.5, "max": 2.0, "default": 1.0},
            "quality_preset": {"supported": True, "values": ["low", "standard", "high"], "default": "standard"},
            "instructions": {"supported": True},
            "profile": {"supported": True},
            "voice_clone": {"supported": True},
        }
        try:
            raw_tts_capabilities = vm.get_tts_capabilities() if hasattr(vm, "get_tts_capabilities") else None
        except Exception:
            raw_tts_capabilities = None
        if raw_tts_capabilities is not None:
            try:
                capability_items = raw_tts_capabilities.to_dict() if hasattr(raw_tts_capabilities, "to_dict") else raw_tts_capabilities
            except Exception:
                capability_items = {}
            if not isinstance(capability_items, dict):
                capability_items = {}
        else:
            capability_items = {}

        tts_capabilities = {
            key: {
                "support": str(value.get("support") or "unsupported"),
                "reason": value.get("reason"),
            }
            for key, value in dict(capability_items or {}).items()
            if isinstance(value, dict)
        }
        raw_engine_id = getattr(adapter, "engine_id", None) or getattr(vm, "_tts_engine_name", None)
        engine_id = _norm_engine_id(raw_engine_id) if raw_engine_id else None
        return {
            "kind": "tts",
            "engine_id": engine_id,
            "provider_id": engine_id,
            "active_profile": _voice_profile_to_dict(active_profile) if active_profile is not None else None,
            "active_model": _active_tts_model(vm, catalog, tts_models),
            "active_tts_provider": tts_providers[0] if tts_providers else None,
            "active_stt_provider": stt_providers[0] if stt_providers else None,
            "profiles": profiles,
            "voices": profiles + cloned_voices,
            "cloned_voices": cloned_voices,
            "tts_providers": tts_providers,
            "stt_providers": stt_providers,
            "available_providers": available_providers,
            "available_tts_providers": available_providers["tts"],
            "available_stt_providers": available_providers["stt"],
            "available_cloning_providers": available_providers["cloning"],
            "tts_models": tts_models,
            "stt_models": stt_models,
            "tts_models_by_provider": tts_models_by_provider,
            "stt_models_by_provider": stt_models_by_provider,
            "tts_model_roles_by_provider": {
                provider: "language" if _tts_provider_uses_language_models(provider) else "model"
                for provider in tts_providers
            },
            "tts_model_variants": tts_model_variants,
            "stt_engine_variants": stt_engine_variants,
            "tts_voices_by_provider": tts_voices_by_provider,
            "tts_profiles_by_provider": tts_profiles_by_provider,
            "tts_catalog_by_provider": tts_catalog_by_provider,
            # Providers we tried and could not reach inside the discovery budget, so a
            # consumer reading the flat lists above can tell a short list from a
            # complete one without walking the per-provider catalog. Present because
            # this path did probe; empty here means "checked, all reachable".
            "unreachable_tts_providers": sorted(unreachable_providers),
            "stt_catalog_by_provider": stt_catalog_by_provider,
            "tts_formats_by_provider": {provider: _tts_formats_for_provider(provider) for provider in tts_providers},
            "stt_formats_by_provider": {provider: _stt_formats_for_provider(provider) for provider in stt_providers},
            "controls": controls,
            "tts_capabilities": tts_capabilities,
            "speech_request_contract": "speech_request_v1",
            "compatibility_catalog": self.compatibility_catalog(),
            "catalog": _json_safe(catalog),
            "catalogs": catalogs,
        }

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ):
        provider_id, requested_model = _resolve_tts_provider_request(provider, model)
        voice_name = ""
        if isinstance(voice, dict):
            for value in (
                voice.get("voice_id"),
                voice.get("voice"),
                voice.get("profile_id"),
                voice.get("id"),
                voice.get("name"),
            ):
                if isinstance(value, str) and value.strip():
                    voice_name = value.strip()
                    break
            if not provider_id:
                provider_id = _profile_provider_id(voice)
            if not requested_model:
                model_from_voice = _profile_model_id(voice)
                if model_from_voice:
                    requested_model = model_from_voice
        vm = self._get_vm_for_provider(tts_provider=provider_id, tts_model=requested_model)
        lk = self._vm_lock(vm)
        with lk:
            requested_language = _tts_model_language_selector(provider_id, requested_model)
            model_name = "" if requested_language else (str(requested_model or "").strip() if isinstance(requested_model, str) else "")
            profile_name = str(_kwargs.get("profile") or "").strip() if isinstance(_kwargs.get("profile"), str) else ""
            if not voice_name and isinstance(voice, str) and voice.strip():
                voice_name = str(voice).strip()
            voice_name = voice_name or None
            quality_preset = str(_kwargs.get("quality_preset") or _kwargs.get("quality") or "").strip()
            instructions_value = str(_kwargs.get("instructions") or "").strip()
            speed_value = _kwargs.get("speed")
            provider_id = _norm_engine_id(
                provider_id
                or getattr(vm, "_abstractvoice_tts_engine", None)
                or getattr(vm, "_tts_engine_name", None)
                or getattr(getattr(vm, "tts_adapter", None), "engine_id", None)
            )
            piper_voice_is_profile = False
            sentinel = object()
            old_vm_model = getattr(vm, "tts_model", sentinel)
            old_language = getattr(vm, "language", sentinel)
            old_profile_id = ""
            adapter = getattr(vm, "tts_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            old_speed = getattr(vm, "speed", sentinel)
            old_tts_quality = sentinel
            old_cloned_quality = sentinel
            applied_profile = False
            try:
                if speed_value is not None and hasattr(vm, "set_speed"):
                    try:
                        vm.set_speed(float(speed_value))
                    except Exception:
                        pass
                if requested_language and hasattr(vm, "set_language"):
                    try:
                        vm.set_language(str(requested_language))
                    except Exception:
                        pass
                if (profile_name or voice_name) and hasattr(vm, "get_active_profile"):
                    try:
                        old_profile = vm.get_active_profile(kind="tts")
                        old_profile_id = str(getattr(old_profile, "profile_id", "") or "").strip()
                    except Exception:
                        old_profile_id = ""
                if model_name:
                    try:
                        setattr(vm, "tts_model", model_name)
                    except Exception:
                        pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                    if provider_id == "piper":
                        language = _piper_language_for_model(model_name)
                        if language and hasattr(vm, "set_language"):
                            vm.set_language(language)
                if provider_id == "piper":
                    for candidate in (profile_name, model_name, voice_name):
                        language = _piper_language_for_model(str(candidate or ""))
                        if not language:
                            continue
                        if hasattr(vm, "set_language"):
                            vm.set_language(language)
                        if voice_name and str(candidate) == voice_name:
                            piper_voice_is_profile = True
                        break
                is_cloned_voice = False
                if voice_name and hasattr(vm, "get_cloned_voice"):
                    try:
                        is_cloned_voice = bool(vm.get_cloned_voice(voice_name))
                    except Exception:
                        is_cloned_voice = False
                if quality_preset:
                    if not is_cloned_voice and hasattr(vm, "get_tts_quality_preset"):
                        try:
                            old_tts_quality = vm.get_tts_quality_preset()
                        except Exception:
                            old_tts_quality = sentinel
                    if is_cloned_voice and hasattr(vm, "get_cloned_tts_quality_preset"):
                        try:
                            old_cloned_quality = vm.get_cloned_tts_quality_preset()
                        except Exception:
                            old_cloned_quality = sentinel
                    if not is_cloned_voice and hasattr(vm, "set_tts_quality_preset"):
                        try:
                            vm.set_tts_quality_preset(quality_preset)
                        except Exception:
                            pass
                    if is_cloned_voice and hasattr(vm, "set_cloned_tts_quality"):
                        try:
                            vm.set_cloned_tts_quality(quality_preset)
                        except Exception:
                            pass
                profile_candidate = "" if is_cloned_voice else (profile_name or voice_name or "")
                if profile_candidate and hasattr(vm, "set_profile"):
                    try:
                        applied_profile = bool(vm.set_profile(profile_candidate, kind="tts"))
                    except TypeError:
                        try:
                            applied_profile = bool(vm.set_profile(profile_candidate))
                        except Exception:
                            applied_profile = False
                    except Exception:
                        applied_profile = False
                speak_kwargs = {
                    "format": str(format),
                    "voice": None if applied_profile or piper_voice_is_profile else voice_name,
                }
                if instructions_value:
                    try:
                        audio = vm.speak_to_bytes(
                            str(text),
                            instructions=instructions_value,
                            **speak_kwargs,
                        )
                    except TypeError as e:
                        if "instructions" not in str(e):
                            raise
                        audio = vm.speak_to_bytes(str(text), **speak_kwargs)
                else:
                    audio = vm.speak_to_bytes(str(text), **speak_kwargs)
            finally:
                if applied_profile and old_profile_id and hasattr(vm, "set_profile"):
                    try:
                        vm.set_profile(old_profile_id, kind="tts")
                    except TypeError:
                        try:
                            vm.set_profile(old_profile_id)
                        except Exception:
                            pass
                    except Exception:
                        pass
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "tts_model", old_vm_model)
                    except Exception:
                        pass
                if old_language is not sentinel and getattr(vm, "language", None) != old_language:
                    try:
                        vm.set_language(old_language)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass
                if old_speed is not sentinel and hasattr(vm, "set_speed"):
                    try:
                        vm.set_speed(float(old_speed))
                    except Exception:
                        try:
                            setattr(vm, "speed", old_speed)
                        except Exception:
                            pass
                if old_tts_quality is not sentinel and hasattr(vm, "set_tts_quality_preset"):
                    try:
                        vm.set_tts_quality_preset(old_tts_quality)
                    except Exception:
                        pass
                if old_cloned_quality is not sentinel and hasattr(vm, "set_cloned_tts_quality"):
                    try:
                        vm.set_cloned_tts_quality(old_cloned_quality)
                    except Exception:
                        pass
            tts_metrics = None
            try:
                if hasattr(vm, "pop_last_tts_metrics"):
                    tts_metrics = vm.pop_last_tts_metrics()
            except Exception:
                tts_metrics = None

        merged_meta: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            merged_meta.update(metadata)
        if isinstance(tts_metrics, dict) and tts_metrics:
            merged_meta["abstractvoice_tts"] = dict(tts_metrics)

        return self._maybe_store_audio(
            audio,
            artifact_store=artifact_store,
            fmt=str(format),
            run_id=run_id,
            tags=tags,
            metadata=merged_meta if merged_meta else None,
        )

    def tts_stream(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        format: str = "wav",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        profile: Optional[str] = None,
        speed: Optional[float] = None,
        instructions: Optional[str] = None,
        quality_preset: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
        **_kwargs: Any,
    ):
        """Yield transport-safe TTS stream events without storing run artifacts.

        Runtime is responsible for run-scoped final artifact truth. This method
        only adapts AbstractVoice chunk semantics to Core's optional stream
        capability.
        """

        fmt = str(format or "wav").strip().lower()
        if fmt == "wave":
            fmt = "wav"
        if fmt != "wav":
            raise ValueError("AbstractVoice TTS streaming currently emits wav segment chunks only")
        instructions_value = str(instructions or _kwargs.get("instructions") or "").strip()
        if instructions_value:
            raise ValueError("AbstractVoice TTS streaming does not support instructions yet; use buffered TTS")

        provider_id, requested_model = _resolve_tts_provider_request(provider, model)
        voice_name = ""
        if isinstance(voice, dict):
            for value in (
                voice.get("voice_id"),
                voice.get("voice"),
                voice.get("profile_id"),
                voice.get("id"),
                voice.get("name"),
            ):
                if isinstance(value, str) and value.strip():
                    voice_name = value.strip()
                    break
            if not provider_id:
                provider_id = _profile_provider_id(voice)
            if not requested_model:
                model_from_voice = _profile_model_id(voice)
                if model_from_voice:
                    requested_model = model_from_voice

        vm = self._get_vm_for_provider(tts_provider=provider_id, tts_model=requested_model)
        lk = self._vm_lock(vm)

        def _events():
            chunk_count = 0
            cancelled = False
            tts_metrics = None
            with lk:
                requested_language = _tts_model_language_selector(provider_id, requested_model)
                model_name = "" if requested_language else (str(requested_model or "").strip() if isinstance(requested_model, str) else "")
                profile_name = str(profile or _kwargs.get("profile") or "").strip()
                local_voice_name = voice_name
                if not local_voice_name and isinstance(voice, str) and voice.strip():
                    local_voice_name = str(voice).strip()
                local_voice_name = local_voice_name or None
                quality_value = str(quality_preset or _kwargs.get("quality") or "").strip()
                speed_value = speed if speed is not None else _kwargs.get("speed")
                active_provider = _norm_engine_id(
                    provider_id
                    or getattr(vm, "_abstractvoice_tts_engine", None)
                    or getattr(vm, "_tts_engine_name", None)
                    or getattr(getattr(vm, "tts_adapter", None), "engine_id", None)
                )
                piper_voice_is_profile = False
                sentinel = object()
                old_vm_model = getattr(vm, "tts_model", sentinel)
                old_language = getattr(vm, "language", sentinel)
                old_profile_id = ""
                adapter = getattr(vm, "tts_adapter", None)
                old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
                old_speed = getattr(vm, "speed", sentinel)
                old_tts_quality = sentinel
                old_cloned_quality = sentinel
                applied_profile = False
                try:
                    if speed_value is not None and hasattr(vm, "set_speed"):
                        try:
                            vm.set_speed(float(speed_value))
                        except Exception:
                            pass
                    if requested_language and hasattr(vm, "set_language"):
                        try:
                            vm.set_language(str(requested_language))
                        except Exception:
                            pass
                    if (profile_name or local_voice_name) and hasattr(vm, "get_active_profile"):
                        try:
                            old_profile = vm.get_active_profile(kind="tts")
                            old_profile_id = str(getattr(old_profile, "profile_id", "") or "").strip()
                        except Exception:
                            old_profile_id = ""
                    if model_name:
                        try:
                            setattr(vm, "tts_model", model_name)
                        except Exception:
                            pass
                        if adapter is not None:
                            try:
                                setattr(adapter, "model_id", model_name)
                            except Exception:
                                pass
                        if active_provider == "piper":
                            language = _piper_language_for_model(model_name)
                            if language and hasattr(vm, "set_language"):
                                vm.set_language(language)
                    if active_provider == "piper":
                        for candidate in (profile_name, model_name, local_voice_name):
                            language = _piper_language_for_model(str(candidate or ""))
                            if not language:
                                continue
                            if hasattr(vm, "set_language"):
                                vm.set_language(language)
                            if local_voice_name and str(candidate) == local_voice_name:
                                piper_voice_is_profile = True
                            break
                    is_cloned_voice = False
                    if local_voice_name and hasattr(vm, "get_cloned_voice"):
                        try:
                            is_cloned_voice = bool(vm.get_cloned_voice(local_voice_name))
                        except Exception:
                            is_cloned_voice = False
                    if quality_value:
                        if not is_cloned_voice and hasattr(vm, "get_tts_quality_preset"):
                            try:
                                old_tts_quality = vm.get_tts_quality_preset()
                            except Exception:
                                old_tts_quality = sentinel
                        if is_cloned_voice and hasattr(vm, "get_cloned_tts_quality_preset"):
                            try:
                                old_cloned_quality = vm.get_cloned_tts_quality_preset()
                            except Exception:
                                old_cloned_quality = sentinel
                        if not is_cloned_voice and hasattr(vm, "set_tts_quality_preset"):
                            try:
                                vm.set_tts_quality_preset(quality_value)
                            except Exception:
                                pass
                        if is_cloned_voice and hasattr(vm, "set_cloned_tts_quality"):
                            try:
                                vm.set_cloned_tts_quality(quality_value)
                            except Exception:
                                pass
                    profile_candidate = "" if is_cloned_voice else (profile_name or local_voice_name or "")
                    if profile_candidate and hasattr(vm, "set_profile"):
                        try:
                            applied_profile = bool(vm.set_profile(profile_candidate, kind="tts"))
                        except TypeError:
                            try:
                                applied_profile = bool(vm.set_profile(profile_candidate))
                            except Exception:
                                applied_profile = False
                        except Exception:
                            applied_profile = False

                    stream_voice = None if applied_profile or piper_voice_is_profile else local_voice_name
                    for audio_chunk, sample_rate in vm.speak_to_audio_chunks(
                        str(text),
                        voice=stream_voice,
                        cancel_event=cancel_event,
                    ):
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled = True
                            break
                        audio_bytes = _audio_chunk_to_wav_segment_bytes(audio_chunk, sample_rate)
                        if not audio_bytes:
                            continue
                        yield {
                            "type": "audio",
                            "schema": "abstractvoice.tts_stream.audio.v1",
                            "sequence": int(chunk_count),
                            "content_type": "audio/wav",
                            "format": "wav",
                            "sample_rate": int(sample_rate or 0),
                            "channels": 1,
                            "audio": audio_bytes,
                            "size_bytes": len(audio_bytes),
                            "provider": active_provider or None,
                            "model": model_name or requested_model or None,
                            "voice": local_voice_name,
                            "profile": profile_name or None,
                            "delivery": "abstractvoice_audio_chunk",
                        }
                        chunk_count += 1
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                finally:
                    if applied_profile and old_profile_id and hasattr(vm, "set_profile"):
                        try:
                            vm.set_profile(old_profile_id, kind="tts")
                        except TypeError:
                            try:
                                vm.set_profile(old_profile_id)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    if old_vm_model is not sentinel:
                        try:
                            setattr(vm, "tts_model", old_vm_model)
                        except Exception:
                            pass
                    if old_language is not sentinel and getattr(vm, "language", None) != old_language:
                        try:
                            vm.set_language(old_language)
                        except Exception:
                            pass
                    if adapter is not None and old_adapter_model is not sentinel:
                        try:
                            setattr(adapter, "model_id", old_adapter_model)
                        except Exception:
                            pass
                    if old_speed is not sentinel and hasattr(vm, "set_speed"):
                        try:
                            vm.set_speed(float(old_speed))
                        except Exception:
                            try:
                                setattr(vm, "speed", old_speed)
                            except Exception:
                                pass
                    if old_tts_quality is not sentinel and hasattr(vm, "set_tts_quality_preset"):
                        try:
                            vm.set_tts_quality_preset(old_tts_quality)
                        except Exception:
                            pass
                    if old_cloned_quality is not sentinel and hasattr(vm, "set_cloned_tts_quality"):
                        try:
                            vm.set_cloned_tts_quality(old_cloned_quality)
                        except Exception:
                            pass
                    try:
                        if hasattr(vm, "pop_last_tts_metrics"):
                            tts_metrics = vm.pop_last_tts_metrics()
                    except Exception:
                        tts_metrics = None

            yield {
                "type": "done" if not cancelled else "cancelled",
                "schema": "abstractvoice.tts_stream.done.v1",
                "ok": not cancelled,
                "cancelled": bool(cancelled),
                "chunks": int(chunk_count),
                "single_chunk": int(chunk_count) == 1,
                "format": "wav",
                "content_type": "audio/wav",
                "metrics": _json_safe(tts_metrics) if isinstance(tts_metrics, dict) else None,
                "delivery": "abstractvoice_audio_chunks",
                "native_streaming": None,
            }

        return _events()

    def stt(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        provider_id, requested_model = _resolve_stt_provider_request(provider, model)
        vm = self._get_vm_for_provider(stt_provider=provider_id, stt_model=requested_model)
        lk = self._vm_lock(vm)
        with lk:
            model_name = str(requested_model or "").strip() if isinstance(requested_model, str) else ""
            sentinel = object()
            old_vm_model = getattr(vm, "stt_model", sentinel)
            old_whisper_model = getattr(vm, "whisper_model", sentinel)
            adapter = getattr(vm, "stt_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            try:
                if model_name:
                    try:
                        setattr(vm, "stt_model", model_name)
                    except Exception:
                        pass
                    if provider_id in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                        try:
                            setattr(vm, "whisper_model", model_name)
                            setattr(vm, "stt_adapter", None)
                            adapter = None
                        except Exception:
                            pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                if isinstance(audio, str):
                    return vm.transcribe_file(str(audio), language=language)

                if isinstance(audio, dict):
                    import os
                    import tempfile

                    audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                    suffix = self._suffix_for_audio_ref(audio, artifact_store=artifact_store)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                        tmp_file.write(bytes(audio_bytes))
                        tmp_path = tmp_file.name
                    try:
                        return vm.transcribe_file(tmp_path, language=language)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                return vm.transcribe_from_bytes(bytes(audio_bytes), language=language)
            finally:
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "stt_model", old_vm_model)
                    except Exception:
                        pass
                if old_whisper_model is not sentinel:
                    try:
                        setattr(vm, "whisper_model", old_whisper_model)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass


class _AudioCapability(_BaseVoice):
    backend_id = "abstractvoice:stt"

    def available_providers(self, task: Any = None) -> Dict[str, Any]:
        """Return selectable STT provider ids without constructing heavy runtimes."""

        normalized_task = str(task or "").strip().lower()
        if normalized_task and normalized_task not in {
            "audio",
            "stt",
            "transcribe",
            "transcription",
            "speech_to_text",
            "speech-to-text",
            "audio_transcription",
            "audio-transcription",
        }:
            return {
                "stt": [],
                "providers": [],
                "stt_providers": [],
                "known_stt_providers": _known_stt_provider_ids(),
                "active_stt_provider": self._configured_provider_id(kind="stt"),
                "details": {"stt": {}},
            }

        stt = self._available_stt_provider_ids()
        return {
            "stt": stt,
            "providers": stt,
            "stt_providers": stt,
            "known_stt_providers": _known_stt_provider_ids(),
            "active_stt_provider": self._configured_provider_id(kind="stt"),
            "details": {"stt": _provider_details("stt", stt)},
        }

    # Alias for callers that use list_* naming conventions.
    def list_available_providers(self, task: Any = None) -> Dict[str, Any]:
        return self.available_providers(task)

    def list_models(
        self,
        task: Any = None,
        provider: Optional[str] = None,
        provider_id: Optional[str] = None,
        kind: Optional[str] = None,
        model: Optional[str] = None,
        **_kwargs: Any,
    ) -> list[str]:
        """List provider-filtered models for STT discovery (import-light, no model loads)."""

        normalized_task = str(task or kind or "").strip().lower()
        if normalized_task and normalized_task not in {
            "audio",
            "stt",
            "transcribe",
            "transcription",
            "speech_to_text",
            "speech-to-text",
            "audio_transcription",
            "audio-transcription",
        }:
            return []

        provider_value: Any = provider if provider is not None else provider_id
        if isinstance(provider_value, dict):
            provider_value = (
                provider_value.get("provider_id")
                or provider_value.get("provider")
                or provider_value.get("engine_id")
                or provider_value.get("engine")
                or provider_value.get("id")
            )

        resolved_provider, requested_model = _resolve_stt_provider_request(provider_value, model)
        requested_model = _normalize_optional_model_id(requested_model)
        if requested_model:
            return [requested_model]

        providers = [resolved_provider] if resolved_provider else self._available_stt_provider_ids()
        model_ids: list[str] = []
        for provider_name in providers:
            normalized_provider = _norm_engine_id(provider_name)
            if normalized_provider in {"openai", "openai-compatible"}:
                model_ids.extend(self._configured_stt_model_ids(normalized_provider))
            elif normalized_provider in {"faster-whisper", "transformers-asr"}:
                model_ids.extend(self._configured_stt_model_ids(normalized_provider))
                model_ids.extend(_stt_model_ids_for_provider(normalized_provider))

        return _dedupe_strings(model_ids)

    def load_resident_model(self, request: Any) -> Dict[str, Any]:
        parsed = self._parse_resident_model_request(request)
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]
        opts = parsed.get("options") or {}

        if task not in {"stt", "audio"}:
            return self._residency_error(
                task=task,
                provider=provider or None,
                model=model,
                code="not_implemented_yet",
                message="Residency warmup is implemented only for STT on the audio backend.",
                state="not_implemented",
                details={"supported": {"task": "stt", "backend": getattr(self, "backend_id", None)}},
            )

        provider_id = _norm_engine_id(provider)
        requested_model = _normalize_optional_model_id(model)
        if not provider_id:
            return self._residency_error(
                task="stt",
                provider=None,
                model=requested_model,
                code="invalid_request",
                message="Provide a local STT provider id (for example faster-whisper or transformers-asr).",
                state="failed",
                details={"supported": {"task": "stt", "provider": "local_stt"}},
            )
        if provider_id in {"openai", "openai-compatible"}:
            return self._residency_error(
                task="stt",
                provider=provider_id,
                model=requested_model,
                code="not_supported",
                message="Residency warmup is supported only for local STT engines (remote providers are excluded).",
                state="not_implemented",
                local=False,
                unloadable=False,
                details={"supported": {"task": "stt", "provider": "local_stt"}},
            )

        if not _local_stt_engine_available(provider_id):
            return self._residency_error(
                task="stt",
                provider=provider_id,
                model=requested_model,
                code="not_implemented_yet",
                message=(
                    "Local STT residency requires the local STT runtime to be installed. "
                    "Install the provider extra (for example abstractvoice[stt] / abstractvoice[apple] / abstractvoice[gpu])."
                ),
                state="not_implemented",
                local=True,
                unloadable=False,
                details={"supported": {"task": "stt", "provider": "local_stt"}},
            )

        warmup = _coerce_bool(opts.get("warmup"), False)
        warmup_audio = opts.get("warmup_audio_path") or opts.get("audio_path") or opts.get("audio")
        warmup_audio_path = str(warmup_audio).strip() if isinstance(warmup_audio, str) and str(warmup_audio).strip() else None
        language = opts.get("language")
        lang = str(language).strip().lower() if isinstance(language, str) and language.strip() else None

        try:
            vm = self._get_vm_for_provider(stt_provider=provider_id, stt_model=requested_model)
            lk = self._vm_lock(vm)
            with lk:
                preload = getattr(vm, "preload_stt_engine", None)
                if not callable(preload):
                    return self._residency_error(
                        task="stt",
                        provider=provider_id,
                        model=requested_model,
                        code="not_implemented_yet",
                        message="STT preload is not available on this VoiceManager build.",
                        state="not_implemented",
                        local=True,
                        unloadable=False,
                    )
                result = preload(
                    warmup=bool(warmup),
                    warmup_audio_path=warmup_audio_path,
                    language=lang,
                )
        except Exception as e:
            return self._engine_residency_entry(
                task="stt",
                provider=provider_id,
                model=requested_model,
                component="stt_engine",
                state="failed",
                loaded=False,
                local=True,
                unloadable=True,
                error={"code": "load_failed", "message": str(e)},
            )

        details = {
            "warmed": bool(result.get("warmed", False)),
            "warm_error": result.get("warm_error"),
        }
        loaded = bool(result.get("resident", False))
        return self._engine_residency_entry(
            task="stt",
            provider=provider_id,
            model=requested_model,
            component="stt_engine",
            state=str(result.get("state") or ("resident" if loaded else "configured")),
            loaded=bool(loaded),
            local=True,
            unloadable=True,
            details=details,
        )

    def list_resident_models(self, filters: Any | None = None) -> list[Dict[str, Any]]:
        parsed = self._parse_resident_model_request(filters or {})
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]
        if task and task not in {"stt", "audio"}:
            return []
        provider_id = _norm_engine_id(provider) if provider else ""

        out: list[Dict[str, Any]] = []
        for cache_key, vm in self._iter_known_vms():
            try:
                lk = self._vm_lock(vm)
                with lk:
                    components = vm.list_resident_components() if hasattr(vm, "list_resident_components") else []
            except Exception:
                continue
            for component in list(components or []):
                if not isinstance(component, dict):
                    continue
                if str(component.get("component") or "").strip().lower() != "stt_engine":
                    continue
                engine = _norm_engine_id(component.get("engine"))
                component_model = component.get("model")
                component_model_s = str(component_model).strip() if isinstance(component_model, str) and component_model.strip() else None
                if provider_id and engine != provider_id:
                    continue
                if model and component_model_s and str(model).strip() and str(model).strip().lower() != "default":
                    if component_model_s.strip().lower() != str(model).strip().lower():
                        continue
                details = {}
                if cache_key is not None:
                    details["cache_key"] = _json_safe(list(cache_key))
                out.append(
                    self._engine_residency_entry(
                        task="stt",
                        provider=engine or None,
                        model=component_model_s,
                        component="stt_engine",
                        state=str(component.get("state") or ("resident" if component.get("resident") else "configured")),
                        loaded=bool(component.get("resident", False)),
                        local=bool(component.get("local", True)),
                        unloadable=bool(component.get("unloadable", True)),
                        details=details,
                    )
                )

        out.sort(key=lambda item: (str(item.get("provider") or ""), str(item.get("model") or ""), str(item.get("state") or "")))
        return out

    def unload_resident_model(self, request: Any) -> Dict[str, Any]:
        parsed = self._parse_resident_model_request(request)
        task = parsed["task"]
        provider = parsed["provider"]
        model = parsed["model"]

        if task not in {"stt", "audio"}:
            return self._residency_error(
                task=task,
                provider=provider or None,
                model=model,
                code="not_implemented_yet",
                message="Residency unload is implemented only for STT on the audio backend.",
                state="not_implemented",
                details={"supported": {"task": "stt", "backend": getattr(self, "backend_id", None)}},
            )

        provider_id = _norm_engine_id(provider)
        requested_model = _normalize_optional_model_id(model)
        if not provider_id:
            return self._residency_error(
                task="stt",
                provider=None,
                model=requested_model,
                code="invalid_request",
                message="Provide a local STT provider id to unload.",
                state="failed",
                details={"supported": {"task": "stt", "provider": "local_stt"}},
            )
        if provider_id in {"openai", "openai-compatible"}:
            return self._residency_error(
                task="stt",
                provider=provider_id,
                model=requested_model,
                code="not_supported",
                message="Residency unload is supported only for local STT engines (remote providers are excluded).",
                state="not_implemented",
                local=False,
                unloadable=False,
                details={"supported": {"task": "stt", "provider": "local_stt"}},
            )

        unloaded_count = 0
        last_error = None
        for _cache_key, vm in self._iter_known_vms():
            try:
                lk = self._vm_lock(vm)
                with lk:
                    engines = self._vm_engine_values(vm, kind="stt")
                    if not (_engine_aliases(provider_id) & engines):
                        continue
                    if requested_model and hasattr(vm, "list_resident_components"):
                        comps = list(vm.list_resident_components() or [])
                        match = False
                        for comp in comps:
                            if not isinstance(comp, dict):
                                continue
                            if str(comp.get("component") or "").strip().lower() != "stt_engine":
                                continue
                            comp_model = comp.get("model")
                            comp_model_s = str(comp_model).strip() if isinstance(comp_model, str) and comp_model.strip() else None
                            if comp_model_s and comp_model_s.strip().lower() == str(requested_model).strip().lower():
                                match = True
                                break
                        if not match:
                            continue
                    unload = getattr(vm, "unload_stt_engine", None)
                    if callable(unload):
                        res = unload()
                        if bool(res.get("unloaded")):
                            unloaded_count += 1
                    else:
                        last_error = "unload_stt_engine_not_available"
            except Exception as e:
                last_error = str(e)
                continue

        details = {"unloaded_count": int(unloaded_count)}
        if last_error and unloaded_count == 0:
            return self._engine_residency_entry(
                task="stt",
                provider=provider_id,
                model=requested_model,
                component="stt_engine",
                state="failed",
                loaded=False,
                local=True,
                unloadable=True,
                details=details,
                error={"code": "unload_failed", "message": str(last_error)},
                unloaded=False,
            )
        return self._engine_residency_entry(
            task="stt",
            provider=provider_id,
            model=requested_model,
            component="stt_engine",
            state="unloaded" if unloaded_count > 0 else "not_loaded",
            loaded=False,
            local=True,
            unloadable=True,
            details=details,
            unloaded=bool(unloaded_count > 0),
        )

    def transcribe(
        self,
        audio: Union[bytes, Dict[str, Any], str],
        *,
        language: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        artifact_store: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_kwargs: Any,
    ) -> str:
        _ = metadata
        provider_id, requested_model = _resolve_stt_provider_request(provider, model)
        vm = self._get_vm_for_provider(stt_provider=provider_id, stt_model=requested_model)
        lk = self._vm_lock(vm)
        with lk:
            model_name = str(requested_model or "").strip() if isinstance(requested_model, str) else ""
            sentinel = object()
            old_vm_model = getattr(vm, "stt_model", sentinel)
            old_whisper_model = getattr(vm, "whisper_model", sentinel)
            adapter = getattr(vm, "stt_adapter", None)
            old_adapter_model = getattr(adapter, "model_id", sentinel) if adapter is not None else sentinel
            try:
                if model_name:
                    try:
                        setattr(vm, "stt_model", model_name)
                    except Exception:
                        pass
                    if provider_id in {"faster-whisper", "faster_whisper", "whisper", "local"}:
                        try:
                            setattr(vm, "whisper_model", model_name)
                            setattr(vm, "stt_adapter", None)
                            adapter = None
                        except Exception:
                            pass
                    if adapter is not None:
                        try:
                            setattr(adapter, "model_id", model_name)
                        except Exception:
                            pass
                if isinstance(audio, str):
                    return vm.transcribe_file(str(audio), language=language)

                if isinstance(audio, dict):
                    import os
                    import tempfile

                    audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                    suffix = self._suffix_for_audio_ref(audio, artifact_store=artifact_store)
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                        tmp_file.write(bytes(audio_bytes))
                        tmp_path = tmp_file.name
                    try:
                        return vm.transcribe_file(tmp_path, language=language)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                audio_bytes = self._resolve_audio_bytes(audio, artifact_store=artifact_store)
                return vm.transcribe_from_bytes(bytes(audio_bytes), language=language)
            finally:
                if old_vm_model is not sentinel:
                    try:
                        setattr(vm, "stt_model", old_vm_model)
                    except Exception:
                        pass
                if old_whisper_model is not sentinel:
                    try:
                        setattr(vm, "whisper_model", old_whisper_model)
                    except Exception:
                        pass
                if adapter is not None and old_adapter_model is not sentinel:
                    try:
                        setattr(adapter, "model_id", old_adapter_model)
                    except Exception:
                        pass


def register(registry: Any) -> None:
    """Register AbstractVoice as an AbstractCore capability plugin."""

    registry.register_voice_backend(
        backend_id=_VoiceCapability.backend_id,
        factory=lambda owner: _VoiceCapability(owner),
        priority=0,
        description="AbstractVoice VoiceManager (TTS+STT).",
        config_hint=(
            "Defaults to OpenAI remote TTS in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_tts_engine/ABSTRACTVOICE_TTS_ENGINE. Use piper or supertonic with "
            "abstractvoice[apple]/[gpu] for local offline TTS."
        ),
    )
    registry.register_audio_backend(
        backend_id=_AudioCapability.backend_id,
        factory=lambda owner: _AudioCapability(owner),
        priority=0,
        description="AbstractVoice STT (speech-to-text).",
        config_hint=(
            "Defaults to OpenAI remote STT in AbstractCore integrations; set OPENAI_API_KEY "
            "or configure voice_stt_engine/ABSTRACTVOICE_STT_ENGINE. Use faster_whisper with abstractvoice[stt], [apple], or [gpu] for local offline STT, or transformers-asr with abstractvoice[stt-hf] for Hugging Face ASR models."
        ),
    )
