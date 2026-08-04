"""Central provider/model compatibility catalog.

This module is the package-owned source of truth for feature compatibility
across TTS, STT, and voice cloning. The goal is to make capability checks
explicit, queryable, and stable for higher-level integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
import pkgutil
from typing import Any, Iterable, Literal

from .adapters.stt_faster_whisper import FasterWhisperAdapter
from .adapters.stt_transformers_asr import TransformersASRAdapter
from .adapters.tts_openai_compatible import _OPENAI_KNOWN_TTS_MODELS, _configured_tts_models
from .adapters.tts_piper import PiperTTSAdapter

CapabilitySupportLevel = Literal["native", "emulated", "conditional", "unsupported"]
CapabilityKind = Literal["tts", "stt", "cloning"]

TTS_COMPATIBILITY_FEATURES: tuple[str, ...] = (
    "speed",
    "quality_preset",
    "instructions",
    "profile",
    "pace",
    "target_duration_s",
    "actions",
    "scene_context",
    "ambient_audio",
    "background_sfx",
    "output_channels",
)

STT_COMPATIBILITY_FEATURES: tuple[str, ...] = (
    "language",
    "prompt",
    "audio_bytes",
    "audio_array",
    "word_timestamps",
)

CLONING_COMPATIBILITY_FEATURES: tuple[str, ...] = (
    "reference_audio",
    "reference_text",
    "reference_text_autofallback",
    "multi_reference_audio",
    "speed",
    "audio_chunks",
)

DEFAULT_OPENAI_STT_MODELS: tuple[str, ...] = (
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
)


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_provider(value: Any, *, kind: CapabilityKind | None = None) -> str:
    text = _norm_text(value).lower().replace("_", "-")
    if text in {"remote", "compatible", "proxy"}:
        return "openai-compatible"
    if kind == "stt" and text in {"faster-whisper", "faster_whisper", "whisper", "local"}:
        return "faster-whisper"
    if kind == "stt" and text in {"transformers_asr", "transformers", "hf_asr", "hf"}:
        return "transformers-asr"
    if kind == "cloning" and text in {"f5-tts", "f5tts", "openf5", "open-f5"}:
        return "f5_tts"
    return text


def _copy_meta(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class CapabilitySupport:
    support: CapabilitySupportLevel
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": str(self.support),
            "reason": self.reason,
            "metadata": _copy_meta(self.metadata),
        }


@dataclass(frozen=True)
class ModelCompatibility:
    model: str
    surfaces: dict[str, dict[str, CapabilitySupport]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "surfaces": {
                str(surface): {
                    str(feature): support.to_dict()
                    for feature, support in dict(features or {}).items()
                }
                for surface, features in dict(self.surfaces or {}).items()
            },
        }


@dataclass(frozen=True)
class ProviderCompatibility:
    kind: CapabilityKind
    provider: str
    default_surfaces: dict[str, dict[str, CapabilitySupport]] = field(default_factory=dict)
    models: dict[str, ModelCompatibility] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "provider": str(self.provider),
            "default_surfaces": {
                str(surface): {
                    str(feature): support.to_dict()
                    for feature, support in dict(features or {}).items()
                }
                for surface, features in dict(self.default_surfaces or {}).items()
            },
            "models": {
                str(model): record.to_dict()
                for model, record in dict(self.models or {}).items()
            },
        }


@dataclass(frozen=True)
class CompatibilityCatalog:
    version: str = "compatibility_v1"
    providers: dict[str, dict[str, ProviderCompatibility]] = field(default_factory=dict)

    def get_provider(self, *, kind: str, provider: str) -> ProviderCompatibility | None:
        normalized_kind = str(kind or "").strip().lower()
        return dict(self.providers or {}).get(normalized_kind, {}).get(
            _norm_provider(provider, kind=normalized_kind if normalized_kind in {"tts", "stt", "cloning"} else None)
        )

    def get_model(
        self,
        *,
        kind: str,
        provider: str,
        model: str | None = None,
    ) -> ModelCompatibility | None:
        record = self.get_provider(kind=kind, provider=provider)
        if record is None:
            return None
        if isinstance(model, str) and model.strip():
            found = dict(record.models or {}).get(model.strip())
            if found is not None:
                return found
        wildcard = dict(record.models or {}).get("*")
        return wildcard

    def support_for(
        self,
        *,
        kind: str,
        provider: str,
        feature: str,
        model: str | None = None,
        surface: str = "default",
    ) -> CapabilitySupport | None:
        feature_name = str(feature or "").strip()
        surface_name = resolve_surface_name(kind, surface)
        model_record = self.get_model(kind=kind, provider=provider, model=model)
        if model_record is not None:
            surface_map = dict(model_record.surfaces or {}).get(surface_name)
            if isinstance(surface_map, dict) and feature_name in surface_map:
                return surface_map[feature_name]
        provider_record = self.get_provider(kind=kind, provider=provider)
        if provider_record is None:
            return None
        surface_map = dict(provider_record.default_surfaces or {}).get(surface_name)
        if isinstance(surface_map, dict):
            return surface_map.get(feature_name)
        return None

    def find_models(
        self,
        *,
        kind: str,
        feature: str,
        surface: str = "default",
        support_in: Iterable[CapabilitySupportLevel] = ("native", "emulated", "conditional"),
    ) -> list[dict[str, Any]]:
        wanted = {str(item) for item in list(support_in or ())}
        surface = resolve_surface_name(kind, surface)
        out: list[dict[str, Any]] = []
        for provider_name, provider_record in dict(self.providers.get(str(kind), {}) or {}).items():
            matched_model = False
            for model_name, model_record in dict(provider_record.models or {}).items():
                support = self.support_for(
                    kind=str(kind),
                    provider=str(provider_name),
                    model=str(model_name),
                    surface=str(surface),
                    feature=str(feature),
                )
                if support is None or str(support.support) not in wanted:
                    continue
                matched_model = True
                out.append(
                    {
                        "kind": str(kind),
                        "provider": str(provider_name),
                        "model": str(model_name),
                        "surface": str(surface),
                        "feature": str(feature),
                        "support": support.to_dict(),
                    }
                )
            if matched_model:
                continue
            support = self.support_for(
                kind=str(kind),
                provider=str(provider_name),
                model=None,
                surface=str(surface),
                feature=str(feature),
            )
            if support is None or str(support.support) not in wanted:
                continue
            out.append(
                {
                    "kind": str(kind),
                    "provider": str(provider_name),
                    "model": None,
                    "surface": str(surface),
                    "feature": str(feature),
                    "support": support.to_dict(),
                }
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": str(self.version),
            "providers": {
                str(kind): {
                    str(provider): record.to_dict()
                    for provider, record in dict(items or {}).items()
                }
                for kind, items in dict(self.providers or {}).items()
            },
        }


def _feature(
    support: CapabilitySupportLevel,
    *,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CapabilitySupport:
    return CapabilitySupport(
        support=str(support),  # type: ignore[arg-type]
        reason=reason,
        metadata=_copy_meta(metadata),
    )


def _surface(features: dict[str, CapabilitySupport]) -> dict[str, CapabilitySupport]:
    return dict(features or {})


def _feature_names_for_kind(kind: CapabilityKind) -> tuple[str, ...]:
    if kind == "tts":
        return TTS_COMPATIBILITY_FEATURES
    if kind == "stt":
        return STT_COMPATIBILITY_FEATURES
    return CLONING_COMPATIBILITY_FEATURES


def _norm_kind(value: Any) -> CapabilityKind:
    text = str(value or "").strip().lower()
    return text if text in ("tts", "stt", "cloning") else "tts"  # type: ignore[return-value]


def resolve_surface_name(kind: Any, surface: Any) -> str:
    """Map the API's ``surface="default"`` to the kind's primary catalog surface.

    No provider publishes a surface literally named "default" (the real names
    are ``bytes``/``playback``, ``transcribe``, ``create``/``speak_bytes``), so
    a literal lookup made every default-argument capability query return None.
    Explicit surface names pass through (lowercased, like every normalizer in
    this module). Note the surfaces can genuinely differ -- e.g. qwen3-tts
    reports ``instructions`` on ``bytes`` but not on ``playback`` -- so
    live-playback consumers should ask about ``surface="playback"`` explicitly.
    """
    name = str(surface or "default").strip().lower() or "default"
    if name != "default":
        return name
    return _default_surface_names(_norm_kind(kind))[0]


def _default_surface_names(kind: CapabilityKind) -> tuple[str, ...]:
    if kind == "tts":
        return ("bytes", "playback")
    if kind == "stt":
        return ("transcribe",)
    return ("create", "speak_bytes")


def _empty_surfaces(kind: CapabilityKind) -> dict[str, dict[str, CapabilitySupport]]:
    features = _feature_names_for_kind(kind)
    return {
        surface_name: _surface({feature_name: _feature("unsupported") for feature_name in features})
        for surface_name in _default_surface_names(kind)
    }


def _read_capability_asset_bytes() -> bytes:
    """Read the capability asset in a way that survives package-name shadowing.

    ``pkgutil.get_data`` keys on the "abstractvoice" package NAME. A directory
    named ``abstractvoice`` reachable from ``sys.path[0]`` (e.g. a serving
    process launched with cwd = the monorepo root) resolves the package as a
    loaderless NAMESPACE package, so ``get_data`` returns None even though this
    module itself was imported from the real install. Resolving the asset
    relative to THIS module file is immune to that shadow, so it goes first;
    ``pkgutil`` stays as the fallback for non-filesystem installs.
    """
    asset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice_model_capabilities.json")
    try:
        with open(asset_path, "rb") as fh:
            return fh.read()
    except OSError:
        pass

    raw = pkgutil.get_data("abstractvoice", "assets/voice_model_capabilities.json")
    if raw is not None:
        return raw

    import sys

    detail = f"looked beside this module ({asset_path}) and via pkgutil.get_data('abstractvoice', ...)"
    package = sys.modules.get("abstractvoice")
    if package is not None and getattr(package, "__file__", None) is None:
        paths = [str(item) for item in list(getattr(package, "__path__", []) or [])]
        detail += (
            "; 'abstractvoice' resolved as a NAMESPACE package"
            + (f" from {paths}" if paths else "")
            + " — a directory named 'abstractvoice' on sys.path (often the current working directory of the"
            " serving process) is shadowing the installed package. Launch from a different cwd or remove"
            " the shadowing entry from sys.path."
        )
    raise RuntimeError(f"Capability asset not found: abstractvoice/assets/voice_model_capabilities.json ({detail})")


@lru_cache(maxsize=1)
def _load_capability_asset() -> dict[str, Any]:
    data = json.loads(_read_capability_asset_bytes().decode("utf-8"))
    _validate_voice_capabilities_json(data)
    return data


def _validate_voice_capabilities_json(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Invalid voice capability asset: top-level JSON must be an object.")
    if not isinstance(data.get("schema_version"), (str, int, float)):
        raise ValueError("Invalid voice capability asset: missing required key 'schema_version'.")
    kinds = data.get("kinds")
    if not isinstance(kinds, dict):
        raise ValueError("Invalid voice capability asset: 'kinds' must be an object.")
    for kind in ("tts", "stt", "cloning"):
        kind_entry = kinds.get(kind)
        if not isinstance(kind_entry, dict):
            raise ValueError(f"Invalid voice capability asset: missing kind entry {kind!r}.")
        providers = kind_entry.get("providers")
        if not isinstance(providers, dict):
            raise ValueError(f"Invalid voice capability asset: kinds[{kind!r}]['providers'] must be an object.")
        for provider_name, provider_entry in providers.items():
            if not isinstance(provider_entry, dict):
                raise ValueError(f"Invalid voice capability asset: provider entry {kind}/{provider_name} must be an object.")
            default_surfaces = provider_entry.get("default_surfaces", {})
            if not isinstance(default_surfaces, dict):
                raise ValueError(
                    f"Invalid voice capability asset: default_surfaces for {kind}/{provider_name} must be an object."
                )
            models = provider_entry.get("models", {})
            if not isinstance(models, dict):
                raise ValueError(f"Invalid voice capability asset: models for {kind}/{provider_name} must be an object.")
            for scope_name, surface_map in [("default_surfaces", default_surfaces)]:
                _validate_surface_map(kind, provider_name, scope_name, surface_map)
            for model_name, model_entry in models.items():
                if not isinstance(model_entry, dict):
                    raise ValueError(
                        f"Invalid voice capability asset: model entry {kind}/{provider_name}/{model_name} must be an object."
                    )
                surface_map = model_entry.get("surfaces", {})
                if not isinstance(surface_map, dict):
                    raise ValueError(
                        f"Invalid voice capability asset: surfaces for {kind}/{provider_name}/{model_name} must be an object."
                    )
                _validate_surface_map(kind, provider_name, f"models/{model_name}", surface_map)


def _validate_surface_map(kind: str, provider: str, label: str, surface_map: dict[str, Any]) -> None:
    allowed_support = {"native", "emulated", "conditional", "unsupported"}
    for surface_name, features in surface_map.items():
        if not isinstance(features, dict):
            raise ValueError(
                f"Invalid voice capability asset: {kind}/{provider}/{label}/{surface_name} must map to an object."
            )
        for feature_name, spec in features.items():
            if isinstance(spec, str):
                support_value = spec
            elif isinstance(spec, dict):
                support_value = str(spec.get("support") or "")
            else:
                raise ValueError(
                    f"Invalid voice capability asset: {kind}/{provider}/{label}/{surface_name}/{feature_name} must be a string or object."
                )
            if support_value not in allowed_support:
                raise ValueError(
                    f"Invalid voice capability asset: unsupported support state {support_value!r} at {kind}/{provider}/{label}/{surface_name}/{feature_name}."
                )


def _provider_asset(kind: CapabilityKind, provider: str) -> dict[str, Any]:
    kinds = dict(_load_capability_asset().get("kinds") or {})
    kind_entry = dict(kinds.get(str(kind)) or {})
    providers = dict(kind_entry.get("providers") or {})
    return dict(providers.get(_norm_provider(provider, kind=kind)) or {})


def _asset_model_ids(kind: CapabilityKind, provider: str) -> list[str]:
    provider_entry = _provider_asset(kind, provider)
    return [str(model_name) for model_name in dict(provider_entry.get("models") or {}).keys() if str(model_name).strip()]


def _support_from_asset(value: Any) -> CapabilitySupport:
    if isinstance(value, str):
        return _feature(value)
    if isinstance(value, dict):
        return _feature(
            str(value.get("support") or "unsupported"),
            reason=(str(value.get("reason")).strip() if value.get("reason") is not None else None),
            metadata=(dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else None),
        )
    return _feature("unsupported")


def _normalize_surface_features(kind: CapabilityKind, raw_features: dict[str, Any]) -> dict[str, CapabilitySupport]:
    out: dict[str, CapabilitySupport] = {}
    for feature_name in _feature_names_for_kind(kind):
        if feature_name in raw_features:
            out[feature_name] = _support_from_asset(raw_features.get(feature_name))
        else:
            out[feature_name] = _feature("unsupported")
    for feature_name, value in dict(raw_features or {}).items():
        name = str(feature_name or "").strip()
        if not name or name in out:
            continue
        out[name] = _support_from_asset(value)
    return out


def _merge_surfaces(
    kind: CapabilityKind,
    base_surfaces: dict[str, dict[str, CapabilitySupport]],
    overrides: dict[str, Any] | None,
) -> dict[str, dict[str, CapabilitySupport]]:
    out = {
        str(surface_name): dict(features or {})
        for surface_name, features in dict(base_surfaces or {}).items()
    }
    for surface_name in _default_surface_names(kind):
        out.setdefault(surface_name, _surface({feature_name: _feature("unsupported") for feature_name in _feature_names_for_kind(kind)}))
    for surface_name, raw_features in dict(overrides or {}).items():
        if not isinstance(raw_features, dict):
            continue
        current = dict(out.get(str(surface_name)) or {})
        current.update(_normalize_surface_features(kind, raw_features))
        out[str(surface_name)] = current
    return out


def _tts_surfaces_for_provider(provider: str) -> dict[str, dict[str, CapabilitySupport]]:
    return _merge_surfaces("tts", _empty_surfaces("tts"), _provider_asset("tts", provider).get("default_surfaces"))


def _tts_surfaces_for_model(
    provider: str,
    model: str,
    base_surfaces: dict[str, dict[str, CapabilitySupport]],
) -> dict[str, dict[str, CapabilitySupport]]:
    provider_entry = _provider_asset("tts", provider)
    model_entry = dict(dict(provider_entry.get("models") or {}).get(_norm_text(model)) or {})
    return _merge_surfaces("tts", base_surfaces, model_entry.get("surfaces"))


def _stt_surfaces_for_provider(provider: str) -> dict[str, dict[str, CapabilitySupport]]:
    return _merge_surfaces("stt", _empty_surfaces("stt"), _provider_asset("stt", provider).get("default_surfaces"))


def _cloning_surfaces_for_provider(provider: str) -> dict[str, dict[str, CapabilitySupport]]:
    return _merge_surfaces(
        "cloning",
        _empty_surfaces("cloning"),
        _provider_asset("cloning", provider).get("default_surfaces"),
    )


def _env_split(key: str) -> list[str]:
    raw = os.environ.get(str(key), "")
    out: list[str] = []
    for item in str(raw).replace("\n", ",").split(","):
        value = item.strip()
        if value:
            out.append(value)
    return out


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        value = _norm_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _known_tts_models(
    *,
    current_provider: str | None = None,
    current_model: str | None = None,
) -> dict[str, list[str]]:
    out = {
        "openai": _asset_model_ids("tts", "openai") or list(_OPENAI_KNOWN_TTS_MODELS),
        "openai-compatible": _configured_tts_models("openai-compatible"),
        "piper": [model for _lang, (_path, model) in PiperTTSAdapter.PIPER_MODELS.items()],
        "supertonic": _asset_model_ids("tts", "supertonic"),
        "omnivoice": _asset_model_ids("tts", "omnivoice"),
        "audiodit": _asset_model_ids("tts", "audiodit"),
        "qwen3-tts": _asset_model_ids("tts", "qwen3-tts"),
    }
    if current_model:
        target_provider = _norm_provider(current_provider, kind="tts") if current_provider else ""
        if target_provider and target_provider in out:
            out.setdefault(target_provider, []).append(str(current_model).strip())
        else:
            for provider in ("openai-compatible", "omnivoice", "audiodit", "qwen3-tts", "supertonic", "piper", "openai"):
                if current_model not in out[provider]:
                    continue
                break
            else:
                out.setdefault("openai-compatible", []).append(str(current_model).strip())
    return {provider: _dedupe(models) for provider, models in out.items()}


def _known_stt_models(
    *,
    current_provider: str | None = None,
    current_model: str | None = None,
) -> dict[str, list[str]]:
    out = {
        "openai": _asset_model_ids("stt", "openai") or list(DEFAULT_OPENAI_STT_MODELS),
        "openai-compatible": _dedupe(
            _asset_model_ids("stt", "openai-compatible")
            + list(DEFAULT_OPENAI_STT_MODELS)
            + _env_split("ABSTRACTVOICE_OPENAI_COMPATIBLE_STT_MODELS")
            + _env_split("ABSTRACTVOICE_REMOTE_STT_MODELS")
            + _env_split("ABSTRACTVOICE_OPENAI_STT_MODELS")
        ),
        "faster-whisper": list(FasterWhisperAdapter.selectable_model_ids()),
        "transformers-asr": list(TransformersASRAdapter.selectable_model_ids()),
    }
    if current_model:
        target_provider = _norm_provider(current_provider, kind="stt") if current_provider else ""
        if target_provider and target_provider in out:
            out.setdefault(target_provider, []).append(str(current_model).strip())
        elif current_model not in out.get("openai-compatible", []):
            out.setdefault("openai-compatible", []).append(str(current_model).strip())
    return {provider: _dedupe(models) for provider, models in out.items()}


def _known_cloning_models(
    *,
    current_provider: str | None = None,
    current_remote_tts_model: str | None = None,
) -> dict[str, list[str]]:
    remote_models = _configured_tts_models("openai-compatible")
    out = {
        "f5_tts": _asset_model_ids("cloning", "f5_tts"),
        "omnivoice": _asset_model_ids("cloning", "omnivoice"),
        "audiodit": _asset_model_ids("cloning", "audiodit"),
        "chroma": _asset_model_ids("cloning", "chroma"),
        "qwen3-tts": _asset_model_ids("cloning", "qwen3-tts"),
        "openai": _asset_model_ids("cloning", "openai") or list(_OPENAI_KNOWN_TTS_MODELS),
        "openai-compatible": _dedupe(remote_models),
    }
    if current_remote_tts_model:
        target_provider = _norm_provider(current_provider, kind="cloning") if current_provider else ""
        if target_provider in out:
            out.setdefault(target_provider, []).append(str(current_remote_tts_model).strip())
        else:
            out.setdefault("openai-compatible", []).append(str(current_remote_tts_model).strip())
    return {provider: _dedupe(models) for provider, models in out.items()}


def _build_provider_record(
    *,
    kind: CapabilityKind,
    provider: str,
    models: Iterable[str],
    surfaces: dict[str, dict[str, CapabilitySupport]],
) -> ProviderCompatibility:
    model_map: dict[str, ModelCompatibility] = {}
    deduped_models = _dedupe(models)
    if not deduped_models:
        model_map["*"] = ModelCompatibility(model="*", surfaces=dict(surfaces))
    else:
        for model_name in deduped_models:
            model_map[str(model_name)] = ModelCompatibility(model=str(model_name), surfaces=dict(surfaces))
    return ProviderCompatibility(
        kind=str(kind),  # type: ignore[arg-type]
        provider=str(provider),
        default_surfaces=dict(surfaces),
        models=model_map,
    )


def build_compatibility_catalog(
    *,
    current_tts_provider: str | None = None,
    current_tts_model: str | None = None,
    current_stt_provider: str | None = None,
    current_stt_model: str | None = None,
    current_cloning_provider: str | None = None,
    current_remote_tts_model: str | None = None,
) -> CompatibilityCatalog:
    providers: dict[str, dict[str, ProviderCompatibility]] = {
        "tts": {},
        "stt": {},
        "cloning": {},
    }

    for provider, models in _known_tts_models(
        current_provider=current_tts_provider,
        current_model=current_tts_model,
    ).items():
        base_surfaces = _tts_surfaces_for_provider(provider)
        providers["tts"][provider] = _build_provider_record(
            kind="tts",
            provider=provider,
            models=models,
            surfaces=base_surfaces,
        )
        for model_name in list(providers["tts"][provider].models.keys()):
            if model_name == "*":
                continue
            providers["tts"][provider].models[model_name] = ModelCompatibility(
                model=str(model_name),
                surfaces=_tts_surfaces_for_model(provider, str(model_name), base_surfaces),
            )

    for provider, models in _known_stt_models(
        current_provider=current_stt_provider,
        current_model=current_stt_model,
    ).items():
        providers["stt"][provider] = _build_provider_record(
            kind="stt",
            provider=provider,
            models=models,
            surfaces=_stt_surfaces_for_provider(provider),
        )

    for provider, models in _known_cloning_models(
        current_provider=current_cloning_provider,
        current_remote_tts_model=current_remote_tts_model,
    ).items():
        providers["cloning"][provider] = _build_provider_record(
            kind="cloning",
            provider=provider,
            models=models,
            surfaces=_cloning_surfaces_for_provider(provider),
        )

    return CompatibilityCatalog(providers=providers)
