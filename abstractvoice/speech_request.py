"""Package-owned speech request and capability helpers.

The public VoiceManager API remains backward compatible, but internally we want
one place where richer speech-generation intent can be normalized before being
handed to an adapter or cloning engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence


SpeechSupportLevel = Literal["native", "emulated", "conditional", "unsupported"]


def _norm_text(value: Any) -> str:
    return str(value or "")


def _norm_opt_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _norm_opt_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _norm_opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _norm_actions(values: Sequence[Any] | None) -> tuple[str, ...]:
    if not values:
        return ()
    out: list[str] = []
    for value in list(values):
        text = str(value or "").strip()
        if text:
            out.append(text)
    return tuple(out)


@dataclass(frozen=True)
class SpeechCapability:
    """Support status for one package-owned speech field."""

    name: str
    support: SpeechSupportLevel
    reason: str | None = None


@dataclass(frozen=True)
class SpeechCapabilities:
    """Best-effort capability map for richer speech requests."""

    fields: dict[str, SpeechCapability] = field(default_factory=dict)

    def support_for(self, name: str) -> SpeechCapability | None:
        return self.fields.get(str(name or "").strip())

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "support": value.support,
                "reason": value.reason,
            }
            for key, value in dict(self.fields or {}).items()
        }


@dataclass(frozen=True)
class SpeechRequest:
    """Normalized package-owned speech request."""

    text: str
    language: str | None = None
    provider: str | None = None
    model: str | None = None
    profile: str | None = None
    voice: str | None = None
    instructions: str | None = None
    speed: float | None = None
    pace: float | None = None
    target_duration_s: float | None = None
    quality_preset: str | None = None
    scene_context: str | None = None
    actions: tuple[str, ...] = ()
    ambient_audio: str | None = None
    background_sfx: bool | None = None
    output_format: str | None = None
    output_channels: int | None = None
    sanitize_syntax: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": str(self.text),
            "language": self.language,
            "provider": self.provider,
            "model": self.model,
            "profile": self.profile,
            "voice": self.voice,
            "instructions": self.instructions,
            "speed": self.speed,
            "pace": self.pace,
            "target_duration_s": self.target_duration_s,
            "quality_preset": self.quality_preset,
            "scene_context": self.scene_context,
            "actions": list(self.actions),
            "ambient_audio": self.ambient_audio,
            "background_sfx": self.background_sfx,
            "output_format": self.output_format,
            "output_channels": self.output_channels,
            "sanitize_syntax": bool(self.sanitize_syntax),
            "metadata": dict(self.metadata or {}),
        }


def build_speech_request(
    text: Any,
    *,
    language: Any = None,
    provider: Any = None,
    model: Any = None,
    profile: Any = None,
    voice: Any = None,
    instructions: Any = None,
    speed: Any = None,
    pace: Any = None,
    target_duration_s: Any = None,
    quality_preset: Any = None,
    scene_context: Any = None,
    actions: Sequence[Any] | None = None,
    ambient_audio: Any = None,
    background_sfx: Any = None,
    output_format: Any = None,
    output_channels: Any = None,
    sanitize_syntax: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> SpeechRequest:
    """Build a normalized SpeechRequest from legacy call sites."""

    bg_sfx: bool | None
    if background_sfx is None:
        bg_sfx = None
    else:
        bg_sfx = bool(background_sfx)

    return SpeechRequest(
        text=_norm_text(text),
        language=_norm_opt_text(language),
        provider=_norm_opt_text(provider),
        model=_norm_opt_text(model),
        profile=_norm_opt_text(profile),
        voice=_norm_opt_text(voice),
        instructions=_norm_opt_text(instructions),
        speed=_norm_opt_float(speed),
        pace=_norm_opt_float(pace),
        target_duration_s=_norm_opt_float(target_duration_s),
        quality_preset=_norm_opt_text(quality_preset),
        scene_context=_norm_opt_text(scene_context),
        actions=_norm_actions(actions),
        ambient_audio=_norm_opt_text(ambient_audio),
        background_sfx=bg_sfx,
        output_format=_norm_opt_text(output_format),
        output_channels=_norm_opt_int(output_channels),
        sanitize_syntax=bool(sanitize_syntax),
        metadata=dict(metadata or {}),
    )
