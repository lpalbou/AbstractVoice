"""AudioDiT TTS adapter (LongCat-AudioDiT).

This adapter is optional and requires:
  pip install "abstractvoice[audiodit]"
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf

from .base import TTSAdapter


_RE_QUOTE = re.compile(r"""["“”‘’]""")
_RE_WS = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    t = str(s or "").lower()
    t = _RE_QUOTE.sub(" ", t)
    t = _RE_WS.sub(" ", t).strip()
    return t


def _prompt_text_prefix_for_seconds(text: str, *, seconds: float, language: str) -> str:
    """Pick a prompt_text prefix roughly matching `seconds` (upstream heuristic)."""
    s = _normalize_text(text)
    if not s:
        return ""
    try:
        target_s = float(seconds)
    except Exception:
        target_s = 4.0
    if not (target_s > 0.0):
        target_s = 4.0

    # Upstream per-character constants.
    en_per = 0.082
    zh_per = 0.21
    lang = str(language or "en").strip().lower()

    compact = re.sub(r"\s+", "", s)
    if not compact:
        return s

    # Count how many non-space characters fit in `target_s`.
    total = 0.0
    n = 0
    for ch in compact:
        if lang == "zh" or ("\u4e00" <= ch <= "\u9fff"):
            total += zh_per
        elif ch.isalpha():
            total += en_per
        else:
            # For non-zh, treat "other" as en (good approximation for English-heavy text).
            total += en_per
        n += 1
        if total >= target_s:
            break

    out: list[str] = []
    seen = 0
    for ch in s:
        out.append(ch)
        if not ch.isspace():
            seen += 1
            if seen >= n:
                break
    return "".join(out).strip() or s


class AudioDiTTTSAdapter(TTSAdapter):
    """TTS adapter that uses LongCat-AudioDiT (AudioDiTModel)."""

    engine_id = "audiodit"

    def __init__(
        self,
        language: str = "en",
        *,
        allow_downloads: bool = True,
        auto_load: bool = False,
        debug_mode: bool = False,
        model_id: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        runtime: Any | None = None,
        settings: Any | None = None,
        max_chars: int = 800,
        use_session_prompt: bool = True,
        session_prompt_seconds: float = 4.0,
    ):
        self._language = str(language or "en").strip().lower()
        self._allow_downloads = bool(allow_downloads)
        self._debug = bool(debug_mode)
        self._sample_rate = 24000

        self._runtime = runtime
        self._settings = settings
        self._max_chars = int(max_chars) if int(max_chars) > 0 else 800

        # Session speaker consistency (upstream methodology: prompt audio + prompt text).
        self._use_session_prompt = bool(use_session_prompt)
        try:
            self._session_prompt_seconds = float(session_prompt_seconds)
        except Exception:
            self._session_prompt_seconds = 4.0
        if not (self._session_prompt_seconds > 0.0):
            self._session_prompt_seconds = 4.0
        self._session_prompt_audio: np.ndarray | None = None
        self._session_prompt_text: str | None = None
        self._session_prompt_sr: int = 24000
        # Cache an encoded prompt latent (torch.Tensor) for session consistency
        # without paying VAE prompt encoding on every turn. Best-effort only.
        self._session_prompt_latent: Any | None = None

        if self._runtime is None:
            try:
                from ..audiodit.runtime import AudioDiTRuntime, AudioDiTSettings
            except Exception as e:
                raise RuntimeError(
                    "AudioDiT requires optional dependencies.\n"
                    "Install with:\n"
                    "  pip install \"abstractvoice[audiodit]\""
                ) from e

            self._settings = self._settings or AudioDiTSettings()
            self._runtime = AudioDiTRuntime(
                model_id=model_id or AudioDiTRuntime.DEFAULT_MODEL_ID,
                revision=revision,
                device=device,
                allow_downloads=bool(self._allow_downloads),
                debug=bool(self._debug),
            )

        # Engine-agnostic quality preset (fast|balanced|high).
        # For AudioDiT this maps to diffusion steps + guidance strength.
        self._quality_preset = "balanced"
        try:
            self.set_quality_preset(getattr(self._settings, "quality_preset", None) or "balanced")
        except Exception:
            # Keep defaults if settings object doesn't match.
            self._quality_preset = "balanced"

        if bool(auto_load):
            # Eagerly load weights/tokenizer so failures are surfaced early.
            try:
                self._runtime._ensure_loaded()  # noqa: SLF001 (internal; best-effort)
            except Exception:
                # Surface errors to caller; explicit engine selection must be actionable.
                raise

    def set_language(self, language: str) -> bool:
        new_lang = str(language or "").strip().lower()
        if new_lang and new_lang != self._language:
            self._language = new_lang
            # Reset prompt on language changes (best-effort).
            self.reset_session_prompt()
        return True

    def get_supported_languages(self) -> list[str]:
        # Upstream LongCat-AudioDiT benchmarks and examples focus on Chinese + English.
        # Other languages may “work” at the tokenizer level but are not guaranteed to
        # be intelligible without a language-specific frontend.
        return ["en", "zh"]

    def get_sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def is_available(self) -> bool:
        # Avoid loading multi-GB weights in a capability probe.
        try:
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                return False
            if importlib.util.find_spec("transformers") is None:
                return False
            if importlib.util.find_spec("einops") is None:
                return False
        except Exception:
            return False
        return True

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "engine": "AudioDiT (LongCat-AudioDiT)",
                "engine_id": "audiodit",
                "sample_rate": self.get_sample_rate(),
                "quality_preset": str(getattr(self, "_quality_preset", "balanced") or "balanced"),
            }
        )
        try:
            if hasattr(self._runtime, "runtime_info"):
                info["runtime"] = dict(self._runtime.runtime_info())
        except Exception:
            pass
        return info

    def set_quality_preset(self, preset: str) -> bool:
        p = str(preset or "").strip().lower()
        if p not in ("fast", "balanced", "high"):
            raise ValueError("preset must be one of: fast|balanced|high")
        self._quality_preset = p

        st = getattr(self, "_settings", None)
        if st is None:
            return False

        # Keep guidance method aligned with upstream-recommended APG by default.
        try:
            if hasattr(st, "guidance_method"):
                st.guidance_method = str(getattr(st, "guidance_method", "apg") or "apg").strip().lower() or "apg"
        except Exception:
            pass

        # Steps dominate speed; guidance strength is a small quality knob.
        try:
            if p == "fast":
                st.steps = 8
                st.cfg_strength = 3.5
            elif p == "balanced":
                st.steps = 16
                st.cfg_strength = 4.0
            else:
                st.steps = 24
                st.cfg_strength = 4.5
        except Exception:
            return False
        return True

    def get_quality_preset(self) -> str | None:
        try:
            return str(getattr(self, "_quality_preset", None) or "").strip() or None
        except Exception:
            return None

    def reset_session_prompt(self) -> None:
        self._session_prompt_audio = None
        self._session_prompt_text = None
        self._session_prompt_sr = int(self.get_sample_rate())
        self._session_prompt_latent = None

    def synthesize(self, text: str) -> np.ndarray:
        prompt_audio = None
        prompt_latent = None
        prompt_text = None
        prompt_sr = None
        if self._use_session_prompt and self._session_prompt_audio is not None and self._session_prompt_text:
            prompt_text = self._session_prompt_text
            prompt_sr = int(self._session_prompt_sr)
            if self._session_prompt_latent is not None:
                prompt_latent = self._session_prompt_latent
            else:
                # Best-effort: pre-encode and cache the prompt latent once.
                try:
                    latent, _, _ = self._runtime.encode_prompt_audio_latent(
                        prompt_audio=self._session_prompt_audio,
                        prompt_audio_sr=int(self._session_prompt_sr),
                        max_prompt_seconds=float(self._session_prompt_seconds),
                    )
                    if latent is not None:
                        self._session_prompt_latent = latent
                        prompt_latent = latent
                except Exception:
                    prompt_latent = None
            if prompt_latent is None:
                # Fallback: pass raw prompt audio (runtime will encode each time).
                prompt_audio = self._session_prompt_audio

        audio, sr = self._runtime.generate(
            text=str(text),
            language=str(self._language),
            prompt_audio_paths=None,
            prompt_audio=prompt_audio,
            prompt_audio_sr=prompt_sr,
            prompt_latent=prompt_latent,
            prompt_text=prompt_text,
            settings=self._settings,
            max_chars=int(self._max_chars),
        )
        self._sample_rate = int(sr)
        out = np.asarray(audio, dtype=np.float32).reshape(-1)

        # Initialize session prompt from the first successful utterance (if enabled).
        if self._use_session_prompt and self._session_prompt_audio is None and out.size and int(sr) > 0:
            try:
                n = int(round(min(8.0, float(self._session_prompt_seconds)) * float(sr)))
                n = max(0, min(int(out.size), int(n)))
                if n >= int(0.8 * float(sr)):
                    self._session_prompt_audio = out[:n].astype(np.float32, copy=False)
                    self._session_prompt_sr = int(sr)
                    self._session_prompt_text = _prompt_text_prefix_for_seconds(
                        str(text),
                        seconds=float(n) / float(sr),
                        language=str(self._language),
                    )
                    self._session_prompt_latent = None
            except Exception:
                pass

        return out

    def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
        if str(format or "wav").strip().lower() != "wav":
            raise ValueError("AudioDiT adapter currently supports WAV output only.")
        audio = self.synthesize(str(text))
        buf = io.BytesIO()
        sf.write(buf, audio, int(self.get_sample_rate()), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize_to_file(self, text: str, output_path: str, format: Optional[str] = None) -> str:
        fmt = (format or Path(output_path).suffix.lstrip(".") or "wav").strip().lower()
        if fmt != "wav":
            raise ValueError("AudioDiT adapter currently supports WAV output only.")
        data = self.synthesize_to_bytes(str(text), format="wav")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return str(out)

