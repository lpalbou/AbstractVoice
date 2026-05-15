"""Minimal Supertonic 3 ONNX runtime.

Design constraints:
- keep the base ``abstractvoice`` install remote-only;
- do not depend on the external ``supertonic`` Python SDK;
- keep model downloads explicit or governed by ``allow_downloads``;
- expose only simple numpy/WAV primitives to the adapter layer.

The model artifacts are downloaded from:
https://huggingface.co/Supertone/supertonic-3
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from unicodedata import category, normalize

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ID = "Supertone/supertonic-3"
DEFAULT_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"

SUPERTONIC_LANGUAGES = [
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
]
SUPERTONIC_UNKNOWN_LANGUAGE = "na"
SUPERTONIC_AVAILABLE_LANGUAGES = SUPERTONIC_LANGUAGES + [SUPERTONIC_UNKNOWN_LANGUAGE]
SUPERTONIC_VOICE_STYLES = ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]

_ONNX_DIR = Path("onnx")
_VOICE_STYLES_DIR = Path("voice_styles")
_REQUIRED_FILES = [
    _ONNX_DIR / "duration_predictor.onnx",
    _ONNX_DIR / "text_encoder.onnx",
    _ONNX_DIR / "vector_estimator.onnx",
    _ONNX_DIR / "vocoder.onnx",
    _ONNX_DIR / "tts.json",
    _ONNX_DIR / "unicode_indexer.json",
    *[_VOICE_STYLES_DIR / f"{style}.json" for style in SUPERTONIC_VOICE_STYLES],
]

_WS_RE = re.compile(r"\s+")
_PUNCT_REPLACEMENTS = {
    "\u2013": "-",
    "\u2011": "-",
    "\u2014": "-",
    "\u00af": " ",
    "_": " ",
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u00b4": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "\u2192": " ",
    "\u2190": " ",
}
_ENDING_PUNCTUATION_RE = re.compile(r"[.!?;:,'\"')\]}]$")


@dataclass(frozen=True)
class SupertonicStyle:
    """Style vectors consumed by Supertonic's ONNX graph."""

    ttl: np.ndarray
    dp: np.ndarray


def get_supertonic_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the AbstractVoice cache directory for Supertonic 3."""

    if cache_dir:
        root = Path(cache_dir).expanduser()
    else:
        root = Path(os.environ.get("ABSTRACTVOICE_SUPERTONIC_CACHE_DIR", "") or "").expanduser()
        if not str(root) or str(root) == ".":
            root = Path.home() / ".cache" / "abstractvoice" / "supertonic-3"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _required_paths(cache_dir: Path) -> list[Path]:
    return [cache_dir / rel for rel in _REQUIRED_FILES]


def is_supertonic_cached(cache_dir: str | os.PathLike[str] | None = None) -> bool:
    """Return True when all required Supertonic 3 files are cached."""

    root = get_supertonic_cache_dir(cache_dir)
    return all(path.exists() and path.stat().st_size > 0 for path in _required_paths(root))


def _download_file(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with requests.get(str(url), stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=str(dest.parent), delete=False) as tmp:
                tmp_path = Path(tmp.name)
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        tmp.write(chunk)
        tmp_path.replace(dest)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def prefetch_supertonic(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    revision: str | None = None,
) -> Path:
    """Download Supertonic 3 ONNX assets and built-in voice styles."""

    root = get_supertonic_cache_dir(cache_dir)
    rev = str(revision or DEFAULT_REVISION).strip() or DEFAULT_REVISION
    base = f"https://huggingface.co/{MODEL_ID}/resolve/{rev}"

    for rel in _REQUIRED_FILES:
        dest = root / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{base}/{rel.as_posix()}"
        logger.info("Downloading Supertonic asset: %s", rel.as_posix())
        _download_file(url, dest)

    return root


def _load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _UnicodeProcessor:
    def __init__(self, indexer_path: Path):
        raw = _load_json(indexer_path)
        if not isinstance(raw, (list, dict)):
            raise ValueError("Supertonic unicode_indexer.json has an unsupported format")
        self._indexer = raw
        self._space_id = self._lookup_char(" ", default=0)

    def _lookup_char(self, char: str, *, default: int | None = None) -> int | None:
        codepoint = ord(str(char)[0])
        idx = None
        if isinstance(self._indexer, list):
            if 0 <= codepoint < len(self._indexer):
                idx = self._indexer[codepoint]
        elif isinstance(self._indexer, dict):
            idx = self._indexer.get(str(codepoint))
            if idx is None:
                idx = self._indexer.get(codepoint)
        try:
            value = int(idx)
        except Exception:
            return default
        if value < 0:
            return default
        return value

    def _remove_emojis_and_controls(self, text: str) -> str:
        out: list[str] = []
        for ch in str(text):
            cat = category(ch)
            if cat.startswith("C") and not ch.isspace():
                continue
            code = ord(ch)
            if (
                0x1F000 <= code <= 0x1FAFF
                or 0x2600 <= code <= 0x27BF
                or 0x1F1E6 <= code <= 0x1F1FF
            ):
                continue
            out.append(ch)
        return "".join(out)

    def preprocess_text(self, text: str, lang: str | None) -> str:
        s = str(text or "")
        try:
            s = normalize("NFKD", s)
        except Exception:
            pass
        s = self._remove_emojis_and_controls(s)
        for old, new in _PUNCT_REPLACEMENTS.items():
            s = s.replace(old, new)
        s = re.sub(r"[\\]", "", s)
        s = s.replace("@", " at ")
        s = s.replace("e.g.,", "for example, ")
        s = s.replace("i.e.,", "that is, ")
        s = re.sub(r" ([,\.!?;:])", r"\1", s)
        s = re.sub(r"(['\"])\1+", r"\1", s)
        s = _WS_RE.sub(" ", s).strip()
        if s and not _ENDING_PUNCTUATION_RE.search(s):
            s += "."
        if lang is not None:
            language = str(lang or SUPERTONIC_UNKNOWN_LANGUAGE).strip().lower()
            if language not in SUPERTONIC_AVAILABLE_LANGUAGES:
                language = SUPERTONIC_UNKNOWN_LANGUAGE
            s = f"<{language}>{s}</{language}>"
        return s

    def __call__(self, texts: list[str], lang: str | None) -> tuple[np.ndarray, np.ndarray]:
        processed = [self.preprocess_text(text, lang) for text in texts]
        lengths = np.array([len(text) for text in processed], dtype=np.int64)
        max_len = int(lengths.max()) if lengths.size else 0
        text_ids = np.zeros((len(processed), max_len), dtype=np.int64)
        for row, text in enumerate(processed):
            ids = []
            for ch in text:
                idx = self._lookup_char(ch, default=self._space_id)
                ids.append(int(idx if idx is not None else 0))
            if ids:
                text_ids[row, : len(ids)] = np.asarray(ids, dtype=np.int64)
        text_mask = _length_to_mask(lengths)
        return text_ids, text_mask


def _length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    if lengths.size <= 0:
        return np.zeros((0, 1, 0), dtype=np.float32)
    ml = int(max_len or int(lengths.max()))
    ids = np.arange(0, ml)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, ml)


def _latent_mask(
    wav_lengths: np.ndarray,
    *,
    base_chunk_size: int,
    chunk_compress_factor: int,
) -> np.ndarray:
    latent_size = int(base_chunk_size) * int(chunk_compress_factor)
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    return _length_to_mask(latent_lengths)


def _clamp_speed(speed: float | None) -> float:
    try:
        value = float(speed)
    except Exception:
        value = 1.05
    return min(2.0, max(0.7, value))


def _quality_steps(total_steps: int | None) -> int:
    try:
        steps = int(total_steps)
    except Exception:
        steps = 8
    return min(100, max(1, steps))


def _array_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if mono.size:
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        mono = np.clip(mono, -1.0, 1.0)
    pcm = (mono * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


class SupertonicRuntime:
    """Runtime wrapper around Supertonic 3 ONNX sessions."""

    def __init__(
        self,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        revision: str | None = None,
        allow_downloads: bool = True,
        intra_op_num_threads: int | None = None,
        inter_op_num_threads: int | None = None,
        providers: list[str] | None = None,
    ):
        self.cache_dir = get_supertonic_cache_dir(cache_dir)
        self.revision = str(revision or DEFAULT_REVISION).strip() or DEFAULT_REVISION
        self.allow_downloads = bool(allow_downloads)
        self.intra_op_num_threads = intra_op_num_threads
        self.inter_op_num_threads = inter_op_num_threads
        self.providers = providers

        self._loaded = False
        self._cfg: dict = {}
        self._processor: _UnicodeProcessor | None = None
        self._dp = None
        self._text_encoder = None
        self._vector_estimator = None
        self._vocoder = None
        self._sample_rate = 24000
        self._base_chunk_size = 1024
        self._chunk_compress_factor = 6
        self._latent_dim = 24
        self._styles: dict[str, SupertonicStyle] = {}

    @property
    def sample_rate(self) -> int:
        return int(self._sample_rate or 24000)

    def ensure_downloaded(self) -> Path:
        return prefetch_supertonic(cache_dir=self.cache_dir, revision=self.revision)

    def is_cached(self) -> bool:
        return is_supertonic_cached(self.cache_dir)

    def _resolve_providers(self):
        import onnxruntime as ort  # type: ignore

        requested = self.providers
        if requested is None:
            raw = os.environ.get("ABSTRACTVOICE_SUPERTONIC_ONNX_PROVIDERS", "")
            requested = [p.strip() for p in raw.split(",") if p.strip()] if raw else ["CPUExecutionProvider"]
        available = set(ort.get_available_providers() or [])
        selected = [p for p in list(requested or []) if p in available]
        return selected or ["CPUExecutionProvider"]

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Supertonic requires ONNX Runtime. Install with:\n"
                '  pip install "abstractvoice[supertonic]"'
            ) from e

        if not self.is_cached():
            if not self.allow_downloads:
                raise FileNotFoundError(
                    "Supertonic 3 artifacts are not cached. Prefetch with:\n"
                    "  abstractvoice-prefetch --supertonic\n"
                    "  python -m abstractvoice download --supertonic"
                )
            self.ensure_downloaded()

        cfg_path = self.cache_dir / _ONNX_DIR / "tts.json"
        cfg = _load_json(cfg_path)
        if not isinstance(cfg, dict):
            raise ValueError("Supertonic tts.json has an unsupported format")

        self._cfg = dict(cfg)
        try:
            self._sample_rate = int(cfg["ae"]["sample_rate"])
            self._base_chunk_size = int(cfg["ae"]["base_chunk_size"])
            self._chunk_compress_factor = int(cfg["ttl"]["chunk_compress_factor"])
            self._latent_dim = int(cfg["ttl"]["latent_dim"])
        except Exception as e:
            raise ValueError("Supertonic tts.json is missing required fields") from e

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self.intra_op_num_threads is not None:
            opts.intra_op_num_threads = int(self.intra_op_num_threads)
        if self.inter_op_num_threads is not None:
            opts.inter_op_num_threads = int(self.inter_op_num_threads)

        providers = self._resolve_providers()

        def _session(rel: Path):
            path = self.cache_dir / rel
            if not path.exists():
                raise FileNotFoundError(str(path))
            return ort.InferenceSession(str(path), sess_options=opts, providers=providers)

        self._dp = _session(_ONNX_DIR / "duration_predictor.onnx")
        self._text_encoder = _session(_ONNX_DIR / "text_encoder.onnx")
        self._vector_estimator = _session(_ONNX_DIR / "vector_estimator.onnx")
        self._vocoder = _session(_ONNX_DIR / "vocoder.onnx")
        self._processor = _UnicodeProcessor(self.cache_dir / _ONNX_DIR / "unicode_indexer.json")
        self._loaded = True

    def load_style(self, voice_style: str) -> SupertonicStyle:
        style_name = str(voice_style or "M1").strip().upper() or "M1"
        if style_name in self._styles:
            return self._styles[style_name]
        if style_name not in SUPERTONIC_VOICE_STYLES:
            raise ValueError(f"Unknown Supertonic voice style: {voice_style}")
        if not self.is_cached():
            if not self.allow_downloads:
                raise FileNotFoundError(
                    "Supertonic 3 voice styles are not cached. Prefetch with: "
                    "abstractvoice-prefetch --supertonic"
                )
            self.ensure_downloaded()
        path = self.cache_dir / _VOICE_STYLES_DIR / f"{style_name}.json"
        raw = _load_json(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid Supertonic voice style: {path}")

        def _vector(section: str) -> np.ndarray:
            item = raw.get(section)
            if not isinstance(item, dict):
                raise ValueError(f"Invalid Supertonic voice style section: {section}")
            dims = item.get("dims")
            data = item.get("data")
            if not isinstance(dims, list) or data is None:
                raise ValueError(f"Invalid Supertonic voice style section: {section}")
            return np.asarray(data, dtype=np.float32).reshape(*[int(x) for x in dims])

        style = SupertonicStyle(ttl=_vector("style_ttl"), dp=_vector("style_dp"))
        self._styles[style_name] = style
        return style

    def _sample_noisy_latent(self, duration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bsz = int(len(duration))
        wav_len_max = float(np.asarray(duration).max()) * float(self.sample_rate)
        wav_lengths = (np.asarray(duration, dtype=np.float32) * float(self.sample_rate)).astype(np.int64)
        chunk_size = int(self._base_chunk_size) * int(self._chunk_compress_factor)
        latent_len = max(1, int(math.ceil(float(wav_len_max) / float(chunk_size))))
        latent_dim = int(self._latent_dim) * int(self._chunk_compress_factor)
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        mask = _latent_mask(
            wav_lengths,
            base_chunk_size=int(self._base_chunk_size),
            chunk_compress_factor=int(self._chunk_compress_factor),
        )
        return noisy_latent * mask, mask

    def _infer_one(
        self,
        text: str,
        *,
        language: str,
        style: SupertonicStyle,
        total_steps: int,
        speed: float,
    ) -> tuple[np.ndarray, float]:
        self._ensure_loaded()
        if self._processor is None or self._dp is None or self._text_encoder is None or self._vector_estimator is None or self._vocoder is None:
            raise RuntimeError("Supertonic runtime is not loaded")

        text_ids, text_mask = self._processor([str(text)], str(language))
        dur, *_ = self._dp.run(
            None,
            {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask},
        )
        dur = np.asarray(dur, dtype=np.float32) / float(_clamp_speed(speed))
        text_emb, *_ = self._text_encoder.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )
        xt, latent_mask = self._sample_noisy_latent(dur)
        total_step_np = np.array([float(total_steps)], dtype=np.float32)
        for step in range(int(total_steps)):
            current_step = np.array([float(step)], dtype=np.float32)
            xt, *_ = self._vector_estimator.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self._vocoder.run(None, {"latent": xt})
        mono = np.asarray(wav, dtype=np.float32).reshape(-1)
        duration_s = float(np.asarray(dur).reshape(-1)[0])
        trim = int(max(0, round(duration_s * float(self.sample_rate))))
        if trim > 0 and trim < mono.size:
            mono = mono[:trim]
        return mono.astype(np.float32, copy=False), duration_s

    def iter_audio_chunks(
        self,
        text: str,
        *,
        language: str = "en",
        voice_style: str = "M1",
        total_steps: int = 8,
        speed: float = 1.05,
        max_chars: int | None = None,
        silence_duration: float = 0.3,
    ) -> Iterable[tuple[np.ndarray, int]]:
        self._ensure_loaded()
        style = self.load_style(voice_style)
        lang = str(language or "en").strip().lower() or "en"
        if lang not in SUPERTONIC_AVAILABLE_LANGUAGES:
            lang = SUPERTONIC_UNKNOWN_LANGUAGE
        steps = _quality_steps(total_steps)
        sp = _clamp_speed(speed)
        mc = int(max_chars or (120 if lang in {"ko", "ja"} else 300))
        silence_samples = max(0, int(float(silence_duration) * float(self.sample_rate)))

        from ..tts.text_chunking import split_text_batches

        chunks = split_text_batches(str(text), max_chars=mc)
        for idx, chunk_text in enumerate(chunks):
            audio, _duration_s = self._infer_one(
                str(chunk_text),
                language=lang,
                style=style,
                total_steps=steps,
                speed=sp,
            )
            if audio.size:
                yield audio, int(self.sample_rate)
            if idx < len(chunks) - 1 and silence_samples > 0:
                yield np.zeros((silence_samples,), dtype=np.float32), int(self.sample_rate)

    def synthesize(
        self,
        text: str,
        *,
        language: str = "en",
        voice_style: str = "M1",
        total_steps: int = 8,
        speed: float = 1.05,
        max_chars: int | None = None,
        silence_duration: float = 0.3,
    ) -> np.ndarray:
        chunks = [
            np.asarray(chunk, dtype=np.float32).reshape(-1)
            for chunk, _sr in self.iter_audio_chunks(
                text,
                language=language,
                voice_style=voice_style,
                total_steps=total_steps,
                speed=speed,
                max_chars=max_chars,
                silence_duration=silence_duration,
            )
            if np.asarray(chunk).size
        ]
        return np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.zeros((0,), dtype=np.float32)

    def synthesize_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        return _array_to_wav_bytes(audio, self.sample_rate)
