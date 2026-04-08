"""Text chunking utilities for streamed TTS.

We use two related concepts:
- **batches**: splitting a *complete* text into short segments (sentence-first)
- **chunker**: incremental segmentation for a *stream* of text deltas (LLM streaming)

Design goals:
- be deterministic and simple (robust across languages)
- prefer natural boundaries (sentence terminators, newlines)
- provide a hard cap (`max_chars`) to bound latency / per-chunk synthesis cost
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_RE_WS = re.compile(r"\s+")

# Sentence terminators (keep it simple + multilingual).
_SENTENCE_TERMINATORS = set(".!?。！？")
# Soft boundaries that are usually safe to cut on for streamed speech.
# These are *not* always sentence ends, but they often represent a natural pause.
_SOFT_TERMINATORS = set(",;:，；：")
_RE_SOFT_END = re.compile(r"(?<=[,;:，；：])\s+")


def split_text_batches(text: str, *, max_chars: int = 240) -> list[str]:
    """Split text into short batches, preferring sentence boundaries."""
    try:
        mc = int(max_chars)
    except Exception:
        mc = 240
    if mc <= 0:
        mc = 240

    s = str(text or "").strip()
    if not s:
        return []

    # Normalize whitespace (stable splitting; good-enough for speech).
    s = _RE_WS.sub(" ", s).strip()
    if not s:
        return []
    if len(s) <= mc:
        return [s]

    # Split on common sentence terminators.
    parts = re.split(r"(?<=[\.\!\?\。\！\？])\s+", s)

    # Ensure each part fits under max_chars (word-based fallback).
    pieces: list[str] = []
    for p in parts:
        p = str(p or "").strip()
        if not p:
            continue
        if len(p) <= mc:
            pieces.append(p)
            continue

        # Fallback for long sentences:
        # 1) try soft boundaries first (commas/semicolons/colons)
        # 2) then word-based chunking.
        soft_parts = re.split(_RE_SOFT_END, p) if ("," in p or "，" in p or ";" in p or "；" in p or ":" in p or "：" in p) else [p]
        for sp in soft_parts:
            sp = str(sp or "").strip()
            if not sp:
                continue
            if len(sp) <= mc:
                pieces.append(sp)
                continue
            cur = ""
            for w in sp.split(" "):
                w = w.strip()
                if not w:
                    continue
                cand = (cur + " " + w).strip() if cur else w
                if len(cand) <= mc:
                    cur = cand
                else:
                    if cur:
                        pieces.append(cur)
                    cur = w
            if cur:
                pieces.append(cur)

    # Merge pieces into batches (avoid overly short chunks).
    batches: list[str] = []
    cur = ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if len(cand) <= mc:
            cur = cand
        else:
            if cur:
                batches.append(cur)
            cur = p
    if cur:
        batches.append(cur)

    return batches


@dataclass(frozen=True)
class TextStreamChunkingConfig:
    max_chars: int = 240
    # Minimum segment size before we emit on a boundary.
    # Keep this small by default to minimize time-to-first-voice in LLM→TTS pipelines.
    min_chars: int = 1


class TextStreamChunker:
    """Incrementally turn text deltas into speakable segments."""

    def __init__(self, *, config: TextStreamChunkingConfig | None = None) -> None:
        self._cfg = config or TextStreamChunkingConfig()
        self._buf = ""

    def push(self, delta: str) -> list[str]:
        self._buf += str(delta or "")
        return self._pop_ready_segments()

    def flush(self) -> list[str]:
        out: list[str] = []
        s = str(self._buf or "").strip()
        self._buf = ""
        if s:
            out.append(s)
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pop_ready_segments(self) -> list[str]:
        out: list[str] = []
        mc = int(self._cfg.max_chars) if int(self._cfg.max_chars) > 0 else 240
        mn = int(self._cfg.min_chars) if int(self._cfg.min_chars) >= 0 else 0

        while True:
            s = self._buf.lstrip()
            if s != self._buf:
                self._buf = s
            if not self._buf:
                break

            cut = self._find_cut_index(self._buf, max_chars=mc, min_chars=mn)
            if cut is None:
                break

            seg = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            if seg:
                out.append(seg)

        return out

    @staticmethod
    def _find_cut_index(buf: str, *, max_chars: int, min_chars: int) -> int | None:
        n = len(buf)
        if n <= 0:
            return None

        # Prefer early, natural boundaries once we have enough content.
        if n >= max(1, min_chars):
            for i, ch in enumerate(buf):
                if ch == "\n" and i + 1 >= max(1, min_chars):
                    return i + 1
                if ch in _SENTENCE_TERMINATORS:
                    j = i + 1
                    if j >= max(1, min_chars):
                        # Only cut if we're at end or next char looks like a boundary.
                        if j >= n or buf[j].isspace():
                            return j
                if ch in _SOFT_TERMINATORS:
                    j = i + 1
                    if j >= max(1, min_chars):
                        # Only cut if we appear to be at a phrase boundary.
                        if j >= n or buf[j].isspace():
                            return j

        # Hard cap: cut at the best boundary <= max_chars.
        if n > max_chars and max_chars > 0:
            upto = min(max_chars, n)

            # Try last sentence boundary.
            for i in range(upto - 1, -1, -1):
                if buf[i] in _SENTENCE_TERMINATORS and i + 1 >= 1:
                    return i + 1

            # Then last soft boundary.
            for i in range(upto - 1, -1, -1):
                if buf[i] in _SOFT_TERMINATORS and i + 1 >= 1:
                    return i + 1

            # Then last whitespace.
            for i in range(upto - 1, -1, -1):
                if buf[i].isspace():
                    return i + 1

            # Final fallback: hard cut.
            return upto

        return None

