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
_RE_SENTENCE_END = re.compile(r"[.!?](?=\s|$)|[。！？]")
_RE_SOFT_END = re.compile(r"[,;:](?=\s|$)|[，；：]")


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

    # Cut at the last natural boundary inside each bounded window. CJK
    # punctuation does not require whitespace. Slice the original normalized
    # text so merging sentences never inserts spaces into continuous scripts.
    batches: list[str] = []
    start = 0
    while start < len(s):
        limit = min(start + mc, len(s))
        end = limit
        if limit < len(s):
            for boundary in (_RE_SENTENCE_END, _RE_SOFT_END):
                cut = max((m.end() for m in boundary.finditer(s, start, limit + 1) if m.end() <= limit), default=0)
                if cut:
                    end = cut
                    break
            else:
                end = max(s.rfind(" ", start, limit + 1), start)
                if end == start:
                    end = limit  # indivisible word / script: enforce the cap
        segment = s[start:end].strip()
        if segment:
            batches.append(segment)
        start = end
        while start < len(s) and s[start].isspace():
            start += 1

    return batches


def split_complete_text_for_streaming(
    text: str,
    *,
    max_chars: int = 240,
    first_max_chars: int = 96,
) -> list[str]:
    """Split a complete text for low-latency streamed TTS delivery.

    The first segment is allowed to cut at an early natural pause to minimize
    time-to-first-audio. The remaining text is then batched with
    `split_text_batches` so complete responses do not produce one provider call
    per comma/colon.
    """

    try:
        mc = int(max_chars)
    except Exception:
        mc = 240
    if mc <= 0:
        mc = 240
    try:
        first_mc = int(first_max_chars)
    except Exception:
        first_mc = 96
    if first_mc <= 0:
        first_mc = 96
    first_mc = min(mc, first_mc)

    s = _RE_WS.sub(" ", str(text or "")).strip()
    if not s:
        return []
    if len(s) <= first_mc:
        return [s]

    cut = TextStreamChunker._find_cut_index(s, max_chars=first_mc, min_chars=1)
    if cut is None:
        return split_text_batches(s, max_chars=mc)

    first = s[:cut].strip()
    rest = s[cut:].strip()
    if not first:
        return split_text_batches(rest, max_chars=mc)
    if not rest:
        return [first]
    return [first] + split_text_batches(rest, max_chars=mc)


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
            for i, ch in enumerate(buf[:max_chars]):
                if ch == "\n" and i + 1 >= max(1, min_chars):
                    return i + 1
                if ch in _SENTENCE_TERMINATORS:
                    j = i + 1
                    if j >= max(1, min_chars):
                        # Only cut if we're at end or next char looks like a boundary.
                        if ch in "。！？" or j >= n or buf[j].isspace():
                            return j
                if ch in _SOFT_TERMINATORS:
                    j = i + 1
                    if j >= max(1, min_chars):
                        # Only cut if we appear to be at a phrase boundary.
                        if ch in "，；：" or j >= n or buf[j].isspace():
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
