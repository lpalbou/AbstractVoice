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


# Streamed delivery of a complete text (round 17, measured on a public gateway 2026-10-17):
# a first segment cut at the EARLIEST pause ("Ah oui,", "Oui.") played 1.3 s of audio, then
# the voice stopped 1.7-3.3 s while the rest of the sentence was made, and a fragment of a
# few words came out in the wrong language ("Ah oui," heard as English "Are we?"). Streaming
# TTS libraries (LiveKit, RealtimeTTS, Pipecat) send the first SENTENCE alone, never a
# fragment, then group sentences so each segment is made before the previous one has played.
STREAM_FIRST_MIN_CHARS = 30
STREAM_MIN_SEGMENT_CHARS = 20
# The pace used to size each segment so it is made before the audio already made has played
# (each segment waits for the previous one: one engine at a time). Measured on the public
# gateway's Supertonic (2026-10-17): ~15 characters of speech per second, synthesis at 0.36x
# real time and ~0.5 s per call; rounded here to the cautious side.
STREAM_CHARS_PER_SECOND = 15.0
STREAM_SYNTH_RATIO = 0.4
STREAM_CALL_OVERHEAD_S = 0.6


def _cut_positions(s: str, start: int, stop: int, pattern: re.Pattern[str]) -> list[int]:
    return [m.end() for m in pattern.finditer(s, start, stop + 1) if m.end() <= stop]


def _choose_cut(s: str, start: int, *, min_len: int, budget: int, max_len: int, earliest: bool) -> int:
    """Where the segment starting at `start` ends, always at a sentence end when there is one
    within `max_len`: the first one at or after `min_len` (`earliest`: the first segment) or the
    last one within `budget` (later segments: whole sentences together); a sentence longer than
    the budget is still taken whole. Only a single sentence longer than `max_len` (the engine's
    cap) is cut, at a clause pause, else a space. A cut never leaves a tail shorter than
    `STREAM_MIN_SEGMENT_CHARS`."""
    n = len(s)
    if not earliest and n - start <= min(budget, max_len):
        return n
    stop = min(start + max_len, n)

    def ok(c: int) -> bool:
        if c - start < min_len:
            return False
        tail = len(s[c:].strip())
        return tail == 0 or tail >= STREAM_MIN_SEGMENT_CHARS

    ends = [c for c in _cut_positions(s, start, stop, _RE_SENTENCE_END) if ok(c)]
    if stop == n and n - start <= max_len:
        ends.append(n)
    if ends:
        if earliest:
            return ends[0]
        within = [c for c in ends if c - start <= budget]
        return within[-1] if within else ends[0]
    if earliest:
        # A short first sentence followed by one longer than the cap: the short one alone (a
        # whole sentence) rather than a first segment cut at a comma.
        shorter = [c for c in _cut_positions(s, start, stop, _RE_SENTENCE_END)
                   if len(s[c:].strip()) == 0 or len(s[c:].strip()) >= STREAM_MIN_SEGMENT_CHARS]
        if shorter:
            return shorter[0]
    cuts = [c for c in _cut_positions(s, start, stop, _RE_SOFT_END) if ok(c)]
    if cuts:
        return cuts[-1]
    spaces = [i for i in range(start + 1, stop + 1) if i < n and s[i] == " " and ok(i)]
    if spaces:
        return spaces[-1]
    return stop  # one indivisible run (a long word, continuous script): enforce the cap


def split_complete_text_for_streaming(
    text: str,
    *,
    max_chars: int = 240,
    first_min_chars: int = STREAM_FIRST_MIN_CHARS,
    chars_per_second: float = STREAM_CHARS_PER_SECOND,
    synth_ratio: float = STREAM_SYNTH_RATIO,
    call_overhead_s: float = STREAM_CALL_OVERHEAD_S,
) -> list[str]:
    """Split a complete text for streamed TTS in whole sentences: soon-ready first audio, no gaps.

    - The first segment is the first sentence; one shorter than `first_min_chars` ("Oui.")
      goes with the next one. It is never cut at a comma.
    - Each later segment groups whole sentences, as many as can be synthesised before the audio
      already made has played (at `chars_per_second`, synthesis at `synth_ratio` of real time
      plus `call_overhead_s` per segment); one longer sentence is still taken whole.
    - Only a sentence longer than `max_chars` (the engine's cap) is cut, at a clause pause.
    - No segment is shorter than `STREAM_MIN_SEGMENT_CHARS` unless the whole text is.
    """

    try:
        mc = int(max_chars)
    except Exception:
        mc = 240
    if mc <= 0:
        mc = 240
    first_min = max(1, min(int(first_min_chars), mc))

    s = _RE_WS.sub(" ", str(text or "")).strip()
    if not s:
        return []

    secs_per_char = 1.0 / max(1.0, float(chars_per_second))
    segments: list[str] = []
    start = 0
    budget = mc
    made_at = 0.0     # when the last segment is synthesised (s)
    played_at = 0.0   # when the audio made so far has played (s)
    while start < len(s):
        first = not segments
        cut = _choose_cut(
            s,
            start,
            min_len=first_min if first else STREAM_MIN_SEGMENT_CHARS,
            budget=budget,
            max_len=mc,
            earliest=first,
        )
        segment = s[start:cut].strip()
        if segment:
            segments.append(segment)
            audio_s = len(segment) * secs_per_char
            made_at += call_overhead_s + synth_ratio * audio_s
            played_at = max(played_at, made_at) + audio_s
            # The next segment must be made before `played_at`, starting now (`made_at`).
            budget_s = played_at - made_at - call_overhead_s
            fits = int(budget_s / max(1e-6, synth_ratio * secs_per_char))
            budget = max(2 * STREAM_MIN_SEGMENT_CHARS, min(mc, fits))
        start = cut
        while start < len(s) and s[start].isspace():
            start += 1
    return segments


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
