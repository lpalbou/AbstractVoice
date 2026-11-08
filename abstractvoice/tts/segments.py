"""Text segments of a streamed synthesis, located in the caller's text (ADR 0009).

A streamed synthesis is a sequence of independent text segments (the engine's own
generations). Segment-aware iterators yield, in delivery order::

    SegmentStart(0, ...), (audio, sr), (audio, sr), SegmentEnd(0), SegmentStart(1, ...), ...

Audio items are the plain ``(audio, sample_rate)`` tuples of the chunk iterators, so
``audio_only`` turns a segment-aware iterator back into a chunk iterator. A
segment's audio speaks exactly its text, whatever the engine sampled: a client that
lost the stream after ``SegmentEnd(k)`` can re-request the caller's text from
``text_end`` of segment k and hear no repeated and no missing word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from ..text_sanitize import sanitize_markdown_with_index

_RE_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class SegmentStart:
    """The audio that follows, up to the matching ``SegmentEnd``, speaks ``text``.

    ``text_start``/``text_end`` locate it in the caller's text (code-point offsets,
    ``[start, end)``); engines leave them None and the voice manager fills them in.
    """

    index: int
    text: str
    text_start: Optional[int] = None
    text_end: Optional[int] = None


@dataclass(frozen=True)
class SegmentEnd:
    """Every audio chunk of segment ``index`` has been yielded."""

    index: int


def audio_only(events: Iterable) -> Iterator:
    """The ``(audio, sample_rate)`` items of a segment-aware iterator (closing it when closed)."""
    try:
        for item in events:
            if isinstance(item, tuple):
                yield item
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()


def single_segment(text: str, chunks: Iterable) -> Iterator:
    """A chunk iterator of an engine that does not report segments: the whole text is segment 0."""
    try:
        yield SegmentStart(0, str(text))
        yield from chunks
        yield SegmentEnd(0)
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            close()


def _collapse_whitespace(text: str, index: list[int]) -> tuple[str, list[int]]:
    """``_RE_WS.sub(" ", text).strip()`` (what the segmenters do) with the index carried along."""
    out: list[str] = []
    kept: list[int] = []
    pos = 0
    for m in _RE_WS.finditer(text):
        out.append(text[pos:m.start()])
        kept.extend(index[pos:m.start()])
        out.append(" ")
        kept.append(index[m.start()])
        pos = m.end()
    out.append(text[pos:])
    kept.extend(index[pos:])
    joined = "".join(out)
    lead = len(joined) - len(joined.lstrip())
    end = len(joined.rstrip())
    return joined[lead:end], kept[lead:end]


class SegmentLocator:
    """Locates the segments of one synthesis, in order, in the caller's text.

    The text an engine speaks is the caller's text prepared for speech (``sanitize``:
    Markdown syntax and code removed, a few marks inserted) and whitespace collapsed;
    segments are consecutive slices of it. Every spoken character carries the index of
    the caller's character it stands for (never decreasing), so each segment maps back
    to an exact span of the caller's text. A span covers the segment's first to last spoken
    character plus the markers directly around them (``**`` before ``Word`` and after
    ``word.``), so the text after ``text_end`` starts where the next segment's speech
    starts.
    """

    def __init__(self, text: str, *, sanitize: bool) -> None:
        self._text = str(text or "")
        if sanitize:
            spoken, index = sanitize_markdown_with_index(self._text)
        else:
            spoken, index = self._text, list(range(len(self._text)))
        self._spoken, self._index = _collapse_whitespace(spoken, index)
        self._cursor = 0  # in the spoken text
        self._end = 0  # in the caller's text: where the previous segment's span ended

    def locate(self, segment_text: str) -> tuple[int, int]:
        """``(text_start, text_end)`` of the next segment; raises if it is not the next slice of the text."""
        needle = _RE_WS.sub(" ", str(segment_text or "")).strip()
        if not needle:
            # Nothing spoken: an empty span where the remaining text starts.
            here = self._index[self._cursor] if self._cursor < len(self._index) else len(self._text)
            return here, here
        pos = self._spoken.find(needle, self._cursor)
        if pos < 0:
            raise RuntimeError(
                f"Streamed speech segment {needle[:40]!r} is not the next part of the request text; "
                "segment offsets would be wrong."
            )
        last = pos + len(needle) - 1
        # Deleted markers right before the first / after the last spoken character ("**" around
        # "word.") belong to the segment: its span is a well-formed piece of the caller's text.
        start = self._index[pos]
        prev = max(self._index[pos - 1] + 1 if pos > 0 else 0, self._end)
        while start > prev and not self._text[start - 1].isspace():
            start -= 1
        end = self._index[last] + 1
        nxt = self._index[last + 1] if last + 1 < len(self._index) else len(self._text)
        while end < nxt and not self._text[end].isspace():
            end += 1
        self._cursor = last + 1
        self._end = end
        return start, end
