"""Markdown to speech text, with every spoken character mapped back to the caller's text.

A reply is read aloud from its Markdown source. A listener should hear what the rendered
reply says, not its syntax: an engine fed raw Markdown pronounces markers ("hash",
"backtick") and, on code, loses its place in the text. Seen live on 2026-11-08: a Qwen3-TTS
clone reading a fenced Python call (``create_llm("endpoint:...", model="qwen/qwen3.8-27b")``)
slurred into one repeated syllable, missed its end of speech and filled the rest of its
segment with silence up to the runaway cap (2 min 10 s), before the next heading played.

The rules follow CommonMark/GFM syntax only; they never guess what plain prose means:

- A fenced code block (``` or ~~~, at any indent, closed by the same fence or by the end
  of the text) is not read, nor is an indented one (4 spaces, not inside a list item).
- Inline code is read as its literal text without the backticks; nothing inside it is
  rewritten. A stray backtick is dropped, and so are the unpaired markers a text resumed in the
  middle of an inline element starts with (``talk*,``, ``here._``, ``~~``, ``](url)``, the
  edge pipes of a table row), so it reads like the rest of the whole.
- A text that starts inside a line (a leading space or tab: a resumed text) has no list
  marker on that line; an ordered marker other than ``1.`` does not interrupt a paragraph
  (CommonMark).
- Headings (ATX ``#``..``######`` and setext), list items, table rows and paragraphs are
  blocks. A block followed by another one ends with a full stop when its text has no closing
  punctuation, so a heading is its own sentence and blocks never run together (the last
  block is left as it is: one line of plain prose is spoken exactly as written).
- Bullet (``-``, ``*``, ``+``) and blockquote (``>``) markers, task boxes (``[ ]``,
  ``[x]``), thematic breaks and table delimiter rows are dropped. An ordered marker ``3.`` or
  ``3)`` is read as ``3:`` (replies refer to "option 2"; a full stop there would end a
  sentence inside the item). Table cells are read in order, separated by commas.
- Emphasis (``**x**``, ``__x__``, ``*x*``, ``_x_``) and strikethrough (``~~x~~``) keep their
  text; a link ``[text](url)`` and an image ``![alt](url)`` are read as their text; an
  autolink ``<url>`` as the url; a backslash escape as the character.
- An arrow (``→``, ``⇒``, ...; ``->``/``=>`` between spaces) is read as a pause (a comma),
  which is right in every language.

``sanitize_markdown_with_index`` also returns, for each output character, the index in the
caller's text it stands for: kept characters their own, a replacement the character it
replaces, inserted punctuation the character it follows. The indices never decrease, so
``tts.segments.SegmentLocator`` maps every spoken segment back to an exact span of the
caller's text (ADR 0009, resume after a break).
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------- block syntax

_FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$")
_ATX = re.compile(r"[ \t]*(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_SETEXT = re.compile(r"^[ \t]*(?:=+|-{2,})[ \t]*$")
_THEMATIC = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_QUOTE = re.compile(r"[ \t]*>[ \t]?")
_LIST = re.compile(r"[ \t]*(?:([-*+])|(\d{1,9})([.)]))(?:([ \t]+)(?:\[[ xX]\][ \t]+)?|$)")
# ``_QUOTE``, ``_ATX`` and ``_LIST`` are matched at a position (after quote markers): no ``^``, which
# would only match at the start of the line.
_TABLE_DELIM_CHARS = frozenset("|:- \t")

# ---------------------------------------------------------------------------- inline syntax

# A code span: a backtick run, then the shortest content closed by a run of the SAME length.
_CODE_SPAN = re.compile(r"(?<![`\\])(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.DOTALL)
_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")
_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)")
_LINK = re.compile(r"\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_REF_LINK = re.compile(r"\[([^\]\n]+)\]\[[^\]\n]*\]")
_AUTOLINK = re.compile(r"<((?:https?|ftp|mailto):[^\s<>]+)>")
_BOLD_STAR = re.compile(r"\*\*(?=\S)([^\n]+?)(?<=\S)\*\*")
_BOLD_UNDER = re.compile(r"(?<![\w_])__(?=\S)([^\n]+?)(?<=\S)__(?![\w_])")
_STRIKE = re.compile(r"~~(?=\S)([^\n]+?)(?<=\S)~~")
_ITALIC_STAR = re.compile(r"(?<![*\w])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![*\w])")
_ITALIC_UNDER = re.compile(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])")
_STRAY_BACKTICKS = re.compile(r"`+")
# An unpaired emphasis star (its pair is in another block, or before a resume point: "talk*,"): a star
# run touching a word on one side only, or a lone "**"/"__". "2 * 3" and "a*b" are left alone.
_STRAY_STARS = re.compile(
    r"(?<=[^\s*])\*+(?=[\s.,;:!?)\]}\"'»”’]|$)|(?:^|(?<=[\s(\[{\"'«“‘]))\*+(?=[^\s*])|(?<!\S)(?:\*\*+|__+)(?!\S)"
)
# The same for the other inline markers whose pair or head lies before a resume point: an underscore
# touching a word on one side only ("here._ After", never "snake_case"), "~~", and a link's "](url)".
_STRAY_UNDERSCORES = re.compile(r"(?<=[^\s_])_+(?=[\s.,;:!?)\]}\"'»”’]|$)|(?:^|(?<=[\s(\[{\"'«“‘]))_+(?=[^\s_])")
_STRAY_TILDES = re.compile(r"~~")
_STRAY_LINK_TAIL = re.compile(r"\]\((?:[^()\s]|\([^()\s]*\))*\)")
_ARROW = re.compile(
    r"[ \t]*(?:[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f\u2794\u2798-\u27af\u2b05-\u2b07\u2b60-\u2b6f]"
    r"|(?<!\S)(?:-->|==>|->|=>|<-)(?!\S))[ \t]*"
)

# A block whose text ends with one of these (possibly inside closing quotes/brackets) needs no full stop.
_CLOSING_PUNCTUATION = frozenset(".!?…:;,。！？；：，")
_CLOSERS = frozenset(")]}\"'»”’")

Mapped = Tuple[str, List[int]]
# A rendered match: ("keep", start, end) keeps text[start:end]; ("put", literal, at) inserts
# ``literal`` standing for text[at].
Part = tuple


def sanitize_markdown_for_speech(text: str) -> str:
    """The words a listener should hear of a Markdown ``text`` (module docstring: the rules)."""
    return sanitize_markdown_with_index(text)[0]


def sanitize_markdown_with_index(text: str) -> Mapped:
    """``sanitize_markdown_for_speech`` plus, for each output character, its index in ``text``.

    The index list has the output's length and never decreases (module docstring).
    """
    source = str(text or "")
    blocks = [b for b in (_trimmed(*_inline(t, i)) for t, i in _blocks(source)) if b[0]]
    out: List[str] = []
    index: List[int] = []
    for k, (spoken, spoken_index) in enumerate(blocks):
        if k + 1 < len(blocks):
            spoken, spoken_index = _end_block(spoken, spoken_index)
        if out:
            out.append("\n")
            index.append(index[-1])
        out.append(spoken)
        index.extend(spoken_index)
    return "".join(out), index


# ------------------------------------------------------------------------------- blocks


def _lines(source: str) -> Iterable[Tuple[int, str]]:
    """``(offset, line)`` for each line, without its line break."""
    offset = 0
    for raw in source.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        yield offset, line
        offset += len(raw)


def _is_table_delimiter(line: str) -> bool:
    return "|" in line and "-" in line and set(line) <= _TABLE_DELIM_CHARS


def _table_rows(lines: List[Tuple[int, str]]) -> set:
    """Line numbers of GFM table rows: a header line with a pipe, its delimiter row, the rows after."""
    rows: set = set()
    i = 0
    while i + 1 < len(lines):
        header, delim = lines[i][1], lines[i + 1][1]
        if "|" in header and not _is_table_delimiter(header) and _is_table_delimiter(delim):
            rows.update((i, i + 1))
            j = i + 2
            while j < len(lines) and "|" in lines[j][1] and lines[j][1].strip():
                rows.add(j)
                j += 1
            i = j
        else:
            i += 1
    return rows


def _blocks(source: str) -> List[Mapped]:
    """The text of each block, in order, with the source index of every character."""
    lines = list(_lines(source))
    tables = _table_rows(lines)
    blocks: List[Mapped] = []
    current: Optional[Tuple[List[str], List[int]]] = None  # the paragraph or list item being built
    kind = ""  # what ``current`` is ("paragraph" | "item"), or what the last block was once closed
    fence: Optional[Tuple[str, int]] = None  # (fence character, length) while inside a code block

    def close() -> None:
        nonlocal current
        if current is not None and current[0]:
            blocks.append(("".join(current[0]), current[1]))
        current = None

    def span(offset: int, start: int, end: int) -> Mapped:
        return source[offset + start:offset + end], list(range(offset + start, offset + end))

    for number, (offset, line) in enumerate(lines):
        if fence is not None:
            stripped = line.strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= fence[1]:
                fence = None
            continue
        opened = _FENCE_OPEN.match(line)
        if opened and not (opened.group(1)[0] == "`" and "`" in opened.group(2)):
            close()
            fence = (opened.group(1)[0], len(opened.group(1)))
            continue
        if not line.strip():
            close()
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if current is None and kind != "item" and (indent >= 4 or line[:1] == "\t"):
            continue  # an indented code block (it cannot interrupt a paragraph, nor follow a list item)
        stripped = line.strip()
        if number in tables or (stripped[:1] == "|" and stripped.count("|") >= 2):
            # A table row; one that starts with a pipe is a row even when its header is not in this
            # text (a text resumed inside a table).
            close()
            kind = "row"
            if not _is_table_delimiter(line):
                blocks.append(_table_row(source, offset, line))
            continue
        if _SETEXT.match(line) and current is not None:
            close()  # the paragraph above is a heading: its own block
            continue
        if _THEMATIC.match(line):
            close()
            kind = "rule"
            continue
        start = 0
        while True:  # blockquote markers, possibly nested
            quoted = _QUOTE.match(line, start)
            if not quoted:
                break
            start = quoted.end()
        if start and not line[start:].strip():
            close()  # an empty quote line separates paragraphs
            continue
        heading = _ATX.match(line, start)
        if heading:
            close()
            kind = "heading"
            if heading.group(2):
                blocks.append(span(offset, heading.start(2), heading.end(2)))
            continue
        item = _LIST.match(line, start)
        if item and number == 0 and indent and not start:
            item = None  # the text starts inside a line (a resumed text): "2. " there is prose, as it was
        if item and item.group(2) and item.group(2) != "1" and current is not None and kind == "paragraph":
            item = None  # CommonMark: only "1." may interrupt a paragraph ("The year was\n2026. Done")
        if item:
            close()
            kind = "item"
            chars: List[str] = []
            idx: List[int] = []
            if item.group(2):  # ordered: the number is read, its delimiter as a colon
                chars.append(item.group(2))
                idx.extend(range(offset + item.start(2), offset + item.end(2)))
                chars.append(":")
                idx.append(offset + item.start(3))
                if item.group(4):
                    chars.append(" ")
                    idx.append(offset + item.start(4))
            rest, rest_idx = span(offset, item.end(), len(line))
            chars.append(rest)
            idx.extend(rest_idx)
            current = (chars, idx)
            continue
        end = len(line)
        if line.rstrip().endswith("|") and not line.rstrip().endswith("\\|"):
            end = len(line.rstrip()) - 1  # the edge pipe of a table row whose start is before this text
        if line[start:end].lstrip(" \t")[:1] == "|":
            start += len(line[start:end]) - len(line[start:end].lstrip(" \t")) + 1
        text, idx = span(offset, start, end)
        if not text.strip():
            continue
        if current is None:
            current = ([text], idx)
            # The rest of a line (a resumed text) may be the rest of a list item: items may follow it.
            kind = "item" if number == 0 and indent else "paragraph"
        else:  # a continuation line of the same paragraph or list item
            current[0].append("\n" + text)
            current[1].append(offset - 1)
            current[1].extend(idx)
    close()
    return blocks


def _table_row(source: str, offset: int, line: str) -> Mapped:
    """A table row's cells, in order, separated by commas (an escaped pipe stays in its cell)."""
    pipes = [i for i, ch in enumerate(line) if ch == "|" and (i == 0 or line[i - 1] != "\\")]
    bounds = [-1] + pipes + [len(line)]
    chars: List[str] = []
    idx: List[int] = []
    for left, right in zip(bounds, bounds[1:]):
        cell = line[left + 1:right]
        lead = len(cell) - len(cell.lstrip())
        cell = cell.strip()
        if not cell:
            continue
        if chars:
            chars.append(", ")
            idx.extend([offset + left, offset + left])
        start = offset + left + 1 + lead
        chars.append(cell)
        idx.extend(range(start, start + len(cell)))
    return "".join(chars), idx


# ------------------------------------------------------------------------------- inline


def _substitute(text: str, index: List[int], pattern: re.Pattern, render: Callable[[re.Match], Iterable[Part]]) -> Mapped:
    """Replace each match of ``pattern`` by ``render(match)``, carrying the index along."""
    chars: List[str] = []
    idx: List[int] = []
    last = 0
    for m in pattern.finditer(text):
        chars.append(text[last:m.start()])
        idx.extend(index[last:m.start()])
        for part in render(m):
            if part[0] == "keep":
                chars.append(text[part[1]:part[2]])
                idx.extend(index[part[1]:part[2]])
            else:
                chars.append(part[1])
                idx.extend([index[part[2]]] * len(part[1]))
        last = m.end()
    chars.append(text[last:])
    idx.extend(index[last:])
    return "".join(chars), idx


def _group(n: int) -> Callable[[re.Match], Iterable[Part]]:
    return lambda m: [("keep", m.start(n), m.end(n))]


def _arrow(m: re.Match) -> Iterable[Part]:
    at = m.start() + (len(m.group(0)) - len(m.group(0).lstrip(" \t")))
    return [("put", ", ", at)]


_PROSE_PASSES = (
    (_ESCAPE, _group(1)),
    (_IMAGE, _group(1)),
    (_LINK, _group(1)),
    (_REF_LINK, _group(1)),
    (_AUTOLINK, _group(1)),
    (_BOLD_STAR, _group(1)),
    (_BOLD_UNDER, _group(1)),
    (_STRIKE, _group(1)),
    (_ITALIC_STAR, _group(1)),
    (_ITALIC_UNDER, _group(1)),
    (_STRAY_BACKTICKS, lambda m: []),
    (_STRAY_STARS, lambda m: []),
    (_STRAY_UNDERSCORES, lambda m: []),
    (_STRAY_TILDES, lambda m: []),
    (_STRAY_LINK_TAIL, lambda m: []),
    (_ARROW, _arrow),
)


def _prose(text: str, index: List[int]) -> Mapped:
    for pattern, render in _PROSE_PASSES:
        text, index = _substitute(text, index, pattern, render)
    return text, index


def _inline(text: str, index: List[int]) -> Mapped:
    """Inline Markdown of one block: code spans are read literally, everything else as prose.

    Each code span's content is masked (one placeholder per character) while the prose
    passes run, so emphasis around it (``**the `x` flag**``) still pairs up and nothing
    inside it is rewritten; the placeholders are then restored from their source indices.
    """
    code: dict = {}  # source index -> character, for every character inside a code span

    def mask(m: re.Match) -> Iterable[Part]:
        start, end = m.start(2), m.end(2)
        content = text[start:end]
        if len(content) > 2 and content[0] == " " and content[-1] == " " and content.strip():
            start, end = start + 1, end - 1
        for k in range(start, end):
            code[index[k]] = text[k]
        return [("put", _CODE_MASK, k) for k in range(start, end)]

    masked, masked_index = _prose(*_substitute(text, index, _CODE_SPAN, mask))
    restored = [code.get(i, ch) if ch == _CODE_MASK else ch for ch, i in zip(masked, masked_index)]
    return "".join(restored), masked_index


# Stands for one character of a code span while the prose passes run (private use: never Markdown).
_CODE_MASK = "\ue000"


def _trimmed(text: str, index: List[int]) -> Mapped:
    lead = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return text[lead:end], index[lead:end]


def _end_block(text: str, index: List[int]) -> Mapped:
    """A (trimmed, non-empty) block closed by a full stop when it has no closing punctuation."""
    k = len(text) - 1
    while k > 0 and text[k] in _CLOSERS:
        k -= 1
    if text[k] in _CLOSING_PUNCTUATION:
        return text, index
    return text + ".", index + [index[-1]]
