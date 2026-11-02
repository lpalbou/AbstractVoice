"""Lightweight text sanitizers for better speech UX.

Design goals:
- No heavy dependencies (regex only).
- Deterministic and conservative (only strip the most common Markdown syntax).
"""

from __future__ import annotations

import re


_MD_HEADER_RE = re.compile(r"(?m)^[ \t]*#{1,5}(?!#)[ \t]*(\S.*)$")
_MD_BOLD_RE = re.compile(r"\*\*([^*\n]+?)\*\*")
# Avoid matching "**bold**" as italic. The bold pass runs first, but keep this safe.
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def sanitize_markdown_for_speech(text: str) -> str:
    """Remove common Markdown syntax that sounds bad in TTS.

    Intentionally minimal:
    - Strip ATX headers at the start of a line: "#", "##", ... "#####"
    - Strip emphasis markers: "**bold**" and "*italic*"
    """
    return sanitize_markdown_with_index(text)[0]


def sanitize_markdown_with_index(text: str) -> tuple[str, list[int]]:
    """``sanitize_markdown_for_speech`` plus, for each kept character, its index in ``text``.

    Every pass only deletes characters (each match is replaced by its first group), so the
    result is a subsequence of ``text`` and the index list maps it back exactly.
    """
    out = str(text or "")
    index = list(range(len(out)))
    for pattern in (_MD_HEADER_RE, _MD_BOLD_RE, _MD_ITALIC_RE):
        kept: list[int] = []
        last = 0
        for m in pattern.finditer(out):
            kept.extend(range(last, m.start()))
            kept.extend(range(m.start(1), m.end(1)))
            last = m.end()
        kept.extend(range(last, len(out)))
        out = "".join(out[i] for i in kept)
        index = [index[i] for i in kept]
    return out, index
