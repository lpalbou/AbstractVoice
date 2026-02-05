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
    if not text:
        return ""

    out = str(text)
    out = _MD_HEADER_RE.sub(r"\1", out)
    out = _MD_BOLD_RE.sub(r"\1", out)
    out = _MD_ITALIC_RE.sub(r"\1", out)
    return out
