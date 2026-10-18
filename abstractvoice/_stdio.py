"""Console output that never crashes on a character the terminal's code page lacks.

On Windows a redirected or piped stream (``> file``, ``| Select-String``, a CI log) uses the
ANSI code page, often cp1252, which has no emoji, arrows or box-drawing characters. The
command-line entry points call `safe_stdio()` first, so such a character is written as a
replacement character instead of ending the command with ``UnicodeEncodeError``. UTF-8
streams and interactive consoles are left as they are.
"""

from __future__ import annotations

import sys


def safe_stdio() -> None:
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = str(getattr(stream, "encoding", "") or "").lower().replace("_", "-")
        if encoding in {"utf-8", "utf8", "utf-8-sig"}:
            continue
        try:
            reconfigure(errors="replace")
        except Exception:  # a stream that cannot be reconfigured keeps its behavior
            pass
