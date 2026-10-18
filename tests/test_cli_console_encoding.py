from __future__ import annotations

import importlib
import io
import subprocess
import sys

import pytest

from abstractvoice._stdio import safe_stdio


def test_safe_stdio_writes_a_replacement_on_a_cp1252_stream(monkeypatch) -> None:
    """Windows pipes and redirects use the ANSI code page (often cp1252), which has no emoji."""
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict"))
    safe_stdio()
    print("❌ Application error: → retry")
    sys.stdout.flush()
    assert raw.getvalue().decode("cp1252").strip() == "? Application error: ? retry"


@pytest.mark.parametrize(
    "module",
    ["abstractvoice.examples.voice_cli", "abstractvoice.examples.cli_repl", "abstractvoice.prefetch", "abstractvoice.__main__"],
)
def test_every_console_entry_point_makes_stdio_safe_first(module) -> None:
    import ast
    import inspect

    mod = importlib.import_module(module)
    body = ast.parse(inspect.getsource(mod.main).lstrip()).body[0].body
    calls = [n for n in ast.walk(ast.Module(body=body[:4], type_ignores=[])) if isinstance(n, ast.Call)]
    assert any(getattr(c.func, "id", "") == "safe_stdio" for c in calls), module


def test_an_error_with_an_emoji_does_not_crash_a_cp1252_pipe() -> None:
    env = {k: v for k, v in __import__("os").environ.items() if k not in {"PYTHONIOENCODING", "PYTHONUTF8", "OPENAI_API_KEY"}}
    env["PYTHONIOENCODING"] = "cp1252"
    out = subprocess.run(
        [sys.executable, "-m", "abstractvoice.examples.voice_cli", "tts", "--tts-engine", "openai", "--prompt", "x", "--output", "x.wav"],
        capture_output=True, timeout=120, env=env,
    )
    err = (out.stdout + out.stderr).decode("cp1252", "replace")
    assert "UnicodeEncodeError" not in err
    assert "Application error" in err
