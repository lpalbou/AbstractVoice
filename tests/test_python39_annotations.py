"""Python 3.9 evaluates annotations at definition time: no `X | Y` there.

`requires-python = ">=3.9"`, but CI's 3.9 job has no torch, so a vendored model
module that uses `str | None` in a signature (evaluated at import on 3.9 without
`from __future__ import annotations`) is never imported there, and the
TypeError only shows up on a 3.9 machine with torch installed. This check reads
every package module's syntax tree instead, so it runs on any Python, torch or not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import abstractvoice

PACKAGE = Path(abstractvoice.__file__).resolve().parent


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _annotations(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _pep604_unions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if _has_future_annotations(tree):
        return []
    return [
        f"{path.relative_to(PACKAGE.parent)}:{annotation.lineno}"
        for annotation in _annotations(tree)
        if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(annotation))
    ]


def test_no_runtime_evaluated_pep604_annotations():
    offenders = [site for path in sorted(PACKAGE.rglob("*.py")) for site in _pep604_unions(path)]
    assert offenders == [], "use Optional/Union or `from __future__ import annotations`: " + ", ".join(offenders)
