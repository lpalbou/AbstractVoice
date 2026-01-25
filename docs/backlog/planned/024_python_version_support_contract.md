## Task 024: Fix Python version support contract

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P0  

---

## Main goals

- Ensure `abstractvoice` installs and imports on the **advertised** Python versions.
- Make the supported Python range consistent across packaging metadata, docs, and CI.

## Secondary goals

- Reduce avoidable support burden from mismatched version metadata.

---

## Context / problem

`pyproject.toml` currently declares `requires-python = ">=3.8"` and includes classifiers for Python 3.8/3.9, but the codebase uses newer typing syntax that breaks on older interpreters:

- **PEP 585 built-in generics** like `list[str]` are used in runtime-evaluated annotations (e.g. `abstractvoice/adapters/base.py`), which requires **Python 3.9+** unless annotations are postponed.
- **PEP 604 union types** like `str | None` are used in modules without `from __future__ import annotations` (e.g. `abstractvoice/examples/cli_repl.py`, `abstractvoice/recognition.py`), which requires **Python 3.10+**.

Impact:
- Users on Python 3.8/3.9 may be able to install the package but hit import-time failures.
- Downstream tooling (pip, Poetry, CI matrices) cannot rely on metadata.

---

## Constraints

- Keep the integrator-facing `VoiceManager` contract stable.
- Prefer the **smallest** change that makes metadata and runtime behavior consistent.
- Keep optional features (cloning, AEC, web) opt-in.

---

## Research, options, and references

- **Option A: Maintain Python >=3.8**
  - Replace newer type-hint syntax with `typing.Optional` / `typing.List` / `typing.Union` throughout, and/or ensure postponed evaluation everywhere.
  - Trade-offs: more code churn; more ongoing constraints on language features.
  - References:
    - `https://peps.python.org/pep-0585/` (built-in generics; 3.9+)
    - `https://peps.python.org/pep-0604/` (union operators; 3.10+)
    - `https://peps.python.org/pep-0563/` (postponed evaluation of annotations)

- **Option B: Raise minimum Python to 3.10**
  - Update `requires-python` and classifiers to match current code and dependencies.
  - Trade-offs: drops older runtimes, but matches modern syntax and reduces maintenance.
  - References:
    - `https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#requires-python`
    - `https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#python-requires`

---

## Decision

**Chosen approach**: Option B — require **Python >=3.10**.

**Why**:
- The repository already uses 3.10-era typing syntax; raising the floor is the least risky, lowest-churn fix.
- Python 3.8 is EOL and many modern ML/audio dependencies increasingly assume newer runtimes.

---

## Dependencies

- **ADRs**: none
- **Backlog tasks**: none

---

## Implementation plan

- Update `pyproject.toml`:
  - Set `requires-python = ">=3.10"`.
  - Remove 3.8/3.9 classifiers; keep 3.10–3.12.
- Update `README.md` and `docs/installation.md` to state supported Python versions.
- Add/adjust CI (or documented local checks) to run `pytest -q` on 3.10/3.11/3.12.
- Optional: add a small import-time guard that provides a clear error message if imported on Python <3.10.

---

## Success criteria

- Installing on Python <3.10 is blocked by packaging metadata.
- Importing `abstractvoice` succeeds on Python 3.10+.
- Docs and metadata agree on the supported Python range.

---

## Test plan

- `pytest -q`
- Packaging sanity:
  - `python -m build`
  - `python -m pip install dist/*.whl`

