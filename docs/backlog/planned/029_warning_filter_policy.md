## Task 029: Revisit global warning filters applied at import time

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Avoid changing global warning behavior for integrators just by importing `abstractvoice`.
- Keep the CLI output clean without muting warnings in downstream applications.

## Secondary goals

- Make warning suppression an explicit, documented choice (CLI flag or env var).

---

## Context / problem

`abstractvoice/__init__.py` currently applies global `warnings.filterwarnings(...)` rules at import time (e.g., suppressing PyTorch `torch.load` FutureWarnings and `pkg_resources` deprecation warnings).

This has two downsides:
- It affects the entire host process, not just AbstractVoice, which can surprise integrators.
- It may hide warnings that users actually want to see during development or security reviews.

---

## Constraints

- Keep end-user REPL/CLI experience low-noise by default.
- Avoid breaking imports for users who currently rely on “quiet by default” behavior.

---

## Research, options, and references

- **Option A: Keep global filters**
  - Lowest change, but continues to affect downstream apps unexpectedly.
- **Option B: Move filters behind explicit CLI/REPL switches**
  - Apply warning suppression only in entry points (`abstractvoice cli`, `python -m abstractvoice ...`).
  - References:
    - `https://docs.python.org/3/library/warnings.html`

---

## Decision

**Chosen approach**: Option B — do not set global filters in library import path; make it an explicit CLI/REPL behavior.

**Why**:
- Libraries should not modify global interpreter warning configuration unless explicitly requested.

---

## Dependencies

- **ADRs**: none
- **Backlog tasks**: none

---

## Implementation plan

- Remove or gate warning filters in `abstractvoice/__init__.py`.
- Add an explicit “quiet warnings” option in CLI entry points:
  - environment variable (e.g., `ABSTRACTVOICE_SUPPRESS_WARNINGS=1`) and/or CLI flag.
- Update docs to explain how to enable/disable warning suppression.

---

## Success criteria

- Importing `abstractvoice` does not modify warning filters by default.
- CLI still supports a low-noise mode without affecting integrators.

---

## Test plan

- `pytest -q`
- Manual:
  - In a Python REPL, import `abstractvoice` and verify warning filters are unchanged unless explicitly enabled.

