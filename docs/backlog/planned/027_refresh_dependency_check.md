## Task 027: Refresh dependency checker to match current engines and constraints

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Make `abstractvoice check-deps` reflect the **current** dependency model (Piper + faster-whisper + optional cloning/AEC/web).
- Remove stale or misleading checks (e.g., old PyTorch compatibility ranges that no longer match the project).

## Secondary goals

- Add actionable diagnostics for common runtime failures (missing audio devices, missing optional extras).

---

## Context / problem

`abstractvoice/dependency_check.py` hard-codes dependency assumptions that appear out-of-sync with current packaging and docs:

- PyTorch compatibility ranges are pinned to old constraints and may not reflect the actual needs of current optional features (e.g., cloning).
- Optional dependency list includes items that are not clearly part of the current supported stack.
- The checker doesn’t explicitly surface “no default audio device” conditions that produce common `sounddevice.PortAudioError` failures.

This reduces trust in the checker and makes troubleshooting harder.

---

## Constraints

- Keep the checker lightweight and offline-capable (no network calls).
- Avoid introducing new heavy dependencies.

---

## Research, options, and references

- **Option A: Keep hard-coded version ranges**
  - Easy, but tends to drift and become misleading as engines change.
- **Option B: Derive checks from installed metadata**
  - Use `importlib.metadata` to report installed versions; focus on presence + basic sanity checks rather than fragile upper bounds.
  - References:
    - `https://docs.python.org/3/library/importlib.metadata.html`

---

## Decision

**Chosen approach**: Option B — prefer metadata-driven reporting + targeted sanity checks.

**Why**:
- Reduces churn and false positives.
- Keeps the output useful across different optional-feature installations.

---

## Dependencies

- **ADRs**: none
- **Backlog tasks**: none

---

## Implementation plan

- Update `DependencyChecker`:
  - Use `importlib.metadata.version()` for installed versions (fall back to module `__version__` if needed).
  - Replace/trim hard-coded PyTorch ranges; instead report detected torch/torchaudio versions if installed and warn only for known-bad combinations.
  - Align optional groups with `pyproject.toml` extras (`voice`, `cloning`, `aec`, `web`).
- Add audio-device diagnostics:
  - If `sounddevice` is installed, attempt `query_devices()` and report default input/output device availability with clear remediation hints.
- Add a small unit test for the checker output structure (no device access required; use monkeypatch).

---

## Success criteria

- `python -m abstractvoice check-deps` produces accurate, actionable output for:
  - base install
  - `abstractvoice[cloning]`
  - missing audio devices (clear message; no crash)

---

## Test plan

- `pytest -q`
- Manual smoke:
  - `python -m abstractvoice check-deps`

