## Task 041: Resolve Python 3.10 all-apple NumPy conflict

**Date**: 2026-05-19
**Status**: Proposed
**Priority**: P1

---

## Main Goals

- Restore a clean dependency resolution story for `abstractvoice[all-apple]` on Python 3.10, or explicitly retire Python 3.10 support for that heavy local profile.
- Keep Python 3.11 and 3.12 `all-apple` installs working.

## Secondary Goals

- Make the failure mode visible in dependency checks and install docs.
- Avoid changing higher-level packages to guess around this resolver conflict.

---

## Context / Problem Statement

`abstractflow[apple]` release testing surfaced an upstream dependency conflict in the full Apple stack on Python 3.10:

- `abstractgateway[apple]` depends on `abstractvoice[all-apple]`.
- `abstractvoice[all-apple]` currently includes `f5-tts>=1.1.0` for Python `>=3.10`.
- Published `f5-tts` versions constrain NumPy to `<=1.26.4` on Python `<3.11`.
- The current AbstractCore all-Apple stack requires NumPy `>=2.1.0`.

That makes the combined local Apple profile unsatisfiable on Python 3.10 while Python 3.11 and 3.12 resolve and test successfully.

---

## Constraints

- Do not weaken the default local Apple install for supported Python versions.
- Do not work around this in Flow, Gateway, or Runtime dependency metadata beyond limiting their release matrix to resolvable Python versions.
- Keep voice engine extras explicit and predictable.

---

## Research & References

- Local resolver check:
  - `uv pip compile pyproject.toml --extra apple --python-version 3.10`
  - Result: `abstractgateway[apple]==0.2.14` cannot be used because `abstractcore[all-apple]>=2.13.15` requires NumPy `>=2.1.0`, while `abstractvoice[all-apple]>=0.10.3` pulls `f5-tts>=1.1.0`, whose Python `<3.11` metadata requires NumPy `<=1.26.4`.
- Python 3.11 and Python 3.12 Flow release jobs resolved the same high-level Apple profile and passed tests.

---

## Decision

Proposed decision: treat Python 3.10 `all-apple` as blocked until AbstractVoice chooses a clean package-level fix.

Candidate fixes:

- Gate the `f5-tts` optional dependency to Python `>=3.11` and document cloning/S2S limitations on Python 3.10.
- Split F5-TTS into a narrower optional extra so `all-apple` can stay installable where the upstream NumPy constraint is incompatible.
- Retire Python 3.10 support for `abstractvoice[all-apple]` and make that explicit in packaging/docs.

---

## Dependencies

- AbstractVoice packaging extras.
- AbstractCore all-Apple NumPy policy.
- Upstream `f5-tts` NumPy metadata for Python `<3.11`.

---

## Implementation Plan

- Reproduce the conflict in AbstractVoice with a resolver-only test or documented check.
- Choose one package-level policy for Python 3.10.
- Update `pyproject.toml` extras and install docs accordingly.
- Add dependency-check messaging so users get an actionable explanation.

---

## Success Criteria

- `abstractvoice[all-apple]` either resolves on Python 3.10 again or clearly rejects that profile with documented Python-version support.
- Python 3.11 and 3.12 `all-apple` installs continue to resolve.
- Flow/Gateway/Runtime do not need dependency workarounds for this specific conflict.

---

## Test Plan

- Resolver checks:
  - `uv pip compile pyproject.toml --extra all-apple --python-version 3.10`
  - `uv pip compile pyproject.toml --extra all-apple --python-version 3.11`
  - `uv pip compile pyproject.toml --extra all-apple --python-version 3.12`
- Existing AbstractVoice test suite.
