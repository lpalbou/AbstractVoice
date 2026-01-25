## Task 028: Add request timeouts and cancellation safety in CLI REPL

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P2  

---

## Main goals

- Prevent the CLI REPL from hanging indefinitely on network requests to the chat API.
- Provide clear, actionable error messages when the API is unreachable or slow.

## Secondary goals

- Make timeouts configurable (CLI flags and/or env vars).

---

## Context / problem

`abstractvoice/examples/cli_repl.py` calls `requests.post(self.api_url, json=payload)` without a timeout. If the endpoint is down or slow, the REPL can block indefinitely, which is a poor UX and makes automation harder.

---

## Constraints

- Keep dependencies minimal (avoid adding a new HTTP client stack unless necessary).
- Preserve current default behavior for healthy local endpoints (e.g., Ollama).

---

## Research, options, and references

- **Option A: Add `timeout=` to `requests.post`**
  - Simple and effective; covers connect + read timeouts when provided as a tuple.
  - References:
    - `https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts`
- **Option B: Switch to an async client (e.g., httpx)**
  - More flexible cancellation, but adds dependencies and complexity.

---

## Decision

**Chosen approach**: Option A — use `requests` timeouts and keep implementation minimal.

**Why**:
- Fixes the core issue with minimal surface-area change.

---

## Dependencies

- **ADRs**: none
- **Backlog tasks**: none

---

## Implementation plan

- Add configurable timeout settings:
  - CLI flags (e.g., `--connect-timeout`, `--read-timeout`) and/or env vars.
- Update the REPL request call to:
  - pass a timeout tuple `(connect_timeout, read_timeout)`
  - handle `requests.exceptions.Timeout` separately with clear messaging
- Add a small unit test (monkeypatch `requests.post`) verifying that a timeout value is passed.

---

## Success criteria

- REPL never blocks indefinitely due to an API request.
- Timeout behavior is configurable and documented in the REPL help output.

---

## Test plan

- `pytest -q`
- Manual:
  - Run `abstractvoice cli` with an invalid `--api-url` and confirm it errors quickly.

