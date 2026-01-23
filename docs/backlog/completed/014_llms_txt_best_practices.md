## Task 014: Update `llms*.txt` (AI agent guidance best practices)

**Status**: Completed  
**Priority**: P1  

---

## Summary

Replaced the legacy, outdated `llms*.txt` content with **spec-aligned** manifests that match the current repo architecture and documentation strategy (single root `README.md`, small public API, internals under `abstractvoice/vm/`, ADRs + backlog workflow).

---

## Changes

- Updated `/llms.txt`:
  - H1 + one-line blockquote
  - short context paragraph (no extra headings)
  - `##` link groups + `## Optional`
  - points to the real “start here” docs and the key code entry points

- Updated `/llms-full.txt`:
  - same structure, but with a broader code map and doc map

- Fixed stale architecture doc references to prevent agents being misdirected:
  - `docs/architecture.md` now points to `abstractvoice/vm/` as the implementation location

---

## References

- llms.txt spec: `https://llmstxt.org/`

