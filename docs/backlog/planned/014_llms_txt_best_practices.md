## Task 014: Update `llms*.txt` (AI agent guidance best practices)

**Status**: Planned  
**Priority**: P1  

---

## Goal

Keep `llms.txt` and `llms-full.txt` as **concise, actionable** manifests for AI agents working on this repo, aligned with current best practices and the `/llms.txt` spec.

---

## Why

- Agents need a small, authoritative entry point (what matters, where to look, what to avoid).
- Our docs structure changed (`abstractvoice/vm/`, ADRs, backlog workflow, etc.).
- We want consistent, low-noise guidance for both humans and agents.

---

## Work items

1. **Align with the llms.txt spec**
   - H1 title, one-line blockquote summary, short context paragraph
   - Sectioned links under `##` headings
   - `## Optional` section for deep/rare content

2. **Repo-specific agent guidance**
   - “Public API first” (`docs/public_api.md`)
   - Default engines (Piper + faster-whisper)
   - How to run tests (`docs/getting_started.md`)
   - Where to put new docs (minimal, no `docs/README.md`)
   - Backlog conventions (`docs/backlog/`)

3. **Validation**
   - Ensure links are correct and current
   - Keep it short (agents should not need to scroll)

---

## References

- `https://llmstxt.org/`

