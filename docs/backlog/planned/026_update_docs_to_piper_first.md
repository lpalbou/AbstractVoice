## Task 026: Update docs that are out-of-sync with Piper-first architecture

**Date**: 2026-01-23  
**Status**: Planned  
**Priority**: P1  

---

## Main goals

- Remove/replace documentation that describes the old Coqui/VITS-based architecture and model selection.
- Ensure docs match the current **Piper-first** design and the public `VoiceManager` contract.

## Secondary goals

- Reduce contributor confusion and prevent implementation decisions based on obsolete guidance.

---

## Context / problem

Core docs currently include outdated content that no longer matches the codebase:

- `docs/architecture.md` describes Coqui `TTS("tts_models/...")` fallbacks and legacy behavior that is not present in the Piper-first implementation.
- `docs/development.md` contains extensive Coqui-specific guidance (sentence segmentation, model hierarchy, espeak-ng dependencies) that conflicts with current `docs/model-management.md` and current adapters.

This creates inconsistent onboarding and increases the risk of regressions by encouraging changes that reintroduce removed dependencies.

---

## Constraints

- Keep the docs concise and aligned with the stable public API (`docs/public_api.md`).
- Preserve useful historical notes in `docs/devnotes/` if needed, but keep “current architecture” docs accurate.

---

## Research, options, and references

- **Option A: Rewrite the docs to reflect Piper-first**
  - Replace Coqui sections with current adapters, caching, and “no surprise downloads” rules.
  - Trade-offs: some effort, but highest long-term clarity.
- **Option B: Move outdated sections into devnotes**
  - Keep them for archaeology, but remove them from “current” docs.
  - Trade-offs: less rewrite, but risks broken links and partial guidance.

References:
- `docs/model-management.md` (already Piper-first and accurate)
- `docs/public_api.md` (stable integrator contract)
- `docs/adr/` (design rationale and constraints)

---

## Decision

**Chosen approach**: Option A, with a small “legacy” note if necessary.

**Why**:
- Contributors should not have to infer which documentation is current.
- The project already maintains devnotes/reports for historical context.

---

## Dependencies

- **ADRs**:
  - `docs/adr/0001-local_assistant_out_of_box.md`
- **Backlog tasks**: none

---

## Implementation plan

- Update `docs/architecture.md`:
  - Replace Coqui model fallback sections with Piper adapter flow and audio playback architecture.
  - Ensure examples reference `VoiceManager(tts_engine="piper")`, `speak_to_bytes()`, and the current voice/language selection path.
- Update `docs/development.md`:
  - Remove Coqui-only sections; replace with Piper-specific operational guidance (voice downloads, caching, chunking strategy if applicable).
  - Link to `docs/model-management.md` for model/voice caching semantics.
- Run a quick consistency sweep:
  - `rg "TTS\\(\" docs/architecture.md docs/development.md` should return no “current” references.

---

## Success criteria

- `docs/architecture.md` and `docs/development.md` no longer instruct contributors to use Coqui APIs or legacy TTS models.
- Docs consistently describe Piper-first behavior and current configuration points.

---

## Test plan

- `pytest -q`
- Doc consistency checks:
  - `rg -n "TTS\\(\" docs/architecture.md docs/development.md || true`

