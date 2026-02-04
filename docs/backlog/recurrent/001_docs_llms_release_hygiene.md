## Recurrent 001: Docs + llms + release hygiene

**Date**: 2026-01-23  
**Status**: Recurrent  
**Applies**: After each development cycle / before marking tasks completed  

---

## Purpose

Keep the repository coherent across code, docs, and release metadata, so third-party integrators and contributors always see a consistent, minimal, accurate story.

---

## Checklist (run before completing a task)

- **Docs consistency**
  - Ensure `README.md` remains the single entry point and links remain accurate.
  - Ensure any changed behavior is reflected in `docs/api.md` and relevant ADRs.
  - Scan for newly outdated docs and update them (keep the docs set minimal).

- **Agent manifests (`llms*.txt`)**
  - If docs layout, entry points, or architecture moved, update `llms.txt` and `llms-full.txt`.
  - Keep `llms.txt` short; push deeper links to `llms-full.txt`.

- **Acknowledgments & licensing**
  - If dependencies changed, update `ACKNOWLEDGMENTS.md`.
  - Verify new deps are permissive (MIT/Apache/BSD). If not, stop and create a backlog item documenting alternatives.

- **Duplication & cleanup**
  - Look for duplicated code/docs and create a cleanup backlog item if needed.
  - Tag unreferenced files for deletion/refactor (prefer fewer files, but no monoliths; avoid >1000-line files).

- **Versioning & changelog**
  - After completing one or more backlog items, bump semantic version and add a `CHANGELOG.md` entry.
