## Task 007: Repo Hygiene (remove committed virtualenv + reduce noise)

**Status**: Planned  
**Priority**: P1  
**Goal**: Keep the repository clean for contributors and CI.

---

## Problem

The repo currently contains a committed virtual environment directory (`.vtest/`),
which creates large diffs, platform-specific noise, and makes it hard to review real changes.

---

## Plan

- ✅ Ensure `.vtest/` is ignored in `.gitignore` (already done)
- ✅ Document the recommended local dev setup in `docs/getting_started.md`
- ⏳ Remove obsolete backup file(s) from the repo (e.g. `abstractvoice/model_manager.py.backup`)
- ⏳ Remove tracked `.vtest/` files from git going forward (`git rm --cached -r .vtest/`)
  - This is destructive to repo history/state; do it as an explicit change with review.

