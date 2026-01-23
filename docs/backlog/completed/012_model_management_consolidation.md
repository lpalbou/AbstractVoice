## Task 012: Model management consolidation (Piper vs legacy Coqui)

**Status**: Completed  
**Priority**: P1  

---

## Summary

Made “model management” **engine-scoped** and non-confusing: Piper voice management stays inside the Piper adapter, while legacy Coqui model management is explicitly isolated behind a clearly named module and is no longer exported as part of the core `abstractvoice` surface.

---

## Changes

- **Engine-scoped module name**
  - Added `abstractvoice/coqui_model_manager.py` as the canonical legacy Coqui model manager.
  - Converted `abstractvoice/simple_model_manager.py` into a thin import façade.

- **Public API clarity**
  - Removed legacy Coqui helpers (`list_models`, `download_model`, `get_status`, `is_ready`) from `abstractvoice.__init__` exports.
  - Tests now import these helpers from `abstractvoice.coqui_model_manager`, matching the intended scope.

- **Internal wiring**
  - Updated internal imports (`vm/mm_mixin.py`, `vm/tts_mixin.py`, `tts/tts_engine.py`, CLI) to reference `coqui_model_manager`.

- **Docs**
  - Updated `docs/model-management.md` to reference `coqui_model_manager.py` and corrected the legacy “essential model” value.
  - Updated internal notes (`CLAUDE.md`, `CHANGELOG.md`) to match the new naming/scope.

---

## Validation

- Tests: **29 passed, 1 skipped**

