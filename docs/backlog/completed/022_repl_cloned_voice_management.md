## Task 022: REPL + API cloned voice management (inspect/use/remove)

**Date**: 2026-01-23  
**Status**: Completed  
**Priority**: P0  

---

## Summary

Cloned voices are now managed like first-class voices:

- REPL can inspect, select, rename, export/import, and delete cloned voices
- `VoiceManager` exposes matching methods so integrators can manage the voice store programmatically

---

## Changes

### Voice store + API

- `abstractvoice/cloning/store.py`
  - Added:
    - `get_voice_dict(voice_id)`
    - `rename_voice(voice_id, new_name)`
    - `delete_voice(voice_id)`

- `abstractvoice/cloning/manager.py`
  - Added:
    - `get_cloned_voice(voice_id)`
    - `rename_cloned_voice(voice_id, new_name)`
    - `delete_cloned_voice(voice_id)`

- `abstractvoice/vm/tts_mixin.py`
  - Exposed via `VoiceManager`:
    - `get_cloned_voice(voice_id)`
    - `rename_cloned_voice(voice_id, new_name)`
    - `delete_cloned_voice(voice_id)`

### REPL commands

Extended `abstractvoice/examples/cli_repl.py`:

- `/clones` now shows source info + which one is selected
- `/clone_info <id|name>` details
- `/clone_ref <id|name>` prints full `reference_text`
- `/clone_rename <id|name> <new_name>`
- `/clone_rm <id|name>`
- `/clone_export <id|name> <path.zip>`
- `/clone_import <path.zip>`
- `/help` updated accordingly

### Tests

- Added `tests/test_voice_clone_store_delete_rename.py`

---

## Validation

- Tests: **35 passed, 2 skipped**

