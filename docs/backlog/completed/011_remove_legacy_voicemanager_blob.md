## Task 011: Remove legacy `voice_manager.py` inline blob (finish Task 005 cleanly)

**Status**: Completed  
**Priority**: P0  

---

## Summary

The large legacy inline implementation inside `abstractvoice/voice_manager.py` was removed so there is only one source of truth: the implementation in `abstractvoice/vm/`.

---

## Result

- `abstractvoice/voice_manager.py` is now ~10 lines and only:
  - imports `VoiceManager` from `abstractvoice/vm/manager.py`
  - defines `__all__`

---

## Validation

- Full test suite: **26 passed, 1 skipped**

