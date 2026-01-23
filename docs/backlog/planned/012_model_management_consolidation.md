## Task 012: Model management consolidation (Piper vs legacy Coqui)

**Status**: Planned  
**Priority**: P1  

---

## Problem

We currently have two “model management” concepts:

- **Piper (default)**: models/voices are managed by the Piper adapter (`abstractvoice/adapters/tts_piper.py`) and its download logic.
- **Legacy Coqui TTS**: model listing/downloading/caching is handled by `abstractvoice/simple_model_manager.py`, and exposed:
  - as a procedural API (`abstractvoice.list_models`, `download_model`, `get_status`, `is_ready`)
  - and as OO methods on `VoiceManager` via **MM** mixin (`abstractvoice/vm/mm_mixin.py`)

This is correct functionally, but confusing conceptually because “model management” doesn’t clearly state *which engine* it applies to.

---

## Goal

Make model management explicit and non-confusing:

- Clearly separate **Piper voice/model management** vs **legacy Coqui model management**
- Ensure docs and public APIs describe scope and limitations precisely
- Remove duplication of catalogs where possible (or explicitly deprecate legacy catalogs)

---

## Proposed work

1. **Docs clarification**
   - Update `docs/model-management.md` to explicitly split Piper vs Coqui sections
   - Ensure `docs/architecture.md` reflects `abstractvoice/vm/` layout and MM mixin

2. **API clarity**
   - Decide whether `simple_model_manager.py` should be renamed to something engine-specific (e.g. `coqui_model_manager.py`)
   - Ensure `VoiceManager` methods and error messages mention which engine they target

3. **Catalog deduplication**
   - Evaluate whether `SimpleModelManager.AVAILABLE_MODELS` should be derived from `abstractvoice/config/voice_catalog.py`
   - If not feasible, document the divergence and mark one source as authoritative

---

## Success criteria

- Users can answer: “What does `download_model()` download?” without reading code.
- Docs and CLI behavior match the default engine story (Piper).

