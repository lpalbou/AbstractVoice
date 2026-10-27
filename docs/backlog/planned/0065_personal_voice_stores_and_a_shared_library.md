## Task 0065: Personal voice stores per owner, and a shared library an administrator deploys to

**Date**: 2026-10-27  
**Status**: Planned  
**Priority**: P1  
**Area**: `cloning/store.py` (`VoiceCloneStore`), AbstractCore plugin (store settings, listing, synthesis lookup), capability catalog  
**Version**: today one store per machine user

---

## Main goals

- A host can give each person their own voice store: the voices they clone or design are theirs, listed, used, renamed and deleted by them only.
- Next to it, a shared library (the built-in voices and the voices an administrator deploys for everyone) that every person lists and speaks with but cannot change.

## Secondary goals

- An administrator can deploy a voice to the library directly, or promote one of their personal voices into it.

---

## Context / problem

Operator, 2026-10-27: "we would like to have proper isolation between the voices cloned for each user ... users should be allowed to register the voices they want, and this should be personal, not shared by all; an admin should still be able to deploy new voices for all." Today every caller of the plugin reads and writes one store, `appdirs.user_data_dir("abstractvoice")` (`cloning/store.py:131`). On a gateway serving several accounts that means one account's clone, rename or delete changes every account's voices, and a recording of someone's voice cloned by one account is available to all. AbstractGateway therefore makes voice changes admin-only for now (its routes `routes/voice_library.py`).

---

## Current code reality (2026-10-27, 0.18.4)

- `VoiceCloneStore(root=None)` already takes a root folder; the plugin constructs it with no argument at several places (`integrations/abstractcore_plugin.py:1866, 3214, 3742`), so it always uses the machine-user default.
- Built-in voices ship in the package and are flagged `tags.builtin`; they are read-only (delete/rename refuse).
- Synthesis resolves `voice=<voice_id>` through the same store.

---

## Constraints

- One plugin instance per host process may serve several people: the store must be chosen per call or per configured instance, never by a process-global switch.
- A voice id stays unique across stores (a personal voice never shadows a library voice silently).
- Default behaviour unchanged when no store is configured (single-user installs keep today's folder).

---

## Research, options, and references

- **Option A**: plugin settings `voice_store_dir` (the personal, writable store) and `voice_library_dirs` (read-only stores, the built-in voices always included). Listing = library + personal (each record says which, `scope: personal|library`); synthesis looks up personal then library; changes only in the personal store. The host (AbstractGateway) builds one AbstractCore client per account already and passes that account's folder.
- **Option B**: one store with an `owner` field, filtered by the host. Weaker: every account's audio sits in one folder, deleting an account means filtering files, and a host bug leaks voices.
- Deploy to everyone: `store.export(voice_id)` / `library.import(...)` (copy the reference audio and record into a library folder the administrator controls).

---

## Decision

**Chosen approach**: Option A (folders are the isolation; the record carries its scope).

---

## Dependencies

- AbstractCore: none expected (plugin settings travel in the capability config).
- AbstractRuntime: the per-account runtime config carries the two settings; the voice library facade lists the scope.
- AbstractGateway backlog 0942 (personal voices per account, admin deploy, entity profiles).
- Related: `0063_cloned_voices_record_no_language.md`.

---

## Implementation plan

- `VoiceCloneStore` roots from the plugin settings; a read-only store mode for library folders.
- Listing, lookup and synthesis across personal + library; changes refused on library voices (422 like built-ins).
- `deploy_to_library(voice_id, library_dir)` (copy) for administrators.
- Catalog/compatibility: declare the scope fields.

---

## Success criteria

- Two plugin configurations with two personal folders never see each other's voices; both see the library.
- A library voice cannot be renamed or deleted through a personal configuration.
- Unconfigured installs behave exactly as 0.18.4.

---

## Test plan

- `pytest -q` (two stores in temp folders, library read-only, id collisions refused)
- Gated E2E: design a voice in store A, speak with it from A, not listed in B.

---

## Report (fill only when completed)

### Summary

### Validation
