## Task 0065: Personal voice stores per owner, and a shared library an administrator deploys to

**Date**: 2026-10-27  
**Status**: Completed (2026-10-27, 0.19.0)  
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
- AbstractRuntime backlog 0851: the per-account runtime config carries the two settings; the voice library facade lists the scope.
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

## Report

### Summary

Completed 2026-10-27 (0.19.0), Option A. `cloning/store.py`: `LayeredVoiceStore` (a personal `VoiceCloneStore` in front
of read-only library stores, `meta.scope` on every record, `LibraryVoiceError` for changes to library voices,
`deploy_to_library`) and `open_voice_store(store_dir, library_dirs)` (unconfigured: the machine user's store as
before; `store_dir` without libraries: the machine user's store becomes the library, so existing voices, built-in ones
included, stay visible). `VoiceManager(voice_store_dir=, voice_library_dirs=)`, `deploy_voice_to_library(voice_id)`.
AbstractCore plugin: settings `voice_store_dir`, `voice_library_dirs`; capabilities with the same engine settings share
one cached `VoiceManager` (one copy of the weights) and each call points it at the caller's store under the manager's
lock (`_VoiceStoreScopedLock`); voice management goes to the configured store; `deploy_voice_to_library(voice_id)`.
An unconfigured capability restores the manager's own store (an embedder's injected store is never replaced).

### Validation

- `tests/test_personal_voice_stores.py`: two personal stores never see each other; library voices listed for everyone,
  read-only, resolvable; deploy keeps the personal voice; defaults; a shared manager speaks and manages in the caller's
  store.
- Real weights (Qwen3-TTS, M-series Mac, 2026-10-27), two plugin capabilities with personal stores and one library,
  11/11: alice designs a voice (10.4 s), lists and speaks it (107,564 bytes); bob does not list it, cannot speak with it
  ("Unknown voice_id") nor delete it; alice deploys it; bob speaks with the library voice and cannot delete it
  ("is a library voice"); one loaded speech engine served both.
- Full suite: 857 passed, 3 skipped.

### Adversarial review (2026-10-27) and fixes

Verdict before fixes: do not ship. Confirmed and fixed: (1) an unconfigured caller could inherit another account's
personal store when the shared manager's cloner was built lazily: the scoped lock now restores the manager's own
store (recorded before any caller's settings) on every release, so a manager never rests on a personal store;
(2) cloned-voice management on a shared manager ran without the lock: now under it; (4) a library voice without a
transcript failed every synthesis (the ASR fallback wrote to the read-only library): its transcript is kept in memory
per process; plus deploy records name no path of the deployer, store index writes are serialised (thread lock +
`fcntl` file lock, atomic replace), the residency lookup reads the caller's store first and other managers only
under their lock. Regression tests in `tests/test_personal_voice_stores.py` (lazy cloner with configured and
unconfigured callers in both orders, unconfigured management, library voice without a transcript, deploy record,
concurrent deploys); dropping the restore-on-release turns the lazy-cloner test red. Real weights rerun: 11/11.
Full suite: 869 passed, 3 skipped.

