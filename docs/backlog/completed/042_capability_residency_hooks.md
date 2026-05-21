## Task 042: Capability-level residency hooks for AbstractCore cloned TTS warmup

**Date**: 2026-05-19  
**Status**: Completed  
**Priority**: P1
**Promoted from**: `docs/backlog/proposed/042_capability_residency_hooks.md`  
**Completed**: 2026-05-19  

---

## Recommendation

Add the AbstractCore-facing residency abstraction now, but keep v1 intentionally small.

The only voice runtime that currently justifies real warm/unload work is the cloned TTS engine path.
Base TTS models are already fast enough in practice that they are not worth designing around yet.
STT warmup can stay as future work until there is a real latency problem to solve.

So the proposal should be:

- expose generic residency hooks at the capability/plugin boundary;
- implement real warm/list/unload behavior for cloned TTS engines only in v1;
- leave non-cloned TTS and STT residency as explicit TODOs.

This gives AbstractCore the abstraction it needs without making AbstractVoice over-engineer paths
that are already fast.

---

## Phase 1 Goals

- let AbstractCore call one residency abstraction through `/acore/models/*`;
- keep the public contract generic enough to grow later;
- support real preload/list/unload for cloned TTS engines;
- avoid spending implementation time on non-cloned TTS and STT until needed.

---

## Why This Scope Makes Sense

The current cold-start pain is concentrated in cloned voice synthesis:

- clone engines can spend noticeable time loading weights, kernels, and prompt state;
- the CLI already contains a clone-specific warmup path that discards a tiny utterance before first
  real use;
- base TTS engines are not currently slow enough to justify residency complexity;
- adding generic preload/unload contracts for every TTS/STT adapter now would cost more than the
  latency it saves.

That means the right move is to define the abstraction now and only implement the expensive path now.

---

## Existing Foundation

AbstractVoice already has most of the internal pieces needed for a clone-focused v1:

- the AbstractCore plugin keeps a process-local `VoiceManager` cache keyed by config;
- clone-engine selection is already part of that cache key;
- `VoiceManager` already exposes clone-engine unload helpers;
- the cloner manager already caches clone-engine instances by engine name;
- the plugin already routes all real work through `VoiceManager`.

Important clarification: the CLI `Preloaded <engine>` path is clone-specific. It is not evidence that
we already have a general TTS/STT preload API. It is simply the one expensive path where warmup is
already worth doing.

---

## Public Contract

AbstractVoice should implement the same optional plugin protocol proposed on the AbstractCore side:

```python
def load_resident_model(request: Mapping[str, Any]) -> Mapping[str, Any]: ...
def list_resident_models(filters: Mapping[str, Any] | None = None) -> list[Mapping[str, Any]]: ...
def unload_resident_model(request: Mapping[str, Any]) -> Mapping[str, Any]: ...
```

Keep the boundary dictionary-shaped so AbstractCore never imports AbstractVoice classes.

For v1, the only supported active warm path is cloned TTS. A practical request shape is:

```json
{
  "task": "tts",
  "provider": "cloned",
  "model": "omnivoice",
  "options": {
    "voice": "myvoice"
  }
}
```

Notes:

- `provider="cloned"` is an AbstractVoice pseudo-provider meaning "speak through the cloned-voice
  path";
- `model` is the clone engine id to warm, unload, or inspect;
- `options.voice` is optional and can be used when a stored cloned voice should be selected for any
  voice-specific prompt-state warmup.

Suggested plugin response shape:

```json
{
  "task": "tts",
  "provider": "cloned",
  "model": "omnivoice",
  "state": "resident | failed",
  "resident": true,
  "local": true,
  "unloadable": true,
  "details": {
    "component": "cloning_engine",
    "voice": "myvoice",
    "engine_cached": true
  },
  "error": null
}
```

AbstractCore should still own `runtime_id`, timestamps, pinning, TTL, and the `/acore/models/*`
response envelope. AbstractVoice should only answer the plugin-level residency question.

---

## Phase 1 Behavior

### Load

`load_resident_model(...)` should:

- recognize cloned TTS requests;
- resolve or create the right cached `VoiceManager` for the requested clone engine;
- warm the clone engine through a public `VoiceManager` helper;
- avoid audible output.

If engine-native preload is not available, an internal discarded synthesis fallback is acceptable for
v1 because this is already how the CLI amortizes clone-engine startup cost. That fallback should stay
an implementation detail and should be reported in returned metadata.

### List

`list_resident_models(...)` should:

- list resident clone engines currently held in process;
- report only real in-memory clone-engine state;
- not pretend that base TTS or STT runtimes are resident just because they are configured.

### Unload

`unload_resident_model(...)` should:

- unload one clone engine when a specific engine/model is targeted;
- optionally support broader clone-engine cleanup later;
- reuse the existing clone-engine unload path rather than inventing a second cache.

---

## Better Internal Seam: VoiceManager Only

For v1, do not extend generic TTS/STT adapter contracts yet.

The right seam is a small set of public `VoiceManager` helpers focused on clone residency:

```python
class VoiceManager:
    def preload_cloning_engine(self, *, engine: str | None = None, voice: str | None = None) -> dict: ...
    def list_resident_components(self) -> list[dict]: ...
    def unload_cloning_engine(self, *, engine: str | None = None) -> dict: ...
```

Why this is the better shape:

- the plugin already talks to `VoiceManager`, not raw adapters;
- clone-engine residency is already implemented behind the cloner manager cache;
- we avoid touching every base TTS/STT adapter for a feature they do not need yet;
- later expansion to base TTS or STT can add more helpers without breaking the plugin contract.

In practice, these helpers should wrap the existing cloner cache, runtime info, and unload behavior
instead of bypassing them.

---

## Explicit TODOs

The following should stay out of scope for v1:

- preload/unload for non-cloned TTS engines such as OmniVoice base TTS, Supertonic, Piper, or
  AudioDiT base paths;
- preload/unload for STT engines such as faster-whisper or transformers-ASR;
- new generic `preload()` / `unload()` methods on all TTS/STT base adapters;
- voice/profile/language-specific residency key design for base TTS and STT.

Those are not rejected. They are deferred until they have a demonstrated payoff.

For now, requests that target non-cloned TTS or STT residency should return a clear structured
`not_implemented_yet` style error rather than pretending warm residency exists.

---

## Capability Routing

AbstractVoice currently registers both a voice capability backend and an audio capability backend.

For v1:

- cloned TTS residency belongs to the voice capability backend;
- the same plugin-level method names can exist on both backends if Core expects them uniformly;
- the audio/STT side may return "not implemented yet" until STT warmup becomes worthwhile.

The important part is that there is still only one process-local residency source of truth inside the
plugin.

---

## Non-Goals

- Do not add a second warm-model endpoint family beside `/acore/models/*`.
- Do not make Core inspect private AbstractVoice globals directly.
- Do not retrofit generic adapter residency hooks before there is a real use case.
- Do not spend time optimizing base TTS residency while base TTS is already fast enough.
- Do not report configured or available backends as resident when they are not actually warm.

---

## Recommended Implementation Plan

1. Add the residency methods to the AbstractVoice capability objects.
2. Add small clone-focused residency helpers on `VoiceManager`.
3. Back those helpers with the existing cloner cache and unload logic.
4. Return clear "not implemented yet" responses for non-cloned TTS and STT requests.
5. Add plugin tests proving warm, reuse, list, and unload for clone engines.

---

## Success Criteria

- AbstractCore can warm a cloned TTS engine through the AbstractVoice plugin.
- Later cloned synthesis reuses the warmed engine instead of paying the full cold-start cost again.
- AbstractCore can list and unload resident clone engines through the same abstraction.
- Non-cloned TTS and STT are clearly marked as deferred rather than being over-designed now.

---

## Validation Ideas

- plugin tests for clone warm then synthesize reuse;
- plugin tests for clone-engine listing after warmup;
- plugin tests for unload of a warmed clone engine;
- failure tests for non-cloned TTS or STT residency requests returning a clear deferred-support
  result.

---

## Report

### Summary

- Added cloned-TTS-only residency hooks on the AbstractCore capability plugin boundary:
  load, list, and unload resident clone engines.
- Kept residency truth narrow: base TTS and STT residency requests return structured
  `not_implemented_yet` responses (no false “loaded” claims).
- Wired residency work through the existing process-local `VoiceManager` and cloner cache so warm
  and unload reuse existing state instead of inventing a second cache.
- Added focused plugin tests covering warm/list/unload and deferred-support responses.
- Shipped in release `0.10.5` (2026-05-19) with a changelog entry describing the residency surface.

### Files touched

- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/vm/tts_mixin.py`
- `abstractvoice/cloning/manager.py`
- `tests/test_abstractcore_plugin.py`
- `CHANGELOG.md`
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- `docs/backlog/completed/042_capability_residency_hooks.md`

### Validation

- Focused: `pytest -q tests/test_abstractcore_plugin.py -k residency` -> 3 passed.

### Follow-ups

- `docs/backlog/completed/0056_normalize_abstractcore_capability_residency_truth.md` (event-truth:
  distinguish first warm vs reuse without broadening residency scope).
