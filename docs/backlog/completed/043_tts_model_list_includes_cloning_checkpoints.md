## Task 043: The TTS model list includes the cloning checkpoints, marked

**Date**: 2026-10-28  
**Status**: Completed (0.19.2)  
**Priority**: P1  

---

## Main goals

- A picker can select the Qwen3-TTS **Base** checkpoint (the one that speaks cloned voices) from the
  provider's TTS model list, shown as "Qwen/Qwen3-TTS-12Hz-0.6B-Base (clones)".

## Secondary goals

- Preset-voice synthesis on a Base checkpoint stays refused with the existing sentence.

---

## Context / problem

Operator, 2026-10-27 (AbstractGateway console, AbstractFramework root backlog 1055): the console's model list for
`qwen3-tts` offered only 0.6B-CustomVoice, 1.7B-CustomVoice and 1.7B-VoiceDesign. `cached_tts_model_ids` dropped the
Base checkpoints (cloning-only) from every TTS listing, so `list_tts_models`, `tts_models_by_provider` and the
pickers built on them never offered the model a cloned voice is spoken with; it appeared only once a clone was
picked.

## What changed

- `local_models.cached_tts_model_ids(..., include_cloning=False)`: `True` keeps the Base checkpoints;
  `local_models.tts_model_role(engine, model_id)` → `"cloning"` (packaged cloning catalog, or the cached config's
  `tts_model_type: base`) / `"tts"`.
- The plugin's selectable TTS models (`list_tts_models`, `voice_catalog`, the light catalog) include them.
- `tts_model_details_by_provider` (both catalogs): `{provider: [{id, role, label}]}`, `role: "cloning"` and
  label "<id> (clones)" for Base. `list_models_detailed(kind=, provider=)` returns the same records.
- Unchanged: `Qwen3TTSAdapter` refuses preset synthesis on a Base checkpoint ("… is a cloning checkpoint; select it
  through voice cloning …"). A Base checkpoint is never the default (`active_model`), and an engine with only a Base
  checkpoint on disk is not reported available (a request with no model would load a CustomVoice checkpoint
  that is not there); its clones are still listed.

## Validation

`tests/test_tts_model_roles.py` (8): the list with and without cloning, the role from the catalog and from an
operator's config, the plugin list and details, the light catalog, the preset refusal. Mutation-checked red: the
cloning checkpoints not included, the details key absent.
