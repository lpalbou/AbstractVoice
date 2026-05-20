# ADR 0006: Voice selectors and clone residency are package-owned semantics

Status: Accepted.

## Context

AbstractVoice now exposes several user-facing selectors and lifecycle controls
that sit above raw model runtime integration:

- `provider`
- `model`
- `profile`
- `voice`
- `instructions`
- `quality_preset`
- clone-engine residency and warmup

These values are easy to misread through the vocabulary of text-model tooling,
where "adapter", "overlay", or "LoRA" often means one generic mechanism.
AbstractVoice does not work that way.

The code already distinguishes multiple different concepts:

- provider and model select a backend/runtime path;
- profile selects an engine-owned preset or remote profile-like voice surface;
- voice may point at a cloned voice id and route synthesis through the clone
  path instead of the base TTS adapter;
- quality presets tune package-owned generation behavior and map differently by
  engine;
- residency currently means clone-engine residency only, not generic TTS or STT
  model warming.

If these concepts are collapsed into a generic adapter/overlay abstraction,
integration code becomes dishonest and request routing breaks. The most obvious
failure is treating a cloned `voice` id as if it were just another `profile`.
Another is pretending that base TTS or STT models are "resident" because a
provider is configured, even though the only meaningful warmed path today is
cloned TTS.

## Decision

- AbstractVoice owns the semantics of `provider`, `model`, `profile`, `voice`,
  `instructions`, `quality_preset`, and clone residency.
- Integration layers may forward these selectors, serialize them, and expose
  catalog data, but they do not redefine what the selectors mean.
- `profile` and `voice` are distinct contracts:
  - `profile` is an engine-owned preset or remote profile-like voice selection
    applied through the active TTS adapter;
  - `voice` is an explicit synthesis target, and when it resolves to a stored
    cloned voice it must route through the clone path rather than through
    `set_profile(...)`.
- `model` identifies backend/model choice. It is not a synonym for profile or
  voice identity.
- `quality_preset` is package-owned. Callers may request it, but they must not
  assume that "high" or "low" has identical implementation details across all
  engines.
- Residency support remains honest about current scope:
  - v1 residency is clone-engine residency on the voice capability path;
  - base TTS and STT residency requests must return a structured
    "not implemented yet" style result rather than being faked.
- If a future backend introduces true weight-overlay semantics analogous to a
  LoRA or model adapter, that concept must be added explicitly as a new voice
  capability concept. It must not silently overload `profile`, `voice`, or
  `model`.

## Consequences

### Positive

- The public vocabulary stays aligned with real package behavior.
- Plugin and API layers can stay thin because they forward package-owned
  semantics instead of inventing their own normalization rules.
- Tests can lock down important distinctions such as profile versus cloned
  voice, and configured provider versus resident runtime.

### Negative

- The API surface is less uniform than a fake one-field `adapter=` abstraction.
- Integrators must learn a small AbstractVoice-specific vocabulary instead of
  flattening everything into one model-layer metaphor.

### Neutral

- Cross-engine portability remains partial by design. Some selectors are stable,
  but their concrete effect is still engine-specific.

## Enforcement

- Keep explicit selector fields in package and integration surfaces. Do not
  replace them with a generic adapter/overlay field without a new ADR.
- Do not apply a cloned `voice` id through `set_profile(...)`.
- Do not treat configured or installable providers as resident models.
- Plugin/catalog surfaces must preserve package-owned provenance such as
  provider id, profile id, voice id, and clone kind instead of collapsing them
  into anonymous labels.

## Validation

- `tests/test_abstractcore_plugin.py`
  - profile ids are applied as profiles and restored correctly;
  - cloned `voice` ids do not get applied as profiles;
  - clone residency works only on the clone path;
  - base TTS and STT residency report `not_implemented`.
- `tests/test_voice_profiles_abstraction.py`
- `tests/test_supertonic_adapter.py`
- `tests/test_piper_adapter.py`
- `tests/test_remote_openai_compatible_adapters.py`

## Backlog links

- `docs/backlog/planned/036_voice_profile_abstraction.md`
- `docs/backlog/completed/039_abstractcore_plugin_voice_catalog_surface.md`
- `docs/backlog/proposed/042_capability_residency_hooks.md`

## Related

- [0005_torch_device_and_dtype_policy.md](./0005_torch_device_and_dtype_policy.md)
- [../api.md](../api.md)
- [../architecture.md](../architecture.md)
- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/vm/tts_mixin.py`
