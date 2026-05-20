# ADR 0007: Directed speech requests and planning are package-owned

Status: Accepted.

## Context

AbstractVoice already owns the user-facing semantics for `provider`, `model`,
`profile`, `voice`, `instructions`, `quality_preset`, and clone residency.
That policy is explicit in ADR 0006.

The Scenema-track proposals introduce the next missing layer above today's
text-first TTS surface:

- richer request semantics such as pacing, target duration, scene context,
  action cues, ambient/background audio, and output layout;
- per-engine capability reporting and explicit degradation instead of silent
  best-effort guessing;
- package-owned planning and request compilation for advanced speech engines.

Current code now has the start of that package-level contract internally:

- `abstractvoice/speech_request.py` defines package-owned `SpeechRequest`,
  `SpeechCapability`, and `SpeechCapabilities` helpers;
- `VoiceManager` compiles current call sites into that internal request layer;
- `VoiceManager.get_tts_capabilities()` already reports support as `native`,
  `emulated`, `conditional`, or `unsupported`.

What does not exist yet is a broadened raw adapter contract for richer directed
speech. `TTSAdapter` is still fundamentally text-first, and the new
directed-speech fields are mostly package-owned metadata and capability status
today rather than engine-wired generation controls.

If future engines add richer speech controls through ad hoc kwargs or if
AbstractCore starts owning speech-planning semantics, the package boundary will
invert. AbstractVoice would become an execution detail for semantics it no
longer controls.

## Decision

- AbstractVoice owns the authoritative directed-speech contract above the raw
  adapter layer.
- That contract is currently represented internally by package-owned request and
  capability types in `abstractvoice/speech_request.py`. Future result metadata
  may use `SpeechResult` or an equivalent typed form when needed.
- The first version must stay small and add only fields with a real caller and
  a real engine or planner consumer. Candidate fields may include:
  - `pace`
  - `target_duration_s`
  - `actions`
  - `scene_context`
  - `ambient_audio`
  - `background_sfx`
  - `output_channels`
- Existing public entry points such as `VoiceManager.speak()`,
  `VoiceManager.speak_to_bytes()`, and current plugin/server payloads should
  compile into that contract internally instead of breaking the current public
  API.
- The raw adapter contract does not need to expand immediately. The package may
  continue compiling richer request intent into the current text-first adapter
  layer until a later ADR or implementation need justifies a wider adapter
  surface.
- Package-owned selector semantics from ADR 0006 remain in force. New directed
  speech fields must not overload `profile`, `voice`, `model`, or
  `instructions` with hidden meanings.
- Every directed-speech field exposed beyond today's stable text-first subset
  must have a package-owned capability state:
  - `native`: the engine/runtime supports the field directly;
  - `emulated`: AbstractVoice or engine glue can implement the effect with a
    documented translation and trade-off;
  - `conditional`: AbstractVoice can forward the field or orchestrate it, but
    provider/model support is endpoint-specific or not yet in the package
    allowlist;
  - `unsupported`: the request cannot be honored.
- Unsupported fields must never disappear silently. The package must either:
  - reject the request clearly; or
  - surface explicit degradation or capability metadata showing what was not
    honored.
- Directed-speech planning and degradation rules stay inside AbstractVoice.
  Integrations such as AbstractCore may serialize requests and results as
  dict/JSON payloads, but they do not define or reinterpret the semantics.
- If future reuse justifies a shared neutral contracts package, only schemas and
  enums may be extracted. Planning logic, degradation policy, and request
  compilation remain AbstractVoice-owned unless a later ADR changes that
  boundary.
- Simple engines are allowed to support only the text-first subset plus the
  stable selectors already defined by current ADRs. They should report richer
  fields as unsupported rather than faking support.

## Consequences

### Positive

- Scenema-class work gets a package-owned seam before engine-specific runtime
  work begins.
- Integrations can stay thin and honest because they forward package-owned
  semantics instead of inventing their own request vocabulary.
- Capability reporting becomes auditable across simple and advanced engines.

### Negative

- AbstractVoice must design and maintain a richer request contract instead of
  leaving that work to integrations or engine-specific wrappers.
- Some future requests will fail explicitly where a hidden best-effort fallback
  might have seemed more convenient.

### Neutral

- The adapter layer may stay narrower than the package-level request contract as
  long as request compilation, capability reporting, and degradation remain
  explicit.

## Enforcement

- Do not add engine-specific directed-speech semantics as undocumented kwargs on
  `VoiceManager`, plugin surfaces, or integration payloads when they belong in
  the package-owned contract.
- Do not silently reinterpret unsupported fields as plain text decoration or as
  generic `instructions` without capability reporting.
- Do not claim support for a directed-speech field merely because the internal
  request object can carry it. Support remains `unsupported` until the package
  wires that field to real engine or planner behavior.
- Keep planning, capability mapping, and degradation rules in AbstractVoice.
  AbstractCore and other integrations should remain serialization and transport
  layers.
- Changes to the directed-speech contract, capability states, or degradation
  rules require an ADR update or a superseding ADR.

## Validation

- Extend `tests/test_abstractcore_plugin.py` so richer request payloads can pass
  through dict/JSON boundaries without redefining package semantics in the
  plugin layer.
- Add request-compilation tests for `VoiceManager.speak()`,
  `VoiceManager.speak_to_bytes()`, and any new directed-speech helper surface.
- Add capability-reporting tests that cover at least one `native`, one
  `emulated`, one `conditional`, and one `unsupported` field.
- Update `docs/api.md` and `docs/architecture.md` when a public directed-speech
  request surface becomes user-facing.

## Backlog links

- `docs/backlog/planned/scenema/044_package_owned_directed_speech_request_and_capabilities.md`
- `docs/backlog/planned/scenema/045_shared_runtime_foundation_for_advanced_speech_engines.md`
- `docs/backlog/planned/scenema/046_scenema_class_python_runtime_spike.md`
- `docs/backlog/planned/scenema/047_device_agnostic_runtime_and_apple_silicon_path.md`

## Related

- [0006_voice_overlays_are_package_owned_not_generic_adapters.md](./0006_voice_overlays_are_package_owned_not_generic_adapters.md)
- [0005_torch_device_and_dtype_policy.md](./0005_torch_device_and_dtype_policy.md)
- [../architecture.md](../architecture.md)
- `abstractvoice/speech_request.py`
- `abstractvoice/vm/tts_mixin.py`
- `abstractvoice/integrations/abstractcore_plugin.py`
- `abstractvoice/examples/web_ui.py`
