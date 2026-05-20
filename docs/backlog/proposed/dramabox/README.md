## Proposed DramaBox track

Status: Proposed.

Purpose:

- keep DramaBox follow-up work grouped as one track instead of scattering
  engine-specific and abstraction-specific notes across root `proposed/`;
- separate reusable foundation ideas from DramaBox-specific runtime work;
- preserve the stop/go gate around the non-vendored LTX-family runtime spike.

Items:

- `043_dramabox_nonvendored_feasibility.md`: predecessor feasibility memo and
  initial product-fit assessment.
- `049_package_owned_expressive_prompt_mapping_and_reference_audio_conditioning.md`:
  reusable planning/conditioning abstraction work.
- `050_advanced_speech_output_layout_and_result_metadata.md`:
  reusable output/result abstraction work.
- `051_ltx_audio_runtime_spike_for_dramabox.md`: stop/go runtime spike for a
  non-vendored LTX-family path.
- `052_dramabox_clone_engine_integration.md`: DramaBox-specific cloning-first
  integration path.
- `053_dramabox_base_tts_engine_and_capability_rollout.md`: DramaBox-specific
  base-TTS follow-on if the clone path succeeds.
- `054_dramabox_apple_silicon_validation_and_explicit_degradation.md`:
  platform validation and support-boundary follow-on.

Reading order:

1. `043_dramabox_nonvendored_feasibility.md`
2. `049_package_owned_expressive_prompt_mapping_and_reference_audio_conditioning.md`
3. `050_advanced_speech_output_layout_and_result_metadata.md`
4. `051_ltx_audio_runtime_spike_for_dramabox.md`
5. `052_dramabox_clone_engine_integration.md`
6. `053_dramabox_base_tts_engine_and_capability_rollout.md`
7. `054_dramabox_apple_silicon_validation_and_explicit_degradation.md`

Governing ADRs:

- `docs/adr/0005_torch_device_and_dtype_policy.md`
- `docs/adr/0006_voice_overlays_are_package_owned_not_generic_adapters.md`
- `docs/adr/0007_directed_speech_requests_and_planning_are_package_owned.md`

Scope:

- reusable directed-speech and output abstractions discovered through the
  DramaBox investigation;
- the LTX-family runtime spike required to know whether a non-vendored path is
  real;
- DramaBox-specific integration and support-boundary follow-ons.

Non-goals:

- this track does not commit AbstractVoice to shipping DramaBox;
- this track does not authorize vendoring the DramaBox repo or copied `ltx2/`
  runtime tree;
- this track does not promise Apple Silicon support before a real runtime spike
  succeeds.
