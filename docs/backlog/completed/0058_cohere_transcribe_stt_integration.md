## Task 058: Add Cohere Transcribe (03-2026) as a first-class local STT backend

**Date**: 2026-05-21  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Add a **clean, robust, efficient** local STT integration for `CohereLabs/cohere-transcribe-03-2026`.
- Keep the integration **offline-first** (`allow_downloads=False` means no network traffic).
- Avoid `trust_remote_code` at runtime (no executing downloaded Hugging Face repo Python).
- Make the model **discoverable** through:
  - `VoiceManager` provider/model listing,
  - `CompatibilityCatalog` (provider/model support queries),
  - AbstractCore plugin discovery surfaces (`available_providers`, `list_models`, etc.).

## Secondary goals

- Add a minimal, opt-in **VAD/noise-gate** prepass (or reuse existing VAD) for silence/noise robustness, since the model is reported as “eager to transcribe” non-speech.
- Keep dependencies minimal and aligned with existing `abstractvoice[stt-hf]` (no extra upstream repos required).

---

## Context / problem

`CohereLabs/cohere-transcribe-03-2026` is a recent open-weights ASR model with strong reported accuracy and throughput, but AbstractVoice currently:

- does not list it among curated Transformers ASR models;
- has no dedicated inference path for its specific processor/model API needs (language prompt format, punctuation toggle, chunking);
- defaults `trust_remote_code=False`, while the Hub repo currently advertises `trust_remote_code=True` for `pipeline(...)` / `AutoModelForSpeechSeq2Seq.from_pretrained(...)`.

We want Cohere Transcribe as an **optional** local STT backend that fits AbstractVoice’s design:

- minimal mandatory dependencies;
- robust, explicit behavior;
- no remote-code execution;
- consistent discovery through AbstractCore integration.

---

## Constraints

- Do not break `VoiceManager` public contract or default remote-first behavior.
- No runtime dependency on external repos (no “clone their repo” requirement).
- Prefer permissive licensing only (Apache-2.0 is acceptable).
- Keep model downloads controlled by `allow_downloads`; no surprise downloads.
- Avoid `trust_remote_code` by default; if we must vendor code, do it explicitly inside `abstractvoice/` with license tracking.

---

## Research, options, and references

Model facts (for planning):

- Cohere model card: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
  - 2B parameter Conformer encoder + Transformer decoder, 16kHz, 14 languages, Apache-2.0.
  - Limitations noted: no language detection, no timestamps/diarization, recommends VAD/noise-gate for silence/noise.
- Cohere blog overview: https://cohere.com/blog/transcribe

Hub packaging details relevant to integration design:

- Cohere repo advertises custom code via `auto_map` (implying `trust_remote_code=True` in generic HF usage helpers).
- Public config examples (useful even if the upstream weights are gated):
  - MLX conversion config (shows `auto_map`, `model_type: cohere_asr`, `supported_languages`, `sample_rate`):
    https://huggingface.co/beshkenadze/cohere-transcribe-03-2026-mlx-fp16/blob/main/config.json
  - ONNX community config (also shows `model_type: cohere_asr`, plus `transformers_version` used for export):
    https://huggingface.co/onnx-community/cohere-transcribe-03-2026-ONNX/blob/main/config.json

### Option A — Use Transformers native support (no vendoring)

Assume `transformers>=5.4.0` has first-class `cohere_asr` support and can load the model with `trust_remote_code=False`.

- Pros:
  - Minimal code to maintain in AbstractVoice.
  - Avoids vendoring large model implementation files.
- Cons:
  - Risky across `transformers` versions: if the installed version lacks `cohere_asr`, the model is unusable.
  - The Hub repo currently includes `auto_map`; some loading paths may still nudge users toward remote code.
  - Gating + transformer version skew makes this brittle for “just works” expectations.

### Option B — Vendor the minimal `cohere_asr` implementation (preferred)

Vendor the small set of Python files from the model repo needed for config/model/processor/tokenizer so we can:

- register config/model/processor with `transformers` Auto* APIs (pattern already used for Qwen3-ASR in `abstractvoice/qwen3_asr/`);
- load with `trust_remote_code=False` consistently;
- keep AbstractVoice behavior deterministic and reviewable.

- Pros:
  - Matches AbstractVoice’s existing “no remote code execution” precedent.
  - Works across more `transformers` versions (within reason) and avoids upstream churn.
- Cons:
  - Needs careful license tracking (Apache-2.0 attribution + third-party license inventory).
  - Vendored code must be kept compatible with our supported `transformers` range.

---

## Decision

**Chosen approach**: Option A (Transformers native `cohere_asr` support) with an explicit `transformers>=5.4.0` requirement and `trust_remote_code=False`.

**Why**:
- The Cohere model card states the model is supported natively in `transformers` and recommends `transformers>=5.4.0`, which avoids vendoring a large model implementation.
- AbstractVoice can keep a small, explicit integration surface while still enforcing “no remote code execution” (`trust_remote_code=False`).

---

## Dependencies

- **Code**:
  - `abstractvoice/adapters/stt_transformers_asr.py` (current local HF ASR adapter)
  - `abstractvoice/qwen3_asr/` (reference implementation for vendoring + Auto* registration)
  - `abstractvoice/integrations/abstractcore_plugin.py` (provider/model discovery plumbing)
  - `abstractvoice/assets/voice_model_capabilities.json` (capability matrix metadata)
- **Backlog tasks**:
  - Completed: `docs/backlog/completed/0057_abstractcore_plugin_audio_discovery_surface.md` (discovery surface patterns)

---

## Implementation plan

- Add a curated model entry:
  - Extend `TransformersASRAdapter.KNOWN_MODELS` with `CohereLabs/cohere-transcribe-03-2026` and an alias like `cohere-transcribe-03-2026`.
- Implement a dedicated Cohere load/infer path (similar to the Qwen3-ASR split):
  - Detect the model id (e.g. contains `cohere-transcribe` or config `model_type == "cohere_asr"` when available).
  - Load via `AutoProcessor` + `AutoModelForSpeechSeq2Seq`/`CohereAsrForConditionalGeneration` with:
    - `trust_remote_code=False`
    - `local_files_only=not allow_downloads`
    - device/dtype resolved via existing `resolve_torch_runtime(...)`.
  - Call the processor with `language=<code>` (required for best results) and support `punctuation` on/off if the processor exposes it.
  - Preserve AbstractVoice’s current output contract (plain text only).
- Silence/noise robustness:
  - Reuse existing VAD (`webrtcvad`) or add a tiny optional VAD/noise-gate wrapper in the STT path (only when enabled), to reduce “hallucinated transcription” on silence.
- Discovery + AbstractCore plugin:
  - Ensure the new curated model id appears in `TransformersASRAdapter.selectable_model_ids()` so:
    - `VoiceManager`/plugin `list_models(kind="stt", provider="transformers-asr")` surfaces it.
  - Confirm `voice_catalog()` includes the model id under `stt_catalog_by_provider["transformers-asr"]`.
- Docs:
  - Add a short mention in `docs/getting-started.md` (or the most relevant doc page) for:
    - selecting `stt_engine="transformers-asr"`,
    - using `stt_model="CohereLabs/cohere-transcribe-03-2026"`,
    - the gating/trust_remote_code policy and prefetch guidance.

---

## Success criteria

- Users can run:
  - `VoiceManager(stt_engine="transformers-asr", stt_model="CohereLabs/cohere-transcribe-03-2026", ...)`
  - without enabling `trust_remote_code`,
  - with `allow_downloads=False` behaving strictly offline.
- AbstractCore discovery shows the model under `transformers-asr` without loading it:
  - `available_providers(kind="stt")` includes `transformers-asr` when dependencies are present.
  - `list_models(kind="stt", provider="transformers-asr")` includes the Cohere model id (and alias).

---

## Test plan

- Add a lightweight unit test for model discovery (no downloads):
  - Assert `TransformersASRAdapter.selectable_model_ids()` contains the Cohere id + alias.
  - Assert AbstractCore plugin `list_models(kind="stt", provider="transformers-asr")` returns the id.
- Add an optional `@pytest.mark.model_download` integration test:
  - If HF access is available (token + gating accepted), transcribe a tiny known sample and assert non-empty text.
  - Keep it skipped by default in CI.

---

## Report (fill only when completed)

### Summary

- Added Cohere Transcribe model discovery and a dedicated inference path under the existing `transformers-asr` STT adapter.
- Updated the capability catalog with a model card entry so it is discoverable through AbstractCore plugin surfaces.
- Bumped Transformers optional-dependency minimum to `transformers>=5.4.0` to match Cohere’s native support requirement.

### Validation

- Unit tests:
  - `pytest -q tests/test_cohere_transcribe_transformers_asr_discovery.py`
  - `pytest -q tests/test_abstractcore_plugin.py`
  - `pytest -q tests/test_dependency_check.py`
- Manual smoke:
  - Transcribed `/Users/albou/Documents/patrick_voice_short2.wav` with `TransformersASRAdapter(model_id="CohereLabs/cohere-transcribe-03-2026", language="en")` and confirmed a coherent non-empty transcript.
