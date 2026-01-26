## Task 030: Evaluate FlashLabs Chroma-4B for optional speech-to-speech + voice cloning

**Date**: 2026-01-26  
**Status**: Planned  
**Priority**: P2 (experimental / GPU-only)  

---

## Main goals

- Decide if **FlashLabs/Chroma-4B** is a meaningfully better **zero-shot voice cloning** option than our current `abstractvoice[cloning]` path (OpenF5/F5-TTS), with acceptable latency and acceptable license terms.
- If promising, integrate it as an **optional engine** without changing the AbstractVoice integrator contract:
  - `VoiceManager.speak(..., voice=<voice_id>)` stays the primary “cloned voice” entry point
  - engine selection is stored inside the cloned-voice record (not exposed in the public API)

## Secondary goals

- Provide an **example** that uses Chroma as an end-to-end **speech-to-speech voice agent** (audio-in → audio-out) while still using AbstractVoice for audio I/O and playback control.
- Preserve the project rule: **no surprise downloads in the REPL** (explicit prefetch only).

---

## Context / problem

AbstractVoice is intentionally modular:
- **STT**: faster-whisper (strong, multilingual, well-understood trade-offs).
- **TTS**: Piper (lightweight, cross-platform, no system deps).
- **Optional voice cloning**: OpenF5/F5-TTS via `abstractvoice[cloning]`.

The remaining gap is “**high-fidelity, low-latency voice cloning**” that is permissively licensed and practical to run.

Chroma 1.0 claims:
- End-to-end spoken dialogue (speech-to-speech) with **streaming generation**
- **High-fidelity zero-shot voice cloning** from a few seconds of reference audio
- **Sub-second responsiveness** and **faster-than-real-time** generation
- Apache-2.0 license signals (code + model repo metadata)

If those claims hold in our environment, Chroma could be valuable as:
- A **higher-quality voice cloning engine** (TTS side) behind the existing AbstractVoice “cloned voice” abstraction
- An **optional local speech-to-speech agent backend** for demos/integrations, without changing core abstractions

---

## Constraints

- **Permissive licensing** for both code and weights (MIT/Apache/BSD compatible).
- **Optional extra only**: base `pip install abstractvoice` must not pull Torch/Transformers/CUDA stacks.
- **No surprise downloads**:
  - Chroma weights/code must not download during `VoiceManager.speak()` in the REPL by default.
  - Provide explicit prefetch entry points (`abstractvoice-prefetch` / `python -m abstractvoice download`).
- **Security**:
  - Chroma requires `trust_remote_code=True` with Transformers.
  - If we use remote code, pin a specific revision and document the risk.
  - Prefer vendoring audited model code as a later hardening step if feasible.
- **Practicality**:
  - GPU is effectively required (paper and upstream docs assume CUDA).
  - Upstream currently depends on a pre-release Transformers (`transformers==5.0.0rc0`).
- **VoiceManager contract stability**: do not expand core `tts_engine` beyond Piper unless explicitly scoped.

---

## Research, options, and references

### What the model card/repo actually says (signals we can trust)

- Hugging Face model metadata (public API):
  - Repo: `https://huggingface.co/FlashLabs/Chroma-4B`
  - License field: `apache-2.0`
  - Gating: `auto` (requires auth to download files)
  - Pipeline tag: `any-to-any`
  - Custom architecture via remote code:
    - `architectures: ["ChromaForConditionalGeneration"]`
    - `auto_map` points to `configuration_chroma.py`, `modeling_chroma.py`, `processing_chroma.py` inside the HF repo
  - HF revision SHA (as of 2026-01-26): `864b4aea0c1359f91af62f1367df64657dc5e90f`

- HF repo contents (important for ops):
  - Weights are split across 3 `.safetensors` shards (~4.3–5.0GB each); total download is large (~14GB+).
  - Audio appears to be 24kHz (consistent with paper + README).

- Upstream README (GitHub):
  - Repo: `https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma`
  - Requirements explicitly call out **Python 3.11+** and **CUDA 12.6** for GPU support.
  - Suggested pins include `torch==2.7.1` and `transformers==5.0.0rc0`.
  - Usage pattern is `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` + `AutoProcessor`.
  - Voice cloning is done by providing `prompt_audio` and `prompt_text` (reference transcript).

### Benchmarks reported in the paper (quality + latency)

Paper: `https://arxiv.org/abs/2601.11141`

Key reported results (from extracted tables):

- **Table 1 (objective speaker similarity / SIM)**:
  - Metric: cosine similarity of speaker embeddings from **WavLM-Large** (SEED-TTS-EVAL protocol, CommonVoice EN).
  - Human baseline: **0.73**
  - F5-TTS: **0.64**
  - Seed-TTS: **0.76**
  - FireRedTTS-2: **0.66**
  - Step-Audio-TTS: **0.66**
  - CosyVoice 3: **0.72**
  - Chroma 1.0: **0.817**
  - Claim: **+10.96% relative** improvement over the human baseline on SIM.
  - Important caveat noted by the authors: Chroma runs at **24kHz** vs **16kHz** for others, which may help preserve speaker cues.

- **Table 2 (subjective pairwise CMOS vs ElevenLabs)**:
  - NCMOS (naturalness preference):
    - Chroma: **24.4%**
    - ElevenLabs: **57.2%**
    - Deuce: **18.3%**
  - SCMOS (speaker similarity preference):
    - Chroma: **40.6%**
    - ElevenLabs: **42.4%**
    - Deuce: **17.0%**
  - Interpretation: ElevenLabs is preferred for naturalness; speaker-similarity preference is near-tied.

- **Table 3 (listener preference: ElevenLabs vs reference audio)**:
  - Reference: **8.0%**
  - ElevenLabs: **92.0%**
  - Author’s point: “naturalness preference” can be dominated by recording quality/cleanliness and not reflect true speaker similarity.

- **Table 4 (latency)**:
  - TTFT (time-to-first-audio-token): **146.87ms**
  - Example generation: 16.58s generation time for 38.80s audio ⇒ **RTF 0.43** (2.3× faster than realtime).
  - Hardware/environment details matter a lot here (paper uses high-end GPUs).

- **Table 5 (URO-Bench basic track: speech understanding / reasoning / oral conversation)**:
  - Chroma (ours), 4B:
    - Rep: **69.05**
    - Sum: **74.12**
    - Gaokao: **38.61**
    - Storal: **71.14**
    - TruthfulQA: **51.69**
    - GSM8K: **22.74**
    - MLC: **60.26**
    - Alpaca: **60.47**
    - CommonVoice: **62.07**
    - WildChat: **64.24**
    - Overall: **57.44**
  - Authors claim Chroma is the only compared model with “personalized voice cloning” and remains competitive despite smaller scale.

### “Is it better than what we have?”

It depends on what “quality” means for AbstractVoice:

- **Voice cloning fidelity (speaker similarity)**: *likely better* than our current default cloning engine.
  - The paper explicitly compares against F5-TTS (SIM 0.64) and reports Chroma at 0.817 under the same SIM metric.
  - Caveat: their evaluation uses a specific protocol and may benefit from 24kHz.

- **Speech naturalness**: uncertain vs Piper/OpenF5; paper shows **ElevenLabs is preferred** for naturalness.
  - We should treat Chroma as “high speaker similarity” first, not “best naturalness”.

- **STT accuracy**: not a clear win.
  - Chroma is not primarily an ASR model; paper does not report standard WER.
  - AbstractVoice already has a strong STT baseline (faster-whisper).

- **End-to-end responsiveness / streaming**: potentially a win for local “voice agent” demos.
  - Chroma’s TTFT/RTF are strong on the paper’s setup, but replication on consumer GPUs is not guaranteed.

---

## Options (integration shapes)

- **Option A — Add Chroma as a `VoiceCloner` engine (recommended first)**
  - Keep Piper as the default TTS.
  - Extend `abstractvoice/cloning/` to support multiple engines:
    - existing: `engine="f5_tts"`
    - new: `engine="chroma"`
  - `voice_id` remains the only thing exposed to `VoiceManager.speak(..., voice=...)`.
  - Pros:
    - Preserves the public contract and existing “cloned voice” UX.
    - The biggest quality upside is specifically cloning.
  - Cons:
    - Heavy optional deps; requires HF auth; `trust_remote_code=True`.

- **Option B — Add Chroma as a full `TTSAdapter` / `tts_engine="chroma"`**
  - Make Chroma a first-class TTS engine choice in `VoiceManager`.
  - Pros: clean configuration; could support text-only “TTS” usage.
  - Cons:
    - Requires relaxing core design that currently enforces Piper-only TTS.
    - Higher risk of dependency conflicts and contract expansion.

- **Option C — Example-only “speech-to-speech agent” integration**
  - Add an `examples/chroma_s2s_agent.py` script:
    - record mic audio with AbstractVoice
    - feed to Chroma
    - play generated audio through AbstractVoice playback pipeline
  - Pros: minimal commitment; quickest to validate quality and latency.
  - Cons: does not immediately improve the shipped cloning API.

- **Option D — Do nothing**
  - Only document Chroma in `docs/voice_cloning_2026.md` as a candidate.

---

## Decision

**Chosen approach**: Phase the work as **Option C → Option A**.

**Why**:
- We should validate quality/latency in our environment before adding a new engine.
- If Chroma is valuable, integrating it behind `VoiceCloner` provides the benefits while keeping the AbstractVoice “voice abstraction” stable and small.

---

## Dependencies

- **Backlog tasks (completed)**:
  - `docs/backlog/completed/009_voice_cloning_adapter_impl.md`
  - `docs/backlog/completed/019_cloning_ux_no_surprise_downloads.md`
  - `docs/backlog/completed/021_streaming_and_cancellation_for_cloned_tts.md`

- **Backlog tasks (planned)**: none
- **ADRs**: none

---

## Implementation plan

### Phase 0: local validation spike (example-only)

- Add `examples/chroma_s2s_agent.py`:
  - Minimal audio-in → audio-out loop.
  - Keep it explicitly “experimental”.
  - Require an opt-in install extra (see Phase 1).
- Document required env:
  - Python 3.11+
  - HF auth (token) for gated download
  - GPU expectation (CUDA)

### Phase 1: packaging + prefetch (explicit downloads)

- Add a new optional dependency group:
  - `abstractvoice[chroma]` (or `abstractvoice[s2s]`) that pulls Torch/Transformers stack.
  - Keep pins close to upstream to avoid breakage (`transformers==5.0.0rc0` is likely required initially).
- Extend explicit download tools:
  - `abstractvoice-prefetch --chroma`
  - `python -m abstractvoice download --chroma`
  - Respect `VoiceManager.allow_downloads` by defaulting to “no download in REPL”.
  - Provide clear error messages when:
    - HF token is missing
    - repo is gated and download fails

### Phase 2: integrate as a cloning engine (behind existing abstraction)

- Add `abstractvoice/cloning/engine_chroma.py` implementing a similar interface to `F5TTSVoiceCloningEngine`:
  - `infer_to_wav_bytes(text, reference_paths, reference_text, ...) -> bytes`
  - `infer_to_audio_chunks(...) -> Iterable[(chunk: np.ndarray, sr: int)]` (optional; can start with full-audio then chunk)
  - `runtime_info()` for debugging (device, dtype, revision, etc.)
- Update `abstractvoice/cloning/manager.py` (`VoiceCloner`) to:
  - Dispatch per voice record (`voice.engine`) to either the F5 engine or the Chroma engine.
  - Continue using stored `reference_text` (and existing STT auto-fallback) because Chroma needs `prompt_text`.
- Add an engine selection mechanism when creating a voice:
  - Prefer adding an optional `engine=` argument to `VoiceCloner.clone_voice(...)` (non-public API).
  - Keep defaults unchanged (`engine="f5_tts"`).

### Phase 3: hardening + security posture

- Pin HF revision by default (to reduce “remote code changed” risk).
- Add docs:
  - Explain `trust_remote_code=True` trade-offs.
  - Recommend using a fixed `revision` and verifying upstream code.
- Optional follow-up: vendor audited model code into AbstractVoice to remove `trust_remote_code` (only if licensing is confirmed and maintenance cost is acceptable).

---

## Success criteria

- **Validation spike**: we can run an end-to-end audio conversation example locally and confirm:
  - model loads reliably
  - audio output is correct 24kHz WAV
  - subjective cloning fidelity is promising on a few internal samples
- **Optional integration**:
  - Base install remains lightweight; no Torch/Transformers dependency.
  - With `abstractvoice[chroma]` installed and Chroma weights cached, `VoiceManager.speak(..., voice=<voice_id>)` works when that `voice_id` was created with the Chroma engine.
  - REPL does not silently download weights; errors instruct explicit prefetch.

---

## Test plan

- `pytest -q`
- Unit tests (CPU-only):
  - Engine dispatch logic in `VoiceCloner` (mock engines; no real model load).
  - Error messaging when `abstractvoice[chroma]` is not installed.
  - Prefetch CLI argument parsing.
- Manual (GPU env):
  - `abstractvoice-prefetch --chroma`
  - Create a cloned voice with Chroma and call:
    - `vm.speak_to_bytes("...", voice=<voice_id>)` and validate WAV header + sample rate.
    - `vm.speak("...", voice=<voice_id>)` and validate playback.

---

## Report (fill only when completed)

### Summary

<what changed and why>

### Validation

- Tests: <result>
