# Voice Cloning (Jan 2026) — Feasibility & Options

## Goal

Add **near‑realtime voice cloning** while respecting:
- **Permissive licensing** (MIT/Apache 2.0 for code + model assets)
- **Simple installs** for the base package (`pip install abstractvoice`)

In practice, this almost certainly means **voice cloning is an optional extra** (GPU/torch-heavy), while the base install stays lightweight and cross‑platform.

## Current status (implementation)

This document started as a feasibility note (Jan 2026). The **current status** below reflects the shipped implementation as of v0.10.x.

- `abstractvoice[omnivoice]` → `omnivoice` engine (recommended/default local cloning backend; omnilingual; prompt-audio cloning + voice design)
- `abstractvoice[cloning]` → `f5_tts` engine (explicit OpenF5 artifacts)
- `abstractvoice[chroma]` → `chroma` engine (Chroma-4B; GPU-heavy)
- `abstractvoice[audiodit]` → `audiodit` engine (LongCat-AudioDiT-1B; prompt-audio cloning)

Operational notes:

- Cloned voices are **engine-bound** (`f5_tts` vs `chroma` vs `audiodit` vs `omnivoice`); selecting a cloned voice uses its stored engine.
- The REPL is offline-first; downloads are explicit via:
  - `python -m abstractvoice download --openf5|--chroma|--audiodit|--omnivoice`
  - REPL: `/cloning_download omnivoice|f5_tts|chroma|audiodit`
- `reference_text` is **optional**:
  - If missing, AbstractVoice auto-generates it via STT (3-pass consensus) on first speak and persists it for the voice.
  - In offline-first contexts, this requires a cached STT model (prefetch: `python -m abstractvoice download --stt small`).
- For best quality, use a clean reference sample and (optionally) set an accurate `reference_text`:
  - `f5_tts`: **6–10s** often works well.
  - `chroma`: upstream examples use longer prompts (≈ **15–30s**), and quality may degrade with very short prompts.
  - `audiodit`: **6–15s** prompt audio is a reasonable starting range; it is currently capped by default.
  - A bad transcript can noticeably degrade cloning quality.
  - The Chroma backend normalizes prompt audio to **mono 24kHz PCM16** and caps extreme prompt lengths for stability.

---

## What “voice cloning” can mean

- **(A) Multi-speaker / voice prompting**: provide a voice prompt or speaker embedding, model synthesizes in that style.
- **(B) Voice conversion**: convert one speaker’s audio into another speaker’s timbre.
- **(C) Fine-tuning**: train/finetune to a specific voice (slow, not “instant”).

For “clone in seconds”, we want (A) and/or (B), not (C).

---

## Candidates (permissive license signals)

### 1) VoxCPM (OpenBMB) — Apache 2.0

- **Claim**: zero-shot voice cloning from short reference audio; streaming inference.
- **Perf**: reported RTF ~0.17 on RTX 4090 (good for realtime).
- **Install**: heavy (PyTorch + GPU recommended); likely optional extra.
- **Languages**: training described as Chinese/English; may not meet full multilingual requirements (FR/DE/ES/RU).

### 2) Dia (Nari Labs) — Apache 2.0

- **Claim**: zero-shot voice cloning from seconds of reference audio; expressive dialogue.
- **Perf**: “realtime” on GPU; reports ~10GB VRAM requirement.
- **Install**: heavy; optional extra.
- **Languages**: unclear breadth; likely not full multilingual coverage.

### 3) MetaVoice‑1B — Apache 2.0

- **Claim**: zero-shot cloning for some English accents; finetuning for others.
- **Install**: heavy; optional extra.
- **Languages**: English focus.

### 4) VibeVoice‑Realtime‑0.5B (Microsoft) — MIT

- **Strength**: realtime/low-latency TTS.
- **Gap**: not clearly a “clone any voice from reference audio” solution (depends on model variant).

### 5) OpenVoice (MyShell)

- **License ambiguity**: repository is commonly labeled MIT, but there are credible reports/issues indicating non‑commercial restrictions.
- **Conclusion**: **do not adopt** unless we can unambiguously confirm permissive terms for both code and weights.

### 6) XTTS‑v2 (Coqui)

- **License**: CPML (non‑permissive / non‑commercial) → **rejected** for our constraints.

---

## Recommendation (current stance)

### Keep the base package clean

- Keep voice cloning and torch-heavy engines behind optional extras.
- Require explicit artifact downloads (`abstractvoice-prefetch ...` or `python -m abstractvoice download ...`) for offline-first deployments.
- Treat each engine as a separate runtime with separate licensing, hardware, and language-quality expectations.

### Use the common API, but test per engine

The shared user-facing shape is implemented:

- `VoiceManager.clone_voice(reference_audio, name=..., engine=...) -> voice_id`
- `VoiceManager.speak(text, voice=voice_id)`
- `VoiceManager.speak_to_bytes(text, voice=voice_id)`
- `export_voice()` / `import_voice()` for portability

Quality, latency, and language coverage still vary by engine and hardware.
Validate the exact model/version you plan to ship.

---

## Ongoing evaluation questions

- Which candidate covers our target languages (EN/FR/DE/ES/RU/ZH) without restrictive licensing?
- Can we support CPU-only “near realtime”, or is GPU a hard requirement?
- Can we standardize “reference audio format” and required length?
