# Voice Cloning (Jan 2026) — Feasibility & Options

## Goal

Add **near‑realtime voice cloning** while respecting:
- **Permissive licensing** (MIT/Apache 2.0 for code + model assets)
- **Simple installs** for the base package (`pip install abstractvoice`)

In practice, this almost certainly means **voice cloning is an optional extra** (GPU/torch-heavy), while the base install stays lightweight and cross‑platform.

## Current status (implementation)

As of late Jan 2026, AbstractVoice includes optional cloning backends:

- `abstractvoice[cloning]` → `f5_tts` engine (OpenF5 artifacts)
- `abstractvoice[chroma]` → `chroma` engine (Chroma-4B; GPU-heavy)

Operational notes:

- Cloned voices are **engine-bound** (`f5_tts` vs `chroma`); selecting a cloned voice uses its stored engine.
- The REPL is offline-first; downloads are explicit via:
  - `python -m abstractvoice download --openf5|--chroma`
  - REPL: `/cloning_download f5_tts|chroma`
- `reference_text` is **optional**:
  - If missing, AbstractVoice auto-generates it via STT (3-pass consensus) on first speak and persists it for the voice.
  - In offline-first contexts, this requires a cached STT model (prefetch: `python -m abstractvoice download --stt small`).
- For best quality, use a clean **6–10s** reference sample and (optionally) set an accurate `reference_text`.
  - A bad transcript can noticeably degrade cloning quality.
  - The Chroma backend normalizes prompt audio to **mono 24kHz** and trims very long prompts for stability.

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

## Recommendation (phased)

### Phase 1 (now): keep base package clean

- Provide a stable, well-documented **cloning API design** without shipping cloning by default.
- Add an optional extra:
  - `pip install abstractvoice[cloning]`
- Choose a permissive model only after confirming:
  - code + weights license are compatible
  - installation story is acceptable (ideally wheels, or documented “GPU required”)
  - we can define language scope honestly (start English-only if needed)

### Phase 2 (implementation)

- Implement one adapter behind a clean interface:
  - `VoiceManager.clone_voice(reference_audio, name=...) -> voice_id`
  - `VoiceManager.speak(text, voice=voice_id)`
  - `export_voice()/import_voice()` for portability

---

## Open questions we must answer before committing

- Which candidate covers our target languages (EN/FR/DE/ES/RU/ZH) without restrictive licensing?
- Can we support CPU-only “near realtime”, or is GPU a hard requirement?
- Can we standardize “reference audio format” and required length?
