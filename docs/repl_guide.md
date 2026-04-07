# REPL Guide (CLI Voice Assistant)

The REPL is the quickest way to validate your installation end‑to‑end.

It is also intentionally **minimal**: it’s a demonstrator/smoke-test harness for the `abstractvoice` library.
For production agent/server use in the AbstractFramework ecosystem, the intended integration is via **AbstractCore** (AbstractVoice provides TTS/STT as a capability backend plugin; see `docs/api.md` → “Integrations”).

## Start

```bash
abstractvoice --debug
abstractvoice --verbose
abstractvoice --voice-mode stop   # enable mic voice input

# Or (from a source checkout):
python -m abstractvoice cli --debug
python -m abstractvoice cli --verbose
python -m abstractvoice cli --voice-mode stop
```

Notes:
- The REPL is **offline-first**: it will not download model weights implicitly.
- Mic voice input is **off by default** for fast startup. Enable with `--voice-mode stop` or in-session: `/voice stop`.
- The REPL includes a minimal OpenAI-compatible LLM HTTP client (`abstractvoice/examples/llm_provider.py`).
- Default LLM provider preset is **Ollama** at `http://localhost:11434` (OpenAI-compatible `POST /v1/chat/completions`).
  - Configure with `--provider` (preferred) or `--api` (legacy), passing a **base URL** (not a full `/v1/...` path), and set the model with `--model`.

## Quick smoke test checklist

### 1) TTS works

- Type a short message to the assistant (it will call your configured LLM endpoint).
- Or use `/speak <text>` to test TTS without calling the LLM.
- Use:
  - `/pause`
  - `/resume`
  - `/stop`

If Piper can’t speak, you likely haven’t cached a Piper model yet (offline mode):

```bash
python -m abstractvoice download --piper en
```

### 2) Language switching (Piper path)

Try:
- `/language fr`
- `/language de`
- `/language es`
- `/language ru`
- `/language zh`

Each language requires its own cached Piper voice model:

```bash
python -m abstractvoice download --piper fr
python -m abstractvoice download --piper de
```

### 3) Disable/enable TTS

- `/tts off` (text‑only)
- `/tts on` (re‑enables voice features)

### 4) Engine selection

- `/tts_engine piper`
- `/tts_engine audiodit` (requires `abstractvoice[audiodit]` + prefetched weights; best results on EN/ZH, other languages not guaranteed)
- `/tts_engine omnivoice` (requires `abstractvoice[omnivoice]` + prefetched weights; supports many languages upstream)
- `/stt_engine faster_whisper`

Engine knobs (base TTS):

- `/tts_quality fast|balanced|high` (best-effort speed/quality preset)
- `/speed 0.9` / `/speed 1.1` (native for OmniVoice; AudioDiT ignores speed)
- `/profile list` / `/profile reload` / `/profile show` / `/profile <id>` (voice profiles for the active base TTS engine)
- OmniVoice-specific parameters + voice design: `/omnivoice` (prints current params + examples)

OmniVoice voice design (`instruct`) and stability (`seed`):

- If available, prefer presets via `/profile <id>` (engine-agnostic). See `/profile list`.
- **`/omnivoice instruct "..."`** enables **voice design mode** (attribute-based speaker). OmniVoice validates `instruct` items upstream.
- **Valid English items** (case-insensitive; comma-separated; one per category):
  - `male`, `female`
  - `child`, `teenager`, `young adult`, `middle-aged`, `elderly`
  - `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`
  - `whisper`
  - `american accent`, `australian accent`, `british accent`, `canadian accent`, `chinese accent`, `indian accent`, `japanese accent`, `korean accent`, `portuguese accent`, `russian accent`
- **Staying on the “same designed voice” across turns**: voice design is stochastic (driven mainly by `position_temperature` / `class_temperature`).
  - Option A (fully deterministic): `/omnivoice position_temperature 0` + `/omnivoice class_temperature 0`
  - Option B (recommended): keep temperatures > 0, but pin a **seed**:
    - `/omnivoice seed 123` (stable across turns)
    - change the seed to “pick another” voice: `/omnivoice seed 124`
    - clear: `/omnivoice seed off`
  - Cross-computer note: you’ll get the closest match when both machines use the **same OmniVoice model snapshot** (and similar `torch/omnivoice` versions). Exact waveform parity across different accelerators/dtypes is not guaranteed, but speaker identity should remain consistent in practice.

### 5) Voice catalog (Piper)

- `/setvoice` lists voices.
- `/setvoice fr.siwis` switches language; voice id is currently best-effort (one default Piper voice per language).

### 6) Voice cloning (optional extra)

Install:

```bash
pip install "abstractvoice[cloning]"
```

For Chroma (GPU-heavy):

```bash
pip install "abstractvoice[chroma]"
```

For AudioDiT (torch/transformers):

```bash
pip install "abstractvoice[audiodit]"
```

For OmniVoice (torch/transformers):

```bash
pip install "abstractvoice[omnivoice]"
```

Downloads (explicit):

```bash
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
python -m abstractvoice download --audiodit
python -m abstractvoice download --omnivoice
```

Commands:
- `/clones` (list cloned voices)
- `/clone <path> [name]` (clone from a WAV file or a folder containing WAVs)
- `/clone-my-voice` (records a short prompt and creates `my_voice`)
- `/tts_voice piper` or `/tts_voice clone <id-or-name>` (choose which voice is used for speaking)
- `/clone_rm <id-or-name>` or `/clone_rm_all --yes` (delete clones)
- `/cloning_status` (check local readiness; no downloads)
- `/cloning_download f5_tts|chroma|audiodit|omnivoice` (explicit downloads)

Notes:
- Cloned voices are **engine-bound** (`f5_tts` vs `chroma` vs `audiodit` vs `omnivoice`). Selecting a clone uses its stored engine.
- The REPL auto-unloads other cloning engines (and unloads Piper voice) when you select a cloned voice to reduce OOM risk.
- `reference_text` is optional: if missing, the REPL will auto-generate it via STT on first speak (requires cached STT model; prefetch with `python -m abstractvoice download --stt small`). You can also set it manually via `/clone_set_ref_text ...`.

## Verbose stats

- Start with `--verbose` or toggle in-session: `/verbose` / `/verbose on|off`.
- After each turn, the REPL prints at most 2 lines with STT/LLM/TTS timings + counts.
- Inspect the exact in-memory LLM chat history (what is sent to the server): `/history [n] [--all] [--full]`.

## Troubleshooting

- If the REPL can’t connect to your LLM server, ensure it’s running and the `--api` URL is correct.

See also:
- `docs/installation.md` (audio devices, extras, common install issues)
- `docs/api.md` (integrator contract; what the library supports)