# REPL Guide (CLI Voice Assistant)

The REPL is the quickest way to validate your installation end‑to‑end.

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
- Default LLM API is Ollama at `http://localhost:11434/api/chat` (configure with `--api` / `--model`).

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
- `/stt_engine faster_whisper`

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

Downloads (explicit):

```bash
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
```

Commands:
- `/clones` (list cloned voices)
- `/clone <path> [name]` (clone from a WAV file or a folder containing WAVs)
- `/clone-my-voice` (records a short prompt and creates `my_voice`)
- `/tts_voice piper` or `/tts_voice clone <id-or-name>` (choose which voice is used for speaking)
- `/clone_rm <id-or-name>` or `/clone_rm_all --yes` (delete clones)
- `/cloning_status` (check local readiness; no downloads)
- `/cloning_download f5_tts|chroma` (explicit downloads)

Notes:
- Cloned voices are **engine-bound** (`f5_tts` vs `chroma`). Selecting a clone uses its stored engine.
- The REPL auto-unloads other cloning engines (and unloads Piper voice) when you select a cloned voice to reduce OOM risk.
- `reference_text` is optional: if missing, the REPL will auto-generate it via STT on first speak (requires cached STT model; prefetch with `python -m abstractvoice download --stt small`). You can also set it manually via `/clone_set_ref_text ...`.

## Verbose stats

- Start with `--verbose` or toggle in-session: `/verbose` / `/verbose on|off`.
- After each turn, the REPL prints at most 2 lines with STT/LLM/TTS timings + counts.

## Troubleshooting

- If the REPL can’t connect to your LLM server, ensure it’s running and the `--api` URL is correct.

See also:
- `docs/installation.md` (audio devices, extras, common install issues)
- `docs/public_api.md` (integrator contract; what the library supports)
