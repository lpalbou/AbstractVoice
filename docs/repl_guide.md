# REPL Guide (CLI Voice Assistant)

The REPL is the quickest way to validate your installation end‑to‑end.

## Start

```bash
abstractvoice cli --debug
```

## Quick smoke test checklist

### 1) TTS works

- Type a short message to the assistant (it will call your configured LLM endpoint).
- Or use `/speak <text>` to test TTS without calling the LLM.
- Use:
  - `/pause`
  - `/resume`
  - `/stop`

### 2) Language switching (Piper path)

Try:
- `/language fr`
- `/language de`
- `/language es`
- `/language ru`
- `/language zh`

### 3) Disable/enable TTS

- `/tts off` (text‑only)
- `/tts on` (re‑enables voice features)

### 4) Engine selection

- `/tts_engine piper`
- `/stt_engine faster_whisper`

### 4) Voice catalog (Piper)

- `/setvoice` lists voices.
- `/setvoice fr.siwis` switches to French (voice id is best-effort).

### 5) Voice cloning (optional extra)

Install:

```bash
pip install "abstractvoice[cloning]"
```

Commands:
- `/clones` (list cloned voices)
- `/clone <path> [name]` (clone from a WAV file or a folder containing WAVs)
- `/clone-my-voice` (records a short prompt and creates `my_voice`)
- `/tts_voice piper` or `/tts_voice clone <id-or-name>` (choose which voice is used for speaking)

## Troubleshooting

- If the REPL can’t connect to your LLM server, ensure it’s running and the `--api` URL is correct.

