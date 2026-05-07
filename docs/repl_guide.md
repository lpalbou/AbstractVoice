# REPL Guide

The `abstractvoice` REPL is the fastest way to validate the package end to end:
local TTS, optional microphone input, optional cloning engines, and an
OpenAI-compatible chat endpoint.

For production agent/server deployments in the AbstractFramework ecosystem,
run AbstractCore Server and let AbstractVoice provide the audio capability
backend. The REPL stays intentionally lightweight and local.

## Start

```bash
abstractvoice --verbose

# From a source checkout:
python -m abstractvoice cli --verbose
```

Microphone input is off by default. Enable it explicitly:

```bash
abstractvoice --voice-mode stop
python -m abstractvoice cli --voice-mode stop
```

Remote audio startup examples:

```bash
OPENAI_API_KEY=... abstractvoice --tts-engine openai --stt-engine openai
abstractvoice --tts-engine openai-compatible --stt-engine openai-compatible --remote-base-url http://localhost:8000/v1
```

Useful startup flags:

- `--verbose`: print compact timing and token/audio stats after each turn.
- `--debug`: print extra diagnostics and save generated debug WAVs.
- `--voice-mode stop|wait|full|ptt|off`: choose the initial microphone mode.
- `--provider <preset-or-url>`: choose an OpenAI-compatible LLM provider.
- `--model <name>`: choose the LLM model.
- `--tts-engine auto|piper|openai|openai-compatible|audiodit|omnivoice`: choose the initial TTS engine.
- `--stt-engine auto|faster_whisper|openai|openai-compatible`: choose the initial STT engine.
- `--tts-model <id>` / `--stt-model <id>`: model ids for remote audio engines.
- `--remote-base-url <url>` / `--remote-api-key <key>`: OpenAI-compatible remote voice endpoint config. `--tts-engine openai` and `--stt-engine openai` default to OpenAI's hosted API and read `OPENAI_API_KEY`.

The default provider preset is Ollama at `http://localhost:11434`.

## First Smoke Tests

### Test TTS without an LLM

```text
/speak hello from AbstractVoice
```

If Piper cannot speak, prefetch the default voice:

```bash
python -m abstractvoice download --piper en
```

### Test a chat turn

Start an OpenAI-compatible chat server, then type a normal message without a
leading slash. For Ollama, the default preset expects an OpenAI-compatible
`/v1/chat/completions` surface at `http://localhost:11434`.

Provider commands:

```text
/provider
/provider ollama
/provider http://localhost:1234
/models
/model <model-name>
/llm_stream on
```

### Test microphone input

```text
/voice stop
```

Speak a short phrase. While TTS is playing, say "ok stop" to interrupt playback.

If microphone startup fails, check OS microphone permission for your terminal or
IDE. On macOS this is usually under System Settings -> Privacy & Security ->
Microphone.

## Voice Modes

`/voice stop` is the best default when using speakers.

- `off`: microphone input disabled.
- `stop`: keep listening; while TTS plays, normal transcriptions are suppressed
  but the stop-phrase detector remains active.
- `wait`: strict turn-taking; microphone processing pauses while TTS plays.
- `full`: interrupt TTS on detected speech; best with a headset or AEC.
- `ptt`: push-to-talk session; SPACE starts/stops capture, ESC exits.

Commands:

```text
/voice off
/voice stop
/voice wait
/voice full
/voice ptt
/aec on
/aec off
```

AEC requires `pip install "abstractvoice[aec]"`.

## Playback And TTS Controls

```text
/tts on
/tts off
/speak <text>
/pause
/resume
/stop
/tts speed 1.1
/tts quality low
/tts quality standard
/tts quality high
/tts delivery buffered
/tts delivery streamed
```

`/tts delivery streamed` lowers time-to-first-audio when the selected engine can
deliver chunks progressively. Pair it with `/llm_stream on` for LLM streaming to
TTS streaming.

## Command Semantics

The REPL has a small preferred command model, with older direct commands kept
for compatibility:

- `/tts ...`: speaking configuration (`on/off`, engine, speed, quality,
  buffered/streamed delivery).
- `/voices ...`: voice discovery and selection. This is the preferred place for
  base TTS, profiles, cloned voices, and raw Piper model listing.
- `/clone...`: create and manage cloned voices.
- `/voice ...`: microphone mode (`off|wait|stop|ptt|full`).

Compatibility/direct commands still work:

- `/profile ...` is the direct profile command; prefer `/voices profiles` and
  `/voices profile <id>` in normal use.
- `/tts_voice ...` is the direct base/cloned selector; prefer `/voices base` and
  `/voices clone <id-or-name>`.
- `/setvoice ...` is the old Piper model selector; prefer `/voices models` for
  listing and `/voices setvoice <language.voice_id>` when you need that legacy
  raw selector.

There is no separate top-level `/profiles` command; use `/voices profiles`.

## Languages And Engines

Piper is the default reliable local TTS engine. It uses one cached voice per
language:

```bash
python -m abstractvoice download --piper fr
```

REPL commands:

```text
/language fr
/voices
/voices profiles
/voices profile <profile_id>
/voices base
/voices clone <id-or-name>
/voices models
/tts engine auto
/tts engine piper
/tts engine openai-compatible
/tts engine audiodit
/tts engine omnivoice
/stt_engine faster_whisper
/stt_engine openai-compatible
/whisper small
```

Engine notes:

- `piper`: default local TTS path; install `abstractvoice[voice]` or
  `abstractvoice[piper]`. Best first choice for reliable local speech.
- `openai` / `openai-compatible`: remote TTS/STT endpoints. Configure
  `OPENAI_API_KEY` for OpenAI or `ABSTRACTVOICE_REMOTE_BASE_URL` for compatible
  servers. Compatible servers may expose `GET /v1/audio/voices`; `/voices
  profiles` lists those remote profile/voice ids and `/voices profile <id>`
  uses the selected id as the remote speech `voice`.
- `audiodit`: optional heavy engine; direct/base TTS can sound distorted in
  `0.8.1`, while AudioDiT cloning remains the better-validated AudioDiT path.
- `omnivoice`: optional heavy engine for omnilingual TTS, voice design, and
  cloning. Stable reusable profiles are still being curated.
- `faster_whisper`: default local STT path; install `abstractvoice[voice]` or
  `abstractvoice[stt]`.

Current caveats are tracked in `docs/known-issues.md`.

## Voice Profiles

`/voices` is the preferred command for voice selection. It shows the current
base/cloned voice state, active profile, cloned voices, and the compatibility
commands that remain available for older workflows.

Voice profiles are engine-local presets. Select the engine first, then list or
apply profiles:

```text
/tts engine omnivoice
/voices profiles
/voices profile <profile_id>
/profile show
/profile reload
```

For OmniVoice, profiles may use either designed voice settings or a persistent
prompt cache. Prompt-cached profiles are the stronger route for keeping a voice
identity stable across turns.

For remote engines, profiles are provider voice ids. `openai` exposes the known
built-in voices such as `alloy` and `nova`; `openai-compatible` can discover
profiles from `GET /v1/audio/voices` (or from `ABSTRACTVOICE_REMOTE_TTS_VOICES`).

Manual OmniVoice voice design:

```text
/tts engine omnivoice
/omnivoice
/omnivoice instruct "female, young adult, moderate pitch"
/omnivoice seed 123
/omnivoice position_temperature 0
/omnivoice class_temperature 0
/speak Bonjour. Ceci est un test.
```

Exact waveform parity across CPU, CUDA, and MPS is not guaranteed. For stronger
portability, use a prompt-cached profile or a cloned voice.

## Voice Cloning

Install the extra and prefetch artifacts for the engine you want:

```bash
pip install "abstractvoice[cloning]"
python -m abstractvoice download --openf5

pip install "abstractvoice[audiodit]"
python -m abstractvoice download --audiodit

pip install "abstractvoice[omnivoice]"
python -m abstractvoice download --omnivoice

pip install "abstractvoice[chroma]"
python -m abstractvoice download --chroma
```

Readiness and downloads from inside the REPL:

```text
/cloning_status
/cloning_download f5_tts
/cloning_download chroma
/cloning_download audiodit
/cloning_download omnivoice
```

Clone from a file:

```text
/clone /path/to/reference.wav my_voice --engine omnivoice --text "Exact transcript of the reference audio."
/voices clone my_voice
/speak This is my cloned voice.
```

Interactive microphone cloning:

```text
/clone myvoice my_voice --engine f5_tts
/clone_use myvoice my_voice --engine f5_tts
```

Clone management:

```text
/clones
/clone_info <id-or-name>
/clone_ref <id-or-name>
/clone_set_ref_text <id-or-name> <exact transcript>
/clone_rename <id-or-name> <new-name>
/clone_rm <id-or-name>
/clone_rm_all --yes
/clone_export <id-or-name> <path>
/clone_import <path>
/clone_quality low|standard|high
/voices base
/voices clone <id-or-name>
```

Good reference audio is short, clean, single-speaker, and trimmed. Start with
4-10 seconds plus an exact transcript.

## History, Memory, And Reset

The REPL has three kinds of local state:

- In-memory LLM messages, sent to the chat provider.
- Terminal command history, used by the up/down arrows.
- Optional `.mem` files created only when you run `/save`.

Commands:

```text
/history
/history 50 --all
/history 10 --full
/clear
/reset
/save my-session
/load my-session
/tokens
```

`/clear` resets the LLM message history. `/reset` also resets the selected voice
state. Neither command deletes saved `.mem` files or terminal command history.

To delete terminal command history, remove `repl_history` from:

```bash
python - <<'PY'
import appdirs
print(appdirs.user_data_dir("abstractvoice"))
PY
```

More cache and reset details are in `docs/faq.md`.

## File Transcription

```text
/transcribe /path/to/audio.wav
```

The default path uses faster-whisper. Prefetch an STT model for offline REPL use:

```bash
python -m abstractvoice download --stt small
```

## Debugging

```text
/verbose on
/debug on
/cloning_status
/provider
/models
```

Debug mode saves synthesized WAVs under `untracked/generated_wavs/` so you can
inspect exactly what the TTS engine produced.

## Command Map

Basics:

- `/help`
- `/exit`, `/q`, `/quit`
- `/clear`
- `/history [n] [--all] [--full]`
- `/reset`
- `/debug [on|off|toggle]`
- `/verbose [on|off]`

TTS:

- `/tts`
- `/tts on|off`
- `/tts engine auto|piper|openai|openai-compatible|audiodit|omnivoice`
- `/tts quality low|standard|high`
- `/tts delivery buffered|streamed`
- `/tts speed <number>`
- `/voices`
- `/voices profiles`
- `/voices profile <profile_id>`
- `/voices base`
- `/voices clone <id-or-name>`
- `/voices models`
- `/omnivoice ...`
- `/language <code>`
- `/speak <text>`
- `/pause`, `/resume`, `/stop`

Compatibility shortcuts that still work:

- `/tts_engine auto|piper|openai|openai-compatible|audiodit|omnivoice`
- `/tts_quality low|standard|high`
- `/tts_delivery buffered|streamed`
- `/speed <number>`
- `/profile ...`
- `/tts_voice ...`
- `/setvoice ...` (prefer `/voices models` for listing and `/voices setvoice ...` for legacy selection)

Voice input:

- `/voice off|wait|stop|ptt|full`
- `/aec on|off [delay_ms]`

Cloning:

- `/cloning_status`
- `/cloning_download f5_tts|chroma|audiodit|omnivoice|openai-compatible`
- `/clone ...`
- `/clone_use ...`
- `/clones`
- `/clone_ref`, `/clone_set_ref_text`, `/clone_info`
- `/clone_rename`, `/clone_rm`, `/clone_rm_all --yes`
- `/clone_export`, `/clone_import`
- `/clone_quality low|standard|high`

Remote clone-compatible services can be used with
`/clone <path> --engine openai-compatible` after setting
`ABSTRACTVOICE_REMOTE_BASE_URL`; no local artifact download is needed.
OpenAI custom voice creation can be selected with `--engine openai`, but it is
org-gated and requires explicit consent configuration.

STT:

- `/stt_engine auto|faster_whisper|openai|openai-compatible|whisper`
- `/whisper <model>`
- `/transcribe <path>`

LLM:

- `/provider [preset-or-url]`
- `/models`
- `/model <name>`
- `/llm_stream on|off`
- `/system <prompt>`
- `/temperature <value>`
- `/max_tokens <n>`
- `/tokens`

AudioDiT utility:

- `/random [seed]`

Compatibility / advanced:

- `/profile list|show|reload|<profile_id>`
- `/tts_voice base|clone <id-or-name>`
- `/setvoice <language.voice_id>` for legacy Piper voice model selection
- `/list_languages`
- `/lang_info`
- `/tts_model`

## Troubleshooting

- Piper cannot speak: run `python -m abstractvoice download --piper en`.
- Mic cannot start: check OS microphone permission and default input device.
- LLM chat fails: run `/provider`, `/models`, and confirm the server is running.
- Optional cloning engine fails: run `/cloning_status` and prefetch with
  `/cloning_download <engine>`.
- AudioDiT direct TTS sounds distorted: use Piper for base TTS or validate the
  AudioDiT cloning path; see `docs/known-issues.md`.
- OmniVoice profiles drift: prefer prompt-cached profiles or cloned voices; see
  `docs/known-issues.md`.

For the supported library contract, use `docs/api.md`.
