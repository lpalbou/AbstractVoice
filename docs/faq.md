# FAQ

## Setup

### What Python versions are supported?

AbstractVoice supports Python `>=3.9`. The lightweight base install and the
remote/OpenAI-compatible path are supported on Python 3.9. Local
Piper/faster-whisper extras, Supertonic 3, and the web example are also
supported on Python 3.9. OpenF5/F5-TTS, Chroma, and OmniVoice require Python
3.10+ because their upstream runtimes do; AEC requires Python 3.11+.

### Do I need system speech dependencies?

The bare package install does not require local speech engines. Install
`abstractvoice[apple]` or `abstractvoice[gpu]` for the platform local
Piper/Supertonic/faster-whisper/cloning path, or compose granular extras such as
`abstractvoice[supertonic,stt,audio-io]`.
Piper runs
through ONNX Runtime, so it does not require system packages such as
`espeak-ng`.

Microphone and speaker I/O use `sounddevice` / PortAudio and are installed by
`abstractvoice[audio-io]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`. On some Linux systems you
may need PortAudio packages from the OS package manager. See
`docs/installation.md` for platform notes.

### I installed the package but `abstractvoice` is not found. What should I check?

Install into the Python environment that runs your shell:

```bash
python -m pip install abstractvoice
python -m abstractvoice cli --verbose
```

From a source checkout, `python -m abstractvoice cli --verbose` is the most
reliable way to start without depending on console-script path setup.

## Local State, History, And Caches

### How do I clear the current REPL conversation history?

Inside the REPL:

```text
/clear
```

`/clear` resets the in-memory LLM message list back to the system prompt. That
is the history sent to the configured OpenAI-compatible chat endpoint.

Related commands:

```text
/history
/history 50 --all
/history 10 --full
/reset
```

`/reset` clears the chat history and also resets the active voice selection back
to the default TTS path.

### Does the REPL automatically persist LLM conversation history?

No. The LLM message history is in memory unless you explicitly save it:

```text
/save my-session
/load my-session
```

`/save my-session` writes `my-session.mem` in the current working directory.
Delete that `.mem` file if you no longer want the saved conversation.

### Why do my previous typed commands still appear with the up arrow?

That is terminal command history, not LLM chat history. The REPL stores it as a
small readline history file so the up/down arrows work across sessions.

Find the app-data directory with:

```bash
python - <<'PY'
import appdirs
print(appdirs.user_data_dir("abstractvoice"))
PY
```

Then delete `repl_history` inside that directory.

Common locations:

- macOS: `~/Library/Application Support/abstractvoice/repl_history`
- Linux: `~/.local/share/abstractvoice/repl_history`
- Windows: `%LOCALAPPDATA%\\abstractvoice\\abstractvoice\\repl_history` or the path printed by `appdirs`

### Where are models, cloned voices, and prompt caches stored?

Default local state:

- Piper voices: `~/.piper/models`
- Supertonic 3: `~/.cache/abstractvoice/supertonic-3`
- faster-whisper models: Hugging Face cache, usually `~/.cache/huggingface`
- OpenF5 artifacts: `~/.cache/abstractvoice/openf5`
- Chroma artifacts and prompt cache: `~/.cache/abstractvoice/chroma`
- Cloned voices: `appdirs.user_data_dir("abstractvoice")/cloned_voices`
- OmniVoice persistent prompt profiles: `appdirs.user_data_dir("abstractvoice")/omnivoice_prompt_cache`
- REPL terminal command history: `appdirs.user_data_dir("abstractvoice")/repl_history`
- Saved REPL memories: wherever you ran `/save`, with a `.mem` extension

### How do I reset local state completely?

Stop any running AbstractVoice process, then remove only the state you want to
purge.

For a full local reset on macOS/Linux:

```bash
rm -rf ~/.piper/models
rm -rf ~/.cache/abstractvoice/supertonic-3
rm -rf ~/.cache/abstractvoice/openf5
rm -rf ~/.cache/abstractvoice/chroma
python - <<'PY'
import shutil
from pathlib import Path
import appdirs

root = Path(appdirs.user_data_dir("abstractvoice"))
for name in ("cloned_voices", "omnivoice_prompt_cache", "repl_history"):
    path = root / name
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
PY
```

Be careful with the Hugging Face cache. It is shared by many AI tools. Prefer
deleting specific model folders under `~/.cache/huggingface/hub` instead of
removing the whole cache unless you really want to reset everything.

## Offline-First Downloads

### Why does the REPL not download models automatically?

The REPL starts `VoiceManager(..., allow_downloads=False)`. Interactive sessions
should not surprise you with multi-GB downloads. Prefetch what you need:

```bash
abstractvoice-prefetch --piper en
abstractvoice-prefetch --supertonic
abstractvoice-prefetch --stt small
```

The library default is remote-first: `VoiceManager()` uses OpenAI audio and
reads `OPENAI_API_KEY`. Local adapters still respect `allow_downloads=True`
when you explicitly select them.

### What should I prefetch first?

Most users should start with:

```bash
python -m abstractvoice download --piper en
python -m abstractvoice download --supertonic
python -m abstractvoice download --stt small
```

Optional heavy engines:

```bash
python -m abstractvoice download --openf5
python -m abstractvoice download --chroma
python -m abstractvoice download --audiodit
python -m abstractvoice download --omnivoice
```

Install the matching extra before prefetching optional engines, for example
`pip install "abstractvoice[omnivoice]"`.

## REPL Usage

### Can I use the REPL without an LLM server?

Yes. Use `/speak <text>` to test the configured TTS engine directly:

```text
/speak hello from AbstractVoice
```

Normal chat messages call the configured OpenAI-compatible LLM endpoint. The
default provider preset is Ollama at `http://localhost:11434`.

### How do I enable microphone input?

Microphone input is off by default. Start with:

```bash
OPENAI_API_KEY=... abstractvoice --voice-mode stop
```

Or enable it inside the REPL:

```text
/voice stop
```

`stop` mode is the recommended hands-free mode on speakers. During TTS it keeps
a stop-phrase detector active, so you can say "ok stop" to cut playback.

### What are the voice modes?

- `off`: no microphone input.
- `stop`: recommended on speakers; normal transcriptions pause during TTS, but
  "ok stop" can interrupt playback.
- `wait`: strict turn-taking; microphone processing pauses during TTS.
- `full`: barge-in on speech; best with a headset or acoustic echo cancellation.
- `ptt`: push-to-talk profile for short intentional captures.

### How do I hide or inspect model reasoning from local LLMs?

The REPL removes `<think>...</think>` blocks before printing, storing, or
speaking an assistant response. This keeps spoken output clean.

Use `/history --full` to inspect the exact message list that remains in memory.

## Library And Server Integration

### Can I use AbstractVoice without the REPL?

Yes. The main API is `VoiceManager`:

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
wav = vm.speak_to_bytes("Hello.", format="wav")
text = vm.transcribe_file("hello.wav")
```

`VoiceManager()` reads `OPENAI_API_KEY` by default. For local/offline use,
install `abstractvoice[supertonic,stt]` or a platform profile, then create
`VoiceManager(tts_engine="supertonic", stt_engine="faster_whisper")`.

For long-lived apps and servers, create one `VoiceManager` per configuration and
reuse it. Heavy engines are expensive to load repeatedly.

### Does AbstractVoice ship an HTTP server?

AbstractVoice ships a small local FastAPI web example:

```bash
pip install "abstractvoice[web]"
OPENAI_API_KEY=... abstractvoice web --port 5000
```

Open `http://127.0.0.1:5000` and use it to test discussion playback,
assistant/user voice selection, TTS, audio-file transcription, and a tiny
example dialogue panel backed by an OpenAI-compatible local provider such as
Ollama. It lazy-loads `VoiceManager` on the first audio request.

If selecting a cloned voice appears slow, that is usually the cloning runtime
loading weights and preparing prompt/runtime caches. The web example shows a
busy overlay during selection/preload and synthesis so the browser does not look
stuck.

For production HTTP APIs, use AbstractCore Server. AbstractVoice is still the
voice I/O library; AbstractCore owns the OpenAI-compatible server surface in the
AbstractFramework ecosystem.

Install both packages in the same environment:

```bash
pip install "abstractcore[server]" abstractvoice
OPENAI_API_KEY=... python -m abstractcore.server.app
```

AbstractCore discovers the AbstractVoice capability plugin and can expose
OpenAI-compatible audio endpoints such as:

- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`

Direct Python integration remains fully supported through `VoiceManager`.

## Languages And Engines

### How do I switch language?

Use `VoiceManager(language="fr")` or:

```python
vm.set_language("fr")
```

In the REPL:

```text
/language fr
```

For Piper, each language needs a cached voice:

```bash
python -m abstractvoice download --piper fr
```

### Which TTS engine should I use?

- Remote OpenAI-compatible engines are the lightest path for server and plugin
  deployments, and hosted OpenAI is the default `VoiceManager()` path.
- Supertonic is the recommended fixed-profile local ONNX TTS path; install
  `abstractvoice[supertonic]`, `abstractvoice[apple]`, or
  `abstractvoice[gpu]`, prefetch `--supertonic`, then select
  `tts_engine="supertonic"`.
- Piper remains available as a smaller local TTS fallback; install
  `abstractvoice[piper]`, `abstractvoice[apple]`, or `abstractvoice[gpu]`.
- faster-whisper is the local STT path; install `abstractvoice[stt]`,
  `abstractvoice[apple]`, or `abstractvoice[gpu]`.
- OmniVoice is the recommended/default local cloning backend. OpenF5, Chroma,
  and AudioDiT remain optional heavier alternatives for explicit selection.
- AudioDiT is best treated as an EN/ZH-focused experimental TTS/cloning engine
  in this integration.
- OmniVoice is the main optional path for omnilingual speech and voice design,
  but stable reusable voice profiles still need more validation.

For current caveats, see `docs/known-issues.md`.

### OmniVoice fails to import with `operator torchvision::nms does not exist`. What now?

OmniVoice uses the torch/torchaudio/torchvision stack. If another package
installed an incompatible `torchvision`, reinstall the matching build. For the
torch 2.8 family this is typically:

```bash
python -m pip install --upgrade --force-reinstall "torchvision==0.23.*"
```

If you do not need `torchvision`, uninstalling it can also avoid the import path:

```bash
python -m pip uninstall torchvision
```

## Voice Cloning

### Is voice cloning included in the base install?

No. Voice cloning is optional:

```bash
pip install "abstractvoice[cloning]"   # OpenF5
pip install "abstractvoice[chroma]"    # Chroma, GPU-heavy
pip install "abstractvoice[audiodit]"  # AudioDiT
pip install "abstractvoice[omnivoice]" # OmniVoice, recommended/default
```

Artifacts are still downloaded explicitly with `python -m abstractvoice download ...`
or `abstractvoice-prefetch ...`.

### How should I prepare reference audio?

Good reference audio matters more than most generation knobs:

- Use one speaker.
- Avoid music, effects, and background noise.
- Start with 4-10 seconds of clean speech.
- Trim leading and trailing silence.
- Provide an exact transcript when the engine accepts `reference_text`.

Engine notes:

- OpenF5 normalizes to 24 kHz mono and clips references to 15 seconds.
- AudioDiT normalizes to 24 kHz and clips prompt audio to 15 seconds.
- OmniVoice currently supports one reference audio file for a clone prompt.
- Chroma normalizes prompt audio to 24 kHz mono and clips to 30 seconds.

### How can I remove background music from a reference clip before cloning?

AbstractVoice does not bundle a music-removal model. A practical reproducible
workflow is:

1. Use `ffmpeg` to extract audio from the source video or container.
2. Use `demucs` to separate the track into `vocals.wav` and `no_vocals.wav`.
3. Use the isolated vocal stem as your cloning reference, then trim it to the
   cleanest 6-15 second speech segment you can find.

Install the tools first:

```bash
python -m pip install demucs
```

You also need a working `ffmpeg` binary on `PATH` from your OS package manager
or Homebrew.

Example: extract the audio from a local MP4 into WAV:

```bash
ffmpeg -y -i input.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 extracted.wav
```

Example: run vocal separation:

```bash
demucs --two-stems=vocals -o separated extracted.wav
```

That writes two files under a model-named output directory, typically:

- `separated/htdemucs/extracted/vocals.wav`
- `separated/htdemucs/extracted/no_vocals.wav`

`--two-stems=vocals` tells Demucs to keep a vocal stem and a combined
everything-else stem. On first run, Demucs may download model weights.

If the isolated vocal stem still has too much bleed, try the older
`mdx_extra_q` model:

```bash
demucs -n mdx_extra_q --two-stems=vocals -o separated extracted.wav
```

After separation, trim to the cleanest speech window:

```bash
ffmpeg -y -i separated/htdemucs/extracted/vocals.wav -ss 00:00:12 -t 8 reference_clean.wav
```

Then clone from the cleaned file. REPL example:

```text
/clone /path/to/reference_clean.wav my_voice --engine omnivoice --text "Exact transcript of the reference audio."
```

Library example:

```python
from abstractvoice import VoiceManager

vm = VoiceManager(cloning_engine="omnivoice")
voice_id = vm.clone_voice(
    "reference_clean.wav",
    name="my_voice",
    reference_text="Exact transcript of the reference audio.",
)
```

Practical notes:

- Separation is not perfect. Some music bleed or reverb can remain.
- A short clean stem usually clones better than a longer noisy one.
- Use only audio you have the rights and consent to clone.

### Can I move cloned voices between machines?

Yes. Use the REPL export/import commands:

```text
/clone_export <id-or-name> <path>
/clone_import <path>
```

The target machine still needs the same cloning engine installed and its weights
prefetched.

### Can OmniVoice make a stable portable voice preset?

There are two levels of stability:

- Designed voices use `instruct`, seed, temperatures, and generation settings.
  This is useful, but exact identity can vary across hardware and model builds.
- Prompt-conditioned voices use reference audio or cached prompt tokens. This is
  the stronger route for reusable voices.

For practical persistence, prefer a profile that builds a persistent prompt
cache, or create/export an OmniVoice clone from reference audio.

## Known Issues And Bug Tracking

### Where are known bugs tracked?

GitHub Issues should be the canonical tracker for active bugs. Use labels such
as `bug`, `known-issue`, `engine:audiodit`, `engine:omnivoice`, and
`release:0.8.1`.

`docs/known-issues.md` is the curated release-facing mirror: it lists the known
issues users should see before choosing an engine, plus the current workaround.
When a bug is fixed, close the GitHub issue, move the note to `CHANGELOG.md`,
and remove it from the active known-issues list.

### Are there known engine caveats in this release?

Yes:

- AudioDiT direct/base TTS can sound distorted in this release. AudioDiT cloning
  is still the better-validated AudioDiT path.
- OmniVoice stable reusable profiles are still being curated. For stronger
  persistence, use cached prompt profiles or exported cloned voices.

See `docs/known-issues.md` for the current list.

## Licensing

### Is AbstractVoice MIT licensed? What about voices and model weights?

The library code is MIT licensed. Model weights and voice files are separate
assets with their own licenses and distribution terms.

Read `docs/voices-and-licenses.md` before redistributing models, voices, or
generated voice assets.
