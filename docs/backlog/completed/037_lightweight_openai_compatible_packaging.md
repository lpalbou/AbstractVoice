## Task 037: Lightweight OpenAI-compatible packaging for plugin/server hosts

**Date**: 2026-05-07  
**Status**: Completed  
**Priority**: P1  

---

## Main goals

- Make `pip install abstractvoice` a lightweight install that supports:
  - remote OpenAI/OpenAI-compatible TTS
  - remote OpenAI/OpenAI-compatible STT
  - remote voice/profile listing
  - compatible remote clone handles
  - AbstractCore capability plugin discovery and registration
- Move local voice runtimes and platform-sensitive audio I/O dependencies behind explicit extras.
- Let AbstractCore publish a media-capable, remote-first Docker image without implicitly installing Piper, faster-whisper, PortAudio/audio-device packages, or optional cloning stacks.

## Secondary goals

- Preserve the public `VoiceManager` API and AbstractCore plugin entry point.
- Preserve local-first desktop/REPL behavior through a clear `abstractvoice[voice]` or `abstractvoice[local]` install.
- Keep Python 3.9 support for the lightweight remote/plugin path where practical.
- Keep optional local engines first-class but explicit.
- Avoid implicit model downloads and surprise network/model runtime weight.

---

## Context / problem

AbstractVoice now has two valid deployment shapes:

- **Local voice library / REPL**: Piper TTS, faster-whisper STT, microphone/playback, optional cloning.
- **Remote/plugin host**: OpenAI-compatible audio adapters loaded inside AbstractCore or another server, with all synthesis/transcription happening upstream.

The code already has much of the runtime abstraction:

- TTS engine selection goes through `abstractvoice/adapters/tts_registry.py`.
- `tts_engine="openai"` and `tts_engine="openai-compatible"` use `OpenAICompatibleTTSAdapter`.
- `stt_engine="openai"` and `stt_engine="openai-compatible"` use `OpenAICompatibleSTTAdapter`.
- Remote voice profiles are exposed through `VoiceManager.get_profiles()`.
- Remote compatible clone creation is routed through `RemoteVoiceCloningEngine`.
- AbstractCore discovers the plugin through:
  - `pyproject.toml` -> `[project.entry-points."abstractcore.capabilities_plugins"]`
  - `abstractvoice = "abstractvoice.integrations.abstractcore_plugin:register"`

The packaging does not match that architecture yet. Current base dependencies include:

- `piper-tts`
- `huggingface_hub`
- `faster-whisper`
- `sounddevice`
- `soundfile`
- `webrtcvad`
- `numpy`
- `requests`
- `appdirs`

Python extras can add dependencies, but cannot remove base dependencies. While the local runtime stack remains in base, `abstractvoice[openai-compatible]` and `abstractvoice[remote]` cannot be lightweight in Docker or CI.

There is also an import-boundary issue: a lightweight install must be able to import `abstractvoice`, load the AbstractCore plugin entry point, and construct remote-only `VoiceManager` instances without importing local engine modules or requiring local audio packages.

---

## Investigation findings

### Runtime abstraction that already exists

- `VoiceManager` accepts `tts_engine`, `stt_engine`, `remote_base_url`, `remote_api_key`, and `remote_timeout_s`.
- TTS `auto` currently resolves to Piper; explicit `openai` and `openai-compatible` routes avoid Piper.
- Remote TTS uses a `requests`-based client and posts to `/audio/speech`.
- Remote STT uses multipart HTTP and posts to `/audio/transcriptions`.
- Remote profile listing uses `/audio/voices` and `/voices`.
- Remote clone support stores a remote voice id locally and reuses it for remote speech.
- Audio playback imports `sounddevice` only when a playback stream is opened, so `speak_to_bytes()` is naturally headless-friendly.

### Import blockers to fix before metadata migration

- `abstractvoice/audio/__init__.py` imports `record_wav`.
- `abstractvoice/audio/recorder.py` imports `soundfile` at module import time.
- `abstractvoice/__init__.py` imports `VoiceManager`, which imports `AdapterTTSEngine`, which reaches `tts_engine.py`, which imports `audio.resample`; importing `abstractvoice.audio.resample` executes `audio/__init__.py`.
- Result: `import abstractvoice` currently requires `soundfile` even for remote-only users.
- `abstractvoice/adapters/__init__.py` eagerly imports `PiperTTSAdapter` and `FasterWhisperAdapter`. This does not necessarily import external `piper` or `faster_whisper`, but it does import local adapter modules and weakens import-light guarantees.

### Docs and tests that currently assume local-first base install

- README and installation docs say `pip install abstractvoice` provides Piper/faster-whisper/audio I/O.
- `docs/dependencies.md` describes Piper/faster-whisper/audio I/O as core dependencies.
- `DependencyChecker.CORE_DEPS` lists the local voice stack as core.
- `tests/test_dependency_check.py` asserts the local voice stack is core.
- `tests/test_fresh_install.py` is really local Piper/model-download coverage, not lightweight-base coverage.

---

## Constraints

- Do not redesign `VoiceManager`.
- Do not move OpenAI-compatible server routes into AbstractVoice; AbstractCore owns production HTTP server routing.
- Keep remote adapters and plugin registration importable from the base package.
- Keep `tts_engine="auto"` behavior stable for local installs: `auto` remains Piper-first.
- For lightweight installs, require explicit remote engine selection or AbstractCore config.
- Keep local audio playback/capture optional for server images.
- Keep cloning engines explicit and avoid installing torch-backed stacks by default.
- Keep install errors actionable and exact: tell users which extra to install.

---

## Research, options, and references

### Option A: Keep base local-first and document custom Docker images

Pros:
- Preserves current `pip install abstractvoice` local desktop behavior.
- Minimal packaging churn.

Cons:
- Does not solve AbstractCore media-image size/runtime sensitivity.
- Remote-only users still install local model and device packages.
- No extra can remove `piper-tts`, `faster-whisper`, `sounddevice`, `soundfile`, or `webrtcvad` from base.

### Option B: Make base lightweight and move local runtimes behind extras

Pros:
- Clean single-distribution model.
- Lets AbstractCore depend on AbstractVoice in a remote-first media image.
- Makes deployment weight explicit.
- Aligns with remote TTS/STT/profile/clone paths already implemented.

Cons:
- Changes user expectation for `pip install abstractvoice`.
- Requires docs, dependency checker, tests, and install hints to move from "local core" to "lightweight core + local extras".

### Option C: Add a separate lightweight distribution

Examples:
- `abstractvoice-core`
- `abstractvoice-remote`

Pros:
- Preserves current `abstractvoice` local-first behavior.
- Gives AbstractCore a small dependency.

Cons:
- Splits package ownership and docs.
- Creates confusing import-path and entry-point ownership questions.
- Increases release complexity for little runtime benefit.

### Option D: Keep base dependencies but rely on `--no-deps` or Docker pruning

Pros:
- Avoids user-facing packaging change.

Cons:
- Fragile and non-standard.
- Pushes dependency correctness onto AbstractCore Docker builds.
- Breaks normal resolver expectations.

---

## Decision

**Chosen approach**: Option B.

Make `abstractvoice` lightweight by default, and make local engines explicit extras. Keep the package single-distribution and preserve the public API.

**Why**:
- It is the only clean single-distribution path that supports AbstractCore's remote-first media image.
- The runtime architecture already supports remote TTS/STT/profile/clone paths.
- The known blockers are import-boundary and packaging/test/docs migration issues, not a `VoiceManager` redesign.

---

## Chosen install profiles

### Base

`abstractvoice`

Recommended base dependencies:

- `numpy>=1.24.0`
- `requests>=2.31.0`
- `appdirs>=1.4.0`

Rationale:
- `numpy` is still part of the adapter/audio-array surface and remote bytes/array helpers.
- `requests` is the HTTP runtime for remote audio.
- `appdirs` is small and keeps remote clone-handle storage usable from the base package. If clone-store path handling is later refactored to avoid `appdirs`, this can be revisited.

Base must not include:

- `piper-tts`
- `faster-whisper`
- `sounddevice`
- `soundfile`
- `webrtcvad`
- torch/torchaudio/torchvision
- `f5-tts`
- `omnivoice`
- Chroma/AudioDiT runtime stacks

### Remote intent extras

- `abstractvoice[openai]`: no-op intent extra for hosted OpenAI audio unless a future OpenAI SDK dependency is adopted.
- `abstractvoice[openai-compatible]`: no-op intent extra for generic compatible providers.
- `abstractvoice[remote]`: alias no-op intent extra.

Keep both `openai` and `openai-compatible`. They are technically redundant while
both are empty and remote adapters use the shared `requests` client, but they
encode different installer intent:

- `openai`: hosted OpenAI API, default OpenAI base URL, OpenAI API key/env semantics, and possible future OpenAI SDK/helper dependency.
- `openai-compatible`: any compatible `/v1` provider, including AbstractCore Server, LM Studio-style gateways, or custom proxies.

Because empty extras add no install weight, keeping both is clearer than forcing
hosted OpenAI and generic compatible providers into one ambiguous extra.

### Local engine extras

- `abstractvoice[piper]` or `abstractvoice[local-tts]`
  - `piper-tts>=1.2.0`
  - Include `huggingface_hub` only if implementation still directly needs it for this path. Current Piper downloads use direct HTTPS, so this should be audited instead of assumed.
- `abstractvoice[stt]` or `abstractvoice[local-stt]`
  - `faster-whisper>=0.10.0`
  - `soundfile>=0.12.1` if local STT/transcode/test paths require it.
- `abstractvoice[audio-io]`
  - `sounddevice>=0.4.6`
  - `soundfile>=0.12.1`
  - `webrtcvad>=2.0.10`
- `abstractvoice[voice]`
  - local TTS + local STT + audio I/O
- `abstractvoice[local]`
  - alias for `voice`

### Compatibility aliases

Keep these working for at least one release cycle:

- `voice-full`: alias for `voice`
- `core-stt`: alias for `stt`
- `audio-only`: alias for `audio-io`

### All extra

Keep `abstractvoice[all]` as a first-class convenience bundle.

Recommended meaning:

- local voice path (`voice`)
- audio effects (`audio-fx`)
- common local cloning/AEC extras where resolver-safe:
  - `cloning`
  - `aec`
- exclude GPU-heavy or highly specialized stacks unless explicitly requested:
  - Chroma
  - AudioDiT
  - OmniVoice

Rationale:
- `all` should mean "common local features" rather than "every possible model stack".
- Keeping Chroma/AudioDiT/OmniVoice explicit avoids surprising torch/transformers installs and Python-version resolver conflicts.
- Existing users already have an `all` extra today, so removing it would be unnecessary churn.

### Web extras

- `abstractvoice[web]`
  - FastAPI local web example only; remote-compatible and engine-agnostic.
- `abstractvoice[web,voice]`
  - web example plus local Piper/faster-whisper/audio I/O.
- `abstractvoice[web-cloning]`
  - web example plus OpenF5 cloning runtime.
- `abstractvoice[web-audiodit]`
  - web example plus AudioDiT runtime.
- `abstractvoice[web-omnivoice]`
  - web example plus OmniVoice runtime.
- `abstractvoice[web-chroma]`
  - web example plus Chroma runtime.
- `abstractvoice[web-full]`
  - explicit heavy web/demo bundle.

### Cloning extras

- Remote compatible clone handles should work from base as long as `appdirs` remains in base.
- `abstractvoice[cloning]` remains the local OpenF5 cloning extra.
- `abstractvoice[chroma]`, `abstractvoice[audiodit]`, and `abstractvoice[omnivoice]` remain explicit heavy engine extras.
- Audit direct `huggingface_hub` imports in OpenF5, Chroma, AudioDiT, and OmniVoice prefetch paths. Do not rely on transitive dependencies if a public prefetch command imports `huggingface_hub` directly.

---

## Install recipes to document

### Remote-only library or plugin host

```bash
pip install abstractvoice
```

```python
from abstractvoice import VoiceManager

vm = VoiceManager(
    tts_engine="openai-compatible",
    stt_engine="openai-compatible",
    remote_base_url="http://localhost:8000/v1",
)
```

### Hosted OpenAI audio

```bash
pip install "abstractvoice[openai]"
export OPENAI_API_KEY=...
```

```python
from abstractvoice import VoiceManager

vm = VoiceManager(tts_engine="openai", stt_engine="openai")
```

### Local desktop / REPL voice

```bash
pip install "abstractvoice[voice]"
abstractvoice-prefetch --piper en
abstractvoice-prefetch --stt small
abstractvoice
```

### Remote web example

```bash
pip install "abstractvoice[web]"
abstractvoice web \
  --tts-engine openai-compatible \
  --stt-engine openai-compatible \
  --remote-base-url http://localhost:8000/v1
```

### Local web lab

```bash
pip install "abstractvoice[web,voice]"
abstractvoice web
```

### Remote compatible cloning

```python
from abstractvoice import VoiceManager

vm = VoiceManager(
    tts_engine="openai-compatible",
    cloning_engine="openai-compatible",
    remote_base_url="http://localhost:8000/v1",
)
voice_id = vm.clone_voice("reference.wav", name="demo", engine="openai-compatible")
wav = vm.speak_to_bytes("Hello.", voice=voice_id)
```

The compatible server must expose the configured clone route, defaulting to `POST /voice/clone`, and return `{"voice_id": "..."}` or `{"id": "..."}`.

### Lightweight CI

```bash
pip install -e ".[test]"
pytest -q tests/test_remote_openai_compatible_adapters.py tests/test_abstractcore_plugin.py
```

### Local voice CI

```bash
pip install -e ".[test,voice]"
pytest -q tests/test_piper_adapter.py tests/test_faster_whisper_adapter.py
```

### AbstractCore remote media image

```bash
pip install "abstractcore[server]" abstractvoice
```

Configure the plugin host with:

- `voice_tts_engine=openai-compatible`
- `voice_stt_engine=openai-compatible`
- `voice_remote_base_url=http://.../v1`
- optional `voice_remote_api_key=...`

---

## Dependencies

- **Backlog tasks**:
  - `docs/backlog/planned/013_openai_compatible_audio_endpoint.md`
  - `docs/backlog/planned/027_refresh_dependency_check.md`
  - `docs/backlog/planned/036_voice_profile_abstraction.md`
  - `docs/backlog/proposed/2026-05-07_lightweight_openai_compatible_packaging.md`
- **Key implementation files**:
  - `pyproject.toml`
  - `abstractvoice/__init__.py`
  - `abstractvoice/adapters/__init__.py`
  - `abstractvoice/audio/__init__.py`
  - `abstractvoice/audio/recorder.py`
  - `abstractvoice/tts/tts_engine.py`
  - `abstractvoice/recognition.py`
  - `abstractvoice/vm/stt_mixin.py`
  - `abstractvoice/integrations/abstractcore_plugin.py`
  - `abstractvoice/dependency_check.py`

---

## Implementation plan

### Phase 1: Add packaging/import tests first

- Add metadata tests proving base dependencies exclude local runtime stacks.
- Add metadata tests proving local extras include expected runtime stacks.
- Add an import-blocking test that simulates missing:
  - `piper`
  - `faster_whisper`
  - `sounddevice`
  - `soundfile`
  - `webrtcvad`
- Assert these still succeed:
  - `import abstractvoice`
  - `import abstractvoice.integrations.abstractcore_plugin`
  - AbstractCore entry point discovery via `importlib.metadata.entry_points(...)`

### Phase 2: Fix import-light boundaries

- Stop `abstractvoice/audio/__init__.py` from importing `record_wav` eagerly, or move `soundfile` import inside `record_wav()`.
- Update `abstractvoice/audio/recorder.py` so both `sounddevice` and `soundfile` failures point to `abstractvoice[audio-io]` or `abstractvoice[voice]`.
- Make `abstractvoice/adapters/__init__.py` import-light:
  - export base interfaces directly
  - expose local adapter classes through lazy `__getattr__`, or stop re-exporting concrete adapters from package import
- Re-run import-blocking tests.
- Consider making `abstractvoice/__init__.py` lazy only if the above changes are not enough to keep base import clean.

### Phase 3: Move dependencies into extras

- Reduce `[project].dependencies` to the selected lightweight base set.
- Add canonical extras:
  - `piper`
  - `local-tts`
  - `stt`
  - `local-stt`
  - `audio-io`
  - `voice`
  - `local`
- Preserve compatibility aliases:
  - `voice-full`
  - `core-stt`
  - `audio-only`
- Keep `all` as a first-class common local bundle:
  - `voice`
  - `audio-fx`
  - resolver-safe `cloning`
  - resolver-safe `aec`
  - not Chroma/AudioDiT/OmniVoice unless explicitly requested
- Keep remote intent extras:
  - `openai`
  - `openai-compatible`
  - `remote`
- Audit `huggingface_hub` direct imports and assign it to the extras whose public commands require it.
- Review Python markers:
  - F5/OpenF5, Chroma, OmniVoice: Python 3.10+
  - AEC: Python 3.11+
  - AudioDiT: keep existing Python 3.9-compatible split only if it still resolves cleanly.

### Phase 4: Make local failures actionable

- Explicit local TTS without `piper-tts` should say:
  - `pip install "abstractvoice[piper]"`
  - or `pip install "abstractvoice[voice]"`
- Explicit local STT without `faster-whisper` should say:
  - `pip install "abstractvoice[stt]"`
  - or `pip install "abstractvoice[voice]"`
- `listen()` / microphone capture failures should say:
  - `pip install "abstractvoice[audio-io]"`
  - or `pip install "abstractvoice[voice]"`
- Playback failures should say:
  - `pip install "abstractvoice[audio-io]"`
  - or `pip install "abstractvoice[voice]"`
- Cloning failures should point to exact engine extras:
  - `abstractvoice[cloning]`
  - `abstractvoice[chroma]`
  - `abstractvoice[audiodit]`
  - `abstractvoice[omnivoice]`

### Phase 5: Update dependency checker

- Replace "core voice stack" with profiles:
  - `base`
  - `remote`
  - `local_tts`
  - `local_stt`
  - `audio_io`
  - `local_voice`
  - `web`
  - `cloning`
- Keep audio device probing optional and skip cleanly when `sounddevice` is absent.
- Update recommendations so missing local packages are not reported as a broken base install.

### Phase 6: Update docs

- README:
  - Split install into lightweight remote/plugin base and local voice install.
  - Update AbstractCore recipe to avoid implying local runtimes are included.
- `docs/installation.md`:
  - Add the install matrix above.
  - Update optional extras descriptions.
  - Update Python-version notes.
- `docs/dependencies.md`:
  - Rename current core dependency section to lightweight base.
  - Move Piper/faster-whisper/audio I/O into extras sections.
- `docs/getting-started.md`:
  - Require `abstractvoice[voice]` before local `/speak` or present a remote-first smoke test.
- `docs/api.md`:
  - Clarify that `tts_engine="auto"` remains Piper-first but requires local extras in lightweight installs.
- `docs/development.md`:
  - Add lightweight CI and local-voice CI commands.

### Phase 7: Test profiles and release notes

- Mark local engine tests so they skip or run only when relevant extras are installed.
- Rename or re-scope fresh-install tests as local voice tests.
- Add a migration note:
  - old local behavior: `pip install abstractvoice`
  - new local behavior: `pip install "abstractvoice[voice]"`
  - remote/plugin behavior: `pip install abstractvoice`

---

## Success criteria

- `pip install abstractvoice` does not install Piper, faster-whisper, sounddevice, soundfile, webrtcvad, torch, F5-TTS, Chroma, AudioDiT, or OmniVoice.
- Base install supports remote/OpenAI-compatible TTS and STT with explicit remote engine selection.
- Base install supports AbstractCore capability plugin discovery and registration.
- Remote-compatible profile listing works from base install.
- Remote-compatible clone handles work from base install if `appdirs` remains in base.
- `pip install "abstractvoice[voice]"` restores the expected local Piper + faster-whisper + microphone/playback path.
- Local runtime failures produce exact extra install hints.
- Python 3.9 can install the lightweight remote/plugin path even when optional engines require Python 3.10+ or 3.11+.

---

## Test plan

- `pytest -q`
- Focused lightweight tests:
  - packaging metadata base/extras tests
  - import-blocking subprocess/import-hook tests
  - `tests/test_remote_openai_compatible_adapters.py`
  - `tests/test_abstractcore_plugin.py`
- Focused local voice tests:
  - `tests/test_piper_adapter.py`
  - `tests/test_faster_whisper_adapter.py`
  - `tests/test_dependency_check.py`
- Subprocess smoke:
  - `python -c "import abstractvoice; import importlib.metadata as m; list(m.entry_points(group='abstractcore.capabilities_plugins'))"`
- Optional live smoke, gated by env:
  - hosted OpenAI TTS/STT
  - configured OpenAI-compatible remote audio server
  - AbstractCore server media image build

---

## Non-goals

- Do not remove local TTS, STT, streaming, voice cloning, AEC, or web-demo capabilities.
- Do not make AbstractVoice own production OpenAI-compatible HTTP server routing.
- Do not make AbstractVoice a standalone LLM server.
- Do not include model weights in Python package metadata.

---

## Report

### Summary

- Implemented Option B: the base package is now lightweight and remote/plugin friendly.
- Moved local Piper, faster-whisper, audio I/O, and local file/audio runtime dependencies into explicit extras.
- Kept `abstractvoice[openai]`, `abstractvoice[openai-compatible]`, and `abstractvoice[remote]` as no-op intent extras because remote adapters use the base `requests` dependency today.
- Added `abstractvoice[all]` as the common local feature bundle while keeping Chroma, AudioDiT, and OmniVoice behind explicit heavy-engine extras.
- Made `abstractvoice.audio` and `abstractvoice.adapters` lazy enough that base imports and AbstractCore plugin discovery do not pull local speech packages.
- Updated dependency diagnostics, install hints, README, installation, getting-started, dependencies, API, FAQ, REPL, changelog, and release metadata for the new install contract.
- During the final independent check, added `soundfile` explicitly to optional engine extras that import it, so those extras no longer rely on the old base dependency set.

### Validation

- Focused: `pytest -q tests/test_dependency_check.py tests/test_lightweight_import_boundaries.py tests/test_remote_openai_compatible_adapters.py tests/test_abstractcore_plugin.py` -> 20 passed.
- Full suite: `pytest -q` -> 135 passed, 8 skipped.
- Known warnings: `webrtcvad`/`pkg_resources` deprecations in the full-mode echo-gate test.
