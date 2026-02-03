# AbstractVoice Architecture

This document explains how AbstractVoice works internally, its components, and how they communicate to provide seamless voice interactions with immediate pause/resume capabilities.

For acronyms used here (TTS/STT/VAD/VM/MM), see `docs/acronyms.md`.

## Overview

AbstractVoice is designed as a modular voice interaction system that bridges text generation systems (LLMs) with voice input/output. The architecture prioritizes real-time responsiveness, especially for pause/resume functionality.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Your App      │    │  VoiceManager   │    │   LLM/API       │
│                 │◄──►│  (Orchestrator) │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Audio Pipeline │
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            ┌─────────────────┐ ┌─────────────────┐
            │   TTSEngine     │ │ VoiceRecognizer │
            │ (Text→Speech)   │ │ (Speech→Text)   │
            └─────────────────┘ └─────────────────┘
                    │                   │
                    ▼                   ▼
        ┌─────────────────────┐ ┌─────────────────┐
        │NonBlockingAudioPlayer│ │ Whisper + VAD   │
        │  (OutputStream)     │ │                 │
        └─────────────────────┘ └─────────────────┘
```

## How AbstractVoice integrates with AbstractCore / AbstractFramework

AbstractVoice is usable in two ways:

1) **Standalone library** (direct Python use): you can use `VoiceManager` directly for STT/TTS.
2) **AbstractCore capability plugin**: when installed alongside `abstractcore`, AbstractVoice registers:
   - `core.voice` (speech-oriented UX surface: STT/TTS)
   - `core.audio` (audio capability surface used by AbstractCore’s `audio_policy`)

This integration is intentionally plugin-based (ADR-0028) so `abstractcore` stays dependency-light by default (ADR-0001).

Implementation:
- Plugin entry: `abstractvoice/abstractvoice/integrations/abstractcore_plugin.py`

### Library mode vs framework mode

- **Library mode**: calls return strings/bytes; the caller decides where to store outputs.
- **Framework mode (runtime/gateway)**: when an `artifact_store` is provided, large audio outputs are stored durably and referenced by artifact id (`{"$artifact":"..."}`).

User-facing policy note:
- Audio input is not “just speech”; AbstractCore defaults to `audio_policy='native_only'` to avoid silent STT on music/signals.
- STT is enabled explicitly via `audio_policy='speech_to_text'` (or configured `audio.strategy`).

See also:
- `docs/guide/capability-audio.md`
- `docs/adr/0028-capabilities-plugins-and-library-framework-modes.md`

## Core Components

### 1. VoiceManager (Orchestrator)

**Location**:
- Public façade: `abstractvoice/voice_manager.py`
- Implementation: `abstractvoice/vm/manager.py` (built from small mixins in `abstractvoice/vm/`)

The central coordinator that manages all voice interactions:

```python
class VoiceManager:
    # See `abstractvoice/vm/manager.py` for the real implementation.
    # Internally, it wires TTS lifecycle callbacks to listening behavior via `abstractvoice/vm/core.py`.
```

**Key Responsibilities:**
- Initializes and coordinates TTS and STT components
- Manages voice modes (full, wait, off)
- Handles TTS interruption during voice recognition
- Provides high-level API for applications

**Communication Patterns:**
- **Callbacks**: Uses callback functions to coordinate between TTS and STT
- **State Management**: Tracks voice modes and playback states
- **Thread Coordination**: Manages pause/resume of listening during TTS

### 2. TTSEngine (Text-to-Speech)

**Location**: `abstractvoice/tts/tts_engine.py`

In AbstractVoice core, this module contains **audio playback utilities** used by the Piper-backed engine facade:

```python
from abstractvoice.tts.tts_engine import NonBlockingAudioPlayer

player = NonBlockingAudioPlayer(sample_rate=22050)
```

**Key Features:**
- **Speed Control**: Time-stretching via librosa (preserves pitch)
- **Immediate Pause/Resume**: Via NonBlockingAudioPlayer

**Text Processing Pipeline:**
1. **Preprocessing**: Clean and normalize input text
2. **Chunking**: Split long text at sentence/paragraph boundaries
3. **Synthesis**: Generate audio using TTS model
4. **Speed Adjustment**: Apply time-stretching if speed ≠ 1.0
5. **Streaming Playback**: Queue audio chunks for real-time playback

### 3. NonBlockingAudioPlayer (Audio Streaming)

**Location**: `abstractvoice/tts/tts_engine.py` (embedded class)

The heart of the immediate pause/resume system:

```python
class NonBlockingAudioPlayer:
    def __init__(self, sample_rate=22050, debug_mode=False):
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.pause_lock = threading.Lock()
```

**Architecture Principles:**
- **Callback-Based**: Uses `sounddevice.OutputStream` with callback function
- **Non-Blocking**: Never calls `sd.stop()` which interferes with terminal I/O
- **Thread-Safe**: All pause/resume operations use proper locking
- **Queue-Based**: Audio chunks are queued and consumed by callback

**How Immediate Pause/Resume Works:**

```python
def _audio_callback(self, outdata, frames, time, status):
    """Called by audio system ~50 times per second"""
    
    # Check pause state (thread-safe, immediate response)
    with self.pause_lock:
        if self.is_paused:
            outdata.fill(0)  # Output silence immediately
            return
    
    # Normal audio output...
    # Get next chunk from queue and output to speakers
```

**Key Benefits:**
- **Immediate Response**: Pause takes effect within ~20ms (next audio callback)
- **No Terminal Interference**: Never calls blocking `sd.stop()`
- **Exact Position Resume**: Continues from precise audio position
- **Seamless Streaming**: Works with ongoing synthesis

### 4. VoiceRecognizer (Speech-to-Text)

**Location**: `abstractvoice/recognition.py`

Handles speech recognition with Voice Activity Detection:

```python
class VoiceRecognizer:
    def __init__(self, transcription_callback, stop_callback=None, whisper_model="tiny"):
        self.whisper = whisper.load_model(whisper_model)
        self.vad = webrtcvad.Vad(2)  # Voice Activity Detection
        
        # TTS interrupt control
        self.tts_interrupt_enabled = True
        self.listening_paused = False
```

**Key Features:**
- **VAD Integration**: Efficient speech detection using WebRTC VAD
- **Whisper Models**: Supports tiny, base, small, medium, large
- **TTS Interrupt Control**: Can pause listening during TTS playback
- **Configurable Sensitivity**: Adjustable VAD aggressiveness (0-3)

**Recognition Pipeline:**
1. **Audio Capture**: Continuous microphone input
2. **VAD Processing**: Detect speech vs silence
3. **Audio Buffering**: Collect speech segments
4. **Whisper Transcription**: Convert speech to text
5. **Callback Execution**: Deliver results to application

## Component Communication

### 1. TTS ↔ STT Coordination

**Problem**: TTS playback can trigger STT (acoustic feedback loop)

**Solution**: Callback-based coordination

```python
# In VoiceManager
def _on_tts_start(self):
    """Called when TTS starts playing"""
    if self._voice_mode == "wait":
        self.voice_recognizer.pause_listening()
    elif self._voice_mode == "full":
        self.voice_recognizer.pause_tts_interrupt()

def _on_tts_end(self):
    """Called when TTS finishes playing"""
    if self._voice_mode == "wait":
        self.voice_recognizer.resume_listening()
    elif self._voice_mode == "full":
        self.voice_recognizer.resume_tts_interrupt()
```

### 2. Application ↔ VoiceManager API

**High-Level Interface:**
```python
# Simple usage
vm = VoiceManager()
vm.speak("Hello world")
vm.pause_speaking()  # Immediate pause
vm.resume_speaking()  # Immediate resume

# With callbacks
def on_speech(text):
    response = generate_response(text)
    vm.speak(response)

vm.listen(on_transcription=on_speech)
```

### 3. Enhanced Audio Lifecycle Callbacks (v0.5.1+)

**Problem**: Applications need to distinguish between synthesis phase and actual audio playback for precise visual feedback.

**Solution**: Dual-layer callback system

```python
# Synthesis Phase Callbacks (existing)
vm.tts_engine.on_playback_start = synthesis_start_callback  # TTS synthesis begins
vm.tts_engine.on_playback_end = synthesis_end_callback      # TTS synthesis completes

# Audio Playback Callbacks (NEW in v0.5.1)
vm.on_audio_start = audio_start_callback    # First audio sample plays
vm.on_audio_end = audio_end_callback        # Last audio sample finishes
vm.on_audio_pause = audio_pause_callback    # Audio playback paused
vm.on_audio_resume = audio_resume_callback  # Audio playback resumed
```

**Timing Precision:**
- **Synthesis callbacks**: Fire during text processing and model inference
- **Audio callbacks**: Fire during actual speaker output with ~20ms precision
- **Use case**: Perfect for system tray icons showing thinking vs speaking states

**Implementation Details:**
```python
# In NonBlockingAudioPlayer._audio_callback()
if frames_to_output > 0 and not self._audio_started:
    self._audio_started = True
    if self.on_audio_start:
        threading.Thread(target=self.on_audio_start, daemon=True).start()

# When queue empty and playback complete
if self.is_playing:
    self.is_playing = False
    self._audio_started = False
    if self.on_audio_end:
        threading.Thread(target=self.on_audio_end, daemon=True).start()
```

### 4. Threading Model

**Main Thread**: Application logic, REPL, user interface
**TTS Synthesis Thread**: Background text-to-speech generation
**Audio Callback Thread**: Real-time audio output (managed by sounddevice)
**STT Thread**: Speech recognition processing
**VAD Thread**: Voice activity detection

**Thread Safety:**
- All pause/resume operations use `threading.Lock()`
- Audio queue uses thread-safe `queue.Queue()`
- State variables protected by locks

## Pause/Resume Implementation Deep Dive

### The Challenge

Traditional audio systems use blocking `sd.play()` + `sd.stop()` which:
- Requires waiting for audio chunks to complete
- `sd.stop()` interferes with terminal I/O
- Cannot resume from exact position

### The Solution: OutputStream Callbacks

**1. Non-Blocking Audio Stream:**
```python
self.stream = sd.OutputStream(
    samplerate=22050,
    channels=1,
    callback=self._audio_callback,  # Called ~50x per second
    blocksize=1024,  # Small buffer for low latency
    dtype=np.float32
)
```

**2. Immediate Pause Response:**
```python
def pause(self):
    """Pause audio playback immediately"""
    with self.pause_lock:
        if self.is_playing and not self.is_paused:
            self.is_paused = True  # Next callback will output silence
            return True
    return False
```

**3. Exact Position Resume:**
```python
def resume(self):
    """Resume audio playback immediately"""
    with self.pause_lock:
        if self.is_paused:
            self.is_paused = False  # Next callback will continue audio
            return True
    return False
```

**4. Audio Callback Logic:**
```python
def _audio_callback(self, outdata, frames, time, status):
    # Immediate pause check (thread-safe)
    with self.pause_lock:
        if self.is_paused:
            outdata.fill(0)  # Silence
            return
    
    # Get next audio chunk from queue
    if self.current_audio is None or self.current_position >= len(self.current_audio):
        try:
            self.current_audio = self.audio_queue.get_nowait()
            self.current_position = 0
        except queue.Empty:
            outdata.fill(0)  # No more audio
            return
    
    # Output audio data
    remaining = len(self.current_audio) - self.current_position
    frames_to_output = min(frames, remaining)
    
    if frames_to_output > 0:
        outdata[:frames_to_output, 0] = self.current_audio[
            self.current_position:self.current_position + frames_to_output
        ]
        self.current_position += frames_to_output
    
    # Fill remaining with silence
    if frames_to_output < frames:
        outdata[frames_to_output:].fill(0)
```

### Performance Characteristics

**Pause Latency**: ~20ms (next audio callback)
**Resume Latency**: ~20ms (next audio callback)
**Memory Usage**: Minimal (small audio queue)
**CPU Usage**: Low (efficient callback-based processing)
**Thread Safety**: Full (all operations protected by locks)

## Voice Modes

AbstractVoice supports different voice interaction modes:

### 1. "off" Mode
- **STT**: Disabled
- **TTS**: Normal operation
- **Use Case**: Text-only applications with TTS

### 2. "full" Mode
- **STT**: Always listening
- **TTS**: Continues during speech recognition
- **Interrupt**: TTS stops when speech detected
- **Use Case**: Conversational AI with interruption

### 3. "wait" Mode (Recommended)
- **STT**: Paused during TTS playback
- **TTS**: Uninterrupted playback
- **Resume**: STT resumes after TTS completes
- **Use Case**: Turn-based conversation (prevents acoustic feedback)

## Error Handling and Fallbacks

### TTS availability (Piper-first)

AbstractVoice core uses Piper for TTS. In offline-first contexts (the REPL by default),
Piper voice weights must be cached ahead of time:

```bash
python -m abstractvoice download --piper en
```

### Audio System Fallbacks
- If OutputStream fails, falls back to legacy `sd.play()` system
- Graceful degradation of pause/resume functionality
- Error logging and user-friendly messages

## Configuration and Customization

### Model Selection
```python
# Automatic (recommended)
vm = VoiceManager()  # Piper TTS + faster-whisper STT (downloads allowed by default)

# Offline-first (no implicit downloads)
vm = VoiceManager(allow_downloads=False)
```

### Performance Tuning
```python
# Development (fast)
vm = VoiceManager(
    whisper_model="tiny"
)

# Production (quality)
vm = VoiceManager(
    whisper_model="base"
)
```

### Audio Parameters
```python
# In NonBlockingAudioPlayer
self.stream = sd.OutputStream(
    samplerate=22050,      # Audio quality vs performance
    blocksize=1024,        # Latency vs stability
    channels=1,            # Mono for efficiency
    dtype=np.float32       # Audio precision
)
```

## Future Enhancements

### Planned Features
1. **Multiple Voice Models**: Support for different voices/languages
2. **Streaming STT**: Real-time transcription during speech
3. **Advanced VAD**: ML-based voice activity detection
4. **Audio Effects**: Reverb, EQ, noise reduction
5. **WebRTC Integration**: Browser-based voice interactions

### Architecture Evolution
- **Plugin System**: Modular TTS/STT backends
- **Cloud Integration**: Remote model inference
- **Caching Layer**: Audio segment caching for performance
- **Metrics Collection**: Performance monitoring and analytics

## Conclusion

AbstractVoice's architecture prioritizes real-time responsiveness through:

1. **Non-blocking audio streaming** via OutputStream callbacks
2. **Immediate pause/resume** without terminal interference
3. **Thread-safe coordination** between TTS and STT components
4. **Intelligent text processing** for high-quality synthesis
5. **Modular design** for easy integration and customization

The key innovation is the NonBlockingAudioPlayer, which provides professional-grade audio control while maintaining compatibility with terminal-based applications. This makes AbstractVoice suitable for both interactive CLI tools and embedded voice applications.
