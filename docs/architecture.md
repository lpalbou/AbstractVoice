# AbstractVoice Architecture

This document explains how AbstractVoice works internally, its components, and how they communicate to provide seamless voice interactions with immediate pause/resume capabilities.

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

## Core Components

### 1. VoiceManager (Orchestrator)

**Location**: `abstractvoice/voice_manager.py`

The central coordinator that manages all voice interactions:

```python
class VoiceManager:
    def __init__(self, tts_model=None, whisper_model="tiny", debug_mode=False):
        self.tts_engine = TTSEngine(model_name=tts_model, debug_mode=debug_mode)
        self.voice_recognizer = VoiceRecognizer(...)
        
        # Set up communication between components
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end
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

Handles text-to-speech synthesis with advanced features:

```python
class TTSEngine:
    def __init__(self, model_name="tts_models/en/ljspeech/vits", debug_mode=False, streaming=True):
        # Initialize TTS model
        self.tts = TTS(model_name=model_name)
        
        # Initialize non-blocking audio player for immediate pause/resume
        self.audio_player = NonBlockingAudioPlayer(sample_rate=22050, debug_mode=debug_mode)
        
        # Set up callbacks
        self.audio_player.playback_complete_callback = self._on_playback_complete
```

**Key Features:**
- **Text Preprocessing**: Normalizes text for better synthesis
- **Intelligent Chunking**: Splits long text at natural boundaries (300 chars)
- **Speed Control**: Time-stretching via librosa (preserves pitch)
- **Streaming Synthesis**: Starts playback while synthesizing remaining chunks
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

### 3. Threading Model

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

### TTS Model Fallbacks
```python
# Automatic fallback chain
try:
    tts = TTS("tts_models/en/ljspeech/vits")  # Best quality (needs espeak-ng)
except:
    tts = TTS("tts_models/en/ljspeech/fast_pitch")  # Fallback (pure Python)
```

### Audio System Fallbacks
- If OutputStream fails, falls back to legacy `sd.play()` system
- Graceful degradation of pause/resume functionality
- Error logging and user-friendly messages

## Configuration and Customization

### Model Selection
```python
# Automatic (recommended)
vm = VoiceManager()  # Uses best available model

# Explicit
vm = VoiceManager(tts_model="tts_models/en/ljspeech/fast_pitch")
```

### Performance Tuning
```python
# Development (fast)
vm = VoiceManager(
    tts_model="tts_models/en/ljspeech/fast_pitch",
    whisper_model="tiny"
)

# Production (quality)
vm = VoiceManager(
    tts_model="tts_models/en/ljspeech/vits",
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
