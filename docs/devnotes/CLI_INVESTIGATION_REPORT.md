# AbstractVoice CLI Structure Investigation Report

## Executive Summary

AbstractVoice has a well-organized CLI structure with two entry points, comprehensive Ollama integration through API requests, and well-designed error handling for dependency failures. The `check-deps` functionality works correctly but is only accessible via `python -m abstractvoice check-deps` from `__main__.py`, not through the `abstractvoice` direct command.

---

## 1. CLI ENTRY POINTS STRUCTURE

### A. pyproject.toml Configuration

```toml
[project.scripts]
abstractvoice = "abstractvoice.examples.voice_cli:main"
abstractvoice-cli = "abstractvoice.__main__:main"
```

**Key Finding**: The two entry points are REVERSED from what one might expect:
- `abstractvoice` → voice_cli.py (direct voice mode)
- `abstractvoice-cli` → __main__.py (examples dispatcher)

### B. Command Execution Paths

#### Path 1: Direct Voice Mode
```
Command: abstractvoice [options]
    ↓
Entry: abstractvoice.examples.voice_cli:main
    ↓
Handler: VoiceREPL class (from cli_repl.py)
    ↓
Result: Launches interactive voice-enabled REPL with Ollama integration
```

**Supported Arguments**:
- `--debug` - Enable debug output
- `--api <url>` - Ollama API endpoint (default: http://localhost:11434/api/chat)
- `--model <name>` - LLM model (default: cogito:3b)
- `--whisper <model>` - Whisper model (tiny, base, small, medium, large)
- `--no-listening` - Disable speech-to-text
- `--system <prompt>` - Custom system prompt
- `--temperature <float>` - 0.0-2.0 (default: 0.4)
- `--max-tokens <int>` - Default: 4096
- `--language <lang>` - en, fr, es, de, it, ru, multilingual (default: en)
- `--tts-model <model>` - Override language default TTS model

#### Path 2: Examples Dispatcher
```
Command: abstractvoice-cli [example] [options]
    ↓
Entry: abstractvoice.__main__:main
    ↓
Handler: Dispatcher routing to:
    - cli → cli_repl.py:main (CLI REPL)
    - web → web_api.py:main (Flask web API)
    - simple → simple_example() (Demo)
    - check-deps → dependency_check.py:check_dependencies()
    ↓
Result: Executes chosen example
```

**Supported Examples**:
- `abstractvoice-cli cli` - Interactive CLI REPL
- `abstractvoice-cli web` - Flask web API
- `abstractvoice-cli simple` - Simple demo
- `abstractvoice-cli check-deps` - Dependency checker

#### Path 3: Python Module
```
Command: python -m abstractvoice [example] [options]
    ↓
Entry: __main__.py:main
    ↓
Result: Same as Path 2 (examples dispatcher)
```

---

## 2. --model PARAMETER HANDLING

### A. Model Parameter Flow

**Voice CLI (abstractvoice command)**:
```
voice_cli.py:parse_args()
    ↓ --model cogito:3b
cli_repl.py:VoiceREPL.__init__(model=args.model)
    ↓
self.model = "cogito:3b"
    ↓
process_query() → POST request to Ollama API with model name
```

**Code Path** (voice_cli.py lines 18-19):
```python
parser.add_argument("--model", default="cogito:3b",
                  help="LLM model name")
```

### B. Model Usage in API Requests

Location: cli_repl.py, lines 162-168

```python
payload = {
    "model": self.model,  # ← User-provided model name
    "messages": self.messages,
    "stream": False,
    "temperature": self.temperature,
    "max_tokens": self.max_tokens
}
response = requests.post(self.api_url, json=payload)
```

### C. Error Handling for Models

**Ollama Integration Points**:
1. **API Communication** (cli_repl.py lines 160-172):
   - Sends model name directly to Ollama endpoint
   - No model validation before sending (Ollama handles validation)
   - Ollama returns error if model not available

2. **Error Recovery** (cli_repl.py lines 238-242):
   ```python
   except Exception as e:
       print(f"Error: {e}")
       if self.debug_mode:
           import traceback
           traceback.print_exc()
   ```
   - Generic exception handler catches any API errors
   - With `--debug`, prints full traceback
   - No automatic model fallback (user must specify valid model)

---

## 3. OLLAMA INTEGRATION IMPLEMENTATION

### A. Overall Architecture

**Design Philosophy**: AbstractVoice treats Ollama as a GENERIC HTTP API endpoint

```
AbstractVoice (Voice I/O)
    ↓
    └─ HTTP POST
       ↓
    Ollama API (http://localhost:11434/api/chat)
       ↓
    LLM Model (cogito:3b by default)
```

### B. Integration Points

**1. Default Configuration** (voice_cli.py lines 16-17):
```python
parser.add_argument("--api", default="http://localhost:11434/api/chat",
                  help="LLM API URL")
```

**2. API Endpoint Format** (Ollama API):
- **Method**: HTTP POST
- **Default URL**: `http://localhost:11434/api/chat`
- **Payload Format**: Ollama-compatible JSON
- **Response Format**: JSON with `message.content` field

**3. Request/Response Handling** (cli_repl.py lines 170-223):

```python
# SENDING REQUEST
response = requests.post(self.api_url, json=payload)
response.raise_for_status()

# PARSING RESPONSE - Dual format support
try:
    response_data = response.json()
    
    # Check for Ollama format
    if "message" in response_data and "content" in response_data["message"]:
        response_text = response_data["message"]["content"].strip()
    
    # Check for OpenAI format (backward compatibility)
    elif "choices" in response_data and len(response_data["choices"]) > 0:
        response_text = response_data["choices"][0]["message"]["content"].strip()
```

### C. Streaming Capability

**Current Implementation**: Non-streaming (line 165)
```python
"stream": False,  # Disable streaming for simplicity
```

**Why Non-Streaming**:
- Simplifies error handling
- Ensures complete response before TTS playback
- Avoids partial sentence speaking

**Streaming Not Implemented**: Future enhancement could enable streaming for:
- Real-time voice playback of AI responses
- Lower latency interaction
- Better handling of long responses

### D. Ollama-Specific Features Used

| Feature | Implementation | Notes |
|---------|---|---|
| Multiple Models | `--model` parameter | Any Ollama-compatible model |
| Temperature Control | `"temperature": self.temperature` | 0.0-2.0 range |
| Max Tokens | `"max_tokens": self.max_tokens` | Default: 4096 |
| Message History | `"messages": self.messages` | Maintains conversation context |
| System Prompt | First message with role="system" | User-configurable |

---

## 4. WHY check-deps IS NOT RECOGNIZED

### Problem Statement

Users expect:
```bash
abstractvoice check-deps  ❌ FAILS - Command not found
```

But it only works via:
```bash
abstractvoice-cli check-deps  ✅ WORKS
python -m abstractvoice check-deps  ✅ WORKS
```

### Root Cause Analysis

**The Issue**: Entry point separation in pyproject.toml

```toml
# ACTUAL MAPPING
abstractvoice = "abstractvoice.examples.voice_cli:main"  ← Only has voice mode logic
abstractvoice-cli = "abstractvoice.__main__:main"         ← Has all examples including check-deps
```

**What Users See in __main__.py**:
```python
# Line 100
parser.add_argument("example", nargs="?", help="Example to run (cli, web, simple, check-deps)")

# Line 113
if args.example == "check-deps":
    from abstractvoice.dependency_check import check_dependencies
    check_dependencies(verbose=True)
```

**The Problem**: This code is in `__main__.py`, but:
- `abstractvoice` command calls `voice_cli.py:main`, NOT `__main__.py:main`
- `voice_cli.py` has NO example dispatcher logic
- Therefore, `abstractvoice check-deps` is interpreted as a voice mode with a positional argument

### Verification

**In voice_cli.py** (lines 37-96):
```python
def main():
    """Entry point for direct voice mode."""
    try:
        args = parse_args()  # ← Parses --model, --api, --language, etc.
        # ... initializes VoiceREPL ...
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting AbstractVoice...")
```

**Key Missing Code**: No example dispatcher like in __main__.py

### Why This Design Choice?

The architecture reflects the intended usage:
- `abstractvoice` = Quick access to voice mode (primary use case)
- `abstractvoice-cli` = Access to all examples/tools (secondary)

This is similar to:
- `git` command directly = Primary tool
- `git help <topic>` = Reference information

---

## 5. ERROR HANDLING FOR MODEL FAILURES

### A. Model Load Failures

**Scenario**: User specifies non-existent model

```bash
abstractvoice --model nonexistent-model
```

**Error Flow**:

1. **Initialization** (voice_cli.py lines 39-96):
   - Command-line argument parsed
   - VoiceREPL initialized successfully
   - No model validation at this stage

2. **First Query** (user types something):
   - Request sent to `http://localhost:11434/api/chat`
   - Ollama rejects with error (model not found)
   - HTTP response error code (usually 400 or 500)

3. **Error Handling** (cli_repl.py lines 238-242):
   ```python
   except Exception as e:
       print(f"Error: {e}")
       if self.debug_mode:
           import traceback
           traceback.print_exc()
   ```

4. **User Experience**:
   ```
   > Hello
   Error: 404 Not Found  (or similar Ollama error)
   ```

### B. API Connection Failures

**Scenario**: Ollama service not running

```bash
# Ollama not started
abstractvoice --api http://localhost:11434/api/chat
> Hello
```

**Error Flow**:

1. Connection attempt fails in `requests.post()`
2. ConnectionError raised
3. Caught by generic exception handler
4. Displays: `Error: Connection refused` (or similar)

### C. Comprehensive Error Handling

**Current Error Handling** (cli_repl.py lines 160-242):

| Error Type | Current Behavior | With --debug |
|---|---|---|
| Model not found (Ollama) | "Error: <ollama message>" | Full traceback shown |
| API unreachable | "Error: Connection refused" | Full traceback shown |
| Invalid JSON response | "Error: Error parsing JSON response: ..." | Full traceback shown |
| Streaming format errors | Attempts to parse as streaming | Full traceback shown |
| Generic exceptions | "Error: <exception message>" | Full traceback shown |

### D. TTS Component Error Handling

**TTS Failures** (separate from Ollama):

Located in voice_manager.py - lazy loading with helpful error messages:

```python
def _import_tts_engine():
    """Import TTSEngine with helpful error message if dependencies missing."""
    try:
        from .tts import TTSEngine
        return TTSEngine
    except ImportError as e:
        if "TTS" in str(e) or "torch" in str(e) or "librosa" in str(e):
            raise ImportError(
                "TTS functionality requires optional dependencies. Install with:\n"
                "  pip install abstractvoice[tts]    # For TTS only\n"
                "  pip install abstractvoice[all]    # For all features\n"
                f"Original error: {e}"
            ) from e
```

### E. Dependency Checking Tool

**Purpose**: Validate entire system before running

```bash
abstractvoice-cli check-deps
# or
python -m abstractvoice check-deps
```

**What It Checks**:
- Core dependencies (numpy, requests)
- PyTorch ecosystem (torch, torchvision, torchaudio)
- Optional dependencies (Whisper, coqui-tts, etc.)
- PyTorch version conflicts
- Python version compatibility

**Output Example**:
```
🔍 AbstractVoice Dependency Check Report
==================================================

🐍 Python: 3.12.0 | 64-bit
🖥️  Platform: darwin

📦 Core Dependencies:
  ✅ numpy: v1.24.3
  ✅ requests: v2.31.0

🔥 PyTorch Ecosystem:
  ✅ torch: v2.1.0
  ✅ torchvision: v0.16.0
  ✅ torchaudio: v2.1.0
  🎉 PyTorch ecosystem looks compatible!

🔧 Optional Dependencies:
  Installed:
    ✅ coqui-tts: v0.27.0
    ✅ openai-whisper: v20231117
  Missing:
    ⭕ sounddevice: not installed
```

---

## 6. ARCHITECTURE SUMMARY DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                     AbstractVoice CLI Layer                  │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
         ┌──────────▼──────────┐  ┌─────▼──────────────┐
         │  abstractvoice      │  │  abstractvoice-cli │
         │  (voice_cli.py)     │  │  (__main__.py)     │
         └──────────┬──────────┘  └─────┬──────────────┘
                    │                    │
        ┌───────────▼──────────┐  ┌──────┴──────────────┐
        │ VoiceREPL (REPL mode)│  │ Example Dispatcher │
        │ + Ollama Integration │  │  - cli             │
        │ + Voice I/O          │  │  - web             │
        │                      │  │  - simple          │
        │ Direct Interaction   │  │  - check-deps      │
        └───────────┬──────────┘  └────────────────────┘
                    │
        ┌───────────▼──────────────────────┐
        │  requests → Ollama HTTP API      │
        │  http://localhost:11434/api/chat │
        └────────────────────────────────┬─┘
                                         │
                         ┌───────────────▼───────────┐
                         │  Ollama LLM Models        │
                         │  (cogito:3b default) │
                         └──────────────────────────┘
```

---

## 7. KEY FINDINGS & RECOMMENDATIONS

### A. Current Implementation Strengths

✅ **Clean Separation of Concerns**:
- Voice mode (voice_cli.py) focuses on voice interaction
- Examples dispatcher (__main__.py) handles utilities
- Ollama integration is stateless and flexible

✅ **Flexible Model Selection**:
- Works with any Ollama-compatible model
- User can override defaults via CLI
- No hardcoded model dependencies

✅ **Comprehensive Error Handling**:
- Lazy imports with helpful error messages
- Dependency checker tool included
- Debug mode provides full traceback

✅ **Ollama Integration**:
- Supports both Ollama and OpenAI API formats
- Configurable endpoint and model
- Message history maintained
- Temperature and token controls

### B. Issues & Limitations

⚠️ **Check-deps Discovery Problem**:
- Users expect `abstractvoice check-deps` to work
- Only `abstractvoice-cli check-deps` works
- Design choice makes sense but is discoverable

⚠️ **No Model Validation**:
- Model names validated only by Ollama
- No pre-flight check for availability
- First query reveals model problems

⚠️ **No Streaming Support**:
- Currently disabled for simplicity
- Could enable real-time TTS playback
- Would require more complex buffering

⚠️ **Russian Language Listed but Not Tested**:
- CLI shows "ru" as option (line 31-32 in voice_cli.py)
- Documentation says Russian not supported
- May fail at runtime with Russian text

### C. Recommendations

**For Users**:
1. Use `python -m abstractvoice check-deps` to validate setup first
2. Specify `--model <name>` when using non-default Ollama models
3. Use `--debug` flag for API connection troubleshooting
4. Ensure Ollama is running: `ollama serve` (if not running as service)

**For Maintainers**:
1. Consider adding example in help showing how to use other Ollama models
2. Add model existence pre-check with helpful error message
3. Document the entry point separation (abstractvoice vs abstractvoice-cli)
4. Consider adding /ollama command in REPL to query available models
5. Remove "ru" from CLI choices if not actually supported, or add Russian TTS

---

## 8. QUICK REFERENCE

### Commands That Work

```bash
# Direct voice mode (Ollama required, running on localhost:11434)
abstractvoice                           # Default: cogito:3b model
abstractvoice --model mistral           # Use Mistral model
abstractvoice --language fr             # French voice

# Examples and utilities (Ollama not required)
abstractvoice-cli cli                   # Interactive REPL example
abstractvoice-cli web                   # Flask web API
abstractvoice-cli simple                # Simple demo
abstractvoice-cli check-deps            # Dependency checker

# Python module form (same as above)
python -m abstractvoice check-deps
python -m abstractvoice simple
```

### Configuration for Ollama

```bash
# Start Ollama (must be running before using abstractvoice)
ollama serve &

# Or if running as service on non-standard port
abstractvoice --api http://192.168.1.100:11434/api/chat

# Pull a different model
ollama pull mistral              # ~26GB
ollama pull neural-chat         # ~12GB
ollama pull openhermes          # ~34GB
```

### Common Ollama Models for Voice Use

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| cogito:3b | 3GB | Very Fast | Good | Fast responses |
| mistral:7b | 26GB | Fast | Excellent | Default quality |
| neural-chat | 12GB | Fast | Good | Conversational |
| openhermes:34b | 34GB | Slow | Excellent | Complex queries |

---

## Appendix A: File Structure Reference

```
abstractvoice/
├── __main__.py                      # Examples dispatcher
├── voice_manager.py                 # TTS/STT orchestrator
├── dependency_check.py              # Dependency validator
├── examples/
│   ├── voice_cli.py                # ENTRY: abstractvoice command
│   ├── cli_repl.py                 # VoiceREPL class (Ollama integration)
│   └── web_api.py                  # Flask web API
├── stt/
│   └── transcriber.py              # Whisper wrapper
├── tts/
│   └── tts_engine.py               # Coqui-TTS wrapper
└── vad/
    └── voice_detector.py           # WebRTC VAD wrapper
```

---

## Appendix B: API Response Format Examples

### Ollama Response (Current)
```json
{
  "model": "cogito:3b",
  "created_at": "2024-10-19T12:34:56Z",
  "message": {
    "role": "assistant",
    "content": "This is the assistant's response."
  },
  "done": true
}
```

### OpenAI Format (Backward Compatible)
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "This is the assistant's response."
      }
    }
  ]
}
```

