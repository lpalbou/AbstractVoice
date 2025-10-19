# AbstractVoice CLI - Quick Reference Guide

## Entry Points Quick Lookup

| Command | What It Does | Requires Ollama | Use When |
|---------|------------|---|---|
| `abstractvoice` | Direct voice mode with Ollama | ✅ YES | You want to start voice interaction immediately |
| `abstractvoice-cli cli` | CLI REPL mode | ✅ YES | You want text-based CLI (no voice required) |
| `abstractvoice-cli web` | Flask web API | ❌ NO | You want REST API endpoint |
| `abstractvoice-cli simple` | Simple demo | ❌ NO | Testing without Ollama |
| `abstractvoice-cli check-deps` | Dependency checker | ❌ NO | Validating your installation |
| `python -m abstractvoice check-deps` | Dependency checker (alternative) | ❌ NO | Same as above |

## Common Usage Patterns

### Pattern 1: Default Voice Mode
```bash
# Prerequisites: Ollama running with default model (granite3.3:2b)
abstractvoice

# Then in the REPL:
> Hello there
> /help              # Show all commands
> /voice off         # Disable voice input
> /exit              # Quit
```

### Pattern 2: Using Different Ollama Model
```bash
# First ensure model is pulled
ollama pull mistral

# Then use it
abstractvoice --model mistral
abstractvoice --model neural-chat
```

### Pattern 3: Different Language
```bash
abstractvoice --language fr        # French
abstractvoice --language es        # Spanish
abstractvoice --language de        # German
abstractvoice --language it        # Italian
```

### Pattern 4: Debug Mode
```bash
abstractvoice --debug              # See model selection, API calls, etc.
```

### Pattern 5: Validation Before Running
```bash
# Check everything is installed correctly
python -m abstractvoice check-deps

# If OK, run
abstractvoice
```

### Pattern 6: Custom Ollama Server
```bash
# If Ollama on different machine/port
abstractvoice --api http://192.168.1.100:11434/api/chat --model mistral
```

## Parameter Reference

### Model Parameters
- `--model <name>` - LLM model name (default: granite3.3:2b)
- `--api <url>` - Ollama endpoint (default: http://localhost:11434/api/chat)
- `--temperature <0.0-2.0>` - Creativity level (default: 0.4)
- `--max-tokens <int>` - Max response length (default: 4096)

### Voice Parameters
- `--language <lang>` - Voice language: en, fr, es, de, it, ru (default: en)
- `--tts-model <model>` - Override language TTS model
- `--whisper <model>` - Speech recognition model: tiny, base, small, medium, large
- `--no-listening` - Disable speech-to-text

### Other
- `--system <prompt>` - Custom system prompt
- `--debug` - Enable debug output

## REPL Commands

Once you start `abstractvoice` or `abstractvoice-cli cli`, you get a REPL with these commands:

```
/help                    - Show all commands
/exit, /q, /quit         - Exit
/clear                   - Clear chat history
/language <lang>         - Switch voice language
/setvoice [id]           - List or set specific voice
/voice off|full|wait     - Control voice input mode
/tts on|off              - Toggle text-to-speech
/speed <number>          - Set speed (0.5-2.0)
/whisper <model>         - Change speech recognition model
/tts_model <model>       - Change TTS model
/model <name>            - Change LLM model
/temperature <val>       - Set temperature (0.0-2.0)
/max_tokens <num>        - Set max tokens
/tokens                  - Show token usage
/save <filename>         - Save chat history
/load <filename>         - Load chat history
/pause                   - Pause current speech
/resume                  - Resume paused speech
/stop                    - Stop voice mode or speech
```

## Troubleshooting

### "Connection refused" Error
```bash
# Ollama not running
ollama serve &           # Start Ollama in background
abstractvoice            # Try again
```

### "Model not found" Error
```bash
# Model not pulled yet
ollama pull mistral      # Or other model
ollama list              # See what you have
abstractvoice --model mistral
```

### "TTS functionality requires optional dependencies"
```bash
# Missing TTS dependencies
pip install abstractvoice[all]

# Or minimal:
pip install abstractvoice[tts]
```

### Voice quality issues
```bash
# Install espeak-ng for premium voices
# macOS
brew install espeak-ng

# Linux
sudo apt-get install espeak-ng

# Windows
# Download espeak-ng-X64.msi from GitHub
```

### Dependency conflicts
```bash
# Check all dependencies
python -m abstractvoice check-deps

# Fix conflicts
pip uninstall torch torchvision torchaudio
pip install abstractvoice[all]
```

## Configuration File Locations

- **Cache**: `~/.cache/huggingface/` - TTS/STT models
- **Settings**: Not used (all command-line)
- **Ollama models**: `~/.ollama/models/` (Ollama installation)

## Performance Tips

### Faster Responses
```bash
# Use smaller/faster model
abstractvoice --model granite3.3:2b    # 2GB, fastest
abstractvoice --model neural-chat      # 12GB, faster
```

### Better Quality
```bash
# Use larger model
abstractvoice --model mistral          # 26GB, better
abstractvoice --model openhermes:34b   # 34GB, best
```

### Lower Temperature = More Focused
```bash
abstractvoice --temperature 0.1        # Very focused, predictable
abstractvoice --temperature 0.7        # Balanced (default)
abstractvoice --temperature 1.5        # Creative, varied
```

## Model Selection Guide

| Use Case | Model | Command |
|----------|-------|---------|
| **Testing** | granite3.3:2b | `abstractvoice` |
| **General Chat** | neural-chat | `--model neural-chat` |
| **Quality** | mistral | `--model mistral` |
| **Best Quality** | openhermes:34b | `--model openhermes:34b` |
| **Code** | neural-chat | `--model neural-chat` |
| **Writing** | mistral | `--model mistral` |

## Advanced: Multiple Instances

```bash
# Terminal 1: English voice, Mistral model
abstractvoice --language en --model mistral

# Terminal 2: French voice, different model  
abstractvoice --language fr --model neural-chat

# Terminal 3: Web API
abstractvoice-cli web
```

## File Structure

- Voice mode entry: `abstractvoice` → `voice_cli.py:main`
- Examples entry: `abstractvoice-cli` → `__main__.py:main`
- Voice REPL: `cli_repl.py` (contains Ollama integration)
- Dependency checker: `dependency_check.py`

## When to Use Each Command

```
✅ Use 'abstractvoice' for:
  - Interactive voice conversations
  - Quick testing
  - Primary use case

✅ Use 'abstractvoice-cli' for:
  - Text-only REPL (/cli)
  - Web API (/web)
  - Examples & utilities (/simple)
  - Dependency checking (/check-deps)

✅ Use 'python -m abstractvoice' for:
  - Running examples programmatically
  - Checking dependencies in scripts
  - Cross-platform compatibility
```

---

**Last Updated**: 2024-10-19
**AbstractVoice Version**: 0.2.1+
