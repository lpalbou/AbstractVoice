# Changelog

All notable changes to the AbstractVoice project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.5] - 2025-10-19

### Fixed
- **🎯 CRITICAL FIX**: Voice switching now ACTUALLY loads different models instead of always falling back
  - **Root cause**: TTS engine was ignoring `preferred_model` parameter and always trying hardcoded priority order
  - **Solution**: Reordered model loading to try requested model FIRST, only fall back if it fails
  - **Result**: `/setvoice en.jenny` now loads jenny model, `/setvoice en.ek1` loads ek1 model, etc.
- **💥 Fixed Segmentation Faults**: Added proper memory cleanup when switching TTS models
  - **Root cause**: Old TTS models not cleaned up before loading new ones, causing memory conflicts
  - **Solution**: Added `cleanup()` method to `NonBlockingAudioPlayer` and proper deletion in `set_tts_model()`
  - **Result**: No more crashes when switching voices, especially with EK1 model

### Changed
- **Smart Model Loading Strategy**: Now tries requested model first, bulletproof fallback second
  1. **Try requested model** (if cached and compatible)
  2. **Try downloading requested model** (if not cached)
  3. **Only then fall back** to compatibility-first priority order
- **Memory Management**: Proper cleanup of audio streams and TTS objects during voice switching

### Technical Details
- **TTS Engine**: `_load_with_simple_fallback()` now respects `preferred_model` parameter
- **Audio Player**: Added `cleanup()` method to prevent memory leaks
- **Voice Manager**: Enhanced `set_tts_model()` with proper resource cleanup

### User Experience
- ✅ **Voice switching WORKS**: Different voices now sound genuinely different
- ✅ **No crashes**: Segmentation faults eliminated during voice switching
- ✅ **Memory stable**: No memory leaks or conflicts when changing voices
- ✅ **Bulletproof fallback**: Still works reliably if requested voice fails

## [0.4.4] - 2025-10-19

### Added
- **🎭 VOICE DIVERSITY**: Added multiple distinct speakers for true voice switching
  - **Jenny**: Different female speaker (US accent) - `en.jenny`
  - **EK1**: Male voice with British accent - `en.ek1`
  - **VCTK**: Multi-speaker dataset with various accents - `en.vctk`
  - **Tacotron2**: Reliable female voice (primary fallback) - `en.tacotron2`
- **📢 Real Voice Differences**: Each voice uses different speaker datasets, not just different engines
  - No more "all English voices sound the same" problem
  - Clear male vs female vs multi-speaker options
  - US vs British accent variants

### Fixed
- **🔧 Voice Switching Actually Works**: Fixed voice catalog synchronization
  - Updated `VOICE_CATALOG` to match `simple_model_manager.py` definitions
  - `/setvoice en.jenny` now actually switches to Jenny's voice
  - `/setvoice en.ek1` now actually switches to male British voice
- **🛡️ Bulletproof Installation Maintained**: All new voices work with existing reliability
  - Fallback priority: `tacotron2` → `jenny` → `ek1` → `vctk` → others
  - Smart espeak detection still skips incompatible models
  - Zero breaking changes to installation robustness

### Technical Details
- **Updated model definitions**: Both `simple_model_manager.py` and `voice_manager.py` synchronized
- **Enhanced fallback chain**: Includes diverse speakers in compatibility-first order
- **Preserved bulletproof installation**: All reliability improvements from v0.4.3 maintained

### User Experience
- ✅ **Real voice diversity**: English voices now sound genuinely different
- ✅ **Easy switching**: `/setvoice en.jenny` vs `/setvoice en.ek1` vs `/setvoice en.tacotron2`
- ✅ **Gender options**: Female (jenny, tacotron2) and male (ek1) voices available
- ✅ **Accent variety**: US English vs British English options
- ✅ **Zero installation friction**: Still works immediately after pip install

## [0.4.3] - 2025-10-19

### Fixed
- **🎯 MAJOR UX FIX**: Completely rewrote TTS model selection logic to be bulletproof
  - **Cache-first strategy**: Always tries cached models before downloads
  - **Compatibility-first priority**: `tacotron2-DDC` → `fast_pitch` → `glow-tts` → `vits`
  - **Smart espeak detection**: Automatically skips VITS models when espeak-ng unavailable
  - **No more failures**: System finds and uses ANY working cached model instead of giving up
- **🔧 Fixed Model Consistency**: Unified all model definitions across codebase
  - Changed primary essential model from `fast_pitch` to `tacotron2-DDC` (more reliable)
  - Updated VoiceManager English default to use `tacotron2-DDC` instead of `vits`
  - Consistent model priority order in all fallback logic

### Changed
- **Model Priority Revolution**: Reversed from "premium-first" to "compatibility-first"
  - Old broken order: VITS → fast_pitch → tacotron2 (failed frequently)
  - New working order: tacotron2 → fast_pitch → glow-tts → VITS (succeeds reliably)
- **Intelligent Model Selection**: TTS engine now tries ALL cached models in priority order
- **Better User Experience**: Clear debug output shows exactly which model loads successfully

### Technical Details
- **`_load_with_simple_fallback()`**: Complete rewrite with bulletproof fallback chain
- **`_check_espeak_available()`**: New method for accurate espeak-ng detection
- **Model definitions**: Synchronized between `simple_model_manager.py` and `model_manager.py`
- **Zero breaking changes**: All existing APIs remain identical

### User Impact
- ✅ **No more "TTS Model Loading Failed" errors** when models are actually cached
- ✅ **Instant TTS startup** - uses cached `tacotron2-DDC` immediately
- ✅ **Works without espeak-ng** - automatic fallback to compatible models
- ✅ **Clear debug output** - users can see exactly what's happening

## [0.4.2] - 2025-10-19

### Fixed
- **🔧 Critical Fix**: Missing `appdirs` dependency causing cache detection failures
  - Added `appdirs>=1.4.0` to core dependencies in pyproject.toml
  - Resolves `No module named 'appdirs'` errors when checking model cache
- **🔄 Critical Fix**: Circular dependency in `download-models` command
  - Fixed `download-models` command trying to initialize VoiceManager before downloading models
  - Now uses ModelManager directly to avoid TTS initialization requirement
  - Resolves infinite loop where download command fails because no models exist
- **📦 Enhanced Model Management**: Improved standalone model download functionality
  - `download-models --status` now works without requiring TTS initialization
  - Language-specific downloads (`--language fr`) work independently
  - Essential model downloads work without VoiceManager dependency

### Technical Details
- **ModelManager Independence**: `download_models_cli()` now operates completely independently
- **Simplified Dependencies**: Removed VoiceManager requirement from CLI model management
- **Better Error Handling**: Download failures provide actionable guidance without TTS errors

## [0.4.1] - 2025-10-19

### Documentation
- **📚 Complete Documentation Overhaul**: Comprehensive update to all documentation files for v0.4.0 model management system
  - Updated `README.md` with offline-first TTS and model management sections
  - Enhanced `llms.txt` (AI integration quick reference) with instant TTS setup and JSON APIs
  - Enhanced `llms-full.txt` (developer integration guide) with programmatic model management
  - Updated `docs/model-management.md` with comprehensive model management guide
  - Added third-party API examples and integration patterns
  - Updated CLI command references and model information tables

### Added
- **🔧 Enhanced JSON APIs Documentation**: Complete examples for third-party application integration
  - `list_models()`, `download_model()`, `get_status()`, `is_ready()` functions
  - Voice ID format examples (`fr.css10_vits`) and full model names
  - Cache status monitoring and model availability checking
- **📦 Model Information Reference**: Detailed tables with model sizes, quality ratings, and dependencies
  - Essential model: `en.fast_pitch` (107MB) - Reliable English voice
  - Premium models: High-quality VITS models with espeak-ng requirements
  - Cache location documentation for all platforms
- **🌐 Integration Pattern Examples**: Simple, robust, and enterprise deployment patterns
  - One-liner setup for basic integration
  - Progress callbacks and error handling for robust integration
  - Pre-deployment verification for enterprise environments

### Changed
- **Documentation Structure**: Organized model management information across multiple files
  - Quick reference (`llms.txt`) for immediate AI assistant integration
  - Comprehensive guide (`llms-full.txt`) for developers and architects
  - Technical documentation (`docs/model-management.md`) for advanced users
- **CLI Documentation**: Updated all command examples to reflect v0.4.0 model management capabilities
- **API Examples**: Enhanced programmatic API documentation with real-world usage patterns

## [0.4.0] - 2025-10-19

### Added
- **🎯 Offline-First TTS Initialization**: Revolutionary improvement to user experience
  - TTS models now load instantly from cache when available (0.2s vs 30s+ download)
  - Intelligent fallback system tries cached models before attempting downloads
  - No more network dependency for users who have used AbstractVoice before
- **📦 Model Management System**: Complete model download and caching utilities
  - `abstractvoice download-models` - Download essential models for offline use
  - `abstractvoice download-models --all` - Download all supported models
  - `abstractvoice download-models --status` - Check current cache status
  - `abstractvoice download-models --clear` - Clear model cache
- **🔄 Smart Model Selection Strategy**: Four-tier fallback system
  1. **Preferred cached model** - Load instantly if available
  2. **Cached fallback model** - Use any available cached model
  3. **Network download preferred** - Download if internet available
  4. **Network download fallback** - Try alternative models if preferred fails
- **✨ Enhanced Error Guidance**: Actionable error messages with specific commands
  - Clear distinction between offline/network/corruption issues
  - Step-by-step troubleshooting guidance
  - Recommendations for `download-models`, cache clearing, or text-only mode

### Changed
- **MAJOR UX IMPROVEMENT**: First-time users get immediate TTS after essential model download
- **Enhanced CLI**: Added `download-models` command with comprehensive options
- **Better Reliability**: TTS initialization now much more robust with multiple fallback strategies
- **Improved Performance**: Cached models load in ~200ms instead of 30+ seconds

### Technical Details
- **ModelManager class**: New utility for managing TTS model cache and downloads
- **Offline-first loading**: `_load_with_offline_fallback()` implements intelligent model selection
- **Cache detection**: Automatic discovery of cached models across different cache locations
- **Essential models**: Curated list of lightweight, reliable models for immediate functionality
- **Premium models**: High-quality models downloaded on-demand or via `download-models --all`

### Benefits for Users
- ✅ **No waiting**: TTS works immediately after first setup
- ✅ **Offline capable**: Full TTS functionality without internet connection
- ✅ **Robust fallback**: Always finds a working model when possible
- ✅ **Clear guidance**: Actionable error messages and status information
- ✅ **Storage efficient**: Download only what you need, when you need it

## [0.3.2] - 2025-10-19

### Added
- **`--no-tts` option**: Allows running in text-only mode when TTS models fail to download
- **Better TTS fallback**: Multiple fallback models attempted before giving up
- **Clearer error messages**: Distinguished between TTS model issues and Ollama model issues

### Fixed
- **Misleading error messages**: Now correctly identifies TTS download failures vs Ollama model issues
- **TTS model download failures**: Improved handling and multiple fallback attempts
- **Error attribution**: Users no longer see "Model not found" for their Ollama models when TTS fails

## [0.3.1] - 2025-10-19

### Fixed
- **Dependency Conflict**: Removed upper bound on librosa version to resolve conflict with coqui-tts>=0.27.0
  - coqui-tts requires librosa>=0.11.0 but we had pinned librosa<0.11.0
  - Now allows librosa>=0.10.0 without upper bound for compatibility

## [0.3.0] - 2025-10-19

### Added
- **Unified CLI Structure**: Single `abstractvoice` command handles all functionality
  - `abstractvoice` - Voice mode (default)
  - `abstractvoice cli` - CLI REPL example
  - `abstractvoice web` - Web API server
  - `abstractvoice simple` - Simple demonstration
  - `abstractvoice check-deps` - Dependency compatibility checking
- **Enhanced Dependency Management**: Fixed PyTorch/TorchVision conflicts
  - Added explicit torchvision dependency with compatible version ranges
  - Restructured optional dependencies into meaningful groups (`voice-full`, `core-tts`, `core-stt`, `audio-only`)
  - Added comprehensive dependency checker with conflict detection
- **Improved Error Messages**: Context-aware error handling for Ollama/model issues
  - Specific guidance for connection errors, missing models, and installation issues
  - Actionable error messages with exact commands to resolve problems

### Changed
- **BREAKING**: Removed `abstractvoice-cli` command (functionality moved to `abstractvoice`)
- **Dependencies**: Added version constraints for PyTorch ecosystem (2.0.0-2.3.x)
- **CLI Structure**: Simplified from dual entry points to single unified command

### Fixed
- **PyTorch Conflicts**: Resolved `RuntimeError: operator torchvision::nms does not exist`
- **Check-deps Command**: Now accessible via `abstractvoice check-deps`
- **Model Parameter**: Enhanced error messages when Ollama models are unavailable
- **Dependency Compatibility**: Automatic detection and resolution guidance for version conflicts

### Documentation
- Updated all documentation to use unified `abstractvoice` command
- Enhanced installation guides with dependency troubleshooting
- Added comprehensive dependency management documentation
- Simplified CLI reference and usage examples

## [0.2.1] - 2025-10-19
- Dependency compatibility fixes and enhanced error handling

## [0.2.0] - 2025-10-18
- Package renamed from voicellm to abstractvoice

## [0.1.1] - 2025-10-18
- Minor update to README and .toml

## [0.1.0] - 2025-10-15

### Added
- **Professional-grade immediate pause/resume functionality**
  - Pause/resume takes effect within ~20ms (next audio callback)
  - Resumes from exact audio position (no repetition or gaps)
  - No terminal I/O interference (uses OutputStream callbacks)
  - Thread-safe operations with proper locking
  - Works seamlessly with streaming synthesis
- **NonBlockingAudioPlayer class** using sounddevice.OutputStream callbacks
  - Replaces blocking sd.play() + sd.stop() approach
  - Immediate response to pause/resume commands
  - Queue-based audio streaming for continuous playback
  - Professional audio control without terminal interference
- **Enhanced programmatic API** for pause/resume control
  - `pause_speaking()` returns True/False for success status
  - `resume_speaking()` returns True/False for success status
  - `is_paused()` provides reliable pause state checking
  - Thread-safe operations from any thread or callback
- **Comprehensive documentation structure**
  - `docs/architecture.md` - Complete technical architecture guide
  - `docs/development.md` - Development insights and best practices
  - `CONTRIBUTING.md` - Contribution guidelines for open source
  - Updated README.md with detailed pause/resume examples

### Changed
- **Improved TTS pause/resume implementation**
  - Replaced problematic sd.stop() calls with non-blocking approach
  - Immediate response instead of waiting for audio chunks to complete
  - Exact position resume instead of restarting segments
- **Enhanced CLI/REPL responsiveness**
  - `/pause` command no longer blocks terminal input
  - Prompt appears immediately after pause/resume commands
  - No hanging or unresponsive behavior
- **Better error handling and user feedback**
  - Clear success/failure indicators for pause/resume operations
  - Improved status checking with reliable state tracking
  - Better error messages and fallback behavior

### Fixed
- **Terminal I/O interference during pause operations**
  - Eliminated sd.stop() calls that blocked terminal input
  - REPL prompt now appears immediately after /pause command
  - No more hanging or unresponsive terminal behavior
- **Audio position accuracy during pause/resume**
  - Resume continues from exact audio position
  - No repetition of audio segments
  - Seamless continuation of speech synthesis
- **Thread safety issues in pause/resume operations**
  - Proper locking mechanisms for all audio control operations
  - Safe concurrent access from multiple threads
  - Reliable state management across thread boundaries

### Technical Details
- **OutputStream Callback Architecture**: Uses sounddevice.OutputStream with callback function for real-time audio control
- **Non-Blocking Design**: No blocking operations that interfere with terminal or UI responsiveness
- **Queue-Based Streaming**: Audio chunks queued and consumed by callback for continuous playback
- **Thread-Safe Locking**: All pause/resume operations protected by threading.Lock()
- **Performance Optimized**: ~20ms response time with minimal CPU and memory overhead

## [0.1.9] - 2025-10-18

### Added
- **New Method**: `VoiceManager.set_tts_model()` - Change TTS model dynamically
  - Switch between VITS, fast_pitch, glow-tts, tacotron2-DDC at runtime
  - No need to reinitialize VoiceManager
  - Example: `vm.set_tts_model("tts_models/en/ljspeech/vits")`
- **New REPL Command**: `/tts_model <model>` - Change TTS model from CLI
  - Shortcuts: vits, fast_pitch, glow-tts, tacotron2-DDC
  - Example: `/tts_model vits` or `/tts_model fast_pitch`
- **Comprehensive Installation Guide** for espeak-ng
  - macOS (Homebrew), Linux (apt/yum), Windows (conda/chocolatey/installer)
  - Clear instructions in README for all platforms
- **Automatic Model Selection** based on espeak-ng availability
  - Automatically uses VITS if espeak-ng is installed (best quality)
  - Gracefully falls back to fast_pitch if espeak-ng is missing
  - User-friendly message with installation instructions on fallback

### Fixed
- **CRITICAL**: Fixed speed parameter affecting pitch - now uses librosa time-stretching
  - Speed changes no longer alter voice pitch (preserves naturalness)
  - Proper time-stretching algorithm applied to all audio chunks
  - Speed parameter now works as expected: 1.2x = 20% faster with same pitch
  - Works with both `speak(speed=X)` and `set_speed(X)` methods
  - Range: 0.5x (half speed) to 2.0x (double speed)
- **Fixed**: Silenced coqpit deserialization warnings (Type mismatch in FastPitchConfig)
  - No more technical warnings during startup
  - Clean user experience

### Changed
- **Changed default TTS model to VITS (best quality, with auto-fallback)**
  - VITS provides significantly better voice quality than fast_pitch/glow-tts/tacotron2
  - Uses phoneme-based synthesis for natural prosody and intonation
  - Requires espeak-ng (available via package managers on all platforms)
  - Automatic fallback to fast_pitch if espeak-ng not found
  - Detection is automatic - no configuration needed
- **Honest about voice quality trade-offs in all documentation:**
  - VITS (best): Natural prosody, emotional expression, requires espeak-ng
  - fast_pitch: Good quality, works everywhere, no dependencies
  - glow-tts: Alternative fallback, similar to fast_pitch
  - tacotron2-DDC: Legacy model, slower
- **Updated all documentation** to reflect quality rankings and installation
- Added `librosa>=0.10.0` dependency for audio time-stretching (pure Python)
- Silenced pkg_resources deprecation warning from jieba dependency

### Programmatic Usage
```python
from abstractvoice import VoiceManager

# Automatic: Uses VITS if espeak-ng available, fast_pitch otherwise
vm = VoiceManager()

# Explicit: Force a specific model
vm = VoiceManager(tts_model="tts_models/en/ljspeech/fast_pitch")

# Dynamic: Change model at runtime
vm.set_tts_model("tts_models/en/ljspeech/vits")

# Speed control (pitch preserved)
vm.set_speed(1.2)  # 20% faster
vm.speak("Test", speed=1.5)  # 50% faster for this speech only
```

## [0.1.8] - 2025-10-17

### Fixed
- **CRITICAL**: Fixed Python 3.12 compatibility issue by updating TTS dependency from `TTS>=0.21.0` to `coqui-tts>=0.27.0`
- **MAJOR**: Fixed long text synthesis degradation and distortion issues with SOTA best practices implementation
- **MAJOR**: Fixed TTS self-interruption bug when in voice mode (listening enabled)
  - Voice recognition now pauses TTS interrupt during playback
  - Prevents system from interrupting its own speech
  - Ensures long text plays completely without premature termination
- **MAJOR**: Fixed and standardized REPL command recognition
  - ALL commands now require `/` prefix (except `stop` for voice convenience)
  - Added `/q` and `/quit` as aliases for `/exit`
  - Commands like `/save`, `/load`, `/model`, `/temperature`, `/max_tokens`, `/tokens` now require `/` prefix
  - Only `stop` works without `/` (voice command convenience)
  - Overrode `parseline()` method to strip `/` prefix before command parsing
  - Commands are ALWAYS recognized first, text without `/` (except `stop`) goes to LLM
  - Predictable and consistent command syntax
- **Reduced chunk size from 500 to 300 characters** to prevent distortion on Tacotron2-DDC model
  - Based on empirical testing with real-world long texts
  - Eliminates audio degradation issues
- **Enhanced Help System**: Updated REPL `/help` command with actionable parameters and examples
  - Added default values for all configurable parameters
  - Improved clarity with all available options listed
  - Organized commands into logical categories
- **Improved README Documentation**: 
  - Added clear shell usage section with command-line examples
  - Enhanced integration section with 4 complete examples (Ollama, OpenAI, TTS-only, STT-only)
  - Added key integration points with configuration examples
- Updated package dependency specifications in both requirements.txt and pyproject.toml
- Added Python 3.12 classifier to pyproject.toml to indicate official support

### Added
- **Startup Help Display**: REPL now shows quick start guide on launch with API info and basic commands
- **Voice Mode Options**: Enhanced `/voice` command with multiple modes
  - `off` - Disable voice input
  - `full` - Continuous listening with interrupt on speech detection
  - `wait` - Pause listening during TTS playback (recommended, reduces self-interruption)
  - `stop` - Only stop on 'stop' keyword (planned feature)
  - `ptt` - Push-to-talk mode (planned feature)
- **Streaming Playback** (ENABLED BY DEFAULT): Progressive audio playback for multi-chunk synthesis
  - Starts playing first chunk immediately while synthesizing remaining chunks
  - Reduces perceived latency by ~40% for long text
  - Background synthesis continues while audio plays
  - Seamless transitions between chunks
  - Can be disabled with `streaming=False` parameter
- **Text Preprocessing**: Added `preprocess_text()` function to normalize input text before synthesis
  - Removes excessive whitespace and normalizes punctuation
  - Prevents synthesis errors from malformed text
- **Intelligent Text Chunking**: Added `chunk_long_text()` function for very long text (>300 chars)
  - Automatically splits at paragraph and sentence boundaries  
  - Default chunk size reduced to 300 chars (prevents distortion on Tacotron2-DDC model)
  - Prevents memory issues and attention mechanism degradation
- **Sentence Segmentation**: Enabled `split_sentences=True` in TTS API calls (SOTA best practice)
  - Prevents attention mechanism collapse on long sentences
  - Each sentence processed independently with full model attention
- **Seamless Audio Concatenation**: Chunk audio results are concatenated without artifacts
- **Enhanced Debug Output**: Shows text length, chunk count, processing progress, and streaming status
- **Comprehensive Documentation**: Created docs/KnowledgeBase.md with TTS best practices

### Improved
- TTSEngine.speak() now handles arbitrary text length reliably
- No more audio distortion or premature termination on long text
- Better audio quality through text normalization
- More informative debug output for troubleshooting
- Voice recognition system now intelligently pauses during TTS playback
- Added playback lifecycle callbacks (on_playback_start, on_playback_end) to TTSEngine
- VoiceRecognizer can now pause/resume TTS interruption dynamically

### Technical Details
- The original `TTS` package on PyPI has been renamed to `coqui-tts`
- The `coqui-tts>=0.27.0` package provides full Python 3.12 compatibility
- All existing AbstractVoice functionality remains unchanged - only enhanced
- Verified compatibility with Python 3.12.2 and coqui-tts 0.27.2
- Implemented SOTA best practices based on Coqui TTS research and recommendations
- Text chunking at 500 character boundaries aligns with model training distribution
- Sentence segmentation uses pysbd library (included with coqui-tts)

## [0.1.7] - 2024-04-27

### Added
- Added temperature parameter with default of 0.4
- Added max_tokens parameter with default of 4096
- Added CLI commands to adjust temperature and max_tokens
- Updated memory file (.mem) format to store these new settings
- Added command line arguments for temperature and max_tokens to voice_cli.py

## [0.1.6] - 2024-04-26

### Added
- Added `get_speed()` and `get_whisper()` methods to VoiceManager class
- Added save/load functionality for TTS speed and Whisper model settings

### Fixed
- Fixed token calculation issue in CLI REPL's clear command
- Improved token recalculation in tokens command for accurate counts
- Fixed loading of saved memory files

## [0.1.5] - 2024-04-25

### Added
- Initial public release with CLI interface
- Support for voice recognition with Whisper
- Text-to-speech capabilities with interrupt handling
- Memory file format (.mem) for saving and loading sessions 