#!/usr/bin/env python3
"""
CLI example using AbstractVoice with a text-generation API.

This example shows how to use AbstractVoice to create a CLI application
that interacts with an LLM API for text generation.
"""

import argparse
import cmd
import json
import re
import shutil
import sys
import importlib.util
import threading
import time
import requests
from abstractvoice import VoiceManager


# ANSI color codes
class Colors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


class VoiceREPL(cmd.Cmd):
    """Voice-enabled REPL for LLM interaction."""
    
    intro = ""  # Will be set in __init__ to include help
    prompt = f"{Colors.GREEN}> {Colors.END}"
    
    # Override cmd module settings
    ruler = ""  # No horizontal rule line
    use_rawinput = True
    
    def __init__(self, api_url="http://localhost:11434/api/chat",
                 model="granite3.3:2b", debug_mode=False, language="en", tts_model=None, disable_tts=False):
        super().__init__()

        # Debug mode
        self.debug_mode = debug_mode

        # API settings
        self.api_url = api_url
        self.model = model
        self.temperature = 0.4
        self.max_tokens = 4096

        # Language settings
        self.current_language = language
        self._initial_tts_model = tts_model

        # Initialize voice manager with language support
        if disable_tts:
            self.voice_manager = None
            print("🔇 TTS disabled - text-only mode")
        else:
            self.voice_manager = VoiceManager(
                language=language,
                tts_model=tts_model,
                debug_mode=debug_mode,
                allow_downloads=False,
                cloned_tts_streaming=False,
            )

        # Current speaking voice:
        # - None => Piper (default, language-driven)
        # - str  => cloned voice_id
        self.current_tts_voice: str | None = None

        # Seed a default cloned voice (HAL9000) if samples are present.
        self._seed_hal9000_voice()
        
        # Settings
        self.use_tts = True
        # Default voice mode: STOP.
        # Rationale: users can interrupt TTS with "ok stop"/"okay stop" without
        # self-feedback loops, and keep the conversation going hands-free.
        self.voice_mode = "stop"  # off, full, wait, stop, ptt
        self.voice_mode_active = False  # Is voice recognition running?
        self._ptt_session_active = False
        self._ptt_recording = False
        self._ptt_busy = False
        
        # System prompt
        self.system_prompt = """
                You are a Helpful Voice Assistant. By design, your answers are short and more conversational, unless specifically asked to detail something.
                You only speak, so never use any text formatting or markdown. Write for a speaker.
                """
        
        # Message history
        self.messages = [{"role": "system", "content": self.system_prompt}]
        
        # Token counting
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        self._count_system_tokens()
        
        if self.debug_mode:
            print(f"Initialized with API URL: {api_url}")
            print(f"Using model: {model}")
        
        # Try to auto-enable voice input in STOP mode (best-effort).
        if self.voice_manager:
            try:
                self.do_voice("stop")
            except Exception:
                # Never block REPL start.
                pass

        # Set intro with help information
        self.intro = self._get_intro()
        
    def _get_intro(self):
        """Generate intro message with help."""
        intro = f"\n{Colors.BOLD}Welcome to AbstractVoice CLI REPL{Colors.END}\n"
        if self.voice_manager:
            lang_name = self.voice_manager.get_language_name()
            intro += f"API: {self.api_url} | Model: {self.model} | Voice: {lang_name}\n"
        else:
            intro += f"API: {self.api_url} | Model: {self.model} | Voice: Disabled\n"
        intro += f"\n{Colors.CYAN}Quick Start:{Colors.END}\n"
        intro += "  • Type messages to chat with the LLM\n"
        intro += "  • Voice input: enabled by default (STOP mode). Use /voice off to disable.\n"
        intro += "  • PTT: /voice ptt then SPACE to capture (ESC exits)\n"
        intro += "  • Use /language <lang> to switch voice language\n"
        intro += "  • Use /clones and /tts_voice to use cloned voices\n"
        intro += "  • Type /help for full command list\n"
        intro += "  • Type /exit or /q to quit\n"
        return intro
        
    def _count_system_tokens(self):
        """Count tokens in the system prompt."""
        self._count_tokens(self.system_prompt, "system")
    
    def parseline(self, line):
        """Parse the line to extract command and arguments.
        
        Override to handle / prefix for commands. This ensures /voice, /help, etc.
        are recognized as commands by stripping the leading / before parsing.
        """
        # Commands still use leading "/". In PTT mode we don't accept typed input.
        s = line.strip()
        if s.startswith("/"):
            return super().parseline(s[1:].strip())
        return super().parseline(line.strip())
        
    def default(self, line):
        """Handle regular text input.
        
        Only 'stop' is recognized as a command without /
        All other commands MUST use / prefix.
        """
        # Skip empty lines
        if not line.strip():
            return
        
        # In PTT mode we do not accept typed input.
        if self.voice_mode == "ptt":
            print("PTT mode: press SPACE to speak, ESC to exit.")
            return

        # Check if in voice mode - don't send to LLM
        if self.voice_mode_active:
            if self.debug_mode:
                print(f"Voice mode active ({self.voice_mode}). Use /voice off to disable.")
            return

        # Interrupt any ongoing TTS playback immediately when the user types.
        # This is the expected “barge-in by typing” UX for a REPL.
        try:
            if self.voice_manager:
                self.voice_manager.stop_speaking()
        except Exception:
            pass
        
        # Everything else goes to LLM
        self.process_query(line.strip())

    # NOTE: PTT is implemented as a dedicated key-loop session (no typing).
        
    def process_query(self, query):
        """Process a query and get a response from the LLM."""
        if not query:
            return

        # If audio is currently playing, stop it so the new request can be handled
        # without overlapping speech.
        try:
            if self.voice_manager:
                self.voice_manager.stop_speaking()
        except Exception:
            pass
            
        # Count user message tokens
        self._count_tokens(query, "user")
        
        # Create the message
        user_message = {"role": "user", "content": query}
        self.messages.append(user_message)
        
        if self.debug_mode:
            print(f"Sending request to API: {self.api_url}")
            
        try:
            # Structure the payload with system prompt outside the messages array
            payload = {
                "model": self.model,
                "messages": self.messages,
                "stream": False,  # Disable streaming for simplicity
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Make API request
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Try to parse response
            try:
                # First, try to parse as JSON
                response_data = response.json()
                
                # Check for different API formats
                if "message" in response_data and "content" in response_data["message"]:
                    # Ollama format
                    response_text = response_data["message"]["content"].strip()
                elif "choices" in response_data and len(response_data["choices"]) > 0:
                    # OpenAI format
                    response_text = response_data["choices"][0]["message"]["content"].strip()
                else:
                    # Some other format
                    response_text = str(response_data).strip()
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"Error parsing JSON response: {e}")
                
                # Handle streaming or non-JSON response
                response_text = response.text.strip()
                
                # Try to extract content from streaming format if possible
                if response_text.startswith("{") and "content" in response_text:
                    try:
                        # Extract the last message if multiple streaming chunks
                        lines = response_text.strip().split("\n")
                        last_complete_line = lines[-1]
                        for i in range(len(lines) - 1, -1, -1):
                            if '"done":true' in lines[i]:
                                last_complete_line = lines[i]
                                break
                                
                        # Parse the message content
                        import json
                        data = json.loads(last_complete_line)
                        if "message" in data and "content" in data["message"]:
                            full_content = ""
                            for line in lines:
                                try:
                                    chunk = json.loads(line)
                                    if "message" in chunk and "content" in chunk["message"]:
                                        full_content += chunk["message"]["content"]
                                except:
                                    pass
                            response_text = full_content.strip()
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error extracting content from streaming response: {e}")
            
            # Count assistant message tokens
            self._count_tokens(response_text, "assistant")
            
            # Add to message history
            self.messages.append({"role": "assistant", "content": response_text})
            
            # Display the response with color
            print(f"{Colors.CYAN}{response_text}{Colors.END}")
            
            # Speak the response if voice manager is available
            if self.voice_manager and self.use_tts:
                try:
                    # UX guard: never trigger big cloning downloads during normal chat.
                    if self.current_tts_voice and not self._is_cloning_runtime_ready():
                        print(
                            "ℹ️  Cloned voice selected but cloning runtime is not ready.\n"
                            "   Run /cloning_status then /cloning_download, or switch back with /tts_voice piper."
                        )
                    else:
                        self._speak_with_spinner_until_audio_starts(response_text)
                except Exception as e:
                    print(f"❌ TTS failed: {e}")
                
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Cannot connect to Ollama API at {self.api_url}")
            print(f"   Please check that Ollama is running and accessible")
            print(f"   Try: ollama serve")
            if self.debug_mode:
                print(f"   Connection error: {e}")
        except requests.exceptions.HTTPError as e:
            if "404" in str(e):
                print(f"❌ Model '{self.model}' not found on Ollama server")
                print(f"   Available models: Try 'ollama list' to see installed models")
                print(f"   To install a model: ollama pull {self.model}")
            else:
                print(f"❌ HTTP error from Ollama API: {e}")
            if self.debug_mode:
                print(f"   Full error: {e}")
        except Exception as e:
            error_msg = str(e).lower()
            if "model file not found" in error_msg or "no such file" in error_msg:
                print(f"❌ Model '{self.model}' not found or not fully downloaded")
                print(f"   Try: ollama pull {self.model}")
                print(f"   Or use an existing model: ollama list")
            elif "connection" in error_msg or "refused" in error_msg:
                print(f"❌ Cannot connect to Ollama at {self.api_url}")
                print(f"   Make sure Ollama is running: ollama serve")
            else:
                print(f"❌ Error: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
    
    def _count_tokens(self, text, role):
        """Count tokens in text."""
        try:
            import tiktoken
            
            # Initialize the tokenizer 
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            
            # Count tokens
            token_count = len(encoding.encode(text))
            
            # Update the token counts based on role
            if role == "system":
                self.system_tokens = token_count
            elif role == "user":
                self.user_tokens += token_count
            elif role == "assistant":
                self.assistant_tokens += token_count
            
            # Calculate total tokens
            total_tokens = self.system_tokens + self.user_tokens + self.assistant_tokens
            
            if self.debug_mode:
                print(f"{role.capitalize()} tokens: {token_count}")
                print(f"Total tokens: {total_tokens}")
                    
        except ImportError:
            # If tiktoken is not available, just don't count tokens
            pass
        except Exception as e:
            if self.debug_mode:
                print(f"Error counting tokens: {e}")
            pass
    
    def _clean_response(self, text):
        """Clean LLM response text."""
        patterns = [
            r"user:.*", r"<\|user\|>.*", 
            r"assistant:.*", r"<\|assistant\|>.*", 
            r"<\|end\|>.*"
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.DOTALL)
            
        return text.strip()

    def do_language(self, args):
        """Switch voice language.

        Usage: /language <lang>
        Available languages: en, fr, es, de, ru, zh
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        if not args:
            current_name = self.voice_manager.get_language_name()
            current_code = self.voice_manager.get_language()
            print(f"Current language: {current_name} ({current_code})")

            print("Available languages:")
            for code in self.voice_manager.get_supported_languages():
                name = self.voice_manager.get_language_name(code)
                print(f"  {code} - {name}")
            return

        language = args.strip().lower()

        # Stop any current voice activity
        if self.voice_mode_active:
            self._voice_stop_callback()
            was_active = True
        else:
            was_active = False

        # Switch language
        old_lang = self.current_language
        if self.voice_manager.set_language(language):
            self.current_language = language
            old_name = self.voice_manager.get_language_name(old_lang)
            new_name = self.voice_manager.get_language_name(language)
            print(f"🌍 Language changed: {old_name} → {new_name}")

            # Test the new language with localized message
            test_messages = {
                'en': "Language switched to English.",
                'fr': "Langue changée en français.",
                'es': "Idioma cambiado a español.",
                'de': "Sprache auf Deutsch umgestellt.",
                'ru': "Язык переключен на русский.",
                'zh': "语言已切换到中文。"
            }
            test_msg = test_messages.get(language, "Language switched.")
            # Respect TTS toggle: if the user disabled TTS, don't speak test messages.
            if getattr(self, "use_tts", True):
                self.voice_manager.speak(test_msg, voice=self.current_tts_voice)

            # Restart voice mode if it was active
            if was_active:
                self.do_voice(self.voice_mode)
        else:
            supported = ', '.join(self.voice_manager.get_supported_languages())
            print(f"Failed to switch to language: {language}")
            print(f"Supported languages: {supported}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def do_setvoice(self, args):
        """Set a specific voice model.

        Usage:
          /setvoice                    # Show all available voices
          /setvoice <voice_id>         # Set voice (format: language.voice_id)

        Examples:
          /setvoice                    # List all Piper voices
          /setvoice fr.siwis           # Switch to French (voice id is best-effort)
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        if not args:
            # Show all available voices with metadata
            print(f"\n{Colors.CYAN}Available Voice Models:{Colors.END}")

            try:
                models = self.voice_manager.list_available_models()

                for language, voices in models.items():
                    # Get language name
                    lang_names = {
                        'en': 'English', 'fr': 'French', 'es': 'Spanish',
                        'de': 'German', 'ru': 'Russian', 'zh': 'Chinese'
                    }
                    lang_name = lang_names.get(language, language.upper())

                    print(f"\n🌍 {lang_name} ({language}):")

                    for voice_id, voice_info in voices.items():
                        cached_icon = "✅" if voice_info.get('cached', False) else "📥"
                        quality_icon = "🔧"
                        size_text = f"{voice_info.get('size_mb', 0)}MB"

                        print(f"  {cached_icon} {quality_icon} {language}.{voice_id}")
                        print(f"      {voice_info['name']} ({size_text})")
                        print(f"      {voice_info['description']}")
                        # Piper has no system deps.

                print(f"\n{Colors.YELLOW}Usage:{Colors.END}")
                print("  /setvoice <language>.<voice_id>")
                print("  Example: /setvoice fr.siwis")
                print("\n📥 = Download needed  ✅ = Ready")

            except Exception as e:
                print(f"❌ Error listing models: {e}")
                print("   (No fallback available)")
            return

        voice_spec = args.strip()

        # Parse language.voice_id format
        if '.' not in voice_spec:
            print(f"❌ Invalid format. Use: language.voice_id")
            print(f"   Example: /setvoice fr.css10_vits")
            print(f"   Run '/setvoice' to see available voices")
            return

        try:
            language, voice_id = voice_spec.split('.', 1)
        except ValueError:
            print(f"❌ Invalid format. Use: language.voice_id")
            return

        # Stop any current voice activity
        if self.voice_mode_active:
            self._voice_stop_callback()
            was_active = True
        else:
            was_active = False

        # Download and set the specific voice using programmatic API
        try:
            print(f"🔄 Setting voice {voice_spec}...")
            success = self.voice_manager.set_voice(language, voice_id)

            if success:
                self.current_language = language
                print(f"✅ Voice set to {voice_spec}")

                test_messages = {
                    'en': 'Voice changed to English.',
                    'fr': 'Voix changée en français.',
                    'es': 'Voz cambiada al español.',
                    'de': 'Stimme auf Deutsch geändert.',
                    'ru': 'Голос изменён на русский.',
                    'zh': '语音已切换到中文。'
                }
                test_msg = test_messages.get(language, f'Voice changed to {language}.')
                if getattr(self, "use_tts", True):
                    self.voice_manager.speak(test_msg, voice=self.current_tts_voice)

                if was_active:
                    self.do_voice(self.voice_mode)
            else:
                print(f"❌ Failed to set voice: {voice_spec}")

        except Exception as e:
            print(f"❌ Error setting voice: {e}")
            print(f"   Run '/setvoice' to see available voices")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def do_lang_info(self, args):
        """Show current language information."""
        info = self.voice_manager.get_language_info()
        print(f"\n{Colors.CYAN}Current Language Information:{Colors.END}")
        print(f"  Language: {info['name']} ({info['code']})")
        print(f"  Model: {info['model']}")
        print(f"  Available models: {list(info['available_models'].keys())}")

        # Check if XTTS supports multiple languages
        if 'xtts' in (info['model'] or '').lower():
            print(f"  ✅ Supports multilingual synthesis")
        else:
            print(f"  ℹ️ Monolingual model")

    def do_list_languages(self, args):
        """List all supported languages."""
        print(f"\n{Colors.CYAN}Supported Languages:{Colors.END}")
        for lang in self.voice_manager.get_supported_languages():
            name = self.voice_manager.get_language_name(lang)
            current = " (current)" if lang == self.current_language else ""
            print(f"  {lang} - {name}{current}")

    def do_voice(self, arg):
        """Control voice input mode.
        
        Modes:
          off  - Disable voice input
          full - Continuous listening, interrupts TTS on speech detection
          wait - Pause listening while TTS is speaking (recommended)
          stop - Keep listening while speaking, but only stop TTS on stop phrase
          ptt  - Push-to-talk (use /ptt to record one utterance)
        """
        arg = (arg or "").lower().strip()
        
        # Handle legacy "on" argument
        if arg == "on":
            arg = "wait"
        
        if arg in ["off", "full", "wait", "stop", "ptt"]:
            if not self.voice_manager:
                print("🔇 Voice features are disabled. Use '/tts on' to enable.")
                return

            # Exit PTT session if running.
            if self._ptt_session_active:
                self._ptt_session_active = False
                self._ptt_recording = False
                self._ptt_busy = False

            # Stop any ongoing mic session.
            try:
                self.voice_manager.stop_listening()
            except Exception:
                pass
            self.voice_mode_active = False

            self.voice_mode = arg
            self.voice_manager.set_voice_mode(arg)

            if arg == "off":
                print("Voice mode disabled.")
                return

            if arg == "ptt":
                # PTT is a dedicated session: no text entry.
                print("Voice mode: PTT - Push-to-talk (no typing).")
                print("SPACE: start/stop recording (transcribe on stop)")
                print("ESC:   exit PTT mode")
                self._run_ptt_session()
                return

            # Continuous listening modes.
            try:
                self.voice_manager.listen(
                    on_transcription=self._voice_callback,
                    # Stop phrase interrupts TTS; keep listening.
                    on_stop=lambda: print("\n⏹️  Stopped speaking.\n"),
                )
                self.voice_mode_active = True
            except Exception as e:
                self.voice_mode_active = False
                self.voice_mode = "off"
                print(f"❌ Failed to start microphone listening: {e}")
                print("   Tip: check microphone permissions/device availability.")
                return

            if arg == "wait":
                print("Voice mode: WAIT - Listens continuously except while speaking.")
                print("Use /voice off to disable.")
            elif arg == "stop":
                print("Voice mode: STOP - Always listens; stop phrase stops TTS.")
                print("Use /voice off to disable.")
            elif arg == "full":
                print("Voice mode: FULL - Interrupts TTS on any speech (best with AEC/headset).")
                print("Use /voice off to disable.")
        else:
            print("Usage: /voice off | full | wait | stop | ptt")
            print("  off  - Disable voice input")
            print("  full - Continuous listening, interrupts TTS on speech")
            print("  wait - Listen except while speaking")
            print("  stop - Always listen; stop phrase stops TTS")
            print("  ptt  - Push-to-talk (no typing; SPACE triggers capture)")

    def do_ptt(self, arg):
        """Push-to-talk: record a single utterance, then process it.

        Usage:
          /ptt
        """
        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return
        print("❌ /ptt is deprecated. Use: /voice ptt (then SPACE)")
        return

        # Ensure we are not already listening.
        try:
            self.voice_manager.stop_listening()
        except Exception:
            pass

        return

    def _run_ptt_session(self) -> None:
        """PTT mode key loop (no typing).

        Clean semantics:
        - SPACE toggles recording (start/stop)
        - on stop: transcribe immediately and send to the LLM
        - ESC exits PTT mode (returns to STOP mode)

        This avoids relying on VAD end-of-utterance, which is fragile when speaker
        echo is present (common on laptop speakers).
        """
        if not self.voice_manager:
            return
        self._ptt_session_active = True
        self._ptt_recording = False
        self._ptt_busy = False

        # Lazy imports: keep REPL startup snappy.
        import io
        import wave

        try:
            import sounddevice as sd
        except Exception as e:
            print(f"❌ PTT requires sounddevice: {e}")
            self._ptt_session_active = False
            return

        sr = 16000
        frames: list[bytes] = []
        stream = {"obj": None}

        def _status_line(msg: str) -> None:
            try:
                sys.stdout.write("\r" + (" " * 80) + "\r")
                sys.stdout.write(msg)
                sys.stdout.flush()
            except Exception:
                pass

        def _clear_status() -> None:
            try:
                sys.stdout.write("\r" + (" " * 80) + "\r")
                sys.stdout.flush()
            except Exception:
                pass

        def _start_recording() -> None:
            nonlocal frames
            if self._ptt_recording:
                return
            if self._ptt_busy:
                return
            frames = []

            # Interrupt any speech immediately.
            try:
                self.voice_manager.stop_speaking()
            except Exception:
                pass

            def _cb(indata, _frames, _time, status):
                if status and self.debug_mode:
                    pass
                try:
                    frames.append(indata.copy().tobytes())
                except Exception:
                    pass

            try:
                stream["obj"] = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="int16",
                    callback=_cb,
                    blocksize=int(sr * 0.03),
                )
                stream["obj"].start()
                self._ptt_recording = True
                _status_line("🎙️  Recording… (SPACE to send, ESC to exit)")
            except Exception as e:
                self._ptt_recording = False
                stream["obj"] = None
                _clear_status()
                print(f"\n❌ Failed to start microphone stream: {e}\n")

        def _stop_recording_and_send() -> None:
            if not self._ptt_recording:
                return
            self._ptt_recording = False
            _clear_status()

            try:
                if stream["obj"] is not None:
                    try:
                        stream["obj"].stop()
                    except Exception:
                        pass
                    try:
                        stream["obj"].close()
                    except Exception:
                        pass
            finally:
                stream["obj"] = None

            pcm = b"".join(frames)
            if len(pcm) < int(sr * 0.25) * 2:
                print("\n…(too short, try again)\n")
                return

            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
            wav_bytes = buf.getvalue()

            self._ptt_busy = True
            try:
                text = (self.voice_manager.transcribe_from_bytes(wav_bytes, language=self.current_language) or "").strip()
            except Exception as e:
                self._ptt_busy = False
                print(f"\n❌ Transcription failed: {e}\n")
                return
            self._ptt_busy = False

            if not text:
                print("\n…(no transcription)\n")
                return

            print(f"\n> {text}\n")
            self.process_query(text)

        # Platform key read.
        import sys
        if sys.platform == "win32":
            import msvcrt

            while self._ptt_session_active:
                ch = msvcrt.getwch()
                if ch == "\x1b":  # ESC
                    break
                if self._ptt_busy:
                    continue
                if ch == " ":
                    if not self._ptt_recording:
                        _start_recording()
                    else:
                        _stop_recording_and_send()
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while self._ptt_session_active:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":  # ESC
                        break
                    if self._ptt_busy:
                        continue
                    if ch == " ":
                        if not self._ptt_recording:
                            _start_recording()
                        else:
                            _stop_recording_and_send()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        self._ptt_session_active = False
        self._ptt_recording = False
        self._ptt_busy = False
        try:
            if stream["obj"] is not None:
                stream["obj"].stop()
                stream["obj"].close()
        except Exception:
            pass
        _clear_status()
        # Restore to STOP after exiting PTT.
        try:
            self.do_voice("stop")
        except Exception:
            pass
    
    def _voice_callback(self, text):
        """Callback for voice recognition."""
        # Print what the user said
        print(f"\n> {text}")
        # NOTE: stop phrases are handled by the stop_callback path (interrupt TTS).
        # We do not use "stop" to exit voice mode; use /voice off explicitly.
        
        # Mode-specific handling
        if self.voice_mode == "stop":
            # In 'stop' mode, don't interrupt TTS - just queue the message
            # But since we're in callback, TTS interrupt is already paused
            pass
        elif self.voice_mode == "ptt":
            # In PTT mode, process immediately
            pass
        # 'full' mode has default behavior
        
        # Process the user's query
        self.process_query(text)
    
    def _voice_stop_callback(self):
        """Callback when voice mode is stopped."""
        self.voice_mode = "off"
        self.voice_mode_active = False
        self.voice_manager.stop_listening()
        print("Voice mode disabled.")
    
    def do_tts(self, arg):
        """Toggle text-to-speech."""
        arg = arg.lower().strip()
        
        if arg == "on":
            self.use_tts = True
            if self.voice_manager is None:
                # Re-enable voice features (TTS/STT) by creating a VoiceManager.
                self.voice_manager = VoiceManager(
                    language=self.current_language,
                    tts_model=self._initial_tts_model,
                    debug_mode=self.debug_mode,
                    allow_downloads=False,
                    cloned_tts_streaming=False,
                )
            print("TTS enabled" if self.debug_mode else "")
        elif arg == "off":
            self.use_tts = False
            print("TTS disabled" if self.debug_mode else "")
        else:
            print("Usage: /tts on | off")
    
    def do_speed(self, arg):
        """Set the TTS speed multiplier."""
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        if not arg.strip():
            print(f"Current TTS speed: {self.voice_manager.get_speed()}x")
            return
            
        try:
            speed = float(arg.strip())
            if 0.5 <= speed <= 2.0:
                self.voice_manager.set_speed(speed)
                print(f"TTS speed set to {speed}x")
            else:
                print("Speed should be between 0.5 and 2.0")
        except ValueError:
            print("Usage: /speed <number>  (e.g., /speed 1.5)")
    
    def do_tts_model(self, arg):
        """Deprecated: legacy TTS model switching.

        AbstractVoice core is Piper-first; use `/setvoice` (Piper voices) or cloned voices.
        """
        print("❌ /tts_model is not supported (Piper-first core).")
        print("   Use /setvoice for Piper voices, or /tts_voice clone <id> for cloned voices.")
    
    def do_whisper(self, arg):
        """Change Whisper model."""
        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return
        model = arg.strip()
        if not model:
            print(f"Current Whisper model: {self.voice_manager.get_whisper()}")
            return
        
        self.voice_manager.set_whisper(model)            

    def do_speak(self, arg):
        """Speak a text immediately (without calling the LLM).

        Usage:
          /speak Hello world
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        text = arg.strip()
        if not text:
            print("Usage: /speak <text>")
            return

        try:
            self._speak_with_spinner_until_audio_starts(text)
        except Exception as e:
            print(f"❌ Speak failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()

    def _speak_with_spinner_until_audio_starts(self, text: str) -> None:
        """REPL UX: show spinner while waiting for first audio, then stop.

        This avoids corrupting the `cmd` prompt while still giving feedback during
        long cloned-TTS synthesis. Once playback starts, the prompt is displayed
        normally so the user can interrupt anytime by typing.
        """
        if not self.voice_manager:
            return

        is_clone = bool(self.current_tts_voice)
        ind = self._busy_indicator(enabled=is_clone)
        try:
            if is_clone:
                ind.start()
            self.voice_manager.speak(text, voice=self.current_tts_voice)

            if not is_clone:
                return

            # Wait until audio playback actually starts (or synthesis ends without audio).
            vm = self.voice_manager
            while True:
                try:
                    playing = bool(vm.is_speaking())
                    synth_active = bool(
                        getattr(vm, "_cloned_synthesis_active", None) and vm._cloned_synthesis_active.is_set()
                    )
                except Exception:
                    playing, synth_active = False, False

                if playing:
                    break

                # If synthesis is no longer active and we aren't playing, stop the spinner
                # (either done very quickly or failed).
                if not synth_active:
                    break

                time.sleep(0.05)
        finally:
            try:
                ind.stop()
            except Exception:
                pass
            # After audio starts (or we give up), show a prompt line so users know
            # they can type to interrupt / ask another question immediately.
            try:
                if is_clone:
                    sys.stdout.write("\n" + self.prompt)
                    sys.stdout.flush()
            except Exception:
                pass

    class _busy_indicator:
        """A minimal, discreet spinner (no extra lines)."""

        def __init__(self, enabled: bool = False):
            self.enabled = bool(enabled)
            self._stop = threading.Event()
            self._thread = None

        def start(self):
            if not self.enabled:
                return
            if self._thread and self._thread.is_alive():
                return

            def _run():
                frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                i = 0
                t0 = time.time()
                # Small delay so fast operations don't flash.
                time.sleep(0.25)
                if self._stop.is_set():
                    return
                # Hide cursor for a cleaner look.
                try:
                    sys.stdout.write("\033[?25l")
                    sys.stdout.flush()
                except Exception:
                    pass
                while not self._stop.is_set():
                    elapsed = time.time() - t0
                    sys.stdout.write(f"\r(synthesizing {elapsed:0.1f}s) {frames[i % len(frames)]}")
                    sys.stdout.flush()
                    i += 1
                    time.sleep(0.1)

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

        def stop(self):
            if not self.enabled:
                return
            self._stop.set()
            try:
                if self._thread:
                    self._thread.join(timeout=0.5)
            except Exception:
                pass
            # Clear spinner line.
            try:
                sys.stdout.write("\r" + (" " * 60) + "\r")
                # Restore cursor.
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()
            except Exception:
                pass

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.stop()
            return False

    # NOTE: We intentionally do not keep a background spinner running while the REPL
    # is waiting for user input (it corrupts the prompt line). Instead, we show a
    # spinner only until the first audio actually starts, then stop it so the prompt
    # stays usable for interruption-by-typing.

    def do_clones(self, arg):
        """List cloned voices in the local store."""
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        try:
            voices = self.voice_manager.list_cloned_voices()
            if not voices:
                print("No cloned voices yet. Use /clone <path> or /clone-my-voice.")
                return
            print(f"\n{Colors.CYAN}Cloned voices:{Colors.END}")
            for v in voices:
                vid = v.get("voice_id") or v.get("voice", "")
                name = v.get("name", "")
                src = (v.get("meta") or {}).get("reference_text_source", "")
                src_txt = f" [{src}]" if src else ""
                current = " (current)" if self.current_tts_voice == vid else ""
                print(f"  - {name}: {vid}{src_txt}{current}")
        except Exception as e:
            print(f"❌ Error listing cloned voices: {e}")

    def _resolve_clone_id(self, wanted: str) -> str | None:
        voices = self.voice_manager.list_cloned_voices()
        for v in voices:
            vid = v.get("voice_id") or ""
            name = v.get("name") or ""
            if wanted == vid or vid.startswith(wanted) or wanted == name:
                return vid
        return None

    def do_clone_info(self, arg):
        """Show details for a cloned voice.

        Usage:
          /clone_info <id-or-name>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        wanted = arg.strip()
        if not wanted:
            print("Usage: /clone_info <id-or-name>")
            return
        vid = self._resolve_clone_id(wanted)
        if not vid:
            print(f"❌ Unknown cloned voice: {wanted}. Use /clones to list.")
            return
        try:
            info = self.voice_manager.get_cloned_voice(vid)
            meta = info.get("meta") or {}
            print(f"\n{Colors.CYAN}Cloned voice info:{Colors.END}")
            print(f"  id:   {info.get('voice_id')}")
            print(f"  name: {info.get('name')}")
            print(f"  engine: {info.get('engine')}")
            print(f"  refs: {len(info.get('reference_files') or [])}")
            print(f"  ref_text_source: {meta.get('reference_text_source','')}")
            rt = (info.get('reference_text') or '').strip()
            if rt:
                short = (rt[:200] + "…") if len(rt) > 200 else rt
                print(f"  reference_text: {short}")
            else:
                print("  reference_text: (missing)")
        except Exception as e:
            print(f"❌ Error: {e}")

    def do_clone_ref(self, arg):
        """Print the full reference_text for a cloned voice.

        Usage:
          /clone_ref <id-or-name>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        wanted = arg.strip()
        if not wanted:
            print("Usage: /clone_ref <id-or-name>")
            return
        vid = self._resolve_clone_id(wanted)
        if not vid:
            print(f"❌ Unknown cloned voice: {wanted}. Use /clones to list.")
            return
        info = self.voice_manager.get_cloned_voice(vid)
        print((info.get("reference_text") or "").strip())

    def do_clone_rename(self, arg):
        """Rename a cloned voice.

        Usage:
          /clone_rename <id-or-name> <new_name>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        parts = arg.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /clone_rename <id-or-name> <new_name>")
            return
        vid = self._resolve_clone_id(parts[0])
        if not vid:
            print(f"❌ Unknown cloned voice: {parts[0]}. Use /clones to list.")
            return
        self.voice_manager.rename_cloned_voice(vid, parts[1])
        print("✅ Renamed.")

    def do_clone_rm(self, arg):
        """Remove a cloned voice from the store.

        Usage:
          /clone_rm <id-or-name>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        wanted = arg.strip()
        if not wanted:
            print("Usage: /clone_rm <id-or-name>")
            return
        vid = self._resolve_clone_id(wanted)
        if not vid:
            print(f"❌ Unknown cloned voice: {wanted}. Use /clones to list.")
            return
        # If currently selected, switch back to Piper.
        if self.current_tts_voice == vid:
            self.current_tts_voice = None
        self.voice_manager.delete_cloned_voice(vid)
        print("✅ Deleted.")

    def do_clone_export(self, arg):
        """Export a cloned voice bundle (.zip).

        Usage:
          /clone_export <id-or-name> <path.zip>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        parts = arg.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /clone_export <id-or-name> <path.zip>")
            return
        vid = self._resolve_clone_id(parts[0])
        if not vid:
            print(f"❌ Unknown cloned voice: {parts[0]}. Use /clones to list.")
            return
        out = self.voice_manager.export_voice(vid, parts[1])
        print(f"✅ Exported: {out}")

    def do_clone_import(self, arg):
        """Import a cloned voice bundle (.zip).

        Usage:
          /clone_import <path.zip>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        path = arg.strip()
        if not path:
            print("Usage: /clone_import <path.zip>")
            return
        vid = self.voice_manager.import_voice(path)
        print(f"✅ Imported as: {vid}")

    def do_clone(self, arg):
        """Clone a voice from a reference file or folder.

        Usage:
          /clone <path> [name]
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        parts = arg.strip().split()
        if not parts:
            print("Usage: /clone <path> [name]")
            return
        path = parts[0]
        name = parts[1] if len(parts) > 1 else None
        try:
            voice_id = self.voice_manager.clone_voice(path, name=name)
            print(f"✅ Cloned voice created: {voice_id}")
            print("   Use /tts_voice clone <id-or-name> to select it.")
            print("   Tip: set reference text for best quality:")
            print("     /clone_set_ref_text <id-or-name> \"...\"")
            if not self._is_cloning_runtime_ready():
                print("   (Cloning runtime not ready yet; run /cloning_status and /cloning_download first.)")
        except Exception as e:
            print(f"❌ Clone failed: {e}")

    def do_clone_set_ref_text(self, arg):
        """Set the reference transcript for a cloned voice (quality fix).

        Usage:
          /clone_set_ref_text <id-or-name> <text...>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        parts = arg.strip().split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: /clone_set_ref_text <id-or-name> <text...>")
            return

        wanted, text = parts[0], parts[1]
        voices = self.voice_manager.list_cloned_voices()
        match = None
        for v in voices:
            vid = v.get("voice_id") or ""
            name = v.get("name") or ""
            if wanted == vid or vid.startswith(wanted) or wanted == name:
                match = vid
                break
        if not match:
            print(f"❌ Unknown cloned voice: {wanted}. Use /clones to list.")
            return

        try:
            self.voice_manager.set_cloned_voice_reference_text(match, text)
            print("✅ Updated reference text.")
        except Exception as e:
            print(f"❌ Failed to update reference text: {e}")

    def do_tts_voice(self, arg):
        """Select which voice is used for speaking.

        Usage:
          /tts_voice piper
          /tts_voice clone <voice_id_or_name>
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        parts = arg.strip().split()
        if not parts:
            current = self.current_tts_voice or "piper"
            print(f"Current TTS voice: {current}")
            print("Usage: /tts_voice piper | /tts_voice clone <id-or-name>")
            return

        if parts[0] == "piper":
            self.current_tts_voice = None
            print("✅ Using Piper (default) voice")
            return

        if parts[0] != "clone" or len(parts) < 2:
            print("Usage: /tts_voice piper | /tts_voice clone <id-or-name>")
            return

        wanted = parts[1]
        match = self._resolve_clone_id(wanted)
        if not match:
            print(f"❌ Unknown cloned voice: {wanted}. Use /clones to list.")
            return

        # Do not allow selecting a cloned voice unless the runtime is ready.
        if not self._is_cloning_runtime_ready():
            print("❌ Cloning runtime is not ready (would trigger large downloads).")
            print("   Run /cloning_status and /cloning_download, or use /tts_voice piper.")
            return

        # Allow selecting voices without reference_text; we will auto-fallback at speak-time
        # if the STT model is already cached locally (no downloads in REPL).

        self.current_tts_voice = match
        print(f"✅ Using cloned voice: {match}")

    def do_clone_my_voice(self, arg):
        """Interactive voice cloning from microphone.

        This records a short prompt to WAV and adds it to the voice store.
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        prompt = "Good evening, Dave."
        seconds = 6.0
        print("You will record a short reference sample for voice cloning.")
        print(f"Please read this aloud (once): {prompt}")
        input("Press Enter to start recording...")
        try:
            import appdirs
            from pathlib import Path
            from abstractvoice.audio import record_wav

            out_dir = Path(appdirs.user_data_dir("abstractvoice")) / "recordings"
            out_path = out_dir / "my_voice.wav"
            record_wav(out_path, seconds=seconds, sample_rate=24000, channels=1)
            voice_id = self.voice_manager.clone_voice(str(out_path), name="my_voice", reference_text=prompt)
            print(f"✅ Recorded and cloned: {voice_id}")
            print("   Use /tts_voice clone <id-or-name> to select it.")
        except Exception as e:
            print(f"❌ /clone-my-voice failed: {e}")

    def do_cloning_status(self, arg):
        """Show whether cloning runtime is ready locally (no downloads)."""
        if importlib.util.find_spec("f5_tts") is None:
            print("Cloning runtime not installed in this environment (missing: f5_tts).")
            print("Install: pip install \"abstractvoice[cloning]\"")
            return
        try:
            import torch

            mps = False
            try:
                mps = bool(torch.backends.mps.is_available())
            except Exception:
                mps = False
            print(f"torch: {getattr(torch, '__version__', '?')}")
            print(f"cuda_available: {bool(torch.cuda.is_available())}")
            print(f"mps_available:  {mps}")
        except Exception:
            pass
        if self._is_openf5_cached():
            print("✅ OpenF5 artifacts: present (cached)")
        else:
            print("ℹ️  OpenF5 artifacts: not present (will require ~5.4GB download)")
            print("Run: /cloning_download")
        try:
            if self.voice_manager:
                info = self.voice_manager.get_cloning_runtime_info()
                if info:
                    print(f"cloning_resolved_device: {info.get('resolved_device')}")
                    print(f"cloning_model_param_device: {info.get('model_param_device','?')}")
                    print(f"cloning_quality_preset: {info.get('quality_preset')}")
        except Exception:
            pass

    def do_clone_quality(self, arg):
        """Set cloned TTS quality preset (speed/quality tradeoff).

        Usage:
          /clone_quality fast|balanced|high
        """
        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return
        preset = (arg or "").strip().lower()
        if preset not in ("fast", "balanced", "high"):
            print("Usage: /clone_quality fast|balanced|high")
            return
        try:
            self.voice_manager.set_cloned_tts_quality(preset)
            print(f"✅ Cloned TTS quality preset: {preset}")
        except Exception as e:
            print(f"❌ Failed to set preset: {e}")

    def do_cloning_download(self, arg):
        """Explicitly download cloning artifacts (this may take a long time)."""
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return
        if importlib.util.find_spec("f5_tts") is None:
            print("❌ Cloning runtime not installed in this environment (missing: f5_tts).")
            print("   Install: pip install \"abstractvoice[cloning]\"")
            return
        try:
            cloner = self.voice_manager._get_voice_cloner()  # REPL convenience
            engine = cloner._get_engine()  # explicit download is an engine concern
            print("Downloading OpenF5 artifacts (~5.4GB). This is a one-time cache per machine.")
            engine.ensure_openf5_artifacts_downloaded()
            print("✅ Download complete.")
        except Exception as e:
            print(f"❌ Download failed: {e}")

    def _is_openf5_cached(self) -> bool:
        """Heuristic local check that avoids importing huggingface_hub."""
        from pathlib import Path
        import os

        root = Path(os.path.expanduser("~/.cache/abstractvoice/openf5"))
        if not root.exists():
            return False
        cfg = next(iter(root.rglob("*.yaml")), None) or next(iter(root.rglob("*.yml")), None)
        ckpt = next(iter(root.rglob("*.pt")), None)
        vocab = next(iter(root.rglob("vocab*.txt")), None) or next(iter(root.rglob("*.txt")), None)
        return bool(cfg and ckpt and vocab)

    def _is_cloning_runtime_ready(self) -> bool:
        return importlib.util.find_spec("f5_tts") is not None and self._is_openf5_cached()

    def _seed_hal9000_voice(self):
        """Seed a default 'hal9000' cloned voice if sample WAVs are present."""
        if not self.voice_manager:
            return
        try:
            from pathlib import Path

            sample_dir = Path("audio_samples") / "hal9000"
            if not sample_dir.exists():
                return

            # If already present, do nothing.
            existing_hal = None
            for v in self.voice_manager.list_cloned_voices():
                if (v.get("name") or "").lower() == "hal9000":
                    existing_hal = v.get("voice_id")
                    break

            # Seed from the clean short WAV sample to avoid noisy auto-transcriptions.
            # This avoids repeated artifacts like "how are you hal" bleeding into outputs.
            if existing_hal is None:
                ref = sample_dir / "hal9000_hello.wav"
                if ref.exists():
                    existing_hal = self.voice_manager.clone_voice(
                        str(ref),
                        name="hal9000",
                        reference_text="Hello, Dave.",
                    )
                else:
                    existing_hal = self.voice_manager.clone_voice(str(sample_dir), name="hal9000")
                if self.debug_mode:
                    print(f"Seeded cloned voice 'hal9000': {existing_hal}")

            # Do NOT auto-select here; selecting a clone without explicit user action
            # can cause surprise multi-GB downloads. Users can opt in via /tts_voice.
        except Exception:
            # Best-effort only; never block REPL start.
            return

    def do_tts_engine(self, arg):
        """Select TTS engine: auto|piper.

        This recreates the internal VoiceManager instance.
        """
        engine = arg.strip().lower()
        if engine not in ("auto", "piper"):
            print("Usage: /tts_engine auto|piper")
            return

        if self.voice_manager:
            try:
                self.voice_manager.cleanup()
            except Exception:
                pass

        self.voice_manager = VoiceManager(
            language=self.current_language,
            tts_model=self._initial_tts_model,
            debug_mode=self.debug_mode,
            tts_engine=engine,
            allow_downloads=False,
            cloned_tts_streaming=False,
        )
        print(f"✅ TTS engine set to: {engine}")

    def do_aec(self, arg):
        """Enable/disable optional AEC (echo cancellation) for true barge-in.

        Usage:
          /aec on [delay_ms]
          /aec off
        """
        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return

        parts = arg.strip().split()
        if not parts:
            enabled = bool(getattr(self.voice_manager, "_aec_enabled", False))
            delay = int(getattr(self.voice_manager, "_aec_stream_delay_ms", 0))
            print(f"AEC: {'on' if enabled else 'off'} (delay_ms={delay})")
            print("Usage: /aec on [delay_ms] | /aec off")
            return

        if parts[0] == "off":
            try:
                self.voice_manager.enable_aec(False)
                print("✅ AEC disabled")
            except Exception as e:
                print(f"❌ AEC disable failed: {e}")
            return

        if parts[0] != "on":
            print("Usage: /aec on [delay_ms] | /aec off")
            return

        delay_ms = 0
        if len(parts) > 1:
            try:
                delay_ms = int(parts[1])
            except Exception:
                print("Usage: /aec on [delay_ms] | /aec off")
                return

        try:
            self.voice_manager.enable_aec(True, stream_delay_ms=delay_ms)
            print(f"✅ AEC enabled (delay_ms={delay_ms}).")
            print("Tip: use /voice full for barge-in behavior when AEC is enabled.")
        except Exception as e:
            print(f"❌ AEC enable failed: {e}")

    def do_stt_engine(self, arg):
        """Select STT engine: auto|faster_whisper|whisper.

        This recreates the internal VoiceManager instance.
        """
        engine = arg.strip().lower()
        if engine not in ("auto", "faster_whisper", "whisper"):
            print("Usage: /stt_engine auto|faster_whisper|whisper")
            return

        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return

        # Recreate VoiceManager preserving current TTS engine preference.
        # If the current engine is unknown, let it auto-select.
        tts_engine = getattr(self.voice_manager, "_tts_engine_preference", "auto")

        try:
            self.voice_manager.cleanup()
        except Exception:
            pass

        self.voice_manager = VoiceManager(
            language=self.current_language,
            tts_model=self._initial_tts_model,
            debug_mode=self.debug_mode,
            tts_engine=tts_engine,
            stt_engine=engine,
            allow_downloads=False,
            cloned_tts_streaming=False,
        )
        print(f"✅ STT engine set to: {engine}")

    def do_transcribe(self, arg):
        """Transcribe an audio file via the library STT path (faster-whisper by default).

        Usage:
          /transcribe path/to/audio.wav

        Notes:
        - This is the simplest way to validate STT without requiring microphone capture.
        - The default engine is faster-whisper; legacy openai-whisper remains optional.
        """
        if not self.voice_manager:
            print("🔇 Voice features are disabled. Use '/tts on' to enable.")
            return

        path = arg.strip()
        if not path:
            print("Usage: /transcribe <path/to/audio.wav>")
            return

        try:
            text = self.voice_manager.transcribe_file(path)
            print(f"{Colors.CYAN}{text}{Colors.END}")
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
    
    def do_clear(self, arg):
        """Clear chat history."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # Reset token counters
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        # Recalculate system tokens
        self._count_system_tokens()
        print("History cleared")
    
    def do_system(self, arg):
        """Set the system prompt."""
        if arg.strip():
            self.system_prompt = arg.strip()
            self.messages = [{"role": "system", "content": self.system_prompt}]
            print(f"System prompt set to: {self.system_prompt}")
        else:
            print(f"Current system prompt: {self.system_prompt}")
    
    def do_exit(self, arg):
        """Exit the REPL."""
        self.voice_manager.cleanup()
        if self.debug_mode:
            print("Goodbye!")
        return True
    
    def do_q(self, arg):
        """Alias for exit."""
        return self.do_exit(arg)
    
    def do_quit(self, arg):
        """Alias for exit."""
        return self.do_exit(arg)
    
    def do_stop(self, arg):
        """Stop voice recognition or TTS playback."""
        # If in voice mode, exit voice mode
        if self.voice_mode_active:
            self._voice_stop_callback()
            return
            
        # Even if not in voice mode, stop any ongoing TTS
        if self.voice_manager:
            self.voice_manager.stop_speaking()
            # Do not show the "Stopped speech playback" message
            return
    
    def do_pause(self, arg):
        """Pause current TTS playback.
        
        Usage: /pause
        """
        if self.voice_manager:
            if self.voice_manager.pause_speaking():
                print("TTS playback paused. Use /resume to continue.")
            else:
                print("No active TTS playback to pause.")
        else:
            print("Voice manager not initialized.")
    
    def _reset_terminal(self):
        """Reset terminal state to prevent I/O blocking."""
        import sys
        import os
        
        try:
            # Flush all output streams
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Force terminal to reset input state
            if hasattr(sys.stdin, 'flush'):
                sys.stdin.flush()
            
            # On Unix-like systems, reset terminal
            if os.name == 'posix':
                os.system('stty sane 2>/dev/null')
                
        except Exception:
            # Ignore errors in terminal reset
            pass
    
    def do_resume(self, arg):
        """Resume paused TTS playback.
        
        Usage: /resume
        """
        if self.voice_manager:
            if self.voice_manager.is_paused():
                result = self.voice_manager.resume_speaking()
                if result:
                    print("TTS playback resumed.")
                else:
                    print("TTS was paused but playback already completed.")
                # Reset terminal after resume operation
                self._reset_terminal()
            else:
                print("No paused TTS playback to resume.")
        else:
            print("Voice manager not initialized.")
            
        # If neither voice mode nor TTS is active - don't show any message
        pass
    
    def do_help(self, arg):
        """Show help information."""
        print("Commands:")
        print("  /exit, /q, /quit    Exit REPL")
        print("  /clear              Clear history")
        print("  /tts on|off         Toggle TTS")
        print("  /voice <mode>       Voice input: off|full|wait|stop|ptt")
        print("  /voice ptt          Push-to-talk session (SPACE captures, ESC exits)")
        print("  /language <lang>    Switch voice language (en, fr, es, de, ru, zh)")
        print("  /setvoice [id]      List Piper voices or set one (lang.voice_id)")
        print("  /lang_info          Show current language information")
        print("  /list_languages     List all supported languages")
        print("  /speed <number>     Set TTS speed (0.5-2.0, default: 1.0, pitch preserved)")
        print("  /tts_voice ...      Select Piper vs cloned voice (see below)")
        print("  /tts_engine <e>     Switch TTS engine: auto|piper")
        print("  /whisper <model>    Switch Whisper model: tiny|base|small|medium|large")
        print("  /stt_engine <e>     Switch STT engine: auto|faster_whisper|whisper (whisper is optional extra)")
        print("  /speak <text>       Speak text (no LLM call)")
        print("  /transcribe <path>  Transcribe an audio file (faster-whisper by default)")
        print("  /system <prompt>    Set system prompt")
        print("  /stop               Stop voice mode or TTS playback")
        print("  /pause              Pause current TTS playback")
        print("  /resume             Resume paused TTS playback")
        print("  /aec on|off         Optional echo cancellation for true barge-in (requires [aec])")
        print("  /tokens             Display token usage stats")
        print("  /help               Show this help")
        print("  /clones             List cloned voices")
        print("  /clone_info <id>    Show cloned voice details")
        print("  /clone_ref <id>     Show cloned voice reference text")
        print("  /clone_rename ...   Rename a cloned voice")
        print("  /clone_rm <id>      Delete a cloned voice")
        print("  /clone_export ...   Export a cloned voice (.zip)")
        print("  /clone_import ...   Import a cloned voice (.zip)")
        print("  /clone <path> [nm]  Add a cloned voice from WAV/FLAC/OGG")
        print("  /clone-my-voice     Record a short prompt and clone it")
        print("  /tts_voice piper    Speak with Piper (default)")
        print("  /tts_voice clone X  Speak with a cloned voice (requires cloning runtime + cache)")
        print("  /cloning_status     Show cloning readiness (no downloads)")
        print("  /cloning_download   Explicitly download OpenF5 artifacts (~5.4GB)")
        print("  /clone_quality      Set cloned TTS speed/quality: fast|balanced|high")
        print("  /save <filename>    Save chat history to file")
        print("  /load <filename>    Load chat history from file")
        print("  /model <name>       Change the LLM model")
        print("  /temperature <val>  Set temperature (0.0-2.0, default: 0.7)")
        print("  /max_tokens <num>   Set max tokens (default: 4096)")
        print("  stop                (deprecated) use /voice off or say 'stop' during STOP mode")
        print("  <message>           Send to LLM (text mode)")
        print()
        print("Note: ALL commands must start with / except 'stop'")
        print("In STOP mode, say 'stop' / 'ok stop' to stop speaking (does not exit voice mode).")
    
    def emptyline(self):
        """Handle empty line input."""
        # Do nothing when an empty line is entered
        pass

    def do_tokens(self, arg):
        """Display token usage information."""
        try:
            # Always recalculate tokens to ensure accuracy
            self._reset_and_recalculate_tokens()
            
            total_tokens = self.system_tokens + self.user_tokens + self.assistant_tokens
            
            print(f"{Colors.YELLOW}Token usage:{Colors.END}")
            print(f"  System prompt: {self.system_tokens} tokens")
            print(f"  User messages: {self.user_tokens} tokens")
            print(f"  AI responses:  {self.assistant_tokens} tokens")
            print(f"  {Colors.BOLD}Total:         {total_tokens} tokens{Colors.END}")
        except Exception as e:
            if self.debug_mode:
                print(f"Error displaying token count: {e}")
            print("Token counting is not available.")
            pass

    def do_save(self, filename):
        """Save chat history to file."""
        try:
            # Add .mem extension if not specified
            if not filename.endswith('.mem'):
                filename = f"{filename}.mem"
                
            # Prepare memory file structure
            memory_data = {
                "header": {
                    "timestamp_utc": self._get_current_timestamp(),
                    "model": self.model,
                    "version": __import__('abstractvoice').__version__  # Get version from package __init__.py
                },
                "system_prompt": self.system_prompt,
                "token_stats": {
                    "system": self.system_tokens,
                    "user": self.user_tokens,
                    "assistant": self.assistant_tokens,
                    "total": self.system_tokens + self.user_tokens + self.assistant_tokens
                },
                "settings": {
                    "tts_speed": self.voice_manager.get_speed(),
                    "whisper_model": self.voice_manager.get_whisper(),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                "messages": self.messages
            }
            
            # Save to file with pretty formatting
            with open(filename, 'w') as f:
                json.dump(memory_data, f, indent=2)
                
            print(f"Chat history saved to {filename}")
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving chat history: {e}")
            print(f"Failed to save chat history to {filename}")
    
    def _get_current_timestamp(self):
        """Get current timestamp in the format YYYY-MM-DD HH-MM-SS."""
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H-%M-%S")

    def do_load(self, filename):
        """Load chat history from file."""
        try:
            # Add .mem extension if not specified
            if not filename.endswith('.mem'):
                filename = f"{filename}.mem"
                
            if self.debug_mode:
                print(f"Attempting to load from: {filename}")
                
            with open(filename, 'r') as f:
                memory_data = json.load(f)
                
            if self.debug_mode:
                print(f"Successfully loaded JSON data from {filename}")
            
            # Handle both formats: new .mem format and legacy format (just messages array)
            if isinstance(memory_data, dict) and "messages" in memory_data:
                # New .mem format
                if self.debug_mode:
                    print("Processing .mem format with messages")
                
                # Update model if specified
                if "header" in memory_data and "model" in memory_data["header"]:
                    old_model = self.model
                    self.model = memory_data["header"]["model"]
                    print(f"Model changed from {old_model} to {self.model}")
                
                # Update system prompt
                if "system_prompt" in memory_data:
                    self.system_prompt = memory_data["system_prompt"]
                    if self.debug_mode:
                        print(f"Updated system prompt: {self.system_prompt}")
                
                # Load messages
                if "messages" in memory_data and isinstance(memory_data["messages"], list):
                    self.messages = memory_data["messages"]
                    if self.debug_mode:
                        print(f"Loaded {len(self.messages)} messages")
                else:
                    print("Invalid messages format in memory file")
                    return
                    
                # Recompute token stats if available
                self._reset_and_recalculate_tokens()
                
                # Restore settings if available
                if "settings" in memory_data:
                    try:
                        settings = memory_data["settings"]
                        
                        # Restore TTS speed
                        if "tts_speed" in settings:
                            speed = settings.get("tts_speed", 1.0)
                            self.voice_manager.set_speed(speed)
                            # Don't need to update the voice manager immediately as the
                            # speed will be used in the next speak() call
                            print(f"TTS speed set to {speed}x")
                        
                        # Restore Whisper model
                        if "whisper_model" in settings:
                            whisper_model = settings.get("whisper_model", "tiny")
                            self.voice_manager.set_whisper(whisper_model)
                            
                        # Restore temperature
                        if "temperature" in settings:
                            temp = settings.get("temperature", 0.4)
                            self.temperature = temp
                            print(f"Temperature set to {temp}")
                            
                        # Restore max_tokens
                        if "max_tokens" in settings:
                            tokens = settings.get("max_tokens", 4096)
                            self.max_tokens = tokens
                            print(f"Max tokens set to {tokens}")
                            
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error restoring settings: {e}")
                        # Continue loading even if settings restoration fails
                
            elif isinstance(memory_data, list):
                # Legacy format (just an array of messages)
                self.messages = memory_data
                
                # Reset token counts and recalculate
                self._reset_and_recalculate_tokens()
                
                # Extract system prompt if present
                for msg in self.messages:
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        self.system_prompt = msg.get("content", self.system_prompt)
                        break
            else:
                print("Invalid memory file format")
                return
                
            # Ensure there's a system message
            self._ensure_system_message()
                
            print(f"Chat history loaded from {filename}")
            
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except json.JSONDecodeError as e:
            if self.debug_mode:
                print(f"Invalid JSON format in {filename}: {e}")
            print(f"Invalid JSON format in {filename}")
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading chat history: {str(e)}")
                import traceback
                traceback.print_exc()
            print(f"Failed to load chat history from {filename}")
    
    def _reset_and_recalculate_tokens(self):
        """Reset token counts and recalculate for all messages."""
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        
        # Count tokens for all messages
        for msg in self.messages:
            if isinstance(msg, dict) and "content" in msg and "role" in msg:
                self._count_tokens(msg["content"], msg["role"])
    
    def _ensure_system_message(self):
        """Ensure there's a system message at the start of messages."""
        has_system = False
        for msg in self.messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                has_system = True
                break
                
        if not has_system:
            # Prepend a system message if none exists
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})
    
    def do_model(self, model_name):
        """Change the LLM model."""
        if not model_name:
            print(f"Current model: {self.model}")
            return
            
        old_model = self.model
        self.model = model_name
        print(f"Model changed from {old_model} to {model_name}")
        
        # Don't add a system message about model change

    def do_temperature(self, arg):
        """Set the temperature parameter for the LLM."""
        if not arg.strip():
            print(f"Current temperature: {self.temperature}")
            return
            
        try:
            temp = float(arg.strip())
            if 0.0 <= temp <= 2.0:
                old_temp = self.temperature
                self.temperature = temp
                print(f"Temperature changed from {old_temp} to {temp}")
            else:
                print("Temperature should be between 0.0 and 2.0")
        except ValueError:
            print("Usage: temperature <number>  (e.g., temperature 0.7)")
    
    def do_max_tokens(self, arg):
        """Set the max_tokens parameter for the LLM."""
        if not arg.strip():
            print(f"Current max_tokens: {self.max_tokens}")
            return
            
        try:
            tokens = int(arg.strip())
            if tokens > 0:
                old_tokens = self.max_tokens
                self.max_tokens = tokens
                print(f"Max tokens changed from {old_tokens} to {tokens}")
            else:
                print("Max tokens should be a positive integer")
        except ValueError:
            print("Usage: max_tokens <number>  (e.g., max_tokens 2048)")
        
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AbstractVoice CLI Example")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--api", default="http://localhost:11434/api/chat",
                      help="LLM API URL")
    parser.add_argument("--model", default="granite3.3:2b",
                      help="LLM model name")
    parser.add_argument("--language", "--lang", default="en",
                      choices=["en", "fr", "es", "de", "it", "ru", "multilingual"],
                      help="Voice language (en=English, fr=French, es=Spanish, de=German, it=Italian, ru=Russian, multilingual=All)")
    parser.add_argument("--tts-model",
                      help="Specific TTS model to use (overrides language default)")
    return parser.parse_args()


def main():
    """Entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Initialize and run REPL with language support
        repl = VoiceREPL(
            api_url=args.api,
            model=args.model,
            debug_mode=args.debug,
            language=args.language,
            tts_model=args.tts_model
        )
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main() 