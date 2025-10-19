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
import sys
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

        # Initialize voice manager with language support
        if disable_tts:
            self.voice_manager = None
            print("🔇 TTS disabled - text-only mode")
        else:
            self.voice_manager = VoiceManager(
                language=language,
                tts_model=tts_model,
                debug_mode=debug_mode
            )
        
        # Settings
        self.use_tts = True
        self.voice_mode = "off"  # off, full, wait, stop, ptt
        self.voice_mode_active = False  # Is voice recognition running?
        
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
        intro += "  • Use /voice <mode> to enable voice input\n"
        intro += "  • Use /language <lang> to switch voice language\n"
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
        line = line.strip()
        
        # If line starts with /, remove it for command processing
        if line.startswith('/'):
            line = line[1:].strip()
        
        # Call parent parseline to do the actual parsing
        return super().parseline(line)
        
    def default(self, line):
        """Handle regular text input.
        
        Only 'stop' is recognized as a command without /
        All other commands MUST use / prefix.
        """
        # Skip empty lines
        if not line.strip():
            return
        
        # ONLY 'stop' is recognized without / (for voice mode convenience)
        if line.strip().lower() == "stop":
            return self.do_stop("")
        
        # Check if in voice mode - don't send to LLM
        if self.voice_mode_active:
            if self.debug_mode:
                print(f"Voice mode active ({self.voice_mode}). Use /voice off or say 'stop' to exit.")
            return
        
        # Everything else goes to LLM
        self.process_query(line.strip())
        
    def process_query(self, query):
        """Process a query and get a response from the LLM."""
        if not query:
            return
            
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
                self.voice_manager.speak(response_text)
                
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
        Available languages: en, fr, es, de, it
        """
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
                'it': "Lingua cambiata in italiano."
            }
            test_msg = test_messages.get(language, "Language switched.")
            self.voice_manager.speak(test_msg)

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
          /setvoice                    # List all voices with JSON-like info
          /setvoice fr.css10_vits      # Set French CSS10 VITS voice
          /setvoice it.mai_male_vits   # Set Italian male VITS voice
        """
        if not args:
            # Show all available voices with metadata
            print(f"\n{Colors.CYAN}Available Voice Models:{Colors.END}")

            try:
                models = self.voice_manager.list_available_models()

                for language, voices in models.items():
                    # Get language name
                    lang_names = {
                        'en': 'English', 'fr': 'French', 'es': 'Spanish',
                        'de': 'German', 'it': 'Italian'
                    }
                    lang_name = lang_names.get(language, language.upper())

                    print(f"\n🌍 {lang_name} ({language}):")

                    for voice_id, voice_info in voices.items():
                        cached_icon = "✅" if voice_info.get('cached', False) else "📥"
                        quality_icon = "✨" if voice_info['quality'] == 'excellent' else "🔧"
                        size_text = f"{voice_info['size_mb']}MB"

                        print(f"  {cached_icon} {quality_icon} {language}.{voice_id}")
                        print(f"      {voice_info['name']} ({size_text})")
                        print(f"      {voice_info['description']}")
                        if voice_info.get('requires_espeak', False):
                            print(f"      ⚠️ Requires espeak-ng")

                print(f"\n{Colors.YELLOW}Usage:{Colors.END}")
                print("  /setvoice <language>.<voice_id>")
                print("  Example: /setvoice fr.css10_vits")
                print("\n📥 = Download needed  ✅ = Ready  ✨ = High quality  🔧 = Good quality")

            except Exception as e:
                print(f"❌ Error listing models: {e}")
                # Fallback to old method
                self.voice_manager.list_voices()
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

            # Use the programmatic download API
            success = self.voice_manager.download_model(voice_spec)

            if success:
                # Now set the language to match
                success = self.voice_manager.set_language(language)

                if success:
                    # Update current language
                    self.current_language = language

                    print(f"✅ Voice set to {voice_spec}")

                    # Test the voice
                    test_messages = {
                        'en': 'Voice changed to English.',
                        'fr': 'Voix changée en français.',
                        'es': 'Voz cambiada al español.',
                        'de': 'Stimme auf Deutsch geändert.',
                        'it': 'Voce cambiata in italiano.'
                    }
                    test_msg = test_messages.get(language, f'Voice changed to {language}.')
                    self.voice_manager.speak(test_msg)

                    # Restart voice mode if it was active
                    if was_active:
                        self.do_voice(self.voice_mode)
                else:
                    print(f"❌ Failed to set language: {language}")
            else:
                print(f"❌ Failed to download voice: {voice_spec}")
                print("   Check your internet connection or try a different voice")

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
          stop - Only stops TTS on 'stop' keyword (planned)
          ptt  - Push-to-talk mode (planned)
        """
        arg = arg.lower().strip()
        
        # Handle legacy "on" argument
        if arg == "on":
            arg = "wait"
        
        if arg in ["off", "full", "wait", "stop", "ptt"]:
            # If switching from one mode to another, stop current mode first
            if self.voice_mode_active and arg != "off":
                self._voice_stop_callback()
            
            self.voice_mode = arg
            self.voice_manager.set_voice_mode(arg)
            
            if arg == "off":
                if self.voice_mode_active:
                    self._voice_stop_callback()
            else:
                # Start voice recognition for non-off modes
                self.voice_mode_active = True
                
                # Start listening with callbacks
                self.voice_manager.listen(
                    on_transcription=self._voice_callback,
                    on_stop=lambda: self._voice_stop_callback()
                )
                
                # Print mode-specific instructions
                if arg == "full":
                    print("Voice mode: FULL - Continuous listening, interrupts TTS on speech.")
                    print("Say 'stop' to exit.")
                elif arg == "wait":
                    print("Voice mode: WAIT - Pauses listening while speaking (recommended).")
                    print("Say 'stop' to exit.")
                elif arg == "stop":
                    print("Voice mode: STOP (Planned) - Only stops TTS on 'stop' keyword.")
                    print("Currently same as WAIT mode.")
                elif arg == "ptt":
                    print("Voice mode: PTT (Planned) - Push-to-talk functionality.")
                    print("Currently same as WAIT mode.")
        else:
            print("Usage: /voice off | full | wait | stop | ptt")
            print("  off  - Disable voice input")
            print("  full - Continuous listening, interrupts TTS on speech")
            print("  wait - Pause listening while speaking (recommended)")
            print("  stop - Only stop TTS on 'stop' keyword (planned)")
            print("  ptt  - Push-to-talk mode (planned)")
    
    def _voice_callback(self, text):
        """Callback for voice recognition."""
        # Print what the user said
        print(f"\n> {text}")
        
        # Check if the user said 'stop' to exit voice mode
        if text.lower() == "stop":
            self._voice_stop_callback()
            # Don't process "stop" as a query
            return
        
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
            print("TTS enabled" if self.debug_mode else "")
        elif arg == "off":
            self.use_tts = False
            print("TTS disabled" if self.debug_mode else "")
        else:
            print("Usage: /tts on | off")
    
    def do_speed(self, arg):
        """Set the TTS speed multiplier."""
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
        """Change TTS model.
        
        Available models (quality ranking):
          vits          - BEST quality (requires espeak-ng)
          fast_pitch    - Good quality (works everywhere)
          glow-tts      - Alternative fallback
          tacotron2-DDC - Legacy
        
        Usage:
          /tts_model vits
          /tts_model fast_pitch
        """
        model_shortcuts = {
            'vits': 'tts_models/en/ljspeech/vits',
            'fast_pitch': 'tts_models/en/ljspeech/fast_pitch',
            'glow-tts': 'tts_models/en/ljspeech/glow-tts',
            'tacotron2-DDC': 'tts_models/en/ljspeech/tacotron2-DDC',
        }
        
        arg = arg.strip()
        if not arg:
            print("Usage: /tts_model <model_name>")
            print("Available models: vits (best), fast_pitch, glow-tts, tacotron2-DDC")
            return
        
        # Get full model name
        model_name = model_shortcuts.get(arg, arg)
        
        print(f"Changing TTS model to: {model_name}")
        try:
            self.voice_manager.set_tts_model(model_name)
            print("✓ TTS model changed successfully")
        except Exception as e:
            print(f"✗ Error changing model: {e}")
    
    def do_whisper(self, arg):
        """Change Whisper model."""
        model = arg.strip()
        if not model:
            print(f"Current Whisper model: {self.voice_manager.get_whisper()}")
            return
        
        self.voice_manager.set_whisper(model)            
    
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
        print("  /language <lang>    Switch voice language (en, fr, es, de, it)")
        print("  /setvoice [id]      List voices or set specific voice (lang.voice_id)")
        print("  /lang_info          Show current language information")
        print("  /list_languages     List all supported languages")
        print("  /speed <number>     Set TTS speed (0.5-2.0, default: 1.0, pitch preserved)")
        print("  /tts_model <model>  Switch TTS model: vits(best)|fast_pitch|glow-tts|tacotron2-DDC")
        print("  /whisper <model>    Switch Whisper model: tiny|base|small|medium|large")
        print("  /system <prompt>    Set system prompt")
        print("  /stop               Stop voice mode or TTS playback")
        print("  /pause              Pause current TTS playback")
        print("  /resume             Resume paused TTS playback")
        print("  /tokens             Display token usage stats")
        print("  /help               Show this help")
        print("  /save <filename>    Save chat history to file")
        print("  /load <filename>    Load chat history from file")
        print("  /model <name>       Change the LLM model")
        print("  /temperature <val>  Set temperature (0.0-2.0, default: 0.7)")
        print("  /max_tokens <num>   Set max tokens (default: 4096)")
        print("  stop                Stop voice mode or TTS (voice command)")
        print("  <message>           Send to LLM (text mode)")
        print()
        print("Note: ALL commands must start with / except 'stop'")
        print("In voice mode, say 'stop' to exit voice mode.")
    
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