#!/usr/bin/env python3
"""
CLI example using AbstractVoice with a text-generation API.

This example shows how to use AbstractVoice to create a CLI application
that interacts with an LLM API for text generation.
"""

import argparse
import cmd
import atexit
import json
import re
import shlex
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
    prompt = "> "
    
    # Override cmd module settings
    ruler = ""  # No horizontal rule line
    use_rawinput = True
    
    def __init__(
        self,
        api_url="http://localhost:11434/api/chat",
        model="cogito:3b",
        debug_mode=False,
        verbose_mode: bool = False,
        language="en",
        tts_model=None,
        voice_mode: str = "off",
        disable_tts=False,
        cloning_engine: str = "f5_tts",
    ):
        super().__init__()

        # Best-effort: enable proper line editing + history (Up/Down arrows).
        # Some Python builds (notably when built without readline/libedit) will
        # otherwise treat arrow keys as escape sequences and corrupt the prompt.
        self._init_readline()

        # Debug mode
        self.debug_mode = debug_mode
        self.verbose_mode = bool(verbose_mode)

        # API settings
        self.api_url = api_url
        self.model = model
        self.temperature = 0.4
        self.max_tokens = 4096

        # Language settings
        self.current_language = language
        self._initial_tts_model = tts_model
        self.cloning_engine = str(cloning_engine or "f5_tts").strip().lower()

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
                cloning_engine=self.cloning_engine,
            )

        # Current speaking voice:
        # - None => Piper (default, language-driven)
        # - str  => cloned voice_id
        self.current_tts_voice: str | None = None

        # When reference_text is auto-generated via ASR ("asr" source), print a
        # ready-to-copy `/clone_set_ref_text ...` hint once per voice for easy correction.
        self._printed_asr_ref_text_hint: set[str] = set()

        # Seed a default cloned voice (HAL9000) if samples are present.
        self._seed_hal9000_voice()
        
        # Settings
        self.use_tts = True
        # Voice input mode (mic). Default: OFF for fast startup + offline-first.
        # Use `--voice-mode stop` (or `/voice stop`) to enable hands-free.
        self.voice_mode = (voice_mode or "off").strip().lower()  # off, full, wait, stop, ptt
        self.voice_mode_active = False  # Is voice recognition running?
        self._ptt_session_active = False
        self._ptt_recording = False
        self._ptt_busy = False
        
        # System prompt
        self.system_prompt = "You are a Helpful Voice Assistant. By design, your answers are short and conversational, unless specifically asked to detail something. You only speak, so never use any text formatting, hinting, *emotions*, emojis or markdown. Incarnate the speaker, never comment your instructions."
        
        # Message history
        self.messages = [{"role": "system", "content": self.system_prompt}]
        
        # Token counting
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        # LLM token totals (best-effort, Ollama API `eval_count`).
        self.total_llm_out_tokens = 0
        # Word counting
        self.system_words = 0
        self.user_words = 0
        self.assistant_words = 0
        # Best-effort tokenizer cache (tiktoken optional).
        self._tiktoken_encoding = None
        self._tiktoken_unavailable = False
        self._count_system_tokens()
        self._count_system_words()

        # Best-effort metrics captured from voice input paths.
        self._pending_stt_metrics: dict | None = None

        if self.debug_mode:
            print(f"Initialized with API URL: {api_url}")
            print(f"Using model: {model}")

        # Optionally auto-start voice input (mic). Keep OFF by default to avoid
        # loading STT models (slow) unless the user explicitly opts in.
        if self.voice_manager and self.voice_mode and self.voice_mode != "off":
            try:
                self.do_voice(self.voice_mode)
            except Exception:
                # Never block REPL start.
                self.voice_mode = "off"
                self.voice_mode_active = False

        # Set intro with help information
        self.intro = self._get_intro()

    def _init_readline(self) -> None:
        """Initialize readline history + make ANSI prompts safe (best-effort)."""
        rl = None
        try:
            import readline as _readline  # type: ignore

            rl = _readline
        except Exception:
            # Windows users may have pyreadline3 installed.
            try:
                import pyreadline3 as _readline  # type: ignore

                rl = _readline
            except Exception:
                rl = None

        if rl is None:
            # Keep prompt simple and avoid ANSI; prevents strange cursor behavior
            # when arrow keys emit escape codes in cooked terminals.
            self.prompt = "> "
            return

        # Keep prompt plain when readline is enabled. ANSI prompts are fragile
        # across readline/libedit builds and can corrupt redraw/history behavior.
        self.prompt = "> "

        # Persist history across sessions (best-effort).
        try:
            from pathlib import Path

            try:
                import appdirs

                hist_dir = Path(appdirs.user_data_dir("abstractvoice"))
            except Exception:
                hist_dir = Path.home() / ".abstractvoice"

            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hist_dir / "repl_history"

            try:
                rl.read_history_file(str(hist_path))
            except FileNotFoundError:
                pass
            except Exception:
                pass

            try:
                rl.set_history_length(2000)
            except Exception:
                pass

            def _save_history():
                try:
                    rl.write_history_file(str(hist_path))
                except Exception:
                    pass

            atexit.register(_save_history)
        except Exception:
            pass

        # Ensure Up/Down arrows traverse history reliably across GNU readline and
        # macOS libedit-backed readline. Some libedit defaults perform prefix
        # search/completion, which can look like text is being appended.
        try:
            doc = getattr(rl, "__doc__", "") or ""
            is_libedit = "libedit" in doc.lower()
            if is_libedit:
                # libedit syntax
                rl.parse_and_bind("bind ^[[A ed-prev-history")
                rl.parse_and_bind("bind ^[[B ed-next-history")
                rl.parse_and_bind("bind ^[[OA ed-prev-history")
                rl.parse_and_bind("bind ^[[OB ed-next-history")
            else:
                # GNU readline syntax
                rl.parse_and_bind('"\\e[A": previous-history')
                rl.parse_and_bind('"\\e[B": next-history')
                rl.parse_and_bind('"\\eOA": previous-history')
                rl.parse_and_bind('"\\eOB": next-history')
        except Exception:
            pass
        
    def _get_intro(self):
        """Generate intro message with help."""
        intro = f"\n{Colors.BOLD}Welcome to AbstractVoice CLI REPL{Colors.END}\n"
        if self.voice_manager:
            lang_name = self.voice_manager.get_language_name()
            mic = (self.voice_mode or "off").upper()
            intro += f"API: {self.api_url} | Model: {self.model} | Voice: {lang_name} | Mic: {mic} | Cloning: {self.cloning_engine}\n"
        else:
            intro += f"API: {self.api_url} | Model: {self.model} | Voice: Disabled\n"
        intro += f"\n{Colors.CYAN}Quick Start:{Colors.END}\n"
        intro += "  • Type messages to chat with the LLM\n"
        intro += "  • Voice input (mic): off by default. Enable: /voice stop  (or start with --voice-mode stop)\n"
        intro += "  • PTT: /voice ptt then SPACE to capture (ESC exits)\n"
        intro += "  • Use /language <lang> to switch voice language\n"
        intro += "  • Use /clones and /tts_voice to use cloned voices\n"
        intro += "  • Type /help for full command list\n"
        intro += "  • Type /exit or /q to quit\n"
        return intro
        
    def _count_system_tokens(self):
        """Count tokens in the system prompt."""
        self._count_tokens(self.system_prompt, "system")

    def _count_system_words(self):
        self.system_words = self._count_words(self.system_prompt)

    def _count_words(self, text: str) -> int:
        s = str(text or "").strip()
        if not s:
            return 0
        # A "word" here is whitespace-delimited for simplicity across languages.
        return len([w for w in re.split(r"\s+", s) if w])

    def _get_tiktoken_encoding(self):
        if getattr(self, "_tiktoken_unavailable", False):
            return None
        enc = getattr(self, "_tiktoken_encoding", None)
        if enc is not None:
            return enc
        try:
            import tiktoken
        except ImportError:
            self._tiktoken_unavailable = True
            return None

        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._tiktoken_unavailable = True
                return None

        self._tiktoken_encoding = enc
        return enc

    def _fmt_s(self, seconds: float | None) -> str:
        try:
            if seconds is None:
                return "--"
            s = float(seconds)
            if s < 0:
                return "--"
            # Keep it compact but readable.
            if s < 10:
                return f"{s:.2f}s"
            return f"{s:.1f}s"
        except Exception:
            return "--"

    def _fmt_num(self, x: float | None, *, digits: int = 2) -> str:
        try:
            if x is None:
                return "--"
            return f"{float(x):.{int(digits)}f}"
        except Exception:
            return "--"

    def _fmt_wtok(self, words: int | None, tokens: int | None) -> str:
        w = int(words) if isinstance(words, int) else (int(words) if words is not None else 0)
        if isinstance(tokens, int):
            return f"{w}w/{int(tokens)}tok"
        return f"{w}w/--tok"

    def _summarize_audio_source(self, source: str) -> tuple[int | None, float | None]:
        """Best-effort: return (file_count, total_seconds) for an audio source path."""
        try:
            from pathlib import Path

            p = Path(str(source)).expanduser()
        except Exception:
            return None, None

        try:
            import soundfile as sf
        except Exception:
            return None, None

        supported = {".wav", ".flac", ".ogg"}
        files = []
        try:
            if p.is_file():
                files = [p]
            elif p.is_dir():
                files = sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() in supported])
            else:
                return None, None
        except Exception:
            return None, None

        total_s = 0.0
        max_files = 25
        for fp in files[:max_files]:
            try:
                info = sf.info(str(fp))
                d = float(getattr(info, "duration", 0.0) or 0.0)
                if d > 0:
                    total_s += d
            except Exception:
                continue

        # If there are too many files, the displayed duration is a lower bound.
        return (int(len(files)) if files else 0), (float(total_s) if total_s > 0 else None)

    def _print_verbose_turn_stats(self, turn: dict) -> None:
        if not bool(getattr(self, "verbose_mode", False)):
            return
        if not isinstance(turn, dict):
            return

        stt = turn.get("stt") if isinstance(turn.get("stt"), dict) else None
        llm = turn.get("llm") if isinstance(turn.get("llm"), dict) else {}
        counts = turn.get("counts") if isinstance(turn.get("counts"), dict) else {}
        tts = turn.get("tts") if isinstance(turn.get("tts"), dict) else None

        in_w = counts.get("in_words")
        out_w = counts.get("out_words")
        in_t = counts.get("in_tokens")
        out_t = counts.get("out_tokens")

        llm_s = llm.get("s")
        api = llm.get("api") if isinstance(llm.get("api"), dict) else {}
        api_prompt_tok = api.get("prompt_eval_count") if isinstance(api.get("prompt_eval_count"), int) else None
        api_out_tok = api.get("eval_count") if isinstance(api.get("eval_count"), int) else None

        # Line 1: STT (if any) + LLM + in/out counts and written speed.
        parts1 = []
        if stt:
            stt_s = stt.get("stt_s")
            stt_a = stt.get("audio_s")
            stt_rtf = stt.get("rtf")
            stt_txt = f"STT {self._fmt_s(stt_s)}"
            if stt_a:
                stt_txt += f"(a{self._fmt_s(stt_a)})"
            if stt_rtf is not None:
                stt_txt += f" rtf{self._fmt_num(stt_rtf, digits=2)}"
            parts1.append(stt_txt)

        if llm_s is not None or api_prompt_tok is not None or api_out_tok is not None:
            llm_txt = f"LLM {self._fmt_s(llm_s)}"
            if api_prompt_tok is not None or api_out_tok is not None:
                p = str(api_prompt_tok) if api_prompt_tok is not None else "--"
                o = str(api_out_tok) if api_out_tok is not None else "--"
                llm_txt += f" (api p{p} o{o})"
            parts1.append(llm_txt)

        in_txt = f"in {self._fmt_wtok(in_w, in_t)}"
        out_txt = f"out {self._fmt_wtok(out_w, out_t)}"

        wps_written = None
        try:
            if isinstance(out_w, int) and out_w > 0 and llm_s and float(llm_s) > 0:
                wps_written = float(out_w) / float(llm_s)
        except Exception:
            wps_written = None

        if wps_written is not None:
            out_txt += f" ({self._fmt_num(wps_written, digits=1)}w/s)"

        parts1.append(in_txt)
        parts1.append(out_txt)

        line1 = " | ".join(parts1)

        # Line 2: TTS (if any) + spoken speed + totals.
        parts2 = []
        if self.voice_manager and self.use_tts:
            if not tts:
                parts2.append("TTS --")
            else:
                eng = str(tts.get("engine") or "").strip().lower()
                if eng == "clone":
                    ce = tts.get("clone_engine")
                    label = f"clone[{ce}]" if ce else "clone"
                elif eng:
                    label = eng
                else:
                    label = "tts"

                err = (tts.get("error") or "").strip()
                if err:
                    # Keep single-line and short.
                    msg = " ".join(err.split())
                    if len(msg) > 120:
                        msg = msg[:120].rstrip() + "…"
                    parts2.append(f"TTS {label} ERR {msg}")
                else:
                    synth_s = tts.get("synth_s")
                    audio_s = tts.get("audio_s")
                    rtf = tts.get("rtf")
                    tts_txt = f"TTS {label} {self._fmt_s(synth_s)}→{self._fmt_s(audio_s)}"
                    if rtf is not None:
                        tts_txt += f" rtf{self._fmt_num(rtf, digits=2)}"

                    # Extra clone streaming details when available.
                    if eng == "clone" and bool(tts.get("streaming")):
                        ttfb_s = tts.get("ttfb_s")
                        if ttfb_s is not None:
                            tts_txt += f" ttfb{self._fmt_s(ttfb_s)}"
                        ch = tts.get("chunks")
                        if isinstance(ch, int):
                            tts_txt += f" ch{ch}"

                    wps_spoken = None
                    try:
                        if isinstance(out_w, int) and out_w > 0 and audio_s and float(audio_s) > 0:
                            wps_spoken = float(out_w) / float(audio_s)
                    except Exception:
                        wps_spoken = None
                    if wps_spoken is not None:
                        tts_txt += f" ({self._fmt_num(wps_spoken, digits=1)}w/s)"

                    parts2.append(tts_txt)
        else:
            parts2.append("TTS off")

        total_words = int(getattr(self, "system_words", 0) + getattr(self, "user_words", 0) + getattr(self, "assistant_words", 0))
        total_tokens = None
        if self._get_tiktoken_encoding() is not None:
            total_tokens = int(getattr(self, "system_tokens", 0) + getattr(self, "user_tokens", 0) + getattr(self, "assistant_tokens", 0))

        tot_txt = f"tot {self._fmt_wtok(total_words, total_tokens)}"
        if isinstance(getattr(self, "total_llm_out_tokens", None), int) and getattr(self, "total_llm_out_tokens") > 0:
            tot_txt += f" (api out {int(getattr(self, 'total_llm_out_tokens'))}tok)"
        parts2.append(tot_txt)

        line2 = " | ".join(parts2)

        # Keep it readable; two lines max.
        print(f"{Colors.YELLOW}{line1}{Colors.END}")
        print(f"{Colors.YELLOW}{line2}{Colors.END}")
    
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
        text = line.strip()
        if not text:
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

        # Shortcut: paste a reference audio path to clone+use a voice.
        # Examples:
        #   audio_samples/hal9000/hal9000_hello.wav
        #   audio_samples/hal9000/hal9000_hello.wav | Hello, Dave.
        if self._maybe_handle_clone_shortcut(text):
            return
        
        # Everything else goes to LLM
        self._pending_stt_metrics = None
        self.process_query(text)

    # NOTE: PTT is implemented as a dedicated key-loop session (no typing).

    def _maybe_handle_clone_shortcut(self, text: str) -> bool:
        """Best-effort: treat a pasted WAV/FLAC/OGG path as `/clone_use`."""
        if not self.voice_manager:
            return False

        raw = (text or "").strip()
        if not raw:
            return False
        if raw.startswith("/"):
            return False

        # Optional transcript with a simple pipe syntax:
        #   path.wav | Hello.
        left, sep, right = raw.partition("|")
        path_str = left.strip()
        ref_text = right.strip() if sep else ""
        reference_text = ref_text or None

        # Strip naive wrapping quotes.
        if (path_str.startswith('"') and path_str.endswith('"')) or (path_str.startswith("'") and path_str.endswith("'")):
            path_str = path_str[1:-1].strip()

        try:
            from pathlib import Path

            p = Path(path_str).expanduser()
        except Exception:
            return False

        if not p.exists():
            return False

        exts = {".wav", ".flac", ".ogg"}
        if p.is_file() and p.suffix.lower() not in exts:
            return False
        if p.is_dir():
            try:
                has_audio = any(x.is_file() and x.suffix.lower() in exts for x in p.iterdir())
            except Exception:
                has_audio = False
            if not has_audio:
                return False

        # Build a `/clone_use` call with a stable name.
        import shlex as _shlex

        default_name = p.stem if p.is_file() else p.name
        args = f"{_shlex.quote(str(p))} {_shlex.quote(default_name)}"
        if reference_text:
            args += f" --text {_shlex.quote(reference_text)}"
        try:
            self.do_clone_use(args)
        except Exception as e:
            print(f"❌ Clone shortcut failed: {e}")
            if self.debug_mode:
                import traceback

                traceback.print_exc()
        return True
        
    def process_query(self, query):
        """Process a query and get a response from the LLM."""
        if not query:
            return

        # Consume any pending STT metrics for this turn (voice/PTT input).
        stt_metrics = getattr(self, "_pending_stt_metrics", None)
        self._pending_stt_metrics = None

        # If audio is currently playing, stop it so the new request can be handled
        # without overlapping speech.
        try:
            if self.voice_manager:
                self.voice_manager.stop_speaking()
        except Exception:
            pass
            
        # Per-turn counts
        user_words = self._count_words(query)
        self.user_words += int(user_words)
        user_tokens = self._count_tokens(query, "user")
        
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
            llm_t0 = time.monotonic()
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Try to parse response
            try:
                # First, try to parse as JSON
                response_data = response.json()
                api_llm_metrics = {}
                try:
                    # Ollama exposes timing + token counts (nanoseconds).
                    # Keep best-effort: if fields are missing, we just omit them.
                    for k in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    ):
                        if k in response_data:
                            api_llm_metrics[k] = response_data.get(k)
                except Exception:
                    api_llm_metrics = {}
                
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
                api_llm_metrics = {}
                
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
            llm_t1 = time.monotonic()
            llm_s = float(llm_t1 - llm_t0)
            
            # Per-turn counts
            assistant_words = self._count_words(response_text)
            self.assistant_words += int(assistant_words)
            assistant_tokens = self._count_tokens(response_text, "assistant")
            
            # Add to message history
            self.messages.append({"role": "assistant", "content": response_text})
            
            # Display the response with color
            print(f"{Colors.CYAN}{response_text}{Colors.END}")
            
            # Record last-turn stats (best-effort; printed only in verbose mode).
            self._last_turn_metrics = {
                "stt": stt_metrics,
                "llm": {
                    "s": llm_s,
                    "api": api_llm_metrics,
                },
                "counts": {
                    "in_words": int(user_words),
                    "out_words": int(assistant_words),
                    "in_tokens": int(user_tokens) if isinstance(user_tokens, int) else None,
                    "out_tokens": int(assistant_tokens) if isinstance(assistant_tokens, int) else None,
                },
            }
            try:
                out_tok = api_llm_metrics.get("eval_count") if isinstance(api_llm_metrics, dict) else None
                if isinstance(out_tok, int) and out_tok >= 0:
                    self.total_llm_out_tokens += int(out_tok)
            except Exception:
                pass

            # Speak the response if voice manager is available
            if self.voice_manager and self.use_tts:
                try:
                    # UX guard: never trigger big cloning downloads during normal chat.
                    if self.current_tts_voice and not self._is_cloning_runtime_ready(voice_id=self.current_tts_voice):
                        print(
                            "ℹ️  Cloned voice selected but cloning runtime is not ready.\n"
                            "   Run /cloning_status then /cloning_download, or switch back with /tts_voice piper."
                        )
                    else:
                        self._speak_with_spinner_until_audio_starts(response_text)
                except Exception as e:
                    print(f"❌ TTS failed: {e}")

            # Capture best-effort TTS metrics (Piper or cloned).
            tts_metrics = None
            try:
                if self.voice_manager and hasattr(self.voice_manager, "pop_last_tts_metrics"):
                    tts_metrics = self.voice_manager.pop_last_tts_metrics()
            except Exception:
                tts_metrics = None

            try:
                if isinstance(getattr(self, "_last_turn_metrics", None), dict):
                    self._last_turn_metrics["tts"] = tts_metrics
            except Exception:
                pass

            # Verbose stats (max 2 lines).
            try:
                if self.verbose_mode and isinstance(getattr(self, "_last_turn_metrics", None), dict):
                    self._print_verbose_turn_stats(self._last_turn_metrics)
            except Exception:
                pass
                
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
        encoding = self._get_tiktoken_encoding()
        if encoding is None:
            return None
        try:
            token_count = len(encoding.encode(str(text or "")))
        except Exception as e:
            if self.debug_mode:
                print(f"Error counting tokens: {e}")
            return None

        # Update the token counts based on role
        if role == "system":
            self.system_tokens = int(token_count)
        elif role == "user":
            self.user_tokens += int(token_count)
        elif role == "assistant":
            self.assistant_tokens += int(token_count)

        if self.debug_mode:
            total_tokens = self.system_tokens + self.user_tokens + self.assistant_tokens
            print(f"{role.capitalize()} tokens: {token_count}")
            print(f"Total tokens: {total_tokens}")
        return int(token_count)
    
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
                    on_stop=lambda: (
                        print("\n⏹️  Stopped speaking.\n") if (self.voice_manager and self.voice_manager.is_speaking()) else None
                    ),
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
        cols = 80
        try:
            cols = int(shutil.get_terminal_size((80, 20)).columns)
        except Exception:
            cols = 80

        def _clear_status() -> None:
            try:
                sys.stdout.write("\r" + (" " * max(10, cols - 1)) + "\r")
                sys.stdout.flush()
            except Exception:
                pass

        def _status_line(msg: str) -> None:
            # Render on a single line (no newline) so SPACE can be pressed repeatedly.
            try:
                _clear_status()
                sys.stdout.write(str(msg)[: max(0, cols - 1)])
                sys.stdout.flush()
            except Exception:
                pass

        def _println(msg: str = "") -> None:
            # When in raw terminal mode, '\n' does NOT reliably return to column 0.
            # Use CRLF explicitly to prevent "diagonal drifting" rendering.
            try:
                _clear_status()
                sys.stdout.write("\r\n" + str(msg) + "\r\n")
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
                _println(f"❌ Failed to start microphone stream: {e}")

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
                _println("…(too short, try again)")
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
                audio_s = 0.0
                try:
                    if sr and sr > 0:
                        audio_s = float(len(pcm)) / float(int(sr) * 2)
                except Exception:
                    audio_s = 0.0

                t0 = time.monotonic()
                text = (self.voice_manager.transcribe_from_bytes(wav_bytes, language=self.current_language) or "").strip()
                t1 = time.monotonic()
                stt_s = float(t1 - t0)
                self._pending_stt_metrics = {
                    "stt_s": stt_s,
                    "audio_s": float(audio_s),
                    "rtf": (stt_s / float(audio_s)) if audio_s else None,
                    "sample_rate": int(sr),
                    "chunks": None,
                    "chunk_ms": None,
                    "profile": "ptt",
                    "ts": time.time(),
                }
            except Exception as e:
                self._ptt_busy = False
                _println(f"❌ Transcription failed: {e}")
                return
            self._ptt_busy = False

            if not text:
                _println("…(no transcription)")
                return

            _println(f"> {text}")
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

                def _run_in_cooked(block):
                    """Run a block with normal tty settings.

                    In raw mode, many terminals treat '\n' as LF without CR, so prints from
                    deeper code paths (LLM responses) can drift/indent. We temporarily
                    restore the terminal mode to keep output rendering stable.
                    """
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old)
                    except Exception:
                        pass
                    try:
                        block()
                    finally:
                        try:
                            tty.setraw(fd)
                        except Exception:
                            pass

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
                            _run_in_cooked(_stop_recording_and_send)
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
        # Ensure we end on a clean line before restoring other modes.
        try:
            sys.stdout.write("\r\n")
            sys.stdout.flush()
        except Exception:
            pass
        # Restore to STOP after exiting PTT.
        try:
            self.do_voice("stop")
        except Exception:
            pass
    
    def _voice_callback(self, text):
        """Callback for voice recognition."""
        # Capture best-effort STT metrics from the recognizer (for verbose stats).
        stt_metrics = None
        try:
            vm = self.voice_manager
            rec = getattr(vm, "voice_recognizer", None) if vm else None
            if rec is not None and hasattr(rec, "pop_last_stt_metrics"):
                stt_metrics = rec.pop_last_stt_metrics()
        except Exception:
            stt_metrics = None
        self._pending_stt_metrics = stt_metrics

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
                    cloning_engine=self.cloning_engine,
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
            if self.verbose_mode:
                out_words = self._count_words(text)
                out_tokens = None
                try:
                    enc = self._get_tiktoken_encoding()
                    if enc is not None:
                        out_tokens = int(len(enc.encode(str(text or ""))))
                except Exception:
                    out_tokens = None

                tts_metrics = None
                try:
                    if hasattr(self.voice_manager, "pop_last_tts_metrics"):
                        tts_metrics = self.voice_manager.pop_last_tts_metrics()
                except Exception:
                    tts_metrics = None

                turn = {
                    "stt": None,
                    "llm": {},
                    "counts": {
                        "in_words": 0,
                        "out_words": int(out_words),
                        "in_tokens": None,
                        "out_tokens": out_tokens,
                    },
                    "tts": tts_metrics,
                }
                self._print_verbose_turn_stats(turn)
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
        if not is_clone:
            # Offline-first: Piper voices must be explicitly cached. Provide a clear
            # message instead of hanging on implicit downloads.
            try:
                a = getattr(self.voice_manager, "tts_adapter", None)
                if a is not None and hasattr(a, "is_available") and not bool(a.is_available()):
                    lang = str(getattr(self, "current_language", "en") or "en").strip().lower()
                    raise RuntimeError(
                        f"Piper voice model for '{lang}' is not available locally.\n"
                        f"Run: python -m abstractvoice download --piper {lang}"
                    )
            except RuntimeError:
                raise
            except Exception:
                pass
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
            # If ASR auto-generated the clone's reference_text, print an easy override command
            # (once). We do this after stopping the spinner to avoid corrupting the prompt line.
            try:
                if is_clone and self.current_tts_voice:
                    self._maybe_print_asr_ref_text_override(self.current_tts_voice)
            except Exception:
                pass
            # Do not print the prompt manually: `cmd` will render it on return,
            # and printing here can result in duplicate prompts (`> >`).

    def _maybe_print_asr_ref_text_override(self, voice_id: str) -> None:
        """If `reference_text` was auto-generated via ASR, print a paste-ready override hint.

        Important: `/clone_set_ref_text` uses a simple `split(maxsplit=1)`, so quoting is not
        interpreted. We therefore print the command *without* quotes to avoid storing them.
        """
        if not self.voice_manager:
            return
        vid = str(voice_id or "").strip()
        if not vid:
            return
        if vid in self._printed_asr_ref_text_hint:
            return
        try:
            info = self.voice_manager.get_cloned_voice(vid) or {}
        except Exception:
            return
        meta = info.get("meta") or {}
        src = str(meta.get("reference_text_source") or "").strip().lower()
        ref_text = str(info.get("reference_text") or "").strip()
        if not ref_text:
            return
        if src != "asr":
            return

        # Mark first so any printing errors won't cause repeated spam.
        self._printed_asr_ref_text_hint.add(vid)

        prefix = vid[:8] if len(vid) >= 8 else vid
        name = str(info.get("name") or "").strip()
        label = f"{name} ({prefix})" if name else prefix
        print("ℹ️  Auto-generated reference transcript (ASR).")
        print(f"   Voice: {label}")
        print("   If you want to correct it, copy/paste and edit the text after the id:")
        print(f"     /clone_set_ref_text {prefix} {ref_text}")

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
                # `\033[2K` clears the entire line (more robust than fixed spaces).
                sys.stdout.write("\r\033[2K\r")
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
                eng = (v.get("engine") or "").strip()
                eng_txt = f" [{eng}]" if eng else ""
                src = (v.get("meta") or {}).get("reference_text_source", "")
                src_txt = f" [{src}]" if src else ""
                current = " (current)" if self.current_tts_voice == vid else ""
                print(f"  - {name}: {vid}{eng_txt}{src_txt}{current}")
            print("Tip: /clone_rm <id-or-name> deletes one; /clone_rm_all --yes deletes all.")
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

    def _resolve_clone_id_by_source(self, source: str, *, engine: str | None = None) -> str | None:
        """Find a cloned voice by its stored meta.source (best-effort)."""
        if not self.voice_manager:
            return None

        try:
            from pathlib import Path

            target = Path(str(source)).expanduser()
            try:
                target_norm = str(target.resolve())
            except Exception:
                target_norm = str(target)
        except Exception:
            target_norm = str(source)

        try:
            voices = self.voice_manager.list_cloned_voices()
        except Exception:
            return None

        wanted_engine = (str(engine).strip().lower() if engine else None) or None
        for v in voices:
            meta = v.get("meta") or {}
            src = meta.get("source")
            if not src:
                continue
            try:
                from pathlib import Path

                p = Path(str(src)).expanduser()
                try:
                    src_norm = str(p.resolve())
                except Exception:
                    src_norm = str(p)
            except Exception:
                src_norm = str(src)

            if src_norm != target_norm:
                continue
            if wanted_engine and (str(v.get("engine") or "").strip().lower() != wanted_engine):
                continue
            return str(v.get("voice_id") or "").strip() or None
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

    def do_clone_rm_all(self, arg):
        """Remove ALL cloned voices from the local store.

        Usage:
          /clone_rm_all --yes
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        confirm = (arg or "").strip().lower()
        if confirm not in ("--yes", "-y", "yes"):
            try:
                n = len(self.voice_manager.list_cloned_voices() or [])
            except Exception:
                n = 0
            if n <= 0:
                print("No cloned voices to delete.")
                return
            print(f"⚠️  This will permanently delete {n} cloned voice(s).")
            print("Re-run with: /clone_rm_all --yes")
            return

        # If currently selected, switch back to Piper.
        self.current_tts_voice = None

        deleted = 0
        failed = 0
        try:
            voices = list(self.voice_manager.list_cloned_voices() or [])
        except Exception as e:
            print(f"❌ Error listing cloned voices: {e}")
            return

        for v in voices:
            vid = str(v.get("voice_id") or v.get("voice") or "").strip()
            if not vid:
                continue
            try:
                self.voice_manager.delete_cloned_voice(vid)
                deleted += 1
            except Exception:
                failed += 1

        if failed:
            print(f"✅ Deleted {deleted} cloned voice(s). ⚠️ Failed: {failed}")
        else:
            print(f"✅ Deleted {deleted} cloned voice(s).")

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
          /clone <path> [name] [--engine f5_tts|chroma] [--text "reference transcript"]
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        try:
            parts = shlex.split(arg.strip())
        except ValueError as e:
            print(f"Usage: /clone <path> [name] [--engine f5_tts|chroma] [--text \"...\"]  (parse error: {e})")
            return

        if not parts:
            print("Usage: /clone <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
            return

        engine = None
        reference_text = None
        pos = []
        i = 0
        while i < len(parts):
            tok = parts[i]
            if tok in ("--engine",):
                if i + 1 >= len(parts):
                    print("Usage: /clone <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
                    return
                engine = parts[i + 1]
                i += 2
                continue
            if tok in ("--text", "--reference-text", "--reference_text"):
                if i + 1 >= len(parts):
                    print("Usage: /clone <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
                    return
                reference_text = parts[i + 1]
                i += 2
                continue
            pos.append(tok)
            i += 1

        if not pos:
            print("Usage: /clone <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
            return

        path = pos[0]
        name = pos[1] if len(pos) > 1 else None
        try:
            t0 = time.monotonic()
            voice_id = self.voice_manager.clone_voice(path, name=name, reference_text=reference_text, engine=engine)
            t1 = time.monotonic()

            eng = ""
            ref_src = ""
            try:
                info = self.voice_manager.get_cloned_voice(voice_id) or {}
                eng = str(info.get("engine") or "").strip()
                ref_src = str((info.get("meta") or {}).get("reference_text_source") or "").strip()
            except Exception:
                eng = ""
                ref_src = ""

            eng_txt = f" (engine: {eng})" if eng else ""
            print(f"✅ Cloned voice created: {voice_id}{eng_txt}")
            print("   Use /tts_voice clone <id-or-name> to select it.")
            print("   Tip: set reference text for best quality:")
            print("     /clone_set_ref_text <id-or-name> \"...\"")
            if not self._is_cloning_runtime_ready(voice_id=voice_id):
                print("   (Cloning runtime not ready yet; run /cloning_status and /cloning_download first.)")
            if str(eng or (engine or self.cloning_engine) or "").strip().lower() == "chroma" and not (reference_text or "").strip():
                print("ℹ️  No reference transcript provided.")
                print("   We will auto-generate it via STT on first speak (offline-first: requires cached STT model).")
                print("   Optional (often best quality): /clone_set_ref_text <id-or-name> \"...\"  (or re-run /clone ... --text \"...\")")

            if self.verbose_mode:
                n_files, ref_audio_s = self._summarize_audio_source(path)
                n_txt = str(n_files) if isinstance(n_files, int) else "--"
                src_txt = ref_src or ("manual" if (reference_text or "").strip() else "--")
                msg = f"CLONE {eng or (engine or self.cloning_engine)} | refs {n_txt} a{self._fmt_s(ref_audio_s)} | ref_text {src_txt} | {self._fmt_s(float(t1 - t0))}"
                print(f"{Colors.YELLOW}{msg}{Colors.END}")
        except Exception as e:
            print(f"❌ Clone failed: {e}")

    def do_clone_use(self, arg):
        """Clone a voice (or reuse an existing one) and immediately select it.

        Usage:
          /clone_use <path> [name] [--engine f5_tts|chroma] [--text "reference transcript"]

        Shortcut:
          - Paste a WAV/FLAC/OGG path directly (optionally: `path.wav | transcript`).
        """
        if not self.voice_manager:
            print("🔇 TTS is disabled. Use '/tts on' to enable voice features.")
            return

        try:
            parts = shlex.split(arg.strip())
        except ValueError as e:
            print(f"Usage: /clone_use <path> [name] [--engine f5_tts|chroma] [--text \"...\"]  (parse error: {e})")
            return

        if not parts:
            print("Usage: /clone_use <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
            return

        engine = None
        reference_text = None
        pos = []
        i = 0
        while i < len(parts):
            tok = parts[i]
            if tok in ("--engine",):
                if i + 1 >= len(parts):
                    print("Usage: /clone_use <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
                    return
                engine = parts[i + 1]
                i += 2
                continue
            if tok in ("--text", "--reference-text", "--reference_text"):
                if i + 1 >= len(parts):
                    print("Usage: /clone_use <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
                    return
                reference_text = parts[i + 1]
                i += 2
                continue
            pos.append(tok)
            i += 1

        if not pos:
            print("Usage: /clone_use <path> [name] [--engine f5_tts|chroma] [--text \"...\"]")
            return

        path = pos[0]
        name = pos[1] if len(pos) > 1 else None

        engine_name = str(engine or self.cloning_engine or "f5_tts").strip().lower()

        # If name isn't provided, use something stable for UX.
        if not name:
            try:
                from pathlib import Path

                p = Path(path)
                name = p.stem if p.is_file() else p.name
            except Exception:
                name = None

        # Reuse a prior clone created from the same source path + engine.
        voice_id = self._resolve_clone_id_by_source(path, engine=engine_name)
        if voice_id:
            if reference_text:
                try:
                    self.voice_manager.set_cloned_voice_reference_text(voice_id, reference_text)
                    print("✅ Reusing cloned voice and updating reference text.")
                except Exception:
                    print("✅ Reusing cloned voice.")
            else:
                print("✅ Reusing cloned voice.")
        else:
            try:
                t0 = time.monotonic()
                voice_id = self.voice_manager.clone_voice(path, name=name, reference_text=reference_text, engine=engine_name)
                t1 = time.monotonic()

                eng = ""
                ref_src = ""
                try:
                    info = self.voice_manager.get_cloned_voice(voice_id) or {}
                    eng = str(info.get("engine") or "").strip()
                    ref_src = str((info.get("meta") or {}).get("reference_text_source") or "").strip()
                except Exception:
                    eng = ""
                    ref_src = ""

                eng_txt = f" (engine: {eng})" if eng else ""
                print(f"✅ Cloned voice created: {voice_id}{eng_txt}")
                if reference_text:
                    print("   (Reference text provided)")
                else:
                    print("   Tip: set reference text for best quality:")
                    print("     /clone_set_ref_text <id-or-name> \"...\"")
                    if str(eng or engine_name or "").strip().lower() == "chroma":
                        print("   ℹ️  No transcript provided; STT auto-fallback runs on first speak (requires cached STT model).")

                if self.verbose_mode:
                    n_files, ref_audio_s = self._summarize_audio_source(path)
                    n_txt = str(n_files) if isinstance(n_files, int) else "--"
                    src_txt = ref_src or ("manual" if (reference_text or "").strip() else "--")
                    msg = f"CLONE {eng or engine_name} | refs {n_txt} a{self._fmt_s(ref_audio_s)} | ref_text {src_txt} | {self._fmt_s(float(t1 - t0))}"
                    print(f"{Colors.YELLOW}{msg}{Colors.END}")
            except Exception as e:
                print(f"❌ Clone failed: {e}")
                return

        # Select if runtime is ready (no surprise downloads).
        if not self._is_cloning_runtime_ready(voice_id=voice_id):
            print("ℹ️  Cloning runtime is not ready (would trigger large downloads).")
            print("   Run /cloning_status and /cloning_download, or use /tts_voice piper.")
            return

        self.current_tts_voice = voice_id
        eng = ""
        try:
            info = self.voice_manager.get_cloned_voice(voice_id) or {}
            eng = str(info.get("engine") or "").strip()
        except Exception:
            eng = ""
        eng_txt = f" (engine: {eng})" if eng else ""
        print(f"✅ Using cloned voice: {voice_id}{eng_txt}")
        if eng and str(eng).strip().lower() != str(self.cloning_engine).strip().lower():
            print(f"ℹ️  Default cloning engine is {self.cloning_engine}; this voice uses {eng}.")
        # Free memory from other cloning engines (important for large backends like Chroma).
        try:
            if hasattr(self.voice_manager, "unload_cloning_engines"):
                self.voice_manager.unload_cloning_engines(keep_engine=str(eng or "").strip().lower() or None)
        except Exception:
            pass
        # Piper is not needed while speaking with a cloned voice; unload it to reduce memory pressure.
        try:
            if hasattr(self.voice_manager, "unload_piper_voice"):
                self.voice_manager.unload_piper_voice()
        except Exception:
            pass

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
            if self.current_tts_voice:
                vid = self.current_tts_voice
                try:
                    info = self.voice_manager.get_cloned_voice(vid) or {}
                    name = (info.get("name") or "").strip()
                    eng = (info.get("engine") or "").strip()
                    label = name or vid
                    suffix = f" (engine: {eng})" if eng else ""
                    print(f"Current TTS voice: {label}{suffix}")
                except Exception:
                    print(f"Current TTS voice: {vid}")
            else:
                print("Current TTS voice: piper")
            print("Usage: /tts_voice piper | /tts_voice clone <id-or-name>")
            return

        if parts[0] == "piper":
            self.current_tts_voice = None
            # Free any heavy cloning engines when switching back to Piper.
            try:
                if hasattr(self.voice_manager, "unload_cloning_engines"):
                    self.voice_manager.unload_cloning_engines()
            except Exception:
                pass
            # If Piper was previously unloaded to save memory, reload it now (offline-first).
            try:
                if self.voice_manager and getattr(self.voice_manager, "tts_adapter", None):
                    a = getattr(self.voice_manager, "tts_adapter", None)
                    if hasattr(a, "is_available") and not bool(a.is_available()):
                        self.voice_manager.set_language(self.current_language)
            except Exception:
                pass
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
        if not self._is_cloning_runtime_ready(voice_id=match):
            print("❌ Cloning runtime is not ready (would trigger large downloads).")
            print("   Run /cloning_status and /cloning_download, or use /tts_voice piper.")
            return

        # Allow selecting voices without reference_text; we will auto-fallback at speak-time
        # if the STT model is already cached locally (no downloads in REPL).

        self.current_tts_voice = match
        eng = ""
        try:
            info = self.voice_manager.get_cloned_voice(match) or {}
            eng = (info.get("engine") or "").strip()
        except Exception:
            eng = ""
        eng_txt = f" (engine: {eng})" if eng else ""
        print(f"✅ Using cloned voice: {match}{eng_txt}")
        if eng and str(eng).strip().lower() != str(self.cloning_engine).strip().lower():
            print(f"ℹ️  Default cloning engine is {self.cloning_engine}; this voice uses {eng}.")
        # Free memory from other cloning engines (e.g. unloading Chroma when switching to F5, or vice-versa).
        try:
            if hasattr(self.voice_manager, "unload_cloning_engines"):
                self.voice_manager.unload_cloning_engines(keep_engine=str(eng or "").strip().lower() or None)
        except Exception:
            pass
        # Piper is not needed while speaking with a cloned voice; unload it to reduce memory pressure.
        try:
            if hasattr(self.voice_manager, "unload_piper_voice"):
                self.voice_manager.unload_piper_voice()
        except Exception:
            pass

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

        print(f"default_cloning_engine: {self.cloning_engine}")

        if importlib.util.find_spec("f5_tts") is None:
            print("ℹ️  OpenF5 runtime: not installed (missing: f5_tts)")
            print("   Install: pip install \"abstractvoice[cloning]\"")
        else:
            if self._is_openf5_cached():
                print("✅ OpenF5 artifacts: present (cached)")
            else:
                print("ℹ️  OpenF5 artifacts: not present (will require ~5.4GB download)")
                print("   Run: /cloning_download f5_tts")

        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            print("ℹ️  Chroma runtime: not installed (missing: transformers/torch)")
            print("   Install: pip install \"abstractvoice[chroma]\"")
        else:
            if self._is_chroma_cached():
                print("✅ Chroma artifacts: present (cached)")
            else:
                print("ℹ️  Chroma artifacts: not present (will require a large download + HF access)")
                print("   Run: /cloning_download chroma")
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

        target = (arg or "").strip().lower() or self.cloning_engine
        engine_name = "f5_tts" if target in ("openf5", "f5", "f5_tts") else target
        if engine_name == "f5_tts":
            if importlib.util.find_spec("f5_tts") is None:
                print("❌ OpenF5 runtime not installed in this environment (missing: f5_tts).")
                print("   Install: pip install \"abstractvoice[cloning]\"")
                return
        elif engine_name == "chroma":
            # Artifacts download uses huggingface_hub and does not require loading the model.
            if importlib.util.find_spec("huggingface_hub") is None:
                print("❌ huggingface_hub is required to download Chroma artifacts.")
                print("   Install: pip install huggingface_hub")
                return
        else:
            print("Usage: /cloning_download [f5_tts|chroma]")
            return

        try:
            cloner = self.voice_manager._get_voice_cloner()  # REPL convenience
            engine = cloner._get_engine(engine_name)  # explicit download is an engine concern
            if engine_name == "f5_tts":
                print("Downloading OpenF5 artifacts (~5.4GB). This is a one-time cache per machine.")
                engine.ensure_openf5_artifacts_downloaded()
            else:
                print("Downloading Chroma artifacts (very large; requires HF access). This is a one-time cache per machine.")
                engine.ensure_chroma_artifacts_downloaded()
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

    def _is_chroma_cached(self) -> bool:
        """Heuristic local check that avoids importing huggingface_hub."""
        from pathlib import Path
        import os

        root = Path(os.path.expanduser("~/.cache/abstractvoice/chroma"))
        if not root.exists():
            return False
        required = [
            "config.json",
            "processor_config.json",
            "model.safetensors.index.json",
            "modeling_chroma.py",
            "processing_chroma.py",
            "configuration_chroma.py",
        ]
        return all((root / name).exists() for name in required)

    def _is_cloning_runtime_ready(self, *, voice_id: str | None = None, engine: str | None = None) -> bool:
        """Return whether the selected cloning engine is ready locally (no downloads)."""
        eng = str(engine or "").strip().lower()
        if not eng and voice_id and self.voice_manager:
            try:
                info = self.voice_manager.get_cloned_voice(voice_id)
                eng = str((info or {}).get("engine") or "").strip().lower()
            except Exception:
                eng = ""
        if not eng:
            eng = str(getattr(self, "cloning_engine", "f5_tts") or "f5_tts").strip().lower()

        if eng == "chroma":
            return (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None
                and self._is_chroma_cached()
            )
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
            cloning_engine=self.cloning_engine,
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
            cloning_engine=self.cloning_engine,
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
        self._clear_history()
        print("History cleared")

    def do_reset(self, arg):
        """Reset the session (history + current voice selection)."""
        try:
            if self.voice_manager:
                self.voice_manager.stop_speaking()
        except Exception:
            pass

        # Reset voice selection back to Piper (default).
        self.current_tts_voice = None
        # Free any heavy cloning engines as part of reset.
        try:
            if self.voice_manager and hasattr(self.voice_manager, "unload_cloning_engines"):
                self.voice_manager.unload_cloning_engines()
        except Exception:
            pass
        # Ensure Piper is ready (in case it was unloaded to save memory).
        try:
            if self.voice_manager and getattr(self.voice_manager, "tts_adapter", None):
                a = getattr(self.voice_manager, "tts_adapter", None)
                if hasattr(a, "is_available") and not bool(a.is_available()):
                    self.voice_manager.set_language(self.current_language)
        except Exception:
            pass

        # Clear chat history.
        self._clear_history()
        print("✅ Reset.")

    def _clear_history(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # Reset token counters
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        # Reset word counters
        self.system_words = 0
        self.user_words = 0
        self.assistant_words = 0
        # Recalculate system tokens
        self._count_system_tokens()
        self._count_system_words()
    
    def do_system(self, arg):
        """Set the system prompt."""
        if arg.strip():
            self.system_prompt = arg.strip()
            self._clear_history()
            print(f"System prompt set to: {self.system_prompt}")
        else:
            print(f"Current system prompt: {self.system_prompt}")
    
    def do_exit(self, arg):
        """Exit the REPL."""
        # Stop any PTT session cleanly.
        self._ptt_session_active = False
        self._ptt_recording = False
        self._ptt_busy = False

        # Stop voice mode / audio best-effort.
        try:
            if self.voice_manager:
                try:
                    self.voice_manager.stop_listening()
                except Exception:
                    pass
                try:
                    self.voice_manager.stop_speaking()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            if self.voice_manager:
                self.voice_manager.cleanup()
        except Exception:
            pass
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

    def do_verbose(self, arg):
        """Toggle verbose per-turn performance stats.

        Usage:
          /verbose            (toggle)
          /verbose on|off
        """
        s = (arg or "").strip().lower()
        if s in ("", "toggle"):
            self.verbose_mode = not bool(getattr(self, "verbose_mode", False))
        elif s in ("on", "1", "true", "yes", "y"):
            self.verbose_mode = True
        elif s in ("off", "0", "false", "no", "n"):
            self.verbose_mode = False
        else:
            print("Usage: /verbose [on|off]")
            return
        print(f"Verbose mode: {'on' if self.verbose_mode else 'off'}")
    
    def do_help(self, arg):
        """Show help information."""
        print("Commands:")
        print("  /exit, /q, /quit    Exit REPL")
        print("  /clear              Clear history")
        print("  /reset              Reset (history + voice)")
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
        print("  /verbose [on|off]   Toggle verbose per-turn stats")
        print("  /help               Show this help")
        print("  /clones             List cloned voices")
        print("  /clone_info <id>    Show cloned voice details")
        print("  /clone_ref <id>     Show cloned voice reference text")
        print("  /clone_rename ...   Rename a cloned voice")
        print("  /clone_rm <id>      Delete a cloned voice")
        print("  /clone_rm_all --yes Delete ALL cloned voices")
        print("  /clone_export ...   Export a cloned voice (.zip)")
        print("  /clone_import ...   Import a cloned voice (.zip)")
        print("  /clone <path> [nm]  Add a cloned voice from WAV/FLAC/OGG")
        print("  /clone_use <path>   Clone+select voice (or reuse)")
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
        print("Shortcut: paste a WAV/FLAC/OGG path to clone+select (optionally: `path | transcript`).")
    
    def emptyline(self):
        """Handle empty line input."""
        # Do nothing when an empty line is entered
        pass

    def do_tokens(self, arg):
        """Display token usage information."""
        try:
            if self._get_tiktoken_encoding() is None:
                print("Token counting is not available (install: pip install tiktoken).")
                return

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
        """Reset token/word counts and recalculate for all messages."""
        self.system_tokens = 0
        self.user_tokens = 0
        self.assistant_tokens = 0
        self.system_words = 0
        self.user_words = 0
        self.assistant_words = 0
        
        # Count tokens for all messages
        for msg in self.messages:
            if isinstance(msg, dict) and "content" in msg and "role" in msg:
                self._count_tokens(msg["content"], msg["role"])
                w = self._count_words(msg["content"])
                r = msg.get("role")
                if r == "system":
                    self.system_words = int(w)
                elif r == "user":
                    self.user_words += int(w)
                elif r == "assistant":
                    self.assistant_words += int(w)
    
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
    parser.add_argument("--verbose", action="store_true", help="Show per-turn performance stats")
    parser.add_argument("--api", default="http://localhost:11434/api/chat",
                      help="LLM API URL")
    parser.add_argument("--model", default="cogito:3b",
                      help="LLM model name")
    parser.add_argument(
        "--cloning-engine",
        default="f5_tts",
        choices=["f5_tts", "chroma"],
        help="Default cloning backend for new voices (f5_tts|chroma)",
    )
    parser.add_argument(
        "--voice-mode",
        default="off",
        choices=["off", "wait", "stop", "full", "ptt"],
        help="Auto-start microphone voice mode (off|wait|stop|full|ptt). Default: off.",
    )
    parser.add_argument(
        "--language",
        "--lang",
        default="en",
        choices=["en", "fr", "de", "es", "ru", "zh"],
        help="Voice language for default Piper TTS (en|fr|de|es|ru|zh).",
    )
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
            verbose_mode=args.verbose,
            language=args.language,
            tts_model=args.tts_model,
            voice_mode=args.voice_mode,
            cloning_engine=args.cloning_engine,
        )
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main() 
