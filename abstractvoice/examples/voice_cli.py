#!/usr/bin/env python3
"""
AbstractVoice voice mode CLI launcher.

This module provides a direct entry point to start AbstractVoice in voice mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from abstractvoice.examples.cli_repl import VoiceREPL
from abstractvoice.examples.llm_provider import PROVIDER_PRESETS, DEFAULT_PROVIDER, DEFAULT_MODEL


def _has_cli_option(argv: list[str], *options: str) -> bool:
    for item in argv:
        for option in options:
            if item == option or item.startswith(f"{option}="):
                return True
    return False


def _looks_like_url(value: str | None) -> bool:
    raw = str(value or "").strip().lower()
    return raw.startswith("http://") or raw.startswith("https://")


def print_examples():
    """Print available examples."""
    print("Available commands:")
    print("  cli            - Command-line REPL example")
    print("  web            - Local FastAPI web example")
    print("  simple         - Simple usage example")
    print("  check-deps     - Check dependency compatibility")
    print("  tts            - One-shot TTS to file")
    print("\nUsage: abstractvoice <command> [--language <lang>] [args...]")
    print("\nSupported local Piper language mapping: en, fr, es, de, ru, zh")
    print("Supertonic supports 31 local TTS languages once selected via /tts engine supertonic.")
    print("Note: OmniVoice supports many additional language codes once selected via /tts engine omnivoice.")
    print("\nExamples:")
    print("  abstractvoice cli --language fr     # French CLI")
    print("  abstractvoice web --port 5000       # Local web example")
    print("  abstractvoice simple --language ru  # Russian simple example")
    print("  abstractvoice check-deps            # Check dependencies")
    print(
        "  abstractvoice --provider openai --model tts-1 --voice alloy "
        "--prompt \"Hello\" --output hello.wav"
    )
    print("  abstractvoice                       # Direct voice mode (default)")


def simple_example():
    """Run a simple example demonstrating basic usage."""
    from abstractvoice import VoiceManager
    import time

    print("Simple AbstractVoice Example")
    print("============================")
    print("This example demonstrates basic TTS and STT functionality.")
    print("(Use --language argument to test different languages)")
    print()

    # Initialize voice manager (can be overridden with --language)
    manager = VoiceManager(debug_mode=True)

    try:
        # TTS example
        print("Speaking a welcome message...")
        manager.speak("Hello! I'm a voice assistant powered by AbstractVoice. "
                     "I can speak and listen to you.")

        # Wait for speech to complete
        while manager.is_speaking():
            time.sleep(0.1)

        print("\nNow I'll listen for 10 seconds. Say something!")

        # Store transcribed text
        transcribed_text = None

        # Callback for speech recognition
        def on_transcription(text):
            nonlocal transcribed_text
            print(f"\nTranscribed: {text}")
            transcribed_text = text

            # If user says stop, stop listening
            if text.lower() == "stop":
                return

            # Otherwise respond
            print("Responding...")
            manager.speak(f"You said: {text}")

        # Start listening
        manager.listen(on_transcription)

        # Listen for 10 seconds or until "stop" is said
        start_time = time.time()
        while time.time() - start_time < 10 and manager.is_listening():
            time.sleep(0.1)

        # Stop listening if still active
        if manager.is_listening():
            manager.stop_listening()
            print("\nDone listening.")

        # If something was transcribed, repeat it back
        if transcribed_text and transcribed_text.lower() != "stop":
            print("\nSaying goodbye...")
            manager.speak("Thanks for trying AbstractVoice! Goodbye!")
            while manager.is_speaking():
                time.sleep(0.1)

        print("\nExample complete!")

    finally:
        # Clean up
        manager.cleanup()


def parse_args(argv: list[str] | None = None):
    """Parse command line arguments."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="AbstractVoice - Voice interactions with AI")

    # Examples and special commands
    parser.add_argument(
        "command",
        nargs="?",
        help="Command to run: cli, web, simple, check-deps, tts (default: voice mode)",
    )

    # Voice mode arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Show per-turn performance stats")
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=(
            f"LLM provider preset ({', '.join(sorted(PROVIDER_PRESETS))}) or base URL; "
            "in one-shot TTS mode, the TTS provider/engine"
        ),
    )
    parser.add_argument("--api", default=None,
                      help="LLM API base URL (overrides --provider)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                      help="LLM model name; in one-shot TTS mode, the TTS model name")
    parser.add_argument(
        "--whisper",
        default="base",
        help="STT model size for faster-whisper (e.g. tiny|base|small|medium|large-v2|large-v3|large).",
    )
    parser.add_argument(
        "--cloning-engine",
        default="omnivoice",
        choices=["omnivoice", "f5_tts", "chroma", "audiodit", "openai", "openai-compatible"],
        help="Default cloning backend for new voices (default: omnivoice; choices: omnivoice|f5_tts|chroma|audiodit|openai|openai-compatible).",
    )
    parser.add_argument(
        "--voice-mode",
        default="off",
        choices=["off", "wait", "stop", "full", "ptt"],
        help="Auto-start microphone voice mode (off|wait|stop|full|ptt). Default: off.",
    )
    parser.add_argument("--no-listening", action="store_true",
                      help="Disable speech-to-text (listening). Alias for --voice-mode off.")
    parser.add_argument("--no-tts", action="store_true",
                      help="Disable text-to-speech (TTS), text-only mode")
    parser.add_argument("--system",
                      help="Custom system prompt")
    parser.add_argument("--temperature", type=float, default=0.4,
                      help="Set temperature (0.0-2.0) for the LLM")
    parser.add_argument("--max-tokens", type=int, default=4096,
                      help="Set maximum tokens for the LLM response")
    parser.add_argument(
        "--language",
        "--lang",
        default="en",
        help="Voice language code (Piper: en/fr/es/de/ru/zh; Supertonic: 31 languages; OmniVoice: many more).",
    )
    parser.add_argument("--tts-model",
                      help="Specific TTS model to use (overrides language default)")
    parser.add_argument("--tts-engine", default="auto", help="Initial TTS engine (auto|supertonic|piper|openai|openai-compatible|audiodit|omnivoice)")
    parser.add_argument("--stt-engine", default="openai", help="Initial STT engine (openai|openai-compatible|faster_whisper|transformers-asr|auto)")
    parser.add_argument("--stt-model", default=None, help="Model id for remote STT engines, or a Hugging Face model id when using transformers-asr (e.g. openai/whisper-large-v3, openai/whisper-large-v3-turbo, Qwen/Qwen3-ASR-1.7B)")
    parser.add_argument("--remote-base-url", default=None, help="Base URL for OpenAI-compatible remote voice endpoints")
    parser.add_argument("--remote-api-key", default=None, help="Bearer API key for remote voice endpoints")
    parser.add_argument("--remote-timeout", type=float, default=None, help="Remote voice request timeout in seconds")
    one_shot = parser.add_argument_group("one-shot TTS")
    one_shot.add_argument("--prompt", help="Text to synthesize and write to --output")
    one_shot.add_argument("--output", help="Output audio file path")
    one_shot.add_argument(
        "--voice",
        help="TTS voice/profile id, or a cloned voice id when no base profile matches",
    )
    one_shot.add_argument(
        "--format",
        dest="output_format",
        default=None,
        help="Audio output format; inferred from --output when omitted",
    )

    args = parser.parse_args(argv)
    args.provider_explicit = _has_cli_option(raw_argv, "--provider")
    args.model_explicit = _has_cli_option(raw_argv, "--model")

    one_shot_requested = args.prompt is not None or args.output is not None
    if one_shot_requested:
        if args.command not in (None, "tts"):
            parser.error("--prompt/--output one-shot TTS cannot be combined with another command")
        if args.prompt is None or args.output is None:
            parser.error("--prompt and --output must be used together")
        if not str(args.prompt).strip():
            parser.error("--prompt cannot be empty")
        if not str(args.output).strip():
            parser.error("--output cannot be empty")
        if args.no_tts:
            parser.error("--no-tts cannot be used with --prompt/--output")
    elif args.command == "tts":
        parser.error("tts requires --prompt and --output")
    return args


def _apply_tts_voice_profile(vm, voice: str | None) -> str | None:
    """Apply a base TTS profile when possible; otherwise return a clone voice id."""
    selected = str(voice or "").strip()
    if not selected:
        return None

    def _resolve_cloned_voice_id() -> str:
        try:
            info = vm.get_cloned_voice(selected)
            if isinstance(info, dict):
                return str(info.get("voice_id") or selected).strip() or selected
        except Exception:
            pass

        try:
            matches = []
            for item in list(vm.list_cloned_voices() or []):
                if not isinstance(item, dict):
                    continue
                voice_id = str(item.get("voice_id") or "").strip()
                name = str(item.get("name") or "").strip()
                if selected == voice_id or selected.lower() == name.lower():
                    matches.append(voice_id or selected)
            if len(matches) == 1:
                return str(matches[0] or selected)
        except Exception:
            pass
        return selected

    try:
        return None if bool(vm.set_profile(selected, kind="tts")) else _resolve_cloned_voice_id()
    except TypeError:
        try:
            return None if bool(vm.set_profile(selected)) else _resolve_cloned_voice_id()
        except Exception:
            return _resolve_cloned_voice_id()
    except Exception:
        return _resolve_cloned_voice_id()


def _run_one_shot_tts(args, *, voice_manager_factory=None) -> str:
    """Run `abstractvoice --prompt ... --output ...` without entering the REPL."""
    if voice_manager_factory is None:
        from abstractvoice import VoiceManager

        voice_manager_factory = VoiceManager

    provider = str(args.tts_engine or "auto").strip() or "auto"
    remote_base_url = args.remote_base_url

    if bool(getattr(args, "provider_explicit", False)):
        requested_provider = str(args.provider or "").strip()
        if _looks_like_url(requested_provider):
            provider = "openai-compatible"
            if not remote_base_url:
                remote_base_url = requested_provider
        elif requested_provider:
            provider = requested_provider

    if (
        not remote_base_url
        and args.api
        and provider.strip().lower().replace("_", "-") == "openai-compatible"
    ):
        remote_base_url = args.api

    tts_model = args.tts_model
    if bool(getattr(args, "model_explicit", False)):
        tts_model = str(args.model or "").strip() or None

    vm = voice_manager_factory(
        language=args.language,
        tts_model=tts_model,
        whisper_model=args.whisper,
        debug_mode=bool(args.debug),
        tts_engine=provider,
        stt_engine=args.stt_engine,
        stt_model=args.stt_model,
        remote_base_url=remote_base_url,
        remote_api_key=args.remote_api_key,
        remote_timeout_s=args.remote_timeout,
        allow_downloads=True,
        cloning_engine=args.cloning_engine,
    )
    try:
        voice_for_call = _apply_tts_voice_profile(vm, args.voice)
        out_path = vm.speak_to_file(
            str(args.prompt),
            str(args.output),
            format=args.output_format,
            voice=voice_for_call,
        )
        print(f"Wrote {out_path}")
        return str(out_path)
    finally:
        try:
            vm.cleanup()
        except Exception:
            pass


def main():
    """Entry point for AbstractVoice CLI."""
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "web":
            from abstractvoice.examples.web_ui import main as web_main

            web_main(sys.argv[2:])
            return

        # Parse command line arguments
        args = parse_args()

        # Normalize aliases/compat flags.
        if getattr(args, "no_listening", False):
            args.voice_mode = "off"

        if args.prompt is not None or args.output is not None:
            _run_one_shot_tts(args)
            return

        # Handle special commands and examples
        if args.command == "check-deps":
            from abstractvoice.dependency_check import check_dependencies
            try:
                check_dependencies(verbose=True)
            except Exception as e:
                print(f"❌ Error running dependency check: {e}")
                print("This might indicate a dependency issue.")
                if args.debug:
                    import traceback
                    traceback.print_exc()
            return
        elif args.command == "cli":
            # Import and run CLI REPL example
            repl = VoiceREPL(
                provider=args.provider,
                api_url=args.api,
                model=args.model,
                debug_mode=args.debug,
                verbose_mode=args.verbose,
                language=args.language,
                tts_model=args.tts_model,
                whisper_model=args.whisper,
                tts_engine=args.tts_engine,
                stt_engine=args.stt_engine,
                stt_model=args.stt_model,
                remote_base_url=args.remote_base_url,
                remote_api_key=args.remote_api_key,
                remote_timeout_s=args.remote_timeout,
                voice_mode=args.voice_mode,
                disable_tts=args.no_tts,
                cloning_engine=args.cloning_engine,
            )
            # Set temperature and max_tokens
            repl.temperature = args.temperature
            repl.max_tokens = args.max_tokens
            if args.system:
                repl.system_prompt = args.system
                repl.messages = [{"role": "system", "content": args.system}]
            repl.cmdloop()
            return
        elif args.command == "web":
            try:
                from abstractvoice.examples.web_ui import run_server

                run_server(
                    language=args.language,
                    whisper_model=args.whisper,
                    tts_engine=args.tts_engine,
                    stt_engine=args.stt_engine,
                    tts_model=args.tts_model,
                    stt_model=args.stt_model,
                    cloning_engine=args.cloning_engine,
                    remote_base_url=args.remote_base_url,
                    remote_api_key=args.remote_api_key,
                    remote_timeout_s=args.remote_timeout,
                    debug_mode=args.debug,
                )
            except RuntimeError as e:
                print(f"❌ {e}")
            return
        elif args.command == "simple":
            simple_example()
            return
        elif args.command == "help" or args.command == "--help":
            print_examples()
            return
        elif args.command:
            print(f"Unknown command: {args.command}")
            print_examples()
            return

        # Default behavior: start the REPL (mic OFF unless --voice-mode is set).
        lang_name = {
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "zh": "Chinese",
        }.get(str(args.language), str(args.language))
        print(f"Starting AbstractVoice ({lang_name})…  (type /help once it starts)")

        # Initialize REPL.
        repl = VoiceREPL(
            provider=args.provider,
            api_url=args.api,
            model=args.model,
            debug_mode=args.debug,
            verbose_mode=args.verbose,
            language=args.language,
            tts_model=args.tts_model,
            whisper_model=args.whisper,
            tts_engine=args.tts_engine,
            stt_engine=args.stt_engine,
            stt_model=args.stt_model,
            remote_base_url=args.remote_base_url,
            remote_api_key=args.remote_api_key,
            remote_timeout_s=args.remote_timeout,
            voice_mode=args.voice_mode,
            disable_tts=args.no_tts,
            cloning_engine=args.cloning_engine,
        )
        
        # Set custom system prompt if provided
        if args.system:
            repl.system_prompt = args.system
            repl.messages = [{"role": "system", "content": args.system}]
            if args.debug:
                print(f"System prompt set to: {args.system}")
        
        # Set temperature and max_tokens
        repl.temperature = args.temperature
        repl.max_tokens = args.max_tokens
        if args.debug:
            print(f"Temperature: {args.temperature}")
            print(f"Max tokens: {args.max_tokens}")
        
        # Start the REPL
        repl.cmdloop()
        
    except KeyboardInterrupt:
        print("\nExiting AbstractVoice...")
    except Exception as e:
        error_msg = str(e).lower()

        # Check if it's a TTS-related error (not Ollama model error)
        if "model file not found in the output path" in error_msg:
            print(f"❌ TTS model download failed")
            print(f"   This is a TTS voice model issue, not your Ollama model")
            print(f"   Your Ollama model '{args.model}' is fine")
            print("   Try: pip install --upgrade abstractvoice")
            print(f"   Or check network connectivity for model downloads")
        elif "connection" in error_msg or "refused" in error_msg:
            provider_name = getattr(args, "provider", DEFAULT_PROVIDER)
            print(f"❌ Cannot connect to LLM provider ({provider_name})")
            print(f"   Make sure the server is running. Use --provider to switch.")
        elif "importerror" in error_msg or "no module" in error_msg:
            print(f"❌ Missing dependencies")
            print(f"   Try running: abstractvoice check-deps")
            print(f"   Or install a platform profile: pip install \"abstractvoice[apple]\" or \"abstractvoice[gpu]\"")
        else:
            print(f"❌ Application error: {e}")
            print(f"   Try running with --debug for more details")

        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 
