#!/usr/bin/env python3
"""
AbstractVoice voice mode CLI launcher.

This module provides a direct entry point to start AbstractVoice in voice mode.
"""

import argparse
import time
from abstractvoice.examples.cli_repl import VoiceREPL

def print_examples():
    """Print available examples."""
    print("Available commands:")
    print("  cli            - Command-line REPL example")
    print("  web            - Web API example")
    print("  simple         - Simple usage example")
    print("  check-deps     - Check dependency compatibility")
    print("  download-models - Download TTS models for offline use")
    print("\nUsage: abstractvoice <command> [--language <lang>] [args...]")
    print("\nSupported languages: en, fr, es, de, it, ru, multilingual")
    print("\nExamples:")
    print("  abstractvoice cli --language fr     # French CLI")
    print("  abstractvoice simple --language ru  # Russian simple example")
    print("  abstractvoice check-deps            # Check dependencies")
    print("  abstractvoice download-models       # Download models for offline use")
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

def parse_args():
    """Parse command line arguments."""
    import sys

    # Check if it's a download-models command and handle separately
    if len(sys.argv) > 1 and sys.argv[1] == "download-models":
        # Return early with just the command to handle in main()
        class DownloadModelsArgs:
            command = "download-models"
            # Add dummy attributes to prevent AttributeError
            model = "granite3.3:2b"
            debug = False
        return DownloadModelsArgs()

    parser = argparse.ArgumentParser(description="AbstractVoice - Voice interactions with AI")

    # Examples and special commands
    parser.add_argument("command", nargs="?", help="Command to run: cli, web, simple, check-deps, download-models (default: voice mode)")

    # Voice mode arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--api", default="http://localhost:11434/api/chat",
                      help="LLM API URL")
    parser.add_argument("--model", default="granite3.3:2b",
                      help="LLM model name")
    parser.add_argument("--whisper", default="tiny",
                      help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--no-listening", action="store_true",
                      help="Disable speech-to-text (listening), TTS still works")
    parser.add_argument("--no-tts", action="store_true",
                      help="Disable text-to-speech (TTS), text-only mode")
    parser.add_argument("--system",
                      help="Custom system prompt")
    parser.add_argument("--temperature", type=float, default=0.4,
                      help="Set temperature (0.0-2.0) for the LLM")
    parser.add_argument("--max-tokens", type=int, default=4096,
                      help="Set maximum tokens for the LLM response")
    parser.add_argument("--language", "--lang", default="en",
                      choices=["en", "fr", "es", "de", "it", "ru", "multilingual"],
                      help="Voice language (en=English, fr=French, es=Spanish, de=German, it=Italian, ru=Russian, multilingual=All)")
    parser.add_argument("--tts-model",
                      help="Specific TTS model to use (overrides language default)")
    return parser.parse_args()

def main():
    """Entry point for AbstractVoice CLI."""
    try:
        # Parse command line arguments
        args = parse_args()

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
        elif args.command == "download-models":
            from abstractvoice.simple_model_manager import download_models_cli
            # Pass remaining arguments to download_models_cli
            import sys
            original_argv = sys.argv
            sys.argv = ["download-models"] + sys.argv[2:]  # Remove script name and "download-models"
            try:
                download_models_cli()
            finally:
                sys.argv = original_argv
            return
        elif args.command == "cli":
            # Import and run CLI REPL example
            repl = VoiceREPL(
                api_url=args.api,
                model=args.model,
                debug_mode=args.debug,
                language=args.language,
                tts_model=args.tts_model
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
            from abstractvoice.examples.web_api import main as web_main
            web_main()
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

        # Show language information
        language_names = {
            'en': 'English', 'fr': 'French', 'es': 'Spanish',
            'de': 'German', 'it': 'Italian', 'ru': 'Russian',
            'multilingual': 'Multilingual'
        }
        lang_name = language_names.get(args.language, args.language)
        print(f"Starting AbstractVoice voice interface ({lang_name})...")
        
        # Initialize REPL with language support
        repl = VoiceREPL(
            api_url=args.api,
            model=args.model,
            debug_mode=args.debug,
            language=args.language,
            tts_model=args.tts_model,
            disable_tts=args.no_tts
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
        
        # Change Whisper model if specified
        if args.whisper and args.whisper != "tiny":
            if repl.voice_manager.set_whisper(args.whisper):
                if args.debug:
                    print(f"Using Whisper model: {args.whisper}")
        
        # Start in voice mode automatically unless --no-listening is specified
        if not args.no_listening:
            print("Activating voice mode. Say 'stop' to exit voice mode.")
            # Use the existing voice mode method
            repl.do_voice("on")
        
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
            print(f"   Try: rm -rf ~/.cache/tts && pip install --force-reinstall coqui-tts")
            print(f"   Or check network connectivity for model downloads")
        elif "ollama" in error_msg or "11434" in error_msg:
            print(f"❌ Cannot connect to Ollama at {args.api}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   Your model '{args.model}' exists but Ollama server isn't responding")
        elif "importerror" in error_msg or "no module" in error_msg:
            print(f"❌ Missing dependencies")
            print(f"   Try running: abstractvoice check-deps")
            print(f"   Or install dependencies: pip install abstractvoice[voice-full]")
        elif "espeak" in error_msg or "phoneme" in error_msg:
            print(f"❌ Voice synthesis setup issue")
            print(f"   Install espeak-ng for better voice quality: brew install espeak-ng")
            print(f"   Or this might be a TTS model download issue")
        else:
            print(f"❌ Application error: {e}")
            print(f"   Try running with --debug for more details")
            print(f"   Note: Your Ollama model '{args.model}' appears to be available")

        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 