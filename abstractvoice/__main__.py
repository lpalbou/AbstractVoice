#!/usr/bin/env python3
"""
AbstractVoice - A modular Python library for voice interactions with AI systems.

This module allows running the examples directly.
"""

import sys
import argparse


def print_examples():
    """Print available examples."""
    print("Available examples:")
    print("  cli       - Command-line REPL example")
    print("  web       - Web API example")
    print("  simple    - Simple usage example")
    print("  check-deps - Check dependency compatibility")
    print("  download  - Explicitly prefetch model artifacts")
    print("\nUsage: python -m abstractvoice <example> [--language <lang>] [args...]")
    print("\nSupported languages: en, fr, de, es, ru, zh")
    print("\nExamples:")
    print("  python -m abstractvoice cli --language fr    # French CLI")
    print("  python -m abstractvoice simple --language ru # Russian simple example")
    print("  python -m abstractvoice check-deps           # Check dependencies")


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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AbstractVoice examples")
    parser.add_argument("example", nargs="?", help="Example to run (cli, web, simple, check-deps)")
    parser.add_argument("--language", "--lang", default="en",
                      choices=["en", "fr", "de", "es", "ru", "zh"],
                      help="Voice language for examples")

    # Parse just the first argument and language
    args, remaining = parser.parse_known_args()

    if not args.example:
        print_examples()
        return

    # Handle check-deps specially (doesn't need language)
    if args.example == "check-deps":
        from abstractvoice.dependency_check import check_dependencies
        try:
            check_dependencies(verbose=True)
        except Exception as e:
            print(f"❌ Error running dependency check: {e}")
            print("This might indicate a dependency issue.")
        return

    if args.example == "download":
        dl = argparse.ArgumentParser(description="AbstractVoice explicit downloads")
        dl.add_argument("--stt", dest="stt_model", default=None, help="Prefetch faster-whisper model (e.g. small)")
        dl.add_argument(
            "--openf5",
            action="store_true",
            help="Prefetch OpenF5 artifacts for cloning (~5.4GB, requires abstractvoice[cloning])",
        )
        dl.add_argument(
            "--chroma",
            action="store_true",
            help="Prefetch Chroma-4B artifacts (~14GB+, requires HF access; install abstractvoice[chroma] to run inference)",
        )
        dl.add_argument(
            "--piper",
            dest="piper_language",
            default=None,
            help="Prefetch Piper voice model for a language (e.g. en/fr/de).",
        )
        dl_args = dl.parse_args(remaining)

        if not dl_args.stt_model and not dl_args.openf5 and not dl_args.chroma and not dl_args.piper_language:
            print("Nothing to download. Examples:")
            print("  python -m abstractvoice download --stt small")
            print("  python -m abstractvoice download --openf5")
            print("  python -m abstractvoice download --chroma")
            print("  python -m abstractvoice download --piper en")
            return

        if dl_args.stt_model:
            try:
                from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter

                model = str(dl_args.stt_model).strip()
                print(f"Downloading STT model (faster-whisper): {model}")
                stt = FasterWhisperAdapter(model_size=model, device="cpu", compute_type="int8", allow_downloads=True)
                if not stt.is_available():
                    raise RuntimeError("Model download/load failed.")
                print("✅ STT model ready.")
            except Exception as e:
                print(f"❌ STT download failed: {e}")

        if dl_args.openf5:
            try:
                from abstractvoice.cloning.engine_f5 import F5TTSVoiceCloningEngine

                print("Downloading OpenF5 artifacts (cloning)…")
                engine = F5TTSVoiceCloningEngine(debug=True)
                engine.ensure_openf5_artifacts_downloaded()
                print("✅ OpenF5 artifacts ready.")
            except Exception as e:
                print(f"❌ OpenF5 download failed: {e}")

        if dl_args.chroma:
            try:
                from abstractvoice.cloning.engine_chroma import ChromaVoiceCloningEngine

                print("Downloading Chroma artifacts (cloning)…")
                engine = ChromaVoiceCloningEngine(debug=True)
                engine.ensure_chroma_artifacts_downloaded()
                print("✅ Chroma artifacts ready.")
            except Exception as e:
                print(f"❌ Chroma download failed: {e}")

        if dl_args.piper_language:
            try:
                from abstractvoice.adapters.tts_piper import PiperTTSAdapter

                lang = str(dl_args.piper_language).strip().lower()
                print(f"Downloading Piper voice model: {lang}")
                piper = PiperTTSAdapter(language=lang, allow_downloads=True, auto_load=False)
                if not piper.ensure_model_downloaded(lang):
                    raise RuntimeError("Piper model download failed.")
                print("✅ Piper model ready.")
            except Exception as e:
                print(f"❌ Piper download failed: {e}")
        return

    # Set remaining args as sys.argv for the examples, including language
    if args.language != "en":
        remaining = ["--language", args.language] + remaining
    sys.argv = [sys.argv[0]] + remaining

    if args.example == "cli":
        from abstractvoice.examples.cli_repl import main
        main()
    elif args.example == "web":
        from abstractvoice.examples.web_api import main
        main()
    elif args.example == "simple":
        simple_example()
    else:
        print(f"Unknown example: {args.example}")
        print_examples()


if __name__ == "__main__":
    main() 
