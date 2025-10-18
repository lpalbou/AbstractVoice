#!/usr/bin/env python3
"""
AbstractVoice voice mode CLI launcher.

This module provides a direct entry point to start AbstractVoice in voice mode.
"""

import argparse
import time
from abstractvoice.examples.cli_repl import VoiceREPL

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AbstractVoice Voice Mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--api", default="http://localhost:11434/api/chat",
                      help="LLM API URL")
    parser.add_argument("--model", default="granite3.3:2b",
                      help="LLM model name")
    parser.add_argument("--whisper", default="tiny",
                      help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--no-listening", action="store_true",
                      help="Disable speech-to-text (listening), TTS still works")
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
    """Entry point for direct voice mode."""
    try:
        # Parse command line arguments
        args = parse_args()
        
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
            tts_model=args.tts_model
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
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 