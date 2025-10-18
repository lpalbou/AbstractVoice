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
    print("\nUsage: python -m abstractvoice <example> [args...]")


def simple_example():
    """Run a simple example demonstrating basic usage."""
    from abstractvoice import VoiceManager
    import time
    
    print("Simple AbstractVoice Example")
    print("============================")
    print("This example demonstrates basic TTS and STT functionality.")
    print()
    
    # Initialize voice manager
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
    parser.add_argument("example", nargs="?", help="Example to run (cli, web, simple)")
    
    # Parse just the first argument
    args, remaining = parser.parse_known_args()
    
    if not args.example:
        print_examples()
        return
    
    # Set remaining args as sys.argv for the examples
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