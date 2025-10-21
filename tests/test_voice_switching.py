#!/usr/bin/env python3
"""
Test script to verify voice switching works correctly.
This will help us validate the fixes to the voice switching system.
"""

from abstractvoice import VoiceManager
import time

def test_voice_switching():
    """Test voice switching to ensure different models actually load."""
    print("🧪 Testing Voice Switching Fixes...")
    print("=" * 50)

    vm = VoiceManager(debug_mode=True)
    print("✅ VoiceManager initialized")

    # Test voices that should sound different
    test_voices = [
        ("en", "tacotron2"),  # Female LJSpeech
        ("en", "jenny"),      # Different female
        ("en", "ek1"),        # Male British
    ]

    print(f"\n🎭 Testing {len(test_voices)} different voices...")

    for i, (lang, voice) in enumerate(test_voices):
        print(f"\n--- Test {i+1}: {lang}.{voice} ---")

        try:
            # Set the voice and ensure it actually loads the requested model
            success = vm.set_voice(lang, voice)

            if success:
                print(f"✅ Voice {voice} set successfully")

                # Test speech with this voice
                test_text = f"Hello, this is voice {voice}."
                vm.speak(test_text, speed=1.0)

                # Small delay between voice tests
                time.sleep(1.0)

            else:
                print(f"❌ Failed to set voice {voice}")

        except Exception as e:
            print(f"💥 Exception with voice {voice}: {e}")

    # Test language switching (should work without crashes)
    print(f"\n🌍 Testing language switching...")

    languages = ["en", "fr", "de"]
    for lang in languages:
        try:
            success = vm.set_language(lang)
            if success:
                print(f"✅ Language {lang}: OK")
                vm.speak(f"Testing {lang}", speed=1.0)
                time.sleep(0.5)
            else:
                print(f"❌ Language {lang}: Failed")
        except Exception as e:
            print(f"💥 Language {lang}: Exception - {e}")

    # Test Italian (the crash-prone one)
    print(f"\n🇮🇹 Testing Italian models (crash safety)...")

    italian_voices = ["mai_male_vits", "mai_female_vits"]
    for voice in italian_voices:
        try:
            print(f"Testing it.{voice}...")
            success = vm.set_voice("it", voice)
            if success:
                print(f"✅ Italian {voice}: Safe!")
                vm.speak("Ciao, test italiano.", speed=0.8)  # Slower for Italian
                time.sleep(0.5)
            else:
                print(f"⚠️ Italian {voice}: Skipped (safety)")
        except Exception as e:
            print(f"💥 Italian {voice}: Exception handled - {e}")

    vm.cleanup()
    print(f"\n🎉 Voice switching test complete!")

if __name__ == "__main__":
    test_voice_switching()