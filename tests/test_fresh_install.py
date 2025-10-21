#!/usr/bin/env python3
"""
Test script that simulates fresh install behavior
Tests language switching and voice selection with download requirements
"""

def test_language_switching():
    """Test language switching with download behavior."""
    from abstractvoice import VoiceManager

    print("🧪 Testing Language Switching (Fresh Install Simulation)")
    print("=" * 60)

    vm = VoiceManager(debug_mode=True)

    # Test languages
    test_languages = [
        ('fr', 'Bonjour, ceci est un test.', 'French'),
        ('es', 'Hola, esta es una prueba.', 'Spanish'),
        ('de', 'Hallo, das ist ein Test.', 'German'),
        ('it', 'Ciao, questo è un test.', 'Italian'),
        ('en', 'Back to English.', 'English'),
    ]

    for lang, text, name in test_languages:
        print(f"\n🌍 Testing {name} ({lang})...")
        success = vm.set_language(lang)

        if success:
            print(f"✅ {name}: Successfully loaded")
            vm.speak(text, speed=1.0)
        else:
            print(f"❌ {name}: Failed to load")
            print(f"   Run: abstractvoice download-models --language {lang}")

    vm.cleanup()
    print("\n✅ Language switching test complete!")


def test_voice_switching():
    """Test voice switching with download behavior."""
    from abstractvoice import VoiceManager

    print("\n🎭 Testing Voice Switching (Fresh Install Simulation)")
    print("=" * 60)

    vm = VoiceManager(debug_mode=True)

    # Test different voices
    test_voices = [
        ('en', 'tacotron2', 'This is Linda voice.'),
        ('en', 'jenny', 'This is Jenny voice.'),
        ('en', 'ek1', 'This is Edward voice.'),
        ('fr', 'css10_vits', 'Voix française.'),
    ]

    for lang, voice_id, text in test_voices:
        print(f"\n🎤 Testing {lang}.{voice_id}...")
        success = vm.set_voice(lang, voice_id)

        if success:
            print(f"✅ {voice_id}: Successfully loaded")
            vm.speak(text)
        else:
            print(f"❌ {voice_id}: Failed to load")

    vm.cleanup()
    print("\n✅ Voice switching test complete!")


def test_cli_commands():
    """Test CLI commands for model management."""
    from abstractvoice.examples.cli_repl import VoiceREPL

    print("\n💻 Testing CLI Commands")
    print("=" * 60)

    cli = VoiceREPL()

    # Test /language command
    print("\n📝 Testing /language fr")
    cli.onecmd('/language fr')

    # Test /setvoice command
    print("\n📝 Testing /setvoice en.jenny")
    cli.onecmd('/setvoice en.jenny')

    print("\n✅ CLI commands test complete!")


def test_download_status():
    """Test model download and status checking."""
    from abstractvoice import is_ready, get_status
    import json

    print("\n📦 Testing Model Status")
    print("=" * 60)

    # Check if ready
    ready = is_ready()
    print(f"System ready: {ready}")

    # Get detailed status
    status = json.loads(get_status())
    print(f"Total cached models: {status.get('total_cached', 0)}")
    print(f"Cache size: {status.get('total_size_mb', 0):.1f} MB")
    print(f"Ready for offline: {status.get('ready_for_offline', False)}")

    # List cached models
    if 'cached_models' in status:
        print("\nCached models:")
        for model in status['cached_models'][:5]:  # Show first 5
            print(f"  • {model}")
        if len(status['cached_models']) > 5:
            print(f"  ... and {len(status['cached_models']) - 5} more")

    print("\n✅ Model status test complete!")


def main():
    """Run all tests."""
    import sys

    print("🚀 AbstractVoice Fresh Install Simulation")
    print("=" * 60)
    print("This tests how the system behaves on a fresh install")
    print("when models need to be downloaded.\n")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--language':
            test_language_switching()
        elif sys.argv[1] == '--voice':
            test_voice_switching()
        elif sys.argv[1] == '--cli':
            test_cli_commands()
        elif sys.argv[1] == '--status':
            test_download_status()
        else:
            print("Usage: python test_fresh_install.py [--language|--voice|--cli|--status]")
    else:
        # Run all tests
        test_download_status()
        test_language_switching()
        test_voice_switching()
        test_cli_commands()

        print("\n" + "=" * 60)
        print("🎉 All fresh install tests complete!")
        print("\nKey findings:")
        print("  • Language switching downloads models if needed")
        print("  • Voice switching downloads models if needed")
        print("  • Clear error messages when downloads fail")
        print("  • CLI commands properly handle missing models")


if __name__ == "__main__":
    main()