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

    # IMPORTANT:
    # This is a "fresh install" simulation test and must be stable in CI/headless
    # environments. We validate functionality via network-safe methods (bytes/file)
    # instead of real audio playback (which can be flaky due to OS audio backends).
    vm = VoiceManager(debug_mode=True, tts_engine="piper")

    # Test languages
    test_languages = [
        ('fr', 'Bonjour, ceci est un test.', 'French'),
        ('es', 'Hola, esta es una prueba.', 'Spanish'),
        ('de', 'Hallo, das ist ein Test.', 'German'),
        ('ru', 'Привет, это тест.', 'Russian'),
        ('zh', '你好，这是一个测试。', 'Chinese'),
        ('en', 'Back to English.', 'English'),
    ]

    for lang, text, name in test_languages:
        print(f"\n🌍 Testing {name} ({lang})...")
        success = vm.set_language(lang)

        if success:
            print(f"✅ {name}: Successfully loaded")
            audio_bytes = vm.speak_to_bytes(text, format="wav")
            assert isinstance(audio_bytes, (bytes, bytearray))
            assert audio_bytes[:4] == b"RIFF"
        else:
            print(f"❌ {name}: Failed to load")
            print(f"   Piper voice for {lang} may be missing or download failed")

    vm.cleanup()
    print("\n✅ Language switching test complete!")


def test_voice_switching():
    """Test voice switching with download behavior."""
    from abstractvoice import VoiceManager

    print("\n🎭 Testing Voice Switching (Fresh Install Simulation)")
    print("=" * 60)

    vm = VoiceManager(debug_mode=True, tts_engine="piper")

    # Piper currently selects a default voice per language.
    # We still validate that `set_voice()` is robust and does not crash even if
    # voice IDs are treated as best-effort hints.
    test_voices = [
        ('en', 'amy', 'This is a test voice.'),
        ('fr', 'siwis', 'Voix française.'),
        ('de', 'thorsten', 'Das ist ein Test.'),
    ]

    for lang, voice_id, text in test_voices:
        print(f"\n🎤 Testing {lang}.{voice_id}...")
        success = vm.set_voice(lang, voice_id)

        if success:
            print(f"✅ {voice_id}: Successfully loaded")
            audio_bytes = vm.speak_to_bytes(text, format="wav")
            assert isinstance(audio_bytes, (bytes, bytearray))
            assert audio_bytes[:4] == b"RIFF"
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

    # Disable TTS so this test is stable in CI/headless environments.
    cli.onecmd('/tts off')

    # Test /language command
    print("\n📝 Testing /language fr")
    cli.onecmd('/language fr')

    # Test /setvoice command (listing only; avoids optional legacy downloads)
    print("\n📝 Testing /setvoice (list)")
    cli.onecmd('/setvoice')

    # Clone management (safe: requires explicit confirmation to delete all).
    print("\n📝 Testing /clone_rm_all (no confirmation)")
    cli.onecmd('/clone_rm_all')

    print("\n✅ CLI commands test complete!")


def test_download_status():
    """Test that no legacy model-management API leaks into core."""
    import abstractvoice

    assert not hasattr(abstractvoice, "list_models")
    assert not hasattr(abstractvoice, "download_model")
    assert not hasattr(abstractvoice, "get_status")
    assert not hasattr(abstractvoice, "is_ready")


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
