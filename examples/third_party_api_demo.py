#!/usr/bin/env python3
"""
Third-Party API Demo for AbstractVoice

This example demonstrates all the APIs that third-party applications can use
to integrate AbstractVoice into their applications.
"""

import json
from abstractvoice import VoiceManager, list_models, download_model, get_status, is_ready


def demo_simple_apis():
    """Demonstrate the simple JSON APIs for third-party integration."""
    print("🔧 SIMPLE APIs FOR THIRD-PARTY APPLICATIONS")
    print("=" * 60)

    # 1. Check if essential model is ready
    print("1. Checking essential model availability...")
    ready = is_ready()
    print(f"   Essential model ready: {ready}")

    # 2. Get cache status
    print("\n2. Getting cache status...")
    status_json = get_status()
    status = json.loads(status_json)
    print(f"   Total cached models: {status['total_cached']}")
    print(f"   Cache size: {status['total_size_mb']} MB")
    print(f"   Ready for offline: {status['ready_for_offline']}")
    print(f"   Available languages: {status['available_languages']}")

    # 3. List all available models
    print("\n3. Listing all available models...")
    models_json = list_models()
    models = json.loads(models_json)

    for language, voices in models.items():
        print(f"   {language.upper()}: {len(voices)} voices")
        for voice_id, voice_info in voices.items():
            cached = "✅" if voice_info['cached'] else "📥"
            quality = "✨" if voice_info['quality'] == 'excellent' else "🔧"
            print(f"     {cached} {quality} {language}.{voice_id} ({voice_info['size_mb']}MB)")

    # 4. Download a specific model
    print("\n4. Downloading a specific model...")
    success = download_model('fr.css10_vits')
    print(f"   Downloaded French VITS: {success}")

    return models


def demo_voice_manager_integration():
    """Demonstrate VoiceManager integration with model management."""
    print("\n\n🎯 VOICEMANAGER INTEGRATION")
    print("=" * 60)

    # Initialize VoiceManager
    vm = VoiceManager(debug_mode=False)
    print(f"1. VoiceManager initialized: {vm.get_language_name()}")

    # Check model readiness
    ready = vm.is_model_ready()
    print(f"2. Essential model ready: {ready}")

    # Ensure ready (download if needed)
    if not ready:
        print("3. Downloading essential model...")
        success = vm.ensure_ready()
        print(f"   Download success: {success}")

    # Test TTS
    print("4. Testing TTS...")
    result = vm.speak("Testing AbstractVoice integration!", speed=1.0)
    print(f"   TTS result: {result}")

    # List available models through VoiceManager
    print("5. Getting models via VoiceManager...")
    models = vm.list_available_models()
    print(f"   Available languages: {list(models.keys())}")

    # Download and switch to French
    print("6. Downloading and switching to French...")
    success = vm.download_model('fr.css10_vits')
    if success:
        vm.set_language('fr')
        vm.speak("Bonjour! Je parle français maintenant.")
        print("   ✅ French voice working!")

    # Get cache status
    status = vm.get_cache_status()
    print(f"7. Cache status: {status['total_cached']} models cached")

    vm.cleanup()


def demo_cli_simulation():
    """Simulate how CLI commands use the programmatic APIs."""
    print("\n\n💻 CLI COMMAND SIMULATION")
    print("=" * 60)

    # Simulate /setvoice command
    print("1. Simulating '/setvoice' command...")

    # List models (what CLI shows)
    models_json = list_models()
    models = json.loads(models_json)

    print("   Available voices:")
    for language, voices in models.items():
        lang_names = {
            'en': 'English', 'fr': 'French', 'es': 'Spanish',
            'de': 'German', 'it': 'Italian'
        }
        lang_name = lang_names.get(language, language.upper())
        print(f"   🌍 {lang_name} ({language}):")

        for voice_id, voice_info in voices.items():
            cached = "✅" if voice_info['cached'] else "📥"
            quality = "✨" if voice_info['quality'] == 'excellent' else "🔧"
            print(f"     {cached} {quality} {language}.{voice_id}")
            print(f"       {voice_info['name']} ({voice_info['size_mb']}MB)")

    # Simulate downloading a voice
    print("\n2. Simulating '/setvoice it.mai_male_vits'...")
    success = download_model('it.mai_male_vits')
    print(f"   Download result: {success}")

    # Show final status
    status_json = get_status()
    status = json.loads(status_json)
    print(f"\n3. Final status: {status['total_cached']} models ready")


def demo_web_api_integration():
    """Demonstrate how a web API would use these functions."""
    print("\n\n🌐 WEB API INTEGRATION EXAMPLE")
    print("=" * 60)

    # Simulate REST API endpoints
    endpoints = {
        'GET /api/models': list_models,
        'GET /api/models/status': get_status,
        'GET /api/models/ready': lambda: json.dumps({"ready": is_ready()}),
        'POST /api/models/download': lambda model: json.dumps({"success": download_model(model)})
    }

    for endpoint, func in endpoints.items():
        print(f"\n{endpoint}:")
        try:
            if endpoint == 'POST /api/models/download':
                result = func('en.vits')
            else:
                result = func()

            # Pretty print JSON (truncated for demo)
            if isinstance(result, str):
                data = json.loads(result)
                print(json.dumps(data, indent=2)[:200] + "..." if len(str(data)) > 200 else json.dumps(data, indent=2))
            else:
                print(result)
        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    print("🎉 ABSTRACTVOICE THIRD-PARTY API DEMONSTRATION")
    print("=" * 70)
    print()
    print("This demo shows all the ways third-party applications can")
    print("integrate AbstractVoice into their systems.")
    print()

    try:
        # Run all demonstrations
        models = demo_simple_apis()
        demo_voice_manager_integration()
        demo_cli_simulation()
        demo_web_api_integration()

        print("\n\n✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("📚 KEY TAKEAWAYS FOR THIRD-PARTY DEVELOPERS:")
        print()
        print("🔧 SIMPLE JSON APIs:")
        print("   • is_ready() - Check if TTS is ready")
        print("   • get_status() - Get cache status as JSON")
        print("   • list_models() - Get all models as JSON")
        print("   • download_model() - Download specific model")
        print()
        print("🎯 VOICEMANAGER APIs:")
        print("   • vm.is_model_ready() - Check readiness")
        print("   • vm.ensure_ready() - Download if needed")
        print("   • vm.download_model() - Download specific model")
        print("   • vm.list_available_models() - Get models as dict")
        print()
        print("💻 CLI INTEGRATION:")
        print("   • /setvoice uses download_model() internally")
        print("   • All CLI commands use the same APIs")
        print("   • Perfect consistency between CLI and programmatic use")
        print()
        print("🌐 WEB API READY:")
        print("   • All functions return JSON strings")
        print("   • Easy to wrap in REST API endpoints")
        print("   • Stateless and thread-safe")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()