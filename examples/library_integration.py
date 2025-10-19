#!/usr/bin/env python3
"""
Example: Using AbstractVoice as a Library Dependency with Model Management

This example demonstrates how to use AbstractVoice as a dependency in your
application with proper model management for both offline and online scenarios.
"""

from abstractvoice import VoiceManager
import time


def example_robust_library_integration():
    """
    Example showing robust integration with model availability checking
    and graceful handling of offline/online scenarios.
    """
    print("🔧 Robust Library Integration Example")
    print("=" * 50)

    # Initialize VoiceManager
    vm = VoiceManager(debug_mode=False)

    # Strategy 1: Check if models are available before using TTS
    print("\n📦 Step 1: Check model availability...")
    if vm.check_models_available():
        print("✅ TTS models are ready - can use speech immediately")
        vm.speak("Hello! TTS is ready to use.")
    else:
        print("❌ No TTS models available locally")
        print("   This typically happens on first installation")

        # Strategy 2: Download essential models if needed
        print("\n📥 Step 2: Downloading essential models...")

        def progress_callback(model_name, success):
            status = "✅" if success else "❌"
            print(f"   {status} {model_name}")

        success = vm.download_essential_models(progress_callback)

        if success:
            print("✅ Essential models downloaded successfully!")
            vm.speak("Models downloaded! TTS is now ready.")
        else:
            print("❌ Failed to download essential models")
            print("   Falling back to text-only mode")

    # Strategy 3: Language-specific model management
    print("\n🌍 Step 3: Multi-language support...")

    target_languages = ['fr', 'es', 'de']

    for lang in target_languages:
        print(f"\n🔍 Checking {lang.upper()} models...")

        if vm.check_models_available(lang):
            print(f"✅ {lang.upper()} models ready")
            # Switch to this language and test
            vm.set_language(lang)
            test_messages = {
                'fr': "Bonjour! Les modèles français sont prêts.",
                'es': "¡Hola! Los modelos en español están listos.",
                'de': "Hallo! Die deutschen Modelle sind bereit."
            }
            vm.speak(test_messages[lang])
        else:
            print(f"⚠️ {lang.upper()} models not available")

            # Option: Download on demand
            print(f"📥 Downloading {lang.upper()} models...")
            success = vm.download_language_models(lang, progress_callback)

            if success:
                print(f"✅ {lang.upper()} models downloaded!")
                vm.set_language(lang)
                vm.speak(test_messages[lang])
            else:
                print(f"❌ Failed to download {lang.upper()} models")

    # Strategy 4: Get comprehensive status
    print("\n📊 Step 4: Model status summary...")
    status = vm.get_model_status()

    print(f"📍 Cache location: {status['cache_dir']}")
    print(f"📦 Total models cached: {status['total_cached']}")
    print(f"🚀 Offline ready: {status['offline_ready']}")
    print(f"🌐 Current language ready: {status['current_language_ready']}")
    print(f"📝 Essential models: {len(status['essential_models_cached'])}/{len(status['essential_models_available'])}")
    print(f"✨ Premium models: {len(status['premium_models_cached'])}/{len(status['premium_models_available'])}")

    # Clean up
    vm.cleanup()
    print("\n🎉 Example completed successfully!")


def example_enterprise_deployment():
    """
    Example for enterprise deployment scenarios where you need
    to ensure all models are available before starting the application.
    """
    print("\n🏢 Enterprise Deployment Example")
    print("=" * 50)

    # Pre-deployment model verification
    vm = VoiceManager(debug_mode=False)

    # Check if all required languages are ready
    required_languages = ['en', 'fr', 'es', 'de', 'it']

    print("🔍 Pre-deployment verification...")
    all_ready = True

    for lang in required_languages:
        ready = vm.check_models_available(lang)
        status = "✅" if ready else "❌"
        lang_name = vm.LANGUAGES.get(lang, {}).get('name', lang)
        print(f"   {status} {lang_name} ({lang})")

        if not ready:
            all_ready = False

    if all_ready:
        print("✅ All required languages are ready for production!")
    else:
        print("❌ Some languages missing models")
        print("🔧 Run this command to prepare all models:")
        print("   for lang in en fr es de it; do")
        print("     abstractvoice download-models --language $lang")
        print("   done")

    # Production-ready initialization
    if all_ready:
        print("\n🚀 Starting production voice service...")

        # Test each language briefly
        for lang in required_languages:
            vm.set_language(lang)
            lang_name = vm.get_language_name()
            print(f"✅ {lang_name} voice service ready")

    vm.cleanup()


def example_simple_integration():
    """
    Simple example for basic library users who just want TTS to work.
    """
    print("\n🎯 Simple Integration Example")
    print("=" * 50)

    # One-liner: ensure models are ready and use TTS
    vm = VoiceManager()

    # This will check for models and download if needed
    if vm.ensure_models_ready(auto_download=True):
        print("✅ TTS is ready!")
        vm.speak("Hello! This is the simple integration example.")

        # Switch to French if models are available
        if vm.ensure_models_ready('fr', auto_download=True):
            vm.set_language('fr')
            vm.speak("Bonjour! Exemple d'intégration simple.")
    else:
        print("❌ Could not prepare TTS models")
        print("   Check internet connection or use --no-tts mode")

    vm.cleanup()


if __name__ == "__main__":
    # Run all examples
    example_simple_integration()
    example_robust_library_integration()
    example_enterprise_deployment()

    print("\n" + "="*70)
    print("📚 INTEGRATION PATTERNS SUMMARY")
    print("="*70)
    print()
    print("🎯 SIMPLE: vm.ensure_models_ready(auto_download=True)")
    print("   • One-liner to check and download models")
    print("   • Perfect for simple applications")
    print()
    print("🔧 ROBUST: vm.check_models_available() + conditional download")
    print("   • Check first, download if needed")
    print("   • Better control over download process")
    print("   • Progress callbacks for user feedback")
    print()
    print("🏢 ENTERPRISE: Pre-deployment verification")
    print("   • Verify all required models before production")
    print("   • Fail fast if models are missing")
    print("   • Use abstractvoice download-models in deployment scripts")
    print()
    print("📊 STATUS: vm.get_model_status()")
    print("   • Comprehensive status information")
    print("   • Cache location, model counts, readiness")
    print("   • Perfect for monitoring and debugging")