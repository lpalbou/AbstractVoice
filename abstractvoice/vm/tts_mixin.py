"""TTS + voice/language methods for VoiceManager.

This module intentionally focuses on orchestration and keeps heavy engine details
behind adapters or TTSEngine implementations.
"""

from __future__ import annotations

from .common import import_tts_engine


class TtsMixin:
    def speak(self, text, speed=1.0, callback=None):
        sp = speed if speed != 1.0 else self.speed
        if not self.tts_engine:
            raise RuntimeError("No TTS engine available")
        return self.tts_engine.speak(text, sp, callback)

    # Network/headless-friendly methods
    def speak_to_bytes(self, text: str, format: str = "wav") -> bytes:
        if self.tts_adapter and self.tts_adapter.is_available():
            return self.tts_adapter.synthesize_to_bytes(text, format=format)
        raise NotImplementedError(
            "speak_to_bytes() requires Piper TTS (default engine)."
        )

    def speak_to_file(self, text: str, output_path: str, format: str | None = None) -> str:
        if self.tts_adapter and self.tts_adapter.is_available():
            return self.tts_adapter.synthesize_to_file(text, output_path, format=format)
        raise NotImplementedError(
            "speak_to_file() requires Piper TTS (default engine)."
        )

    def stop_speaking(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.stop()

    def pause_speaking(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.pause()

    def resume_speaking(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.resume()

    def is_paused(self):
        if not self.tts_engine:
            return False
        return self.tts_engine.is_paused()

    def is_speaking(self):
        if self.tts_engine:
            return self.tts_engine.is_active()
        return False

    def set_speed(self, speed):
        if 0.5 <= speed <= 2.0:
            self.speed = speed
            return True
        return False

    def get_speed(self):
        return self.speed

    def _try_init_piper(self, language: str):
        try:
            from ..adapters.tts_piper import PiperTTSAdapter
            adapter = PiperTTSAdapter(language=language)
            if adapter.is_available():
                return adapter
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Piper TTS not available: {e}")
            return None

    def set_tts_model(self, model_name):
        self.stop_speaking()

        if hasattr(self, "tts_engine") and self.tts_engine:
            try:
                if hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                    if hasattr(self.tts_engine.audio_player, "stop"):
                        self.tts_engine.audio_player.stop()
                    self.tts_engine.audio_player.cleanup()

                if hasattr(self.tts_engine, "tts") and self.tts_engine.tts:
                    try:
                        import torch  # type: ignore[import-not-found]
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    del self.tts_engine.tts

                del self.tts_engine
                self.tts_engine = None

                import gc
                gc.collect()
            except Exception as e:
                if self.debug_mode:
                    print(f"Warning: TTS cleanup issue: {e}")
                self.tts_engine = None

        TTSEngine = import_tts_engine()
        self.tts_engine = TTSEngine(model_name=model_name, debug_mode=self.debug_mode)
        self._tts_engine_name = "vits"
        self._wire_tts_callbacks()
        return True

    def get_supported_languages(self):
        return list(self.LANGUAGES.keys())

    def get_language(self):
        return self.language

    def get_language_name(self, language_code=None):
        lang = language_code or self.language
        return self.LANGUAGES.get(lang, {}).get("name", lang)

    def set_language(self, language):
        language = language.lower()
        if language not in self.LANGUAGES:
            if self.debug_mode:
                available = ", ".join(self.LANGUAGES.keys())
                print(f"⚠️ Unsupported language '{language}'. Available: {available}")
            return False

        if language == self.language:
            if self.debug_mode:
                print(f"✓ Already using {self.LANGUAGES[language]['name']} voice")
            return True

        self.stop_speaking()
        if self.voice_recognizer:
            self.voice_recognizer.stop()

        if self._tts_engine_preference in ("auto", "piper"):
            try:
                if self.tts_adapter is None:
                    self.tts_adapter = self._try_init_piper(language)
                else:
                    self.tts_adapter.set_language(language)

                if self.tts_adapter and self.tts_adapter.is_available():
                    if self._tts_engine_name != "piper" or self.tts_engine is None:
                        from ..tts.adapter_tts_engine import AdapterTTSEngine

                        self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                        self._tts_engine_name = "piper"
                        self._wire_tts_callbacks()

                    self.language = language
                    self.speed = 1.0
                    return True
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Piper language switch failed, falling back: {e}")

        selected_model = self._select_best_model(language)

        from ..instant_setup import is_model_cached
        from ..simple_model_manager import download_model

        if not is_model_cached(selected_model):
            if self.debug_mode:
                print(f"📥 Model {selected_model} not cached, downloading...")
            success = download_model(selected_model)
            if not success and language != "en":
                print(f"❌ Cannot switch to {self.LANGUAGES[language]['name']}: Model download failed")
                print(f"   Try: abstractvoice download-models --language {language}")
                return False

        models_to_try = [selected_model]
        if selected_model != self.SAFE_FALLBACK:
            models_to_try.append(self.SAFE_FALLBACK)

        for model_name in models_to_try:
            try:
                if self.debug_mode:
                    lang_name = self.LANGUAGES[language]["name"]
                    print(f"🌍 Loading {lang_name} voice: {model_name}")

                TTSEngine = import_tts_engine()
                self.tts_engine = TTSEngine(model_name=model_name, debug_mode=self.debug_mode)
                self._tts_engine_name = "vits"
                self._wire_tts_callbacks()

                self.language = language
                self.speed = 0.8 if language == "it" else 1.0
                return True
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Model {model_name} failed to load: {e}")
                continue

        print(f"❌ Cannot switch to {self.LANGUAGES[language]['name']}: No working models")
        return False

    def _select_best_model(self, language):
        if language not in self.LANGUAGES:
            return self.SAFE_FALLBACK

        lang_config = self.LANGUAGES[language]

        if "premium" in lang_config:
            try:
                premium_model = lang_config["premium"]
                if self._test_model_compatibility(premium_model):
                    if self.debug_mode:
                        print(f"✨ Using premium quality model: {premium_model}")
                    return premium_model
            except Exception:
                pass

        return lang_config.get("default", self.SAFE_FALLBACK)

    def _test_model_compatibility(self, model_name):
        try:
            TTSEngine = import_tts_engine()
            engine = TTSEngine(model_name=model_name, debug_mode=False)
            engine.cleanup()
            return True
        except Exception:
            return False

    def set_voice(self, language, voice_id):
        language = language.lower()

        if self._tts_engine_preference in ("auto", "piper"):
            try:
                if self.tts_adapter is None:
                    self.tts_adapter = self._try_init_piper(language)
                else:
                    self.tts_adapter.set_language(language)

                if self.tts_adapter and self.tts_adapter.is_available():
                    if self._tts_engine_name != "piper" or self.tts_engine is None:
                        from ..tts.adapter_tts_engine import AdapterTTSEngine

                        self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                        self._tts_engine_name = "piper"
                        self._wire_tts_callbacks()

                    self.language = language
                    self.speed = 1.0
                    if self.debug_mode:
                        print(f"🎭 Piper voice selection: using default {language} voice (requested: {voice_id})")
                    return True
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Piper voice selection failed, falling back: {e}")

        if language not in self.VOICE_CATALOG:
            if self.debug_mode:
                print(f"⚠️ Language '{language}' not available")
            return False

        if voice_id not in self.VOICE_CATALOG[language]:
            if self.debug_mode:
                available = ", ".join(self.VOICE_CATALOG[language].keys())
                print(f"⚠️ Voice '{voice_id}' not available for {language}. Available: {available}")
            return False

        voice_info = self.VOICE_CATALOG[language][voice_id]
        model_name = voice_info["model"]

        from ..instant_setup import is_model_cached
        from ..simple_model_manager import download_model

        if not is_model_cached(model_name):
            print(f"📥 Voice model '{voice_id}' not cached, downloading...")
            success = download_model(model_name)
            if not success:
                print(f"❌ Failed to download voice '{voice_id}'")
                return False

        self.language = language
        self.speed = voice_info.get("speed", 1.0)
        return self.set_tts_model(model_name)

