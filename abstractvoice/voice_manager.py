"""Main Voice Manager class for coordinating TTS and STT components."""

# Lazy imports - heavy dependencies are only imported when needed
def _import_tts_engine():
    """Import TTSEngine with helpful error message if dependencies missing."""
    try:
        from .tts import TTSEngine
        return TTSEngine
    except ImportError as e:
        if "TTS" in str(e) or "torch" in str(e) or "librosa" in str(e):
            raise ImportError(
                "TTS functionality requires optional dependencies. Install with:\n"
                "  pip install abstractvoice[tts]    # For TTS only\n"
                "  pip install abstractvoice[all]    # For all features\n"
                f"Original error: {e}"
            ) from e
        raise

def _import_voice_recognizer():
    """Import VoiceRecognizer with helpful error message if dependencies missing."""
    try:
        from .recognition import VoiceRecognizer
        return VoiceRecognizer
    except ImportError as e:
        if "whisper" in str(e) or "tiktoken" in str(e):
            raise ImportError(
                "Speech recognition functionality requires optional dependencies. Install with:\n"
                "  pip install abstractvoice[stt]    # For speech recognition only\n"
                "  pip install abstractvoice[all]    # For all features\n"
                f"Original error: {e}"
            ) from e
        raise


class VoiceManager:
    """Main class for voice interaction capabilities with multilingual support."""

    # Smart language configuration - high quality stable defaults
    LANGUAGES = {
        'en': {
            'default': 'tts_models/en/ljspeech/vits',          # High quality premium voice
            'premium': 'tts_models/en/ljspeech/vits',          # Use same stable model
            'name': 'English'
        },
        'fr': {
            'default': 'tts_models/fr/css10/vits',             # High quality cleaner audio
            'premium': 'tts_models/fr/css10/vits',             # Use same stable model
            'name': 'French'
        },
        'es': {
            'default': 'tts_models/es/mai/tacotron2-DDC',      # Keep stable Spanish model
            'premium': 'tts_models/es/mai/tacotron2-DDC',      # Same model (reliable)
            'name': 'Spanish'
        },
        'de': {
            'default': 'tts_models/de/thorsten/vits',          # High quality German
            'premium': 'tts_models/de/thorsten/vits',          # Use same stable model
            'name': 'German'
        },
        'it': {
            'default': 'tts_models/it/mai_male/vits',          # Use slower male voice as default
            'premium': 'tts_models/it/mai_male/vits',          # Same stable model
            'name': 'Italian'
        }
    }

    # Universal safe fallback
    SAFE_FALLBACK = 'tts_models/en/ljspeech/fast_pitch'

    # Complete voice catalog with metadata
    VOICE_CATALOG = {
        'en': {
            'vits_premium': {
                'model': 'tts_models/en/ljspeech/vits',
                'quality': 'premium',
                'gender': 'female',
                'accent': 'US English',
                'license': 'Open source (LJSpeech)',
                'requires': 'espeak-ng'
            },
            'fast_pitch_reliable': {
                'model': 'tts_models/en/ljspeech/fast_pitch',
                'quality': 'good',
                'gender': 'female',
                'accent': 'US English',
                'license': 'Open source (LJSpeech)',
                'requires': 'none'
            },
            'vctk_multi': {
                'model': 'tts_models/en/vctk/vits',
                'quality': 'premium',
                'gender': 'multiple',
                'accent': 'British English',
                'license': 'Open source (VCTK)',
                'requires': 'espeak-ng'
            }
        },
        'fr': {
            'css10_vits': {
                'model': 'tts_models/fr/css10/vits',
                'quality': 'premium',
                'gender': 'male',
                'accent': 'France French',
                'license': 'Apache 2.0 (CSS10/LibriVox)',
                'requires': 'espeak-ng'
            },
            'mai_tacotron': {
                'model': 'tts_models/fr/mai/tacotron2-DDC',
                'quality': 'good',
                'gender': 'female',
                'accent': 'France French',
                'license': 'Permissive (M-AILABS/LibriVox)',
                'requires': 'none'
            }
        },
        'es': {
            'mai_tacotron': {
                'model': 'tts_models/es/mai/tacotron2-DDC',
                'quality': 'good',
                'gender': 'female',
                'accent': 'Spain Spanish',
                'license': 'Permissive (M-AILABS)',
                'requires': 'none'
            }
        },
        'de': {
            'thorsten_vits': {
                'model': 'tts_models/de/thorsten/vits',
                'quality': 'premium',
                'gender': 'male',
                'accent': 'Standard German',
                'license': 'Open source (Thorsten)',
                'requires': 'espeak-ng'
            },
            'thorsten_tacotron': {
                'model': 'tts_models/de/thorsten/tacotron2-DDC',
                'quality': 'good',
                'gender': 'male',
                'accent': 'Standard German',
                'license': 'Open source (Thorsten)',
                'requires': 'none'
            }
        },
        'it': {
            'mai_male_vits': {
                'model': 'tts_models/it/mai_male/vits',
                'quality': 'premium',
                'gender': 'male',
                'accent': 'Standard Italian',
                'license': 'Permissive (M-AILABS)',
                'requires': 'espeak-ng',
                'speed': 0.8  # Slow down to fix pace issues
            },
            'mai_female_vits': {
                'model': 'tts_models/it/mai_female/vits',
                'quality': 'premium',
                'gender': 'female',
                'accent': 'Standard Italian',
                'license': 'Permissive (M-AILABS)',
                'requires': 'espeak-ng',
                'speed': 0.8  # Slow down to fix pace issues
            }
        }
    }

    def __init__(self, language='en', tts_model=None, whisper_model="tiny", debug_mode=False):
        """Initialize the Voice Manager with language support.

        Args:
            language: Language code ('en', 'fr', 'es', 'de', 'it')
            tts_model: Specific TTS model name or None for language default
            whisper_model: Whisper model name to use
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        self.speed = 1.0

        # Validate and set language
        language = language.lower()
        if language not in self.LANGUAGES:
            if debug_mode:
                available = ', '.join(self.LANGUAGES.keys())
                print(f"⚠️ Unsupported language '{language}', using English. Available: {available}")
            language = 'en'
        self.language = language

        # Select TTS model with smart detection
        if tts_model is None:
            tts_model = self._select_best_model(self.language)
            if debug_mode:
                lang_name = self.LANGUAGES[self.language]['name']
                print(f"🌍 Using {lang_name} voice: {tts_model}")

        # Initialize TTS engine using lazy import
        TTSEngine = _import_tts_engine()
        self.tts_engine = TTSEngine(
            model_name=tts_model,
            debug_mode=debug_mode
        )
        
        # Set up callbacks to pause/resume voice recognition during TTS playback
        # This prevents the system from interrupting its own speech
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end
        
        # Voice recognizer is initialized on demand
        self.voice_recognizer = None
        self.whisper_model = whisper_model
        
        # State tracking
        self._transcription_callback = None
        self._stop_callback = None
        self._voice_mode = "full"  # full, wait, stop, ptt
    
    def _on_tts_start(self):
        """Called when TTS playback starts - handle based on voice mode."""
        if not self.voice_recognizer:
            return
        
        if self._voice_mode == "full":
            # Full mode: Keep listening but pause interrupt capability
            self.voice_recognizer.pause_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT modes: Pause listening entirely during TTS
            self.voice_recognizer.pause_listening()
    
    def _on_tts_end(self):
        """Called when TTS playback ends - handle based on voice mode."""
        if not self.voice_recognizer:
            return
        
        if self._voice_mode == "full":
            # Full mode: Resume interrupt capability
            self.voice_recognizer.resume_tts_interrupt()
        elif self._voice_mode in ["wait", "stop", "ptt"]:
            # Wait/Stop/PTT modes: Resume listening
            self.voice_recognizer.resume_listening()
    
    def speak(self, text, speed=1.0, callback=None):
        """Convert text to speech and play audio.
        
        Args:
            text: Text to convert to speech
            speed: Speech speed (0.5-2.0)
            callback: Function to call when speech completes
            
        Returns:
            True if speech started, False otherwise
        """
        sp = 1.0
        if speed != 1.0:
            sp = speed
        else:
            sp = self.speed
        
        return self.tts_engine.speak(text, sp, callback)
    
    def stop_speaking(self):
        """Stop current speech playback.
        
        Returns:
            True if stopped, False if no playback was active
        """
        return self.tts_engine.stop()
    
    def pause_speaking(self):
        """Pause current speech playback.
        
        Pauses at chunk boundaries in streaming mode. Can be resumed with resume_speaking().
        
        Returns:
            True if paused, False if no playback was active
        """
        return self.tts_engine.pause()
    
    def resume_speaking(self):
        """Resume paused speech playback.
        
        Returns:
            True if resumed, False if not paused or no playback active
        """
        return self.tts_engine.resume()
    
    def is_paused(self):
        """Check if TTS is currently paused.
        
        Returns:
            True if paused, False otherwise
        """
        return self.tts_engine.is_paused()
    
    def is_speaking(self):
        """Check if TTS is currently active.
        
        Returns:
            True if speaking, False otherwise
        """
        return self.tts_engine.is_active()
    
    def listen(self, on_transcription, on_stop=None):
        """Start listening for speech with callbacks.
        
        Args:
            on_transcription: Callback for transcribed text
            on_stop: Callback when 'stop' command detected
            
        Returns:
            True if started, False if already listening
        """
        # Store callbacks
        self._transcription_callback = on_transcription
        self._stop_callback = on_stop
        
        # Initialize recognizer if not already done
        if not self.voice_recognizer:
            def _transcription_handler(text):
                if self._transcription_callback:
                    self._transcription_callback(text)

            def _stop_handler():
                # Stop listening
                self.stop_listening()
                # Call user's stop callback if provided
                if self._stop_callback:
                    self._stop_callback()

            # Use lazy import for VoiceRecognizer
            VoiceRecognizer = _import_voice_recognizer()
            self.voice_recognizer = VoiceRecognizer(
                transcription_callback=_transcription_handler,
                stop_callback=_stop_handler,
                whisper_model=self.whisper_model,
                debug_mode=self.debug_mode
            )
        
        # Start with TTS interrupt capability
        return self.voice_recognizer.start(
            tts_interrupt_callback=self.stop_speaking
        )
    
    def stop_listening(self):
        """Stop listening for speech.
        
        Returns:
            True if stopped, False if not listening
        """
        if self.voice_recognizer:
            return self.voice_recognizer.stop()
        return False
    
    def is_listening(self):
        """Check if currently listening for speech.
        
        Returns:
            True if listening, False otherwise
        """
        return self.voice_recognizer and self.voice_recognizer.is_running
    
    def set_voice_mode(self, mode):
        """Set the voice mode (full, wait, stop, ptt).
        
        Args:
            mode: Voice mode to use
            
        Returns:
            True if successful
        """
        if mode in ["full", "wait", "stop", "ptt"]:
            self._voice_mode = mode
            return True
        return False
        
    def set_speed(self, speed):
        """Set the TTS speed.
        
        Args:
            speed: Speech speed multiplier (0.5-2.0)
            
        Returns:
            True if successful
        """
        self.speed = speed
        return True
    
    def get_speed(self):
        """Get the TTS speed.
        
        Returns:
            Current TTS speed multiplier
        """
        return self.speed

    def set_tts_model(self, model_name):
        """Change the TTS model.
        
        Available models (all pure Python, cross-platform):
        - "tts_models/en/ljspeech/fast_pitch" (default, recommended)
        - "tts_models/en/ljspeech/glow-tts" (alternative)
        - "tts_models/en/ljspeech/tacotron2-DDC" (legacy)
        
        Args:
            model_name: TTS model name to use
            
        Returns:
            True if successful
            
        Example:
            vm.set_tts_model("tts_models/en/ljspeech/glow-tts")
        """
        # Stop any current speech
        self.stop_speaking()
        
        # Reinitialize TTS engine with new model using lazy import
        TTSEngine = _import_tts_engine()
        self.tts_engine = TTSEngine(
            model_name=model_name,
            debug_mode=self.debug_mode
        )
        
        # Restore callbacks
        self.tts_engine.on_playback_start = self._on_tts_start
        self.tts_engine.on_playback_end = self._on_tts_end
        
        return True
    
    def set_whisper(self, model_name):
        """Set the Whisper model.
        
        Args:
            whisper_model: Whisper model name (tiny, base, etc.)
            
        Returns:
            True if successful
        """
        self.whisper_model = model_name
        if self.voice_recognizer:
            return self.voice_recognizer.change_whisper_model(model_name)
    
    def get_whisper(self):
        """Get the Whisper model.

        Returns:
            Current Whisper model name
        """
        return self.whisper_model

    def set_language(self, language):
        """Set the voice language.

        Args:
            language: Language code ('en', 'fr', 'es', 'de', 'it')

        Returns:
            True if successful, False otherwise
        """
        # Validate language
        language = language.lower()
        if language not in self.LANGUAGES:
            if self.debug_mode:
                available = ', '.join(self.LANGUAGES.keys())
                print(f"⚠️ Unsupported language '{language}'. Available: {available}")
            return False

        # Skip if already using this language
        if language == self.language:
            if self.debug_mode:
                print(f"✓ Already using {self.LANGUAGES[language]['name']} voice")
            return True

        # Stop any current operations
        self.stop_speaking()
        if self.voice_recognizer:
            self.voice_recognizer.stop()

        # Select best model for this language
        selected_model = self._select_best_model(language)
        models_to_try = [selected_model, self.SAFE_FALLBACK]

        for model_name in models_to_try:
            try:
                if self.debug_mode:
                    lang_name = self.LANGUAGES[language]['name']
                    print(f"🌍 Switching to {lang_name} voice: {model_name}")

                # Reinitialize TTS engine
                TTSEngine = _import_tts_engine()
                self.tts_engine = TTSEngine(model_name=model_name, debug_mode=self.debug_mode)

                # Restore callbacks
                self.tts_engine.on_playback_start = self._on_tts_start
                self.tts_engine.on_playback_end = self._on_tts_end

                # Update language and set appropriate speed for Italian voices
                self.language = language

                # Set language-specific speed adjustments
                if language == 'it':
                    self.speed = 0.8  # Slow down Italian voices to fix pace issues
                    if self.debug_mode:
                        print(f"   Speed: {self.speed} (adjusted for optimal Italian pace)")
                else:
                    self.speed = 1.0  # Default speed for other languages

                return True

            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Model {model_name} failed: {e}")
                continue

        # All models failed
        if self.debug_mode:
            print(f"❌ All models failed for language '{language}'")
        return False

    def get_language(self):
        """Get the current voice language.

        Returns:
            Current language code
        """
        return self.language

    def get_supported_languages(self):
        """Get list of supported language codes.

        Returns:
            List of supported language codes
        """
        return list(self.LANGUAGES.keys())

    def get_language_name(self, language_code=None):
        """Get the display name for a language.

        Args:
            language_code: Language code (defaults to current language)

        Returns:
            Language display name
        """
        lang = language_code or self.language
        return self.LANGUAGES.get(lang, {}).get('name', lang)

    def _select_best_model(self, language):
        """Select the best available TTS model for a language.

        Try premium model first (higher quality), fallback to default (reliable).

        Args:
            language: Language code

        Returns:
            Model name string
        """
        if language not in self.LANGUAGES:
            return self.SAFE_FALLBACK

        lang_config = self.LANGUAGES[language]

        # Try premium model first (better quality)
        if 'premium' in lang_config:
            try:
                premium_model = lang_config['premium']
                # Quick test to see if this model type works
                if self._test_model_compatibility(premium_model):
                    if self.debug_mode:
                        print(f"✨ Using premium quality model: {premium_model}")
                    return premium_model
                elif self.debug_mode:
                    print(f"⚠️ Premium model not compatible, using default")
            except Exception:
                if self.debug_mode:
                    print(f"⚠️ Premium model failed, using default")

        # Use reliable default model
        default_model = lang_config.get('default', self.SAFE_FALLBACK)
        if self.debug_mode:
            print(f"🔧 Using reliable default model: {default_model}")
        return default_model

    def _test_model_compatibility(self, model_name):
        """Quick test if a model is compatible with current system.

        Args:
            model_name: TTS model name

        Returns:
            True if compatible, False otherwise
        """
        # For VITS models, check if espeak-ng is available
        if 'vits' in model_name.lower():
            try:
                import subprocess
                result = subprocess.run(['espeak-ng', '--version'],
                                      capture_output=True, timeout=2)
                return result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
                return False

        # For other models, assume they work (they're more compatible)
        return True

    def set_voice_variant(self, language, variant):
        """Set a specific voice variant for a language.

        Args:
            language: Language code ('fr', 'it')
            variant: Variant name ('female', 'alternative', etc.)

        Returns:
            True if successful, False otherwise

        Examples:
            vm.set_voice_variant('it', 'female')  # Use female Italian voice
            vm.set_voice_variant('fr', 'alternative')  # Use original French model
        """
        if language not in self.ALTERNATIVE_MODELS:
            if self.debug_mode:
                available_langs = ', '.join(self.ALTERNATIVE_MODELS.keys())
                print(f"⚠️ No variants available for '{language}'. Languages with variants: {available_langs}")
            return False

        if variant not in self.ALTERNATIVE_MODELS[language]:
            if self.debug_mode:
                available_variants = ', '.join(self.ALTERNATIVE_MODELS[language].keys())
                print(f"⚠️ Variant '{variant}' not available for {language}. Available: {available_variants}")
            return False

        # Get the specific model for this variant
        model_name = self.ALTERNATIVE_MODELS[language][variant]

        if self.debug_mode:
            lang_name = self.LANGUAGES[language]['name']
            print(f"🎭 Switching to {lang_name} {variant} voice: {model_name}")

        # Set the specific model
        return self.set_tts_model(model_name)

    def get_model_info(self):
        """Get information about currently loaded models and system capabilities.

        Returns:
            Dict with model information and system capabilities
        """
        info = {
            'current_language': self.language,
            'language_name': self.get_language_name(),
            'espeak_available': self._test_model_compatibility('test_vits'),
            'supported_languages': self.get_supported_languages()
        }

        # Add model recommendations for each language
        info['models'] = {}
        for lang in self.get_supported_languages():
            selected_model = self._select_best_model(lang)
            lang_config = self.LANGUAGES[lang]
            is_premium = selected_model == lang_config.get('premium', '')

            info['models'][lang] = {
                'name': lang_config['name'],
                'selected_model': selected_model,
                'quality': 'premium' if is_premium else 'default',
                'default_available': lang_config.get('default', ''),
                'premium_available': lang_config.get('premium', '')
            }

        return info

    def browse_voices(self, language=None, quality=None, gender=None):
        """Browse available voices with filtering options.

        Args:
            language: Language code ('en', 'fr', etc.) or None for all
            quality: 'premium', 'good', or None for all
            gender: 'male', 'female', 'multiple', or None for all

        Returns:
            Dict of available voices with metadata
        """
        voices = {}

        # Get languages to check
        languages_to_check = [language] if language else self.VOICE_CATALOG.keys()

        for lang in languages_to_check:
            if lang not in self.VOICE_CATALOG:
                continue

            lang_voices = {}
            for voice_id, voice_info in self.VOICE_CATALOG[lang].items():
                # Apply filters
                if quality and voice_info['quality'] != quality:
                    continue
                if gender and voice_info['gender'] != gender:
                    continue

                # Check if voice is compatible with current system
                compatible = True
                if voice_info['requires'] == 'espeak-ng':
                    compatible = self._test_model_compatibility(voice_info['model'])

                # Add compatibility info
                voice_data = voice_info.copy()
                voice_data['compatible'] = compatible
                lang_voices[voice_id] = voice_data

            if lang_voices:
                voices[lang] = lang_voices

        return voices

    def list_voices(self, language=None):
        """List available voices in a user-friendly format.

        Args:
            language: Language code or None for all languages
        """
        voices = self.browse_voices(language)

        if not voices:
            print("No voices found matching criteria.")
            return

        # License links mapping
        license_links = {
            'CSS10': 'https://github.com/Kyubyong/CSS10',
            'M-AILABS': 'https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/',
            'LJSpeech': 'https://keithito.com/LJ-Speech-Dataset/',
            'VCTK': 'https://datashare.ed.ac.uk/handle/10283/3443',
            'Thorsten': 'https://www.thorsten-voice.de/en/'
        }

        for lang, lang_voices in voices.items():
            lang_name = self.LANGUAGES.get(lang, {}).get('name', lang)
            print(f"\n🌍 {lang_name} ({lang}) - {len(lang_voices)} voices available:")

            for voice_id, voice_info in lang_voices.items():
                quality_icon = "✨" if voice_info['quality'] == 'premium' else "🔧"
                compat_icon = "✅" if voice_info['compatible'] else "⚠️"
                gender_icon = {"male": "👨", "female": "👩", "multiple": "👥"}.get(voice_info['gender'], "🗣️")

                # Show full format: language.voice_id
                full_voice_id = f"{lang}.{voice_id}"
                print(f"  {compat_icon} {quality_icon} {gender_icon} {full_voice_id}")
                print(f"      {voice_info['accent']} - {voice_info['gender']} voice")

                # Extract license name and add link if available
                license_text = voice_info['license']
                license_with_link = license_text
                for dataset_name, link in license_links.items():
                    if dataset_name in license_text:
                        license_with_link = f"{license_text} - {link}"
                        break

                print(f"      License: {license_with_link}")
                if not voice_info['compatible'] and voice_info['requires'] == 'espeak-ng':
                    print(f"      ⚠️ Requires: espeak-ng (install for premium quality)")

    def set_voice(self, language, voice_id):
        """Set a specific voice by ID.

        Args:
            language: Language code
            voice_id: Voice ID from voice catalog

        Returns:
            True if successful

        Example:
            vm.set_voice('fr', 'css10_vits')  # Use CSS10 French VITS voice
            vm.set_voice('it', 'mai_female_vits')  # Use female Italian VITS voice
        """
        if language not in self.VOICE_CATALOG:
            if self.debug_mode:
                print(f"⚠️ Language '{language}' not available")
            return False

        if voice_id not in self.VOICE_CATALOG[language]:
            if self.debug_mode:
                available = ', '.join(self.VOICE_CATALOG[language].keys())
                print(f"⚠️ Voice '{voice_id}' not available for {language}. Available: {available}")
            return False

        voice_info = self.VOICE_CATALOG[language][voice_id]

        # Check compatibility
        if voice_info['requires'] == 'espeak-ng' and not self._test_model_compatibility(voice_info['model']):
            if self.debug_mode:
                print(f"⚠️ Voice '{voice_id}' requires espeak-ng. Install it for premium quality.")
            return False

        # Set the specific voice
        model_name = voice_info['model']
        if self.debug_mode:
            print(f"🎭 Setting {language} voice to: {voice_id}")
            print(f"   Model: {model_name}")
            print(f"   Quality: {voice_info['quality']} | Gender: {voice_info['gender']}")
            print(f"   Accent: {voice_info['accent']}")

        # Switch to the language and specific model
        self.language = language

        # Set voice-specific speed if available
        if 'speed' in voice_info:
            self.speed = voice_info['speed']
            if self.debug_mode:
                print(f"   Speed: {voice_info['speed']} (adjusted for optimal pace)")
        else:
            self.speed = 1.0  # Default speed

        return self.set_tts_model(model_name)
    
    def change_vad_aggressiveness(self, aggressiveness):
        """Change VAD aggressiveness.
        
        Args:
            aggressiveness: New aggressiveness level (0-3)
            
        Returns:
            True if changed, False otherwise
        """
        if self.voice_recognizer:
            return self.voice_recognizer.change_vad_aggressiveness(aggressiveness)
        return False
    
    def cleanup(self):
        """Clean up resources.
        
        Returns:
            True if cleanup successful
        """
        if self.voice_recognizer:
            self.voice_recognizer.stop()
        
        self.stop_speaking()
        return True 