"""STT + listening methods for VoiceManager."""

from __future__ import annotations

from typing import Optional

from .common import import_voice_recognizer


class SttMixin:
    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        stt = self._get_stt_adapter()
        if stt is not None and hasattr(stt, "transcribe_from_bytes"):
            return stt.transcribe_from_bytes(bytes(audio_bytes), language=language)

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        try:
            return self.transcribe_file(tmp_path, language=language)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> str:
        stt = self._get_stt_adapter()
        if stt is not None:
            return stt.transcribe(audio_path, language=language)

        raise RuntimeError(
            "No STT engine is available.\n"
            "Default STT uses OpenAI remote audio; set OPENAI_API_KEY or pass remote_api_key=....\n"
            "For local faster-whisper, install and select the STT engine:\n"
            "  pip install \"abstractvoice[stt]\"\n"
            "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
            "  pip install \"abstractvoice[gpu]\"    # GPU profile\n"
            "  VoiceManager(stt_engine=\"faster_whisper\", ...)"
        )

    def _get_stt_adapter(self):
        existing = getattr(self, "stt_adapter", None)
        if existing is not None:
            return existing if existing.is_available() else None

        pref = str(getattr(self, "_stt_engine_preference", "openai") or "openai").strip().lower().replace("-", "_")
        if pref == "auto":
            pref = "openai"
        if pref in ("openai", "openai_compatible", "remote", "compatible"):
            try:
                from ..adapters.stt_openai_compatible import OpenAICompatibleSTTAdapter

                provider = "openai" if pref == "openai" else "openai-compatible"
                self.stt_adapter = OpenAICompatibleSTTAdapter(
                    provider=provider,
                    language=getattr(self, "language", None),
                    base_url=getattr(self, "remote_base_url", None),
                    api_key=getattr(self, "remote_api_key", None),
                    model_id=getattr(self, "stt_model", None),
                    timeout_s=getattr(self, "remote_timeout_s", None),
                    debug_mode=bool(getattr(self, "debug_mode", False)),
                )
                return self.stt_adapter if self.stt_adapter.is_available() else None
            except Exception as e:
                self.stt_adapter = None
                if self.debug_mode:
                    print(f"⚠️  Remote STT not available: {e}")
                raise

        if pref in ("transformers_asr", "transformers", "hf_asr", "hf"):
            try:
                from ..adapters.stt_transformers_asr import TransformersASRAdapter

                model_id = getattr(self, "stt_model", None)
                if not isinstance(model_id, str) or not model_id.strip():
                    raise ValueError(
                        "Local STT provider 'transformers-asr' requires an explicit Hugging Face model id.\n"
                        "Examples:\n"
                        "  VoiceManager(stt_engine=\"transformers-asr\", stt_model=\"openai/whisper-large-v3\", ...)\n"
                        "  VoiceManager(stt_engine=\"transformers-asr\", stt_model=\"openai/whisper-large-v3-turbo\", ...)\n"
                        "  VoiceManager(stt_engine=\"transformers-asr\", stt_model=\"Qwen/Qwen3-ASR-1.7B\", ...)"
                    )

                self.stt_adapter = TransformersASRAdapter(
                    model_id=str(model_id).strip(),
                    device="auto",
                    dtype=None,
                    allow_downloads=bool(getattr(self, "allow_downloads", True)),
                )
                return self.stt_adapter if self.stt_adapter.is_available() else None
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️  Transformers ASR STT not available: {e}")
                self.stt_adapter = None
                if pref in ("transformers_asr", "transformers", "hf_asr", "hf"):
                    raise RuntimeError(
                        "Local STT provider 'transformers-asr' requires optional dependencies.\n"
                        "Install with:\n"
                        "  pip install \"abstractvoice[stt-hf]\"\n"
                        "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                        "  pip install \"abstractvoice[gpu]\"    # GPU profile"
                    ) from e
                return None

        if pref not in ("auto", "faster_whisper", "faster-whisper"):
            return None

        try:
            from ..compute import best_faster_whisper_device
            from ..adapters.stt_faster_whisper import FasterWhisperAdapter

            device = str(best_faster_whisper_device() or "cpu").strip().lower() or "cpu"
            # Reasonable default mapping:
            # - CPU: INT8 (fast, low memory)
            # - CUDA: INT8 weights + FP16 compute (good speed/memory balance)
            compute_type = "int8_float16" if device == "cuda" else "int8"

            self.stt_adapter = FasterWhisperAdapter(
                model_size=self.whisper_model,
                device=device,
                compute_type=compute_type,
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
            )
            if self.stt_adapter.is_available():
                return self.stt_adapter
            return None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Faster-Whisper STT not available: {e}")
            self.stt_adapter = None
            if pref in ("faster_whisper", "faster-whisper"):
                raise RuntimeError(
                    "Local STT engine 'faster-whisper' requires optional dependencies.\n"
                    "Install with:\n"
                    "  pip install \"abstractvoice[stt]\"\n"
                    "  pip install \"abstractvoice[apple]\"  # Apple profile\n"
                    "  pip install \"abstractvoice[gpu]\"    # GPU profile"
                ) from e
            return None

    def set_whisper(self, model_name):
        self.whisper_model = model_name
        if self.voice_recognizer:
            return self.voice_recognizer.change_whisper_model(model_name)

    def get_whisper(self):
        return self.whisper_model

    def listen(self, on_transcription, on_stop=None, on_audio_level=None):
        self._transcription_callback = on_transcription
        self._stop_callback = on_stop

        if not self.voice_recognizer:
            def _transcription_handler(text):
                if self._transcription_callback:
                    self._transcription_callback(text)

            def _stop_handler():
                # Stop phrase semantics (ADR 0002 Phase 1):
                # - Always stop TTS playback immediately.
                # - Do NOT forcibly stop listening unless the integrator wants that
                #   (they can call stop_listening() inside on_stop).
                self.stop_speaking()
                if self._stop_callback:
                    self._stop_callback()

            stt_adapter = None
            pref = str(getattr(self, "_stt_engine_preference", "openai") or "openai").strip().lower().replace("-", "_")
            if pref in (
                "auto",
                "openai",
                "openai_compatible",
                "remote",
                "compatible",
                "transformers_asr",
                "transformers",
                "hf_asr",
                "hf",
            ):
                stt_adapter = self._get_stt_adapter()

            VoiceRecognizer = import_voice_recognizer()
            self.voice_recognizer = VoiceRecognizer(
                transcription_callback=_transcription_handler,
                stop_callback=_stop_handler,
                whisper_model=self.whisper_model,
                debug_mode=self.debug_mode,
                aec_enabled=bool(getattr(self, "_aec_enabled", False)),
                aec_stream_delay_ms=int(getattr(self, "_aec_stream_delay_ms", 0)),
                language=getattr(self, "language", None),
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
                stt_adapter=stt_adapter,
                audio_level_callback=on_audio_level,
            )
            try:
                if hasattr(self.voice_recognizer, "set_profile"):
                    self.voice_recognizer.set_profile(getattr(self, "_voice_mode", "stop"))
            except Exception:
                pass

        return self.voice_recognizer.start(tts_interrupt_callback=self.stop_speaking)

    def enable_aec(self, enabled: bool = True, *, stream_delay_ms: int = 0) -> bool:
        """Enable optional AEC-based barge-in support.

        Notes:
        - This is opt-in and requires: pip install "abstractvoice[aec]"
        - Intended for `voice_mode="full"` where we want true barge-in.
        """
        self._aec_enabled = bool(enabled)
        self._aec_stream_delay_ms = int(stream_delay_ms)
        if self.voice_recognizer and hasattr(self.voice_recognizer, "enable_aec"):
            return bool(self.voice_recognizer.enable_aec(bool(enabled), stream_delay_ms=int(stream_delay_ms)))
        return True

    def stop_listening(self):
        if self.voice_recognizer:
            return self.voice_recognizer.stop()
        return False

    def pause_listening(self) -> bool:
        if self.voice_recognizer:
            self.voice_recognizer.pause_listening()
            return True
        return False

    def resume_listening(self) -> bool:
        if self.voice_recognizer:
            self.voice_recognizer.resume_listening()
            return True
        return False

    def is_listening(self):
        return self.voice_recognizer and self.voice_recognizer.is_running

    def set_voice_mode(self, mode):
        if mode in ["full", "wait", "stop", "ptt"]:
            self._voice_mode = mode
            # Keep recognizer thresholds aligned with interaction mode.
            try:
                if self.voice_recognizer and hasattr(self.voice_recognizer, "set_profile"):
                    self.voice_recognizer.set_profile(mode)
            except Exception:
                pass
            return True
        return False

    def change_vad_aggressiveness(self, aggressiveness):
        if self.voice_recognizer:
            return self.voice_recognizer.change_vad_aggressiveness(aggressiveness)
        return False
