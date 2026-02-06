"""TTS + voice/language methods for VoiceManager.

This module intentionally focuses on orchestration and keeps heavy engine details
behind adapters.
"""

from __future__ import annotations

import threading
import time

from ..text_sanitize import sanitize_markdown_for_speech

def _resolve_sanitize_syntax_arg(
    sanitize_syntax: bool,
    saninitze_syntax: bool | None,
) -> bool:
    """Resolve sanitize_syntax value, supporting a common misspelling alias.

    `saninitze_syntax` is accepted as an alias for backward-compat / typo tolerance.
    """
    resolved = bool(sanitize_syntax)
    if saninitze_syntax is not None:
        # If caller provided both, treat `saninitze_syntax` as an alias override,
        # but reject the one ambiguous/confusing case we can detect: opt-out
        # via the canonical flag + opt-in via the alias.
        if bool(sanitize_syntax) is False and bool(saninitze_syntax) is True:
            raise ValueError("Pass only one of sanitize_syntax or saninitze_syntax (alias).")
        resolved = bool(saninitze_syntax)

    return resolved


class TtsMixin:
    def _set_last_tts_metrics(self, metrics: dict | None) -> None:
        lock = getattr(self, "_last_tts_metrics_lock", None)
        if lock is None:
            setattr(self, "_last_tts_metrics", metrics)
            return
        try:
            with lock:
                setattr(self, "_last_tts_metrics", metrics)
        except Exception:
            setattr(self, "_last_tts_metrics", metrics)

    def pop_last_tts_metrics(self) -> dict | None:
        lock = getattr(self, "_last_tts_metrics_lock", None)
        if lock is None:
            m = getattr(self, "_last_tts_metrics", None)
            setattr(self, "_last_tts_metrics", None)
            return m
        try:
            with lock:
                m = getattr(self, "_last_tts_metrics", None)
                setattr(self, "_last_tts_metrics", None)
                return m
        except Exception:
            m = getattr(self, "_last_tts_metrics", None)
            setattr(self, "_last_tts_metrics", None)
            return m

    def _get_voice_cloner(self):
        if getattr(self, "_voice_cloner", None) is None:
            try:
                from ..cloning import VoiceCloner
            except Exception as e:
                raise RuntimeError(
                    "Voice cloning is an optional feature.\n"
                    "Install with: pip install \"abstractvoice[cloning]\"\n"
                    f"Original error: {e}"
                ) from e

            # Use a slightly larger STT model for one-time reference-text auto-fallback.
            self._voice_cloner = VoiceCloner(
                debug=bool(getattr(self, "debug_mode", False)),
                whisper_model=getattr(self, "whisper_model", "tiny"),
                reference_text_whisper_model="small",
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
                default_engine=str(getattr(self, "cloning_engine", "f5_tts") or "f5_tts"),
            )
        return self._voice_cloner

    def clone_voice(
        self,
        reference_audio_path: str,
        name: str | None = None,
        *,
        reference_text: str | None = None,
        engine: str | None = None,
    ) -> str:
        return self._get_voice_cloner().clone_voice(
            reference_audio_path,
            name=name,
            reference_text=reference_text,
            engine=engine,
        )

    def list_cloned_voices(self):
        return self._get_voice_cloner().list_cloned_voices()

    def get_cloned_voice(self, voice_id: str):
        return self._get_voice_cloner().get_cloned_voice(voice_id)

    def set_cloned_voice_reference_text(self, voice_id: str, reference_text: str) -> bool:
        """Update a cloned voice's reference transcript (quality fix).

        A bad reference transcript commonly causes repeated/incorrect words in output.
        """
        self._get_voice_cloner().set_reference_text(voice_id, reference_text)
        return True

    def export_voice(self, voice_id: str, path: str) -> str:
        return self._get_voice_cloner().export_voice(voice_id, path)

    def import_voice(self, path: str) -> str:
        return self._get_voice_cloner().import_voice(path)

    def set_cloned_tts_quality(self, preset: str) -> bool:
        """Set cloned TTS quality preset: fast|balanced|high."""
        self._get_voice_cloner().set_quality_preset(preset)
        return True

    def get_cloning_runtime_info(self):
        return self._get_voice_cloner().get_runtime_info()

    def rename_cloned_voice(self, voice_id: str, new_name: str) -> bool:
        self._get_voice_cloner().rename_cloned_voice(voice_id, new_name)
        return True

    def delete_cloned_voice(self, voice_id: str) -> bool:
        self._get_voice_cloner().delete_cloned_voice(voice_id)
        return True

    def unload_cloning_engines(self, *, keep_engine: str | None = None) -> int:
        """Best-effort free memory held by loaded cloning engines.

        This is critical for large backends (e.g. Chroma). It does NOT delete any
        cloned voices; it only releases in-memory model weights.
        """
        try:
            cloner = self._get_voice_cloner()
        except Exception:
            return 0
        try:
            if keep_engine:
                return int(cloner.unload_engines_except(str(keep_engine)))
            return int(cloner.unload_all_engines())
        except Exception:
            return 0

    def unload_piper_voice(self) -> bool:
        """Best-effort release of Piper voice weights/session (keeps audio output ready).

        This helps reduce memory pressure when switching to large cloning backends.
        """
        try:
            adapter = getattr(self, "tts_adapter", None)
            if adapter is None:
                return False
            if hasattr(adapter, "unload"):
                adapter.unload()
                return True
            # Back-compat: drop voice object if present.
            if hasattr(adapter, "_voice"):
                setattr(adapter, "_voice", None)
                return True
        except Exception:
            return False
        return False
    def speak(
        self,
        text,
        speed=1.0,
        callback=None,
        voice: str | None = None,
        *,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ):
        sp = speed if speed != 1.0 else self.speed
        if not self.tts_engine:
            raise RuntimeError("No TTS engine available")

        speak_text = str(text)
        if _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax):
            speak_text = sanitize_markdown_for_speech(speak_text)

        # Optional cloned voice playback:
        # - stream chunks to the player for better perceived latency
        # - support cancellation on stop_speaking() / new input (best-effort)
        if voice:
            import numpy as np

            from ..audio.resample import linear_resample_mono

            # Clear prior metrics for this new utterance.
            self._set_last_tts_metrics(None)

            # Stop any current speech and reset cancel token.
            try:
                self.stop_speaking()
            except Exception:
                pass

            # IMPORTANT: cancellation must be per-utterance.
            # If we reuse/clear the same Event, an old synthesis thread could resume
            # after a new request starts (race), causing "old audio" to continue.
            try:
                old = getattr(self, "_cloned_cancel_event", None)
                if old is not None:
                    old.set()
            except Exception:
                pass
            cancel = threading.Event()
            setattr(self, "_cloned_cancel_event", cancel)

            cloner = self._get_voice_cloner()
            # Prefer playing cloned audio at its native rate (F5 is typically 24kHz).
            target_sr = 24000
            clone_engine_name = ""
            try:
                info = cloner.get_cloned_voice(str(voice)) or {}
                clone_engine_name = str(info.get("engine") or "").strip().lower()
            except Exception:
                clone_engine_name = ""

            def _worker():
                try:
                    synth_active = getattr(self, "_cloned_synthesis_active", None)
                    if synth_active is not None:
                        try:
                            synth_active.set()
                        except Exception:
                            pass

                    # Option: generate full audio first (smooth playback) vs streaming (faster TTFB).
                    if not bool(getattr(self, "cloned_tts_streaming", True)):
                        import io
                        import soundfile as sf

                        t0 = time.monotonic()
                        wav_bytes = cloner.speak_to_bytes(str(speak_text), voice_id=voice, format="wav", speed=sp)
                        t1 = time.monotonic()
                        if cancel.is_set():
                            return
                        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                        mono = np.mean(audio, axis=1).astype(np.float32).reshape(-1)
                        sr = int(sr)

                        try:
                            audio_samples = int(len(mono))
                        except Exception:
                            audio_samples = 0
                        audio_s = (float(audio_samples) / float(sr)) if sr and audio_samples else 0.0
                        synth_s = float(t1 - t0)
                        self._set_last_tts_metrics(
                            {
                                "engine": "clone",
                                "clone_engine": clone_engine_name or None,
                                "voice_id": str(voice),
                                "streaming": False,
                                "synth_s": synth_s,
                                "audio_s": float(audio_s),
                                "rtf": (synth_s / float(audio_s)) if audio_s else None,
                                "sample_rate": int(sr) if sr else None,
                                "audio_samples": int(audio_samples),
                                "ts": time.time(),
                            }
                        )

                        if hasattr(self.tts_engine, "begin_playback"):
                            self.tts_engine.begin_playback(callback=callback, sample_rate=sr)
                        if cancel.is_set():
                            return
                        if hasattr(self.tts_engine, "enqueue_audio"):
                            try:
                                self.tts_engine.enqueue_audio(mono, sample_rate=sr)
                            except TypeError:
                                self.tts_engine.enqueue_audio(mono)
                        elif hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                            try:
                                self.tts_engine.audio_player.play_audio(mono, sample_rate=sr)
                            except TypeError:
                                self.tts_engine.audio_player.play_audio(mono)
                        return

                    # Streaming path: fewer, larger batches reduce audible cuts and overhead.
                    t0 = time.monotonic()
                    first_chunk_t = None
                    total_samples = 0
                    chunks = 0
                    chunks_iter = cloner.speak_to_audio_chunks(
                        str(speak_text),
                        voice_id=voice,
                        speed=sp,
                        max_chars=240,
                    )

                    # Begin a playback session once (so TTS lifecycle hooks are correct).
                    if hasattr(self.tts_engine, "begin_playback"):
                        self.tts_engine.begin_playback(callback=callback, sample_rate=target_sr)

                    for chunk, sr in chunks_iter:
                        if cancel.is_set():
                            break
                        if first_chunk_t is None:
                            first_chunk_t = time.monotonic()
                        mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
                        if int(sr) != target_sr:
                            mono = linear_resample_mono(mono, int(sr), target_sr)
                        try:
                            total_samples += int(len(mono))
                            chunks += 1
                        except Exception:
                            pass

                        if hasattr(self.tts_engine, "enqueue_audio"):
                            try:
                                self.tts_engine.enqueue_audio(mono, sample_rate=target_sr)
                            except TypeError:
                                self.tts_engine.enqueue_audio(mono)
                        elif hasattr(self.tts_engine, "audio_player") and self.tts_engine.audio_player:
                            try:
                                self.tts_engine.audio_player.play_audio(mono, sample_rate=target_sr)
                            except TypeError:
                                self.tts_engine.audio_player.play_audio(mono)
                        else:
                            break

                    t1 = time.monotonic()
                    audio_s = (float(total_samples) / float(target_sr)) if total_samples else 0.0
                    synth_s = float(t1 - t0)
                    ttfb_s = (float(first_chunk_t - t0) if first_chunk_t is not None else None)
                    self._set_last_tts_metrics(
                        {
                            "engine": "clone",
                            "clone_engine": clone_engine_name or None,
                            "voice_id": str(voice),
                            "streaming": True,
                            "cancelled": bool(cancel.is_set()),
                            "synth_s": synth_s,
                            "ttfb_s": ttfb_s,
                            "audio_s": float(audio_s),
                            "rtf": (synth_s / float(audio_s)) if audio_s else None,
                            "sample_rate": int(target_sr),
                            "audio_samples": int(total_samples),
                            "chunks": int(chunks),
                            "ts": time.time(),
                        }
                    )
                except Exception as e:
                    # Best-effort: never crash caller thread.
                    try:
                        self._set_last_tts_metrics(
                            {
                                "engine": "clone",
                                "clone_engine": clone_engine_name or None,
                                "voice_id": str(voice),
                                "error": str(e),
                                "ts": time.time(),
                            }
                        )
                    except Exception:
                        pass
                    if bool(getattr(self, "debug_mode", False)):
                        print(f"⚠️  Cloned TTS failed: {e}")
                finally:
                    try:
                        synth_active = getattr(self, "_cloned_synthesis_active", None)
                        if synth_active is not None:
                            synth_active.clear()
                    except Exception:
                        pass

            threading.Thread(target=_worker, daemon=True).start()
            return True

        ok = self.tts_engine.speak(speak_text, sp, callback)
        # Mirror adapter metrics into the manager for a single "last TTS metrics"
        # source of truth (used by the verbose REPL).
        try:
            m = getattr(self.tts_engine, "last_tts_metrics", None)
            if isinstance(m, dict) and m:
                self._set_last_tts_metrics(dict(m))
        except Exception:
            pass
        return ok

    # Network/headless-friendly methods
    def speak_to_bytes(
        self,
        text: str,
        format: str = "wav",
        voice: str | None = None,
        *,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ) -> bytes:
        """Synthesize to bytes.

        - If `voice` is None: use Piper (default).
        - If `voice` is provided: treat as a cloned voice_id (requires `abstractvoice[cloning]`).
        """
        speak_text = str(text)
        if _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax):
            speak_text = sanitize_markdown_for_speech(speak_text)
        if voice:
            cloner = self._get_voice_cloner()
            return cloner.speak_to_bytes(speak_text, voice_id=voice, format=format, speed=self.speed)

        if self.tts_adapter and self.tts_adapter.is_available():
            return self.tts_adapter.synthesize_to_bytes(speak_text, format=format)
        raise NotImplementedError("speak_to_bytes() requires Piper TTS (default engine).")

    def speak_to_file(
        self,
        text: str,
        output_path: str,
        format: str | None = None,
        voice: str | None = None,
        *,
        sanitize_syntax: bool = True,
        saninitze_syntax: bool | None = None,
    ) -> str:
        sanitize = _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax)
        speak_text = str(text)
        if sanitize:
            speak_text = sanitize_markdown_for_speech(speak_text)
        if voice:
            data = self.speak_to_bytes(speak_text, format=(format or "wav"), voice=voice, sanitize_syntax=False)
            from pathlib import Path

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            return str(out)

        if self.tts_adapter and self.tts_adapter.is_available():
            return self.tts_adapter.synthesize_to_file(speak_text, output_path, format=format)
        raise NotImplementedError("speak_to_file() requires Piper TTS (default engine).")

    def stop_speaking(self):
        if not self.tts_engine:
            return False
        # Best-effort cancel ongoing cloned synthesis.
        try:
            cancel = getattr(self, "_cloned_cancel_event", None)
            if cancel is not None:
                cancel.set()
        except Exception:
            pass
        ok = False
        try:
            # Keep the output stream open when possible; repeatedly reopening
            # PortAudio streams can be flaky on some macOS AUHAL setups.
            try:
                ok = bool(self.tts_engine.stop(close_stream=False))
            except TypeError:
                ok = bool(self.tts_engine.stop())
        finally:
            # CRITICAL: stopping playback abruptly may not trigger the normal
            # playback-end callbacks (PortAudio stream is just closed).
            # If we don't restore recognizer state here, transcriptions can stay
            # paused or listening can remain paused, which breaks STOP/PTT.
            try:
                on_end = getattr(self, "_on_tts_end", None)
                if callable(on_end):
                    on_end()
            except Exception:
                pass
        return ok

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
            adapter = PiperTTSAdapter(
                language=language,
                allow_downloads=bool(getattr(self, "allow_downloads", True)),
                auto_load=True,
            )
            # Return the adapter even if a voice is not yet loaded. This keeps audio
            # playback available for cloning backends while remaining offline-first.
            return adapter if bool(getattr(adapter, "_piper_available", False)) else None
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Piper TTS not available: {e}")
            return None

    def get_supported_languages(self):
        return list(self.LANGUAGES.keys())

    def list_available_models(self, language: str | None = None) -> dict:
        """List available TTS voices/models (Piper-only core).

        Returns a dict shaped for CLI display:
        { "en": { "amy": { ... } }, "fr": { ... } }
        """
        if self.tts_adapter and hasattr(self.tts_adapter, "list_available_models"):
            return self.tts_adapter.list_available_models(language=language)

        # Best-effort: instantiate a temporary Piper adapter to enumerate models.
        try:
            from ..adapters.tts_piper import PiperTTSAdapter

            return PiperTTSAdapter(
                language=(language or "en"),
                allow_downloads=False,
                auto_load=False,
            ).list_available_models(language=language)
        except Exception:
            return {}

    # Backward-compatible alias used by some CLI code.
    def list_voices(self, language: str | None = None) -> dict:
        return self.list_available_models(language=language)

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

        # Piper-only core: switch Piper model for the requested language.
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
                print(f"⚠️ Piper language switch failed: {e}")

        return False

    def set_voice(self, language, voice_id):
        language = language.lower()

        # Piper voice selection is adapter-specific. For now, treat `voice_id` as
        # best-effort metadata and ensure language switching is robust.
        try:
            if not self.set_language(language):
                return False
            if self.debug_mode:
                print(f"🎭 Piper voice selection (best-effort): {language}.{voice_id}")
            return True
        except Exception:
            return False
