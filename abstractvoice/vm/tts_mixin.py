"""TTS + voice/language methods for VoiceManager.

This module intentionally focuses on orchestration and keeps heavy engine details
behind adapters.
"""

from __future__ import annotations

import threading
import time

from ..text_sanitize import sanitize_markdown_for_speech
from ..adapters.tts_registry import create_tts_adapter

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

    def get_cloned_voice_store_dir(self) -> str:
        """Return the on-disk folder where cloned voices are stored."""
        try:
            return str(self._get_voice_cloner().get_store_base_dir())
        except Exception:
            return ""

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
        """Set cloned TTS quality preset: low|standard|high (aliases: fast, balanced)."""
        from ..quality_preset import normalize_quality_preset

        p = normalize_quality_preset(str(preset))
        self._get_voice_cloner().set_quality_preset(p)
        return True

    def get_cloned_tts_quality_preset(self) -> str | None:
        """Return the current cloned TTS quality preset (best-effort)."""
        try:
            cloner = self._get_voice_cloner()
        except Exception:
            return None
        try:
            if hasattr(cloner, "get_quality_preset"):
                return str(cloner.get_quality_preset() or "").strip() or None
        except Exception:
            return None
        return None

    def set_tts_quality_preset(self, preset: str) -> bool:
        """Set base TTS engine quality preset: low|standard|high (best-effort).

        This is an engine-agnostic knob. Engines that don't support quality tuning
        may ignore it and return False.
        """
        if not getattr(self, "tts_adapter", None):
            return False
        from ..quality_preset import normalize_quality_preset

        try:
            p = normalize_quality_preset(str(preset))
        except Exception:
            return False
        try:
            adapter = getattr(self, "tts_adapter", None)
            if adapter is None:
                return False
            if hasattr(adapter, "set_quality_preset"):
                return bool(adapter.set_quality_preset(p))
        except Exception:
            return False
        return False

    def get_tts_quality_preset(self) -> str | None:
        """Return the current base TTS quality preset, if supported."""
        adapter = getattr(self, "tts_adapter", None)
        if adapter is None:
            return None
        try:
            if hasattr(adapter, "get_quality_preset"):
                return adapter.get_quality_preset()
        except Exception:
            return None
        return None

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
                        wav_bytes = cloner.speak_to_bytes(
                            str(speak_text),
                            voice_id=voice,
                            format="wav",
                            speed=sp,
                            language=str(getattr(self, "language", None) or "en"),
                        )
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
                        language=str(getattr(self, "language", None) or "en"),
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

        - If `voice` is None: use the active TTS engine/adapter (default: Piper).
        - If `voice` is provided: treat as a cloned voice_id (requires `abstractvoice[cloning]`).
        """
        # Clear prior metrics for this new utterance.
        self._set_last_tts_metrics(None)

        speak_text = str(text)
        if _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax):
            speak_text = sanitize_markdown_for_speech(speak_text)

        fmt = str(format or "wav").strip().lower() or "wav"

        def _analyze_audio_bytes(b: bytes) -> dict:
            metrics: dict = {}
            try:
                import io

                import soundfile as sf

                info = sf.info(io.BytesIO(bytes(b)))
                try:
                    metrics["sample_rate"] = int(getattr(info, "samplerate", 0) or 0) or None
                except Exception:
                    metrics["sample_rate"] = None
                try:
                    metrics["channels"] = int(getattr(info, "channels", 0) or 0) or None
                except Exception:
                    metrics["channels"] = None
                try:
                    frames = int(getattr(info, "frames", 0) or 0)
                    metrics["audio_frames"] = frames if frames > 0 else None
                except Exception:
                    metrics["audio_frames"] = None
                try:
                    d = float(getattr(info, "duration", 0.0) or 0.0)
                    metrics["audio_s"] = float(d) if d > 0 else None
                except Exception:
                    metrics["audio_s"] = None
            except Exception:
                pass
            return metrics

        t0 = time.monotonic()
        if voice:
            cloner = self._get_voice_cloner()
            out = cloner.speak_to_bytes(
                speak_text,
                voice_id=voice,
                format=fmt,
                speed=self.speed,
                language=str(getattr(self, "language", None) or "en"),
            )
            synth_s = float(time.monotonic() - t0)

            clone_engine_name = None
            try:
                info = cloner.get_cloned_voice(str(voice)) or {}
                clone_engine_name = str(info.get("engine") or "").strip().lower() or None
            except Exception:
                clone_engine_name = None

            metrics = {
                "engine": "clone",
                "clone_engine": clone_engine_name,
                "voice_id": str(voice),
                "streaming": False,
                "synth_s": synth_s,
                "format": fmt,
                "speed": float(getattr(self, "speed", 1.0) or 1.0),
                "language": str(getattr(self, "language", None) or "en"),
                "ts": time.time(),
            }
            metrics.update(_analyze_audio_bytes(bytes(out)))
            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass
            self._set_last_tts_metrics(metrics)
            return out

        if self.tts_adapter and self.tts_adapter.is_available():
            out = self.tts_adapter.synthesize_to_bytes(speak_text, format=fmt)
            synth_s = float(time.monotonic() - t0)
            try:
                engine_id = str(getattr(self.tts_adapter, "engine_id", "") or "").strip().lower()
            except Exception:
                engine_id = ""
            metrics = {
                "engine": engine_id or "tts",
                "synth_s": synth_s,
                "format": fmt,
                "language": str(getattr(self, "language", None) or "en"),
                "ts": time.time(),
            }
            # Best-effort: attach active profile info when supported by the adapter.
            try:
                p = getattr(self.tts_adapter, "get_active_profile", None)
                prof = p() if callable(p) else None
                if prof is not None:
                    metrics["profile_id"] = getattr(prof, "profile_id", None)
                    metrics["profile_label"] = getattr(prof, "label", None)
            except Exception:
                pass
            metrics.update(_analyze_audio_bytes(bytes(out)))
            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass
            self._set_last_tts_metrics(metrics)
            return out
        raise NotImplementedError("speak_to_bytes() requires a functional TTS adapter.")

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
        # Clear prior metrics for this new utterance.
        self._set_last_tts_metrics(None)

        sanitize = _resolve_sanitize_syntax_arg(sanitize_syntax, saninitze_syntax)
        speak_text = str(text)
        if sanitize:
            speak_text = sanitize_markdown_for_speech(speak_text)
        if voice:
            from pathlib import Path

            # For cloned voices, we only have a bytes API; write it out here.
            fmt = str(format or Path(output_path).suffix.lstrip(".") or "wav").strip().lower() or "wav"
            data = self.speak_to_bytes(speak_text, format=fmt, voice=voice, sanitize_syntax=False)
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(bytes(data))
            return str(out)

        if self.tts_adapter and self.tts_adapter.is_available():
            t0 = time.monotonic()
            out_path = self.tts_adapter.synthesize_to_file(speak_text, output_path, format=format)
            synth_s = float(time.monotonic() - t0)

            try:
                engine_id = str(getattr(self.tts_adapter, "engine_id", "") or "").strip().lower()
            except Exception:
                engine_id = ""

            fmt_used = None
            try:
                from pathlib import Path

                fmt_used = str(format or Path(str(out_path)).suffix.lstrip(".") or "wav").strip().lower() or "wav"
            except Exception:
                fmt_used = str(format or "wav").strip().lower() or "wav"

            metrics: dict = {
                "engine": engine_id or "tts",
                "synth_s": synth_s,
                "format": fmt_used,
                "language": str(getattr(self, "language", None) or "en"),
                "ts": time.time(),
            }
            # Best-effort: attach active profile info when supported by the adapter.
            try:
                p = getattr(self.tts_adapter, "get_active_profile", None)
                prof = p() if callable(p) else None
                if prof is not None:
                    metrics["profile_id"] = getattr(prof, "profile_id", None)
                    metrics["profile_label"] = getattr(prof, "label", None)
            except Exception:
                pass

            try:
                import soundfile as sf

                info = sf.info(str(out_path))
                try:
                    metrics["sample_rate"] = int(getattr(info, "samplerate", 0) or 0) or None
                except Exception:
                    metrics["sample_rate"] = None
                try:
                    metrics["channels"] = int(getattr(info, "channels", 0) or 0) or None
                except Exception:
                    metrics["channels"] = None
                try:
                    frames = int(getattr(info, "frames", 0) or 0)
                    metrics["audio_frames"] = frames if frames > 0 else None
                except Exception:
                    metrics["audio_frames"] = None
                try:
                    d = float(getattr(info, "duration", 0.0) or 0.0)
                    metrics["audio_s"] = float(d) if d > 0 else None
                except Exception:
                    metrics["audio_s"] = None
            except Exception:
                pass

            try:
                audio_s = metrics.get("audio_s")
                if isinstance(audio_s, (int, float)) and float(audio_s) > 0:
                    metrics["rtf"] = float(synth_s) / float(audio_s)
            except Exception:
                pass

            self._set_last_tts_metrics(metrics)
            return str(out_path)

        raise NotImplementedError("speak_to_file() requires a functional TTS adapter.")

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
        try:
            sp = float(speed)
        except Exception:
            return False
        if not (0.5 <= sp <= 2.0):
            return False

        # AudioDiT speed control: not supported (avoid degraded/glitchy audio).
        try:
            a = getattr(self, "tts_adapter", None)
            engine_id = str(getattr(a, "engine_id", "") or "").strip().lower()
        except Exception:
            engine_id = ""
        if engine_id == "audiodit" and sp != 1.0:
            # Keep manager speed unchanged (or reset to 1.0 if unset).
            try:
                self.speed = float(getattr(self, "speed", 1.0) or 1.0)
            except Exception:
                self.speed = 1.0
            return False

        self.speed = float(sp)
        return True

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
        language = str(language or "").strip().lower()
        if not language:
            return False

        # Language validation is engine-dependent:
        # - Piper (and auto->Piper) uses a small curated mapping to avoid trying to
        #   load non-existent voices.
        # - Other engines (e.g. OmniVoice) can support many languages; treat the
        #   language code as a pass-through hint and let the engine decide.
        pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto").strip().lower()
        active_engine = ""
        try:
            a = getattr(self, "tts_adapter", None)
            active_engine = str(getattr(a, "engine_id", "") or "").strip().lower()
        except Exception:
            active_engine = ""

        validate_against_catalog = bool(pref in ("", "auto", "piper") or active_engine in ("", "piper"))
        if validate_against_catalog and language not in self.LANGUAGES:
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

        # Switch language on the active TTS adapter (engine-agnostic).
        try:
            if self.tts_adapter is None:
                pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto")
                self.tts_adapter, resolved_engine = create_tts_adapter(
                    engine=pref,
                    language=language,
                    allow_downloads=bool(getattr(self, "allow_downloads", True)),
                    auto_load=False,
                    debug_mode=bool(getattr(self, "debug_mode", False)),
                )
                if self.tts_adapter is None:
                    return False
                # Track which engine is active (used by CLI/tests/metrics).
                self._tts_engine_name = str(resolved_engine)
                if self.tts_engine is None:
                    from ..tts.adapter_tts_engine import AdapterTTSEngine

                    self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                    self._wire_tts_callbacks()

            ok = bool(self.tts_adapter.set_language(language))
            if not ok:
                return False

            # Ensure playback wrapper exists (used for lifecycle callbacks + audio output).
            if self.tts_engine is None and self.tts_adapter is not None:
                from ..tts.adapter_tts_engine import AdapterTTSEngine

                self.tts_engine = AdapterTTSEngine(self.tts_adapter, debug_mode=self.debug_mode)
                if not getattr(self, "_tts_engine_name", None):
                    pref = str(getattr(self, "_tts_engine_preference", "auto") or "auto").strip().lower()
                    self._tts_engine_name = "piper" if pref in ("", "auto") else pref
                self._wire_tts_callbacks()

            self.language = language
            self.speed = 1.0
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ TTS language switch failed: {e}")

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
