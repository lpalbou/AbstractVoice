"""Qwen3-TTS voice cloning engine (12Hz Base checkpoints).

Managed path (VoiceCloner) is ICL cloning: reference audio + reference text.
The manager owns the transcript per ADR 0003 — user-provided text wins, a
missing one is auto-filled once via STT and persisted, and this engine fails
clearly rather than silently degrading when it is absent. The lower-quality
x-vector-only mode exists on the runtime API for callers who explicitly choose
it; it is never inferred from a missing transcript.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf

from ..qwen3_tts.runtime import DEFAULT_BASE_MODEL_ID

logger = logging.getLogger(__name__)

# Operator overrides for the per-generation speaker check (backlog 056).
SPEAKER_CHECK_ENV = "ABSTRACTVOICE_QWEN3_TTS_SPEAKER_CHECK"
MIN_SPEAKER_SIMILARITY_ENV = "ABSTRACTVOICE_QWEN3_TTS_MIN_SPEAKER_SIMILARITY"


class ClonedVoiceNotAppliedError(RuntimeError):
    """A text segment came out in a voice other than the clone, twice (backlog 056).

    One reply is spoken in ONE voice: a segment whose audio does not carry the
    cloned speaker is regenerated once, and if it still does not, synthesis stops
    here and names the segment instead of speaking it in a stock voice.
    """

    def __init__(
        self, *, segment_index: int, segment_count: int, text: str, scores: list, threshold: float, mid_segment: bool = False
    ):
        self.segment_index = int(segment_index)
        self.segment_count = int(segment_count)
        self.text = str(text)
        self.scores = [float(x) for x in scores]
        self.threshold = float(threshold)
        self.mid_segment = bool(mid_segment)
        self.segments: list = []  # per-segment records of the reply up to this one (set by the engine)
        words = " ".join(self.text.split()[:8])
        found = ", ".join(f"{x:.3f}" for x in self.scores)
        if self.mid_segment:
            what = (
                f"its voice changed part-way (speaker similarity {found} < {self.threshold:.3f}); the part already "
                "delivered was in the cloned voice, the rest of the reply was not spoken"
            )
        else:
            what = (
                f"speaker similarity {found} stayed below {self.threshold:.3f} over {len(self.scores)} attempts; "
                "the audio already delivered is in the cloned voice, this segment and the rest of the reply were not spoken"
            )
        super().__init__(
            f"Qwen3-TTS could not apply the cloned voice to text segment {self.segment_index + 1} of "
            f"{self.segment_count} (\"{words}...\"): {what} rather than spoken in another voice."
        )


class _Outbox:
    """A reply's released audio, delivered at the generation's own cadence (0.19.5).

    ``take()`` hands over one chunk per generated chunk, and everything it holds while the
    listener is short of audio: what was sent is compared with real-time playback from
    the first chunk on (the listener starts playing when the first chunk arrives).
    Bursts of held audio are what made 0.19.4 deliver in 4 s lumps.
    """

    def __init__(self, floor_s: float, *, clock=None) -> None:
        import collections
        import time

        self.floor_s = float(floor_s)
        self._clock = clock or time.perf_counter
        self._q = collections.deque()
        self.started_at: Optional[float] = None
        self.sent_s = 0.0

    def put(self, mono, sr: int) -> None:
        self._q.append((mono, int(sr)))

    def lead_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return self.sent_s - (self._clock() - self.started_at)

    def starving(self) -> bool:
        return self.started_at is not None and not self._q and self.lead_s() < self.floor_s

    def _pop(self):
        mono, sr = self._q.popleft()
        if self.started_at is None:
            self.started_at = self._clock()
        self.sent_s += float(mono.size) / float(sr or 24000)
        return mono, sr

    def take(self) -> list:
        out = [self._pop()] if self._q else []
        while self._q and self.lead_s() < self.floor_s:
            out.append(self._pop())
        return out

    def flush(self) -> list:
        out = []
        while self._q:
            out.append(self._pop())
        return out


class Qwen3TTSVoiceCloningEngine:
    """Cloning over :class:`abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime`."""

    engine_id = "qwen3-tts"

    DEFAULT_BASE_MODEL_ID = DEFAULT_BASE_MODEL_ID

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        allow_downloads: bool = True,
        model_id: str | None = None,
        predictor: str | None = None,
        sampler: str | None = None,
        backend: str | None = None,
        mlx_quantization: str | None = None,
        speaker_check: bool | None = None,
        min_speaker_similarity: float | None = None,
    ) -> None:
        from ..qwen3_tts.runtime import Qwen3TTSSettings

        self.debug = bool(debug)
        # Validated now, so a typo fails before any weights load.
        self._settings = Qwen3TTSSettings(
            predictor=predictor, sampler=sampler, backend=backend, mlx_quantization=mlx_quantization
        )
        self._device = str(device or "auto")
        self._allow_downloads = bool(allow_downloads)
        self._model_id = str(model_id or self.DEFAULT_BASE_MODEL_ID)
        self._runtime = None
        # One prompt per (reference audio, reference text): reference encoding
        # costs a codec encode + speaker-embedding pass, and re-paying it per
        # text chunk would dominate chunked synthesis.
        self._prompt_cache_key: Optional[tuple] = None
        self._prompt_cache_value: Any = None
        # Explicit arguments win; else the environment (an operator's escape hatch); else the defaults.
        if speaker_check is None:
            env = os.environ.get(SPEAKER_CHECK_ENV, "").strip().lower()
            if env and env not in ("0", "1", "true", "false", "on", "off", "yes", "no"):
                raise ValueError(f"{SPEAKER_CHECK_ENV} must be 0/1, true/false, on/off or yes/no; got {env!r}")
            speaker_check = self.SPEAKER_CHECK if not env else env in ("1", "true", "on", "yes")
        if min_speaker_similarity is None:
            env = os.environ.get(MIN_SPEAKER_SIMILARITY_ENV, "").strip()
            min_speaker_similarity = float(env) if env else self.MIN_SPEAKER_SIMILARITY
        self.speaker_check = bool(speaker_check)
        self.min_speaker_similarity = float(min_speaker_similarity)
        # One record per text segment of the latest synthesis (backlog 056): what ran, with which
        # voice reference, model and backend, how many attempts, the speaker scores, and how it ended.
        self.last_segments: list = []

    # ------------------------------------------------------- speaker check
    #
    # Backlog 056: the 0.6B Base checkpoint sometimes samples a WHOLE generation
    # in a stock-sounding voice from the same ICL prompt (seen live: generation 9
    # of the 11 of one 8 kB reply; reproduced on real weights for two short-
    # reference clones). Nothing fails and every per-call check stays green, so
    # every generation's audio is scored with the checkpoint's own speaker
    # encoder against the prompt's speaker embedding BEFORE it is delivered:
    #
    # - the opening is held back: released at SPEAKER_CHECK_EARLY_SECONDS when it
    #   is clearly the clone, else decided at SPEAKER_CHECK_SECONDS. A miss
    #   discards that generation and runs it again (SPEAKER_RETRIES); a second
    #   miss raises ClonedVoiceNotAppliedError naming the segment.
    # - after the opening (0.19.6): chunks are released as generated, paced by the
    #   reply's _Outbox (never a held block), and each SPEAKER_CHECK_SECONDS window
    #   is scored for the segment record only (`window_scores`, `low_windows`). The
    #   0.19.5 stop line on those windows (MIN_BLOCK_SPEAKER_SIMILARITY) stopped
    #   ordinary long replies on hal9000-qwen2; it stops nothing until a measured
    #   line exists. A later segment's opening is checked while the previous one
    #   plays; the listener never waits on it below LEAD_FLOOR_SECONDS in hand.
    #
    # Thresholds, from the real-weights calibration in backlog 056 (Qwen3-TTS
    # 0.6B Base, MLX 8-bit, cosine vs the prompt's reference embedding, 70+
    # generations of five clones): the clone's 4 s openings score >= 0.956 and
    # its later 4 s blocks >= 0.948; stock-voice openings <= 0.941 at 4 s (none
    # reaches 0.975 at 2 s), stock-voice and runaway blocks <= 0.92. The reply's
    # first opening costs up to SPEAKER_CHECK_SECONDS of first-audio latency
    # (about 0.5 s or 1.1 s at the MLX backend's real-time factor); later
    # openings are checked while the previous segment plays.
    SPEAKER_CHECK = True
    SPEAKER_CHECK_EARLY_SECONDS = 2.0
    SPEAKER_CHECK_EARLY_ACCEPT = 0.975
    SPEAKER_CHECK_SECONDS = 4.0
    MIN_SPEAKER_SIMILARITY = 0.95
    # Windows after the opening below this are counted in `low_windows` (logged, not enforced; 0.19.6).
    MIN_BLOCK_SPEAKER_SIMILARITY = 0.93
    # Shorter audio is too short to score reliably and passes unchecked.
    SPEAKER_CHECK_MIN_SECONDS = 1.0
    SPEAKER_RETRIES = 1
    # Audio the listener must have in hand before a chunk may wait for its score (0.19.5).
    LEAD_FLOOR_SECONDS = 1.5
    _clock = None  # time source of the delivery pacing (tests inject a virtual one)

    def _speaker_score(self, runtime, prompt, held: list) -> Optional[float]:
        """Similarity of the held audio to the clone, or None when too short to judge."""
        if not held:
            return None
        sr = int(held[-1][1]) or 24000
        audio = np.concatenate([a for a, _ in held])
        if audio.size < int(self.SPEAKER_CHECK_MIN_SECONDS * sr):
            return None
        return float(runtime.speaker_similarity(audio, clone_prompt=prompt, sample_rate=sr))

    def _checked_generation(
        self, runtime, prompt, *, index: int, count: int, text: str, generate, sink: Optional[list] = None,
        outbox: Optional["_Outbox"] = None, record: Optional[dict] = None,
    ):
        """Yield one segment's audio once its opening scores as the cloned voice.

        ``generate()`` starts one generation of ``text`` and returns its ``(audio, sr)``
        iterator; a rejected generation is closed (releasing the model) before the next.

        - The opening (``SPEAKER_CHECK_EARLY_SECONDS`` / ``SPEAKER_CHECK_SECONDS``) is
          scored before any of it is released; a miss regenerates it (``SPEAKER_RETRIES``),
          a second miss raises ``ClonedVoiceNotAppliedError``.
        - After the opening, chunks are released as generated, through the reply's
          ``outbox`` at the generation's cadence; each 4 s window is scored for the
          record only (0.19.6).
        - The listener is never starved for a later segment's opening: below
          ``LEAD_FLOOR_SECONDS`` in hand a chunk goes out before the opening decision
          ("provisional"; never for the reply's first opening), and a miss then raises.
        """
        if not self.speaker_check:
            for audio, sr in generate():
                mono = np.asarray(audio, dtype=np.float32).reshape(-1)
                if record is not None:
                    record["audio_s"] = round(record["audio_s"] + mono.size / float(int(sr) or 24000), 2)
                yield mono, int(sr)
            return
        box = outbox if outbox is not None else _Outbox(self.LEAD_FLOOR_SECONDS, clock=self._clock)
        threshold = self.min_speaker_similarity
        block_threshold = self.MIN_BLOCK_SPEAKER_SIMILARITY
        scores: list = []

        def scored(chunks: list) -> Optional[float]:
            value = self._speaker_score(runtime, prompt, chunks)
            if value is not None and sink is not None:
                sink.append(round(float(value), 3))
            return value

        def fail(score: float, limit: float, *, heard: bool):
            scores.append(score)
            return ClonedVoiceNotAppliedError(
                segment_index=index, segment_count=count, text=text, scores=scores, threshold=limit, mid_segment=heard,
            )

        window = {"start": 0}

        def log_windows(chunks: list, sizes: list, sr: int, *, final: bool) -> None:
            """Score each completed SPEAKER_CHECK_SECONDS window after the opening, for the record only."""
            need = int(self.SPEAKER_CHECK_SECONDS * sr)
            while window["start"] < len(chunks):
                rest = sum(sizes[window["start"]:])
                if rest < need and not (final and rest >= int(self.SPEAKER_CHECK_MIN_SECONDS * sr)):
                    return
                end, acc = window["start"], 0
                while end < len(chunks) and acc < need:
                    acc += sizes[end]
                    end += 1
                value = self._speaker_score(runtime, prompt, chunks[window["start"]:end])
                window["start"] = end
                if value is None or record is None:
                    continue
                record.setdefault("window_scores", []).append(round(float(value), 3))
                if value < block_threshold:
                    record["low_windows"] = int(record.get("low_windows", 0)) + 1
                    logger.info(
                        "Qwen3-TTS clone: text segment %d/%d, window at %.1f s scored %.3f (< %.3f); logged, not enforced",
                        index + 1, count, sum(sizes[: window["start"]]) / float(sr) - acc / float(sr), value, block_threshold,
                    )

        for attempt in range(1 + int(self.SPEAKER_RETRIES)):
            retrying = attempt < int(self.SPEAKER_RETRIES)
            window["start"] = 0
            if record is not None:
                record["window_scores"], record["low_windows"] = [], 0
            # The reply's very first opening may hold (first-audio latency); later ones only
            # while the listener has enough audio in hand.
            strict_opening = box.started_at is None
            source = iter(generate())
            try:
                chunks: list = []  # this generation's chunks, in order
                sizes: list = []
                released = 0  # chunks handed to the outbox
                verified = 0  # chunks covered by a passing score
                provisional = False
                opened = False
                early_done = False
                rejected = False
                for audio, sr in source:
                    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
                    chunks.append((mono, int(sr)))
                    sizes.append(mono.size)
                    total = sum(sizes)
                    sr = int(sr) or 24000
                    if not opened:
                        if not early_done and total >= int(self.SPEAKER_CHECK_EARLY_SECONDS * sr):
                            early_done = True
                            score = scored(chunks)
                            if score is not None and score >= self.SPEAKER_CHECK_EARLY_ACCEPT:
                                opened, verified = True, len(chunks)
                        if not opened and total >= int(self.SPEAKER_CHECK_SECONDS * sr):
                            score = scored(chunks)
                            if score is not None and score < threshold:
                                if provisional:
                                    raise fail(score, threshold, heard=True)
                                scores.append(score)
                                logger.warning(
                                    "Qwen3-TTS clone: text segment %d/%d came out in another voice "
                                    "(speaker similarity %.3f < %.3f); %s",
                                    index + 1, count, score, threshold, "regenerating it" if retrying else "giving up",
                                )
                                rejected = True
                                break
                            opened, verified = True, len(chunks)
                    else:
                        # After the opening: released as generated (0.19.6). Each 4 s window is scored
                        # for the record only; it stops nothing until a measured line exists.
                        verified = len(chunks)
                        log_windows(chunks, sizes, sr, final=False)
                    # Hand verified chunks over; then keep the listener fed.
                    while released < verified:
                        box.put(*chunks[released])
                        released += 1
                    if box.starving() and released < len(chunks) and not (strict_opening and not opened):
                        box.put(*chunks[released])
                        released += 1
                        provisional = True
                        if record is not None:
                            record["provisional_s"] = round(record.get("provisional_s", 0.0) + sizes[released - 1] / sr, 2)
                    yield from box.take()
                if rejected:
                    continue
                # End of the generation. Opened: log the last window. A generation shorter than the
                # opening window is decided here, on all of it (enforced, like any opening).
                if opened:
                    log_windows(chunks, sizes, int(chunks[-1][1]) or 24000, final=True)
                elif chunks:
                    score = scored(chunks)
                    if score is not None and score < threshold:
                        if provisional:
                            raise fail(score, threshold, heard=True)
                        scores.append(score)
                        logger.warning(
                            "Qwen3-TTS clone: text segment %d/%d came out in another voice (speaker similarity %.3f < %.3f); %s",
                            index + 1, count, score, threshold, "regenerating it" if retrying else "giving up",
                        )
                        continue
                while released < len(chunks):
                    box.put(*chunks[released])
                    released += 1
                if record is not None and chunks:
                    record["audio_s"] = round(sum(sizes) / float(int(chunks[-1][1]) or 24000), 2)
                if outbox is None:
                    yield from box.flush()
                else:
                    yield from box.take()
                return
            except ClonedVoiceNotAppliedError:
                # What was verified (or already heard) is the clone: deliver it, then stop loudly.
                close = getattr(source, "close", None)
                if close is not None:
                    close()
                yield from box.flush()
                raise
            finally:
                close = getattr(source, "close", None)
                if close is not None:
                    close()
        yield from box.flush()
        raise ClonedVoiceNotAppliedError(
            segment_index=index, segment_count=count, text=text, scores=scores, threshold=threshold
        )

    def _recorded_segment(
        self, runtime, prompt, *, index: int, count: int, text: str, ref_path: str, generate, outbox=None
    ):
        """``_checked_generation`` plus one traceable record per segment in ``last_segments`` and the log."""
        info = {}
        try:
            info = dict(runtime.runtime_info())
        except Exception:
            info = {}
        record = {
            "index": int(index),
            "count": int(count),
            "chars": len(str(text)),
            "first_words": " ".join(str(text).split()[:6]),
            "model": info.get("model_id") or self._model_id,
            "backend": info.get("backend"),
            "dtype": info.get("dtype"),
            "reference": "/".join(str(ref_path).replace("\\", "/").split("/")[-2:]),
            "speaker_check": bool(self.speaker_check),
            "attempts": 0,
            "scores": [],
            "audio_s": 0.0,
            "provisional_s": 0.0,
            "window_scores": [],  # 4 s windows after the opening: logged, not enforced (0.19.6)
            "low_windows": 0,
            "status": "running",
            "error": None,
        }
        self.last_segments.append(record)

        def counted():
            record["attempts"] += 1
            return generate()

        try:
            for audio, sr in self._checked_generation(
                runtime, prompt, index=index, count=count, text=text, generate=counted, sink=record["scores"],
                outbox=outbox, record=record,
            ):
                yield audio, sr
            record["status"] = "ok" if record["attempts"] <= 1 else "regenerated"
        except GeneratorExit:
            record["status"] = "cancelled"
            raise
        except ClonedVoiceNotAppliedError as e:
            record["status"], record["error"] = "failed", str(e)
            # The loud failure carries the reply's records so far, so it reads like a success's metrics.
            e.segments = [dict(r) for r in self.last_segments]
            raise
        except Exception as e:
            record["status"], record["error"] = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            logger.info(
                "Qwen3-TTS clone segment %d/%d: %s, %d chars, %.1f s audio, %d attempt(s), scores %s, %s/%s, ref %s%s",
                index + 1, count, record["status"], record["chars"], record["audio_s"], record["attempts"],
                record["scores"], record["model"], record["backend"], record["reference"],
                f", error: {record['error']}" if record["error"] else "",
            )

    # -------------------------------------------------------------- lifecycle

    def _get_runtime(self):
        if self._runtime is None:
            from ..qwen3_tts.runtime import Qwen3TTSRuntime

            self._runtime = Qwen3TTSRuntime(
                model_id=self._model_id,
                device=self._device,
                allow_downloads=self._allow_downloads,
                debug=self.debug,
                settings=self._settings,
            )
        return self._runtime

    def preload(self) -> dict:
        runtime = self._get_runtime()
        runtime._ensure_loaded()
        return dict(runtime.runtime_info())

    def unload(self) -> bool:
        if self._runtime is not None:
            self._runtime.unload()
        self._prompt_cache_key = None
        self._prompt_cache_value = None
        return True

    def is_loaded(self) -> bool:
        return bool(self._runtime is not None and self._runtime.is_loaded)

    def runtime_info(self) -> dict:
        try:
            return dict(self._get_runtime().runtime_info())
        except Exception:
            return {"model_id": self._model_id, "loaded": False}

    def set_quality_preset(self, preset: str) -> None:
        try:
            self._get_runtime().settings.apply_quality_preset(str(preset))
        except Exception:
            pass

    # -------------------------------------------------------------- inference

    def _clone_prompt(self, ref_path: str, reference_text: str):
        try:
            mtime = os.path.getmtime(ref_path)
        except OSError:
            mtime = None
        key = (str(ref_path), str(reference_text), mtime)
        if self._prompt_cache_key == key and self._prompt_cache_value is not None:
            return self._prompt_cache_value
        runtime = self._get_runtime()
        if str(ref_path).lower().endswith(".npz"):
            # A stored clone prompt (the built-in voice pack): codes + speaker embedding,
            # rebuilt for this runtime's backend instead of re-encoding reference audio.
            with np.load(str(ref_path)) as arrays:
                prompt = runtime.clone_prompt_from_arrays(
                    ref_codes=arrays["ref_codes"],
                    speaker_embedding=arrays["speaker_embedding"],
                    ref_text=str(reference_text),
                    x_vector_only=False,
                )
        else:
            prompt = runtime.build_clone_prompt(
                ref_audio=str(ref_path),
                ref_text=str(reference_text),
                x_vector_only=False,  # ICL; the manager guarantees the transcript.
            )
        self._prompt_cache_key = key
        self._prompt_cache_value = prompt
        return prompt

    def _one_shot(self, *, text: str, reference_paths: Iterable[str], reference_text: str, language: str | None):
        """The whole text as ONE clone generation (no segmentation, no truncation)."""
        from ..adapters.tts_qwen3_tts import _qwen_language
        from ..qwen3_tts.runtime import one_shot_max_new_tokens

        max_new_tokens = one_shot_max_new_tokens(str(text))
        if not reference_text or not str(reference_text).strip():
            raise RuntimeError("Missing reference_text for Qwen3-TTS cloning.")
        ref_paths = list(reference_paths or [])
        if len(ref_paths) != 1:
            raise RuntimeError("Qwen3-TTS cloning currently supports exactly one reference audio file.")
        runtime = self._get_runtime()
        prompt = self._clone_prompt(str(ref_paths[0]), str(reference_text))
        qwen_language = _qwen_language(language)
        language_arg = qwen_language if qwen_language != "Auto" else None
        self.last_segments = []
        yield from self._recorded_segment(
            runtime, prompt, index=0, count=1, text=str(text), ref_path=str(ref_paths[0]),
            generate=lambda: [runtime.synthesize_clone(
                str(text), clone_prompt=prompt, language=language_arg, max_new_tokens=max_new_tokens
            )],
        )

    def infer_to_audio_chunks(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        max_chars: int = 200,
        language: str | None = None,
    ):
        """Chunked synthesis: early playback + per-utterance cancellation (ADR 0004)."""
        _ = speed  # Reported unsupported in the capability catalog.
        if not reference_text or not str(reference_text).strip():
            raise RuntimeError(
                "Missing reference_text for Qwen3-TTS cloning.\n"
                "If you're using VoiceCloner/VoiceManager, reference_text should be auto-generated and cached.\n"
                "If you're calling this engine directly, provide reference_text or set it via the voice store."
            )

        ref_paths = list(reference_paths or [])
        if len(ref_paths) != 1:
            raise RuntimeError(
                "Qwen3-TTS cloning currently supports exactly one reference audio file.\n"
                "Provide a single WAV/FLAC/OGG file when creating the voice clone."
            )

        from ..adapters.tts_qwen3_tts import _qwen_language
        from ..tts.text_chunking import split_text_batches

        runtime = self._get_runtime()
        prompt = self._clone_prompt(str(ref_paths[0]), str(reference_text))
        qwen_language = _qwen_language(language)
        language_arg = qwen_language if qwen_language != "Auto" else None

        # Native frame streaming (MLX backend, ADR 0008): first audio no longer
        # depends on the segment length, so the reply is ONE generation up to the
        # native bound and the cloned voice cannot drift between sentences.
        native = bool(getattr(runtime, "supports_native_streaming", False))
        if native:
            from ..qwen3_tts.runtime import NATIVE_STREAM_MAX_CHARS

            bound = max(int(max_chars), int(NATIVE_STREAM_MAX_CHARS))
        else:
            bound = int(max_chars)

        segments = [chunk for chunk in split_text_batches(str(text), max_chars=bound) if chunk.strip()]
        self.last_segments = []
        outbox = _Outbox(self.LEAD_FLOOR_SECONDS, clock=self._clock)  # one per reply: continuous delivery across segments
        for index, chunk in enumerate(segments):
            if native:
                generate = lambda chunk=chunk: runtime.stream_clone(chunk, clone_prompt=prompt, language=language_arg)  # noqa: E731
            else:
                generate = lambda chunk=chunk: [runtime.synthesize_clone(chunk, clone_prompt=prompt, language=language_arg)]  # noqa: E731
            yield from self._recorded_segment(
                runtime, prompt, index=index, count=len(segments), text=chunk, ref_path=str(ref_paths[0]),
                generate=generate, outbox=outbox,
            )
        yield from outbox.flush()

    # `one_shot=True` on infer_to_wav_bytes: the whole text as ONE generation.
    supports_one_shot = True

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str],
        reference_text: str,
        speed: Optional[float] = None,
        language: str | None = None,
        one_shot: bool = False,
    ) -> bytes:
        chunks = []
        sr_out = 24000
        if one_shot:
            source = self._one_shot(
                text=text, reference_paths=reference_paths, reference_text=reference_text, language=language
            )
        else:
            source = self.infer_to_audio_chunks(
                text=text,
                reference_paths=reference_paths,
                reference_text=reference_text,
                speed=speed,
                language=language,
            )
        for chunk, sr in source:
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr_out = int(sr)
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        if speed is not None and float(speed) != 1.0 and audio.size:
            # Pitch-preserving time-stretch after generation (Qwen3-TTS has no speed control).
            from ..tts.tts_engine import apply_speed_without_pitch_change

            audio = apply_speed_without_pitch_change(audio, float(speed), sr=int(sr_out))
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()
