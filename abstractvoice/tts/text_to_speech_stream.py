"""Bridge a streaming text source into streamed TTS audio chunks.

This module is intentionally engine-agnostic:
- it performs incremental segmentation of incoming text deltas
- it delegates "segment -> audio chunks" to a caller-provided function
- it delivers audio chunks to a caller-provided sink (playback enqueue, network, etc.)

Primary use case: LLM streaming -> TTS streaming (low time-to-first-audio).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from .text_chunking import TextStreamChunker, TextStreamChunkingConfig


AudioChunk = Tuple[np.ndarray, int]


@dataclass(frozen=True)
class TextToSpeechStreamConfig:
    chunking: TextStreamChunkingConfig = field(default_factory=TextStreamChunkingConfig)
    # When paused, avoid generating unbounded audio queues.
    pause_poll_s: float = 0.05


class TextToSpeechStream:
    """A push-based text stream that yields audio chunks progressively."""

    def __init__(
        self,
        *,
        iter_audio_chunks_for_segment: Callable[[str], Iterable[AudioChunk]],
        on_audio_chunk: Callable[[np.ndarray, int], None],
        cancel_event: threading.Event,
        is_paused: Optional[Callable[[], bool]] = None,
        on_metrics: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        config: TextToSpeechStreamConfig | None = None,
    ) -> None:
        self._iter_audio_chunks_for_segment = iter_audio_chunks_for_segment
        self._on_audio_chunk = on_audio_chunk
        self._cancel = cancel_event
        self._is_paused = is_paused
        self._on_metrics = on_metrics
        self._on_error = on_error
        self._cfg = config or TextToSpeechStreamConfig()

        self._buf_lock = threading.Lock()
        self._pending: str = ""
        self._has_data = threading.Event()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "TextToSpeechStream":
        if self._thread is not None:
            return self
        t = threading.Thread(target=self._run, daemon=True)
        self._thread = t
        t.start()
        return self

    def push(self, delta: str) -> bool:
        if self._closed.is_set():
            return False
        if self._cancel.is_set():
            return False
        d = str(delta or "")
        if not d:
            return True
        with self._buf_lock:
            self._pending += d
        self._has_data.set()
        return True

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._has_data.set()

    def cancel(self) -> None:
        try:
            self._cancel.set()
        except Exception:
            pass
        self.close()

    def join(self, timeout: float | None = None) -> bool:
        t = self._thread
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        chunker = TextStreamChunker(config=self._cfg.chunking)
        t0 = time.monotonic()
        first_audio_t: float | None = None
        chunks = 0
        total_audio_s = 0.0

        try:
            while not self._cancel.is_set():
                # Wait for more text (or close/cancel).
                self._has_data.wait(timeout=0.1)
                self._has_data.clear()

                # Consume the current pending buffer (coalesces many small deltas).
                with self._buf_lock:
                    item = self._pending
                    self._pending = ""

                if (not item) and self._closed.is_set():
                    break
                if not item:
                    continue

                for seg in chunker.push(str(item)):
                    if self._cancel.is_set():
                        break
                    first_audio_t, seg_chunks, seg_audio_s = self._emit_segment(seg, first_audio_t=first_audio_t)
                    chunks += int(seg_chunks)
                    total_audio_s += float(seg_audio_s)

            # Flush remainder.
            if not self._cancel.is_set():
                for seg in chunker.flush():
                    if self._cancel.is_set():
                        break
                    first_audio_t, seg_chunks, seg_audio_s = self._emit_segment(seg, first_audio_t=first_audio_t)
                    chunks += int(seg_chunks)
                    total_audio_s += float(seg_audio_s)

        except Exception as e:
            if callable(self._on_error):
                try:
                    self._on_error(e)
                except Exception:
                    pass
        finally:
            t1 = time.monotonic()
            synth_s = float(t1 - t0)
            ttfb_s = float(first_audio_t - t0) if first_audio_t is not None else None
            if callable(self._on_metrics):
                try:
                    self._on_metrics(
                        {
                            "streaming": True,
                            "cancelled": bool(self._cancel.is_set()),
                            "synth_s": synth_s,
                            "ttfb_s": ttfb_s,
                            "audio_s": float(total_audio_s),
                            "rtf": (synth_s / float(total_audio_s)) if total_audio_s else None,
                            "chunks": int(chunks),
                            "ts": time.time(),
                        }
                    )
                except Exception:
                    pass

    def _emit_segment(self, seg: str, *, first_audio_t: float | None) -> tuple[float | None, int, float]:
        seg_text = str(seg or "").strip()
        if not seg_text:
            return first_audio_t, 0, 0.0

        seg_chunks = 0
        seg_audio_s = 0.0

        for chunk, sr in self._iter_audio_chunks_for_segment(seg_text):
            if self._cancel.is_set():
                break

            # If paused, avoid unbounded audio queue growth.
            if callable(self._is_paused):
                try:
                    while (not self._cancel.is_set()) and bool(self._is_paused()):
                        time.sleep(float(self._cfg.pause_poll_s))
                except Exception:
                    pass
            if self._cancel.is_set():
                break

            mono = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if mono.size <= 0:
                continue

            try:
                sr_i = int(sr) if sr else 0
            except Exception:
                sr_i = 0
            if sr_i > 0:
                seg_audio_s += float(len(mono)) / float(sr_i)

            # Mark first-audio time at the point of first delivery to the sink.
            if first_audio_t is None:
                first_audio_t = time.monotonic()

            try:
                self._on_audio_chunk(mono, int(sr_i))
            except Exception as e:
                if callable(self._on_error):
                    try:
                        self._on_error(e)
                    except Exception:
                        pass
                break

            seg_chunks += 1

        return first_audio_t, int(seg_chunks), float(seg_audio_s)

