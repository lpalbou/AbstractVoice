"""Backlog 056: one cloned reply is spoken in ONE voice. Weight-free.

A Qwen3-TTS clone generation whose audio does not carry the cloned speaker is
never delivered on its first take: a generation that opens under the clone bar is
regenerated once, a second miss is spoken anyway in the cloned voice (operator ruling
2026-11-05/07), and a window that dips under the block bar after the opening is recorded
(``low_windows``) while the reply keeps playing: nothing stops a reply after its opening
(operator ruling 2026-11-07: a hard failure is worse than a small difference).
"""
from __future__ import annotations

import numpy as np
import pytest

from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine
from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings

SR = 24000
STEP = int(0.96 * SR)  # one MLX stream chunk (12 codec frames)
CLONE, UNSURE, STOCK = 0.98, 0.96, 0.90
# Three ~800-char paragraphs: three generations on the native (MLX) path, more on the 120-char torch path.
TEXT = " ".join(
    " ".join(f"Paragraph {p} sentence {i} says something long enough to be spoken." for i in range(12))
    for p in range(3)
)


def _tag(segment, attempt, chunk):
    return 0.1 * (segment + 1) + 0.01 * attempt + 0.0001 * chunk


def _untag(value):
    value = float(value)
    segment = int(value * 10 + 1e-6) - 1
    rest = value - 0.1 * (segment + 1)
    attempt = int(rest * 100 + 1e-6)
    chunk = int(round((rest - 0.01 * attempt) * 10000))
    return segment, attempt, chunk


class _Gen:
    """One generation: 0.96 s chunks whose samples encode (segment, attempt, chunk index)."""

    def __init__(self, text, segment, attempt, seconds):
        self.text, self.segment, self.attempt, self.seconds = text, segment, attempt, seconds
        self.closed = False
        self.yielded = 0

    def __iter__(self):
        try:
            n = int(self.seconds * SR)
            for k, start in enumerate(range(0, n, STEP)):
                self.yielded += 1
                yield np.full(min(STEP, n - start), _tag(self.segment, self.attempt, k), dtype=np.float32), SR
        finally:
            self.closed = True


class _RT:
    """Fake runtime. ``voice(segment, attempt, chunk)`` is the speaker similarity of that chunk's audio."""

    def __init__(self, voice, *, native=True, seconds=8.0):
        self.settings = Qwen3TTSSettings()
        self.supports_native_streaming = native
        self.voice = voice
        self.seconds = seconds
        self.gens = []
        self.attempts = {}
        self.texts = []

    def build_clone_prompt(self, *, ref_audio, ref_text, x_vector_only):
        return "prompt"

    def _start(self, text):
        if text not in self.texts:
            self.texts.append(text)
        attempt = self.attempts.get(text, 0)
        self.attempts[text] = attempt + 1
        gen = _Gen(text, self.texts.index(text), attempt, self.seconds)
        self.gens.append(gen)
        return gen

    def stream_clone(self, text, *, clone_prompt, language=None, **kw):
        assert clone_prompt == "prompt"
        return iter(self._start(text))

    def synthesize_clone(self, text, *, clone_prompt, language=None, **kw):
        assert clone_prompt == "prompt"
        return np.concatenate([a for a, _ in self._start(text)]), SR

    def speaker_similarity(self, audio, *, clone_prompt, sample_rate=SR):
        # The scored audio mixes its chunks: the worst one decides, like a voice change inside a real block.
        return min(float(self.voice(*_untag(v))) for v in np.unique(np.asarray(audio).reshape(-1)))


def _engine(rt, **kw):
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False, **kw)
    engine._runtime = rt
    return engine


def _speak(engine, text=TEXT):
    return engine.infer_to_audio_chunks(text=text, reference_paths=["ref.wav"], reference_text="the transcript")


def _delivered(chunks):
    """(segment, attempt) of the delivered audio, in order, consecutive duplicates removed."""
    out = []
    for audio, _sr in chunks:
        segment, attempt, _ = _untag(audio[0])
        if not out or out[-1] != (segment, attempt):
            out.append((segment, attempt))
    return out


@pytest.mark.parametrize("native", [True, False], ids=["mlx-native-stream", "torch-segments"])
def test_a_segment_in_another_voice_is_regenerated_and_never_delivered(native):
    # Segment 1's first attempt comes out in a stock voice (the 2026-10-29 symptom).
    rt = _RT(lambda seg, att, k: STOCK if (seg, att) == (1, 0) else CLONE, native=native)
    delivered = _delivered(list(_speak(_engine(rt))))
    assert len({s for s, _ in delivered}) >= 2
    assert (1, 0) not in delivered  # the stock-voice generation never reaches the listener
    assert (1, 1) in delivered  # its regeneration does
    assert all(att == 0 for seg, att in delivered if seg != 1)
    assert all(g.closed for g in rt.gens)  # the rejected generation released the model


@pytest.mark.parametrize("native", [True, False], ids=["mlx-native-stream", "torch-segments"])
def test_a_segment_that_misses_twice_is_spoken_in_the_cloned_voice(native):
    # Operator ruling 2026-11-05/07: retry once; if the take still scores under the bar, speak it in the
    # cloned voice anyway (a small difference beats another voice or an unspoken reply). Nothing raises.
    rt = _RT(lambda seg, att, k: STOCK if seg == 1 else CLONE, native=native)
    engine = _engine(rt)
    delivered = _delivered(list(_speak(engine)))
    assert (1, 0) not in delivered  # the first take was never played: it was regenerated
    assert (1, 1) in delivered  # the second take is spoken, not dropped
    assert len({s for s, _ in delivered}) >= 3  # and the rest of the reply follows
    rec = engine.last_segments[1]  # retried exactly once, then kept
    assert rec["status"] == "kept_below_threshold" and rec["attempts"] == 2 and max(rec["scores"]) < 0.95
    assert all(g.closed for g in rt.gens)


def test_an_unsure_opening_is_held_until_the_full_check_window():
    rt = _RT(lambda seg, att, k: UNSURE, native=True)
    stream = _speak(_engine(rt))
    next(stream)
    # Not clearly the clone at 2 s, so nothing plays before 4 s (five 0.96 s chunks) are scored.
    assert rt.gens[0].yielded == 5
    stream.close()

    rt = _RT(lambda seg, att, k: CLONE, native=True)
    stream = _speak(_engine(rt))
    next(stream)
    assert rt.gens[0].yielded == 3  # clearly the clone at 2 s: released early
    assert not rt.gens[0].closed
    stream.close()
    assert rt.gens[0].closed  # a consumer that stops reading releases the model


def test_a_voice_change_part_way_is_recorded_and_the_reply_keeps_playing():
    # The opening is the clone; from chunk 5 on (the second 4 s block) the voice scores as another one.
    # Nothing stops: the whole segment is heard, and the record says where it dipped and how far.
    rt = _RT(lambda seg, att, k: STOCK if (seg == 0 and k >= 5) else CLONE, native=True, seconds=12.0)
    engine = _engine(rt)
    delivered = [audio for audio, _ in _speak(engine)]
    assert rt.attempts[rt.texts[0]] == 1  # the opening was the clone: never regenerated
    assert max(_untag(a[0])[2] for a in delivered if _untag(a[0])[0] == 0) == 12  # every chunk of segment 0 played
    rec = engine.last_segments[0]
    assert rec["status"] == "ok" and rec["attempts"] == 1
    assert rec["low_windows"] and all(abs(w["score"] - STOCK) < 1e-3 and w["score"] < w["limit"] for w in rec["low_windows"])
    assert rec["low_windows"][0]["at_s"] == pytest.approx(3 * STEP / SR, abs=0.01)  # first window holding chunk 5
    assert all(r["low_windows"] == [] for r in engine.last_segments[1:])  # the other segments were the clone


def test_buffered_and_one_shot_synthesis_are_checked_too():
    rt = _RT(lambda seg, att, k: CLONE if att > 0 else STOCK, native=True, seconds=5.0)
    wav = _engine(rt).infer_to_wav_bytes(text="A short reply.", reference_paths=["ref.wav"], reference_text="t", one_shot=True)
    assert wav and rt.attempts["A short reply."] == 2

    # Both takes under the bar: the second is kept and returned (operator ruling 2026-11-05/07).
    rt = _RT(lambda seg, att, k: STOCK, native=True, seconds=5.0)
    engine = _engine(rt)
    wav = engine.infer_to_wav_bytes(text="A short reply.", reference_paths=["ref.wav"], reference_text="t")
    assert wav and rt.attempts["A short reply."] == 2 and engine.last_segments[-1]["status"] == "kept_below_threshold"


def test_too_short_to_judge_passes_unchecked_and_the_check_can_be_disabled():
    rt = _RT(lambda seg, att, k: STOCK, native=True, seconds=0.5)
    assert len(list(_speak(_engine(rt), "Yes."))) == 1  # 0.5 s: below SPEAKER_CHECK_MIN_SECONDS

    rt = _RT(lambda seg, att, k: STOCK, native=True)
    assert len(list(_speak(_engine(rt, speaker_check=False)))) > 0


def test_the_prompt_cache_is_per_voice_and_audio_is_never_cached():
    """No per-text audio cache: the same text in two voices builds each voice's prompt and runs a generation each time."""
    rt = _RT(lambda seg, att, k: CLONE, native=True, seconds=1.5)
    builds = []
    rt.build_clone_prompt = lambda *, ref_audio, ref_text, x_vector_only: builds.append(ref_audio) or "prompt"
    engine = _engine(rt)
    for ref in ("a.wav", "b.wav", "a.wav"):
        list(engine.infer_to_audio_chunks(text="Same words.", reference_paths=[ref], reference_text="t"))
    assert builds == ["a.wav", "b.wav", "a.wav"]
    assert len(rt.gens) == 3


def test_runtime_speaker_similarity_compares_against_the_prompt_embedding():
    """`Qwen3TTSRuntime.speaker_similarity`: cosine vs the prompt's own speaker embedding, lock-free."""
    from types import SimpleNamespace

    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    rt = Qwen3TTSRuntime("Qwen/Qwen3-TTS-12Hz-0.6B-Base", allow_downloads=False)
    seen = []

    class _MLX:
        def speaker_embedding(self, wav):
            seen.append(wav.size)
            return np.array([1.0, 1.0, 0.0], dtype=np.float32) if wav.mean() > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    rt._mlx, rt._backend = _MLX(), "mlx"
    prompt = SimpleNamespace(ref_codes=None, speaker_embedding=np.array([2.0, 2.0, 0.0], dtype=np.float32), x_vector_only_mode=True, ref_text=None)
    assert rt.speaker_similarity(np.full(SR, 0.1, dtype=np.float32), clone_prompt=prompt) == pytest.approx(1.0)
    assert rt.speaker_similarity(np.full(SR, -0.1, dtype=np.float32), clone_prompt=prompt) == pytest.approx(0.0)
    # Other sample rates are resampled to the encoder's 24 kHz.
    rt.speaker_similarity(np.full(16000, 0.1, dtype=np.float32), clone_prompt=prompt, sample_rate=16000)
    assert seen[-1] == SR
    # Callable while a clone stream holds the generation lock (it is scored between chunks).
    with rt._owned():
        assert rt.speaker_similarity(np.full(SR, 0.1, dtype=np.float32), clone_prompt=prompt) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="another checkpoint"):
        rt.speaker_similarity(np.full(SR, 0.1, dtype=np.float32), clone_prompt=SimpleNamespace(
            ref_codes=None, speaker_embedding=np.ones(5, dtype=np.float32), x_vector_only_mode=True, ref_text=None))


def test_the_check_is_a_constructor_setting_never_the_environment(monkeypatch):
    # Root backlog 1066: settings, not environment variables. The variables of 0.20.x are ignored.
    monkeypatch.setenv("ABSTRACTVOICE_QWEN3_TTS_SPEAKER_CHECK", "0")
    monkeypatch.setenv("ABSTRACTVOICE_QWEN3_TTS_MIN_SPEAKER_SIMILARITY", "0.5")
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    assert engine.speaker_check is True and engine.min_speaker_similarity == Qwen3TTSVoiceCloningEngine.MIN_SPEAKER_SIMILARITY
    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False, speaker_check=False, min_speaker_similarity=0.9)
    assert engine.speaker_check is False and engine.min_speaker_similarity == 0.9
    with pytest.raises(ValueError, match="min_speaker_similarity"):
        Qwen3TTSVoiceCloningEngine(allow_downloads=False, min_speaker_similarity=1.5)


def test_every_segment_leaves_a_traceable_record_and_metrics_carry_them():
    rt = _RT(lambda seg, att, k: STOCK if (seg, att) == (1, 0) else CLONE, native=True)
    rt.runtime_info = lambda: {"model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "backend": "mlx", "dtype": "bfloat16+8bit"}
    engine = _engine(rt)
    list(_speak(engine))
    recs = engine.last_segments
    assert [r["index"] for r in recs] == list(range(len(rt.texts)))
    assert recs[1]["status"] == "regenerated" and recs[1]["attempts"] == 2 and min(recs[1]["scores"]) < 0.95
    assert all(r["status"] == "ok" and r["attempts"] == 1 for i, r in enumerate(recs) if i != 1)
    assert all(r["model"].endswith("-Base") and r["backend"] == "mlx" and r["reference"] == "ref.wav" for r in recs)
    assert all(r["audio_s"] > 0 and r["chars"] > 0 and r["first_words"].startswith("Paragraph") for r in recs)

    # A voice change part-way is recorded, never a stop: every segment is spoken, the record says where it dipped.
    rt = _RT(lambda seg, att, k: STOCK if (seg == 1 and k >= 5) else CLONE, native=True, seconds=12.0)
    engine = _engine(rt)
    list(_speak(engine))
    recs = engine.last_segments
    assert len(recs) == len(rt.texts) == 3 and [r["status"] for r in recs] == ["ok", "ok", "ok"]
    assert recs[1]["low_windows"] and all(w["score"] < w["limit"] for w in recs[1]["low_windows"])
    assert recs[0]["low_windows"] == [] and recs[2]["low_windows"] == []

    # VoiceManager stream metrics (what lands in the artifact's terminal_event.metrics) include them.
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.vm.tts_mixin import TtsMixin

    rt = _RT(lambda seg, att, k: CLONE, native=True)
    engine = _engine(rt)

    class _Cloner(VoiceCloner):
        def __init__(self):
            pass

        def speak_to_audio_segments(self, text, *, voice_id, speed=None, max_chars=120, language=None):
            self._last_chunks_engine = engine
            return engine.infer_to_audio_segments(text=text, reference_paths=["ref.wav"], reference_text="t")

    class _VM(TtsMixin):
        language = "en"

        def __init__(self):
            self._metrics = None

        def _get_voice_cloner(self):
            return _Cloner()

        def _effective_clone_speed(self, **kw):
            return 1.0, "qwen3-tts"

        def _set_last_tts_metrics(self, m):
            self._metrics = m

    vm = _VM()
    list(vm.speak_to_audio_chunks(TEXT, voice="sisko"))
    assert [r["status"] for r in vm._metrics["segments"]] == ["ok"] * len(rt.texts)


def test_clone_records_carry_the_reference_length(tmp_path):
    import soundfile as sf

    from abstractvoice.cloning.store import VoiceCloneStore
    from abstractvoice.integrations.abstractcore_plugin import _clone_profile_tags

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(int(2.5 * 16000), dtype=np.float32), 16000)
    store = VoiceCloneStore(tmp_path / "store", builtin_voices=False)
    vid = store.create_voice([str(ref)], name="short", engine="qwen3-tts", reference_text="hello there")
    assert store.get_voice_dict(vid)["reference_seconds"] == 2.5
    listed = [v for v in store.list_voices() if v["voice_id"] == vid][0]
    assert listed["reference_seconds"] == 2.5
    assert _clone_profile_tags(listed)["reference_seconds"] == 2.5
    assert _clone_profile_tags({**listed, "meta": None}) == {"reference_seconds": 2.5}
