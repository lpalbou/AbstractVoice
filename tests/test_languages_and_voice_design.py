"""Language catalog, voice navigation and description-designed voices, without model weights."""
import importlib.util
import io
import subprocess
import sys
import wave

import numpy as np
import pytest

from abstractvoice.languages import language_catalog, language_entry, languages_by_provider, speakable_languages


# ---------------------------------------------------------------------------
# Language catalog
# ---------------------------------------------------------------------------


def test_catalog_covers_every_kind_and_known_engine():
    cat = language_catalog()
    assert set(cat) == {"tts", "stt", "cloning"}
    assert {"supertonic", "piper", "qwen3-tts", "omnivoice", "audiodit", "openai", "openai-compatible"} <= set(cat["tts"])
    assert {"faster-whisper", "mlx-whisper", "transformers-asr"} <= set(cat["stt"])
    assert {"qwen3-tts", "omnivoice", "f5_tts", "chroma"} <= set(cat["cloning"])
    qwen = cat["tts"]["qwen3-tts"]
    assert {"en", "fr", "de", "zh", "ja"} <= set(qwen["languages"])
    assert qwen["open"] is False and qwen["declared"] is True


def test_omnivoice_languages_come_from_its_package_when_installed():
    languages = language_entry("tts", "omnivoice")["languages"]
    if importlib.util.find_spec("omnivoice") is None:
        assert languages == ["en", "fr", "de", "es", "ru", "zh", "ja", "ko"]  # fallback without the package
    else:
        assert len(languages) > 100


def test_remote_engines_are_open_and_undeclared_lists_say_so():
    assert language_entry("tts", "openai") == {"provider": "openai", "languages": [], "open": True, "declared": True}
    chroma = language_entry("cloning", "chroma")
    assert chroma["declared"] is False and chroma["languages"] == []


def test_provider_names_are_normalized_and_filters_apply():
    cat = language_catalog(kind="tts", provider="Qwen3_TTS")
    assert list(cat) == ["tts"] and list(cat["tts"]) == ["qwen3-tts"]


def test_provider_alone_returns_the_kinds_that_engine_serves():
    assert list(language_catalog(provider="piper")) == ["tts"]
    assert list(language_catalog(provider="qwen3-tts")) == ["tts", "cloning"]
    assert list(language_catalog(provider="faster-whisper")) == ["stt"]


def test_unknown_provider_and_kind_fail_loudly():
    with pytest.raises(KeyError, match="known: .*supertonic"):
        language_catalog(kind="tts", provider="nope")
    with pytest.raises(KeyError, match="unknown provider 'nope'"):
        language_catalog(provider="nope")
    with pytest.raises(ValueError, match="kind must be one of"):
        language_catalog(kind="music")
    with pytest.raises(KeyError):
        language_entry("stt", "nope")


def test_languages_by_provider_leaves_out_engines_it_does_not_know():
    out = languages_by_provider("tts", ["piper", "third-party"])
    assert list(out) == ["piper"] and "en" in out["piper"]


def test_speakable_languages_fall_back_to_cloning_then_empty():
    assert "fr" in speakable_languages("qwen3-tts")
    assert speakable_languages("f5_tts") == ["en", "zh"]  # a cloning-only engine
    assert speakable_languages("openai") == [] and speakable_languages("third-party") == []


def test_catalog_does_not_import_torch():
    code = (
        "import sys\n"
        "from abstractvoice.languages import language_catalog\n"
        "language_catalog()\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'transformers' not in sys.modules, 'transformers imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ---------------------------------------------------------------------------
# Warm-up cut and designed reference rendering
# ---------------------------------------------------------------------------

SR = 24000


def _tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _gap(seconds):
    return np.zeros(int(seconds * SR), np.float32)


def test_reference_start_is_where_the_reference_opening_is_heard():
    from abstractvoice.cloning.design import reference_start_s, starts_with_reference

    words = [(" Hello", 0.0), (" there,", 0.5), (" before", 0.9), (" we", 1.2), (" go.", 1.4),
             (" Good", 1.6), (" mornin", 1.9), (" everyone.", 2.4)]
    # A misheard word ("mornin") is tolerated.
    assert reference_start_s(words, reference_text="Good morning everyone.") == 1.6
    # Scripts without spaces: characters carry their word's start.
    zh = [("这是开场白", 0.0), ("大家好欢迎光临", 1.2), ("很高兴", 2.0)]
    assert reference_start_s(zh, reference_text="大家好，欢迎光临。很高兴你们能来。") == 1.2
    assert reference_start_s([], reference_text="a b") is None
    assert reference_start_s([(" Something", 0.0), (" else", 0.4)], reference_text="Good morning everyone.") is None
    assert starts_with_reference([(" Good", 0.0), (" morning", 0.3)], reference_text="Good morning everyone.")
    assert not starts_with_reference([(" we", 0.0), (" go.", 0.2), (" Good", 0.5)], reference_text="Good morning everyone.")


def test_warmup_cut_takes_the_last_pause_before_the_reference_start():
    from abstractvoice.cloning.design import find_warmup_cut

    # Pauses at ~1.0 s (inside the warm-up), ~2.0 s (the boundary) and ~3.4 s (inside the reference).
    audio = np.concatenate([_tone(0.9), _gap(0.2), _tone(0.8), _gap(0.2), _tone(1.2), _gap(0.6), _tone(4.0)])
    cut = find_warmup_cut(audio, SR, reference_start=2.15)
    assert 1.9 * SR <= cut <= 2.1 * SR
    # A pause well after the heard start would cut the reference's first words: never chosen.
    cut = find_warmup_cut(audio, SR, reference_start=2.8)
    assert cut <= 2.8 * SR


def test_warmup_cut_without_a_pause_uses_the_quietest_instant_before_the_start():
    from abstractvoice.cloning.design import find_warmup_cut

    quiet = np.concatenate([_tone(2.0), _tone(0.05, amp=0.01), _tone(3.0)])  # a dip, not a silence
    cut = find_warmup_cut(quiet, SR, reference_start=2.1)
    assert 2.0 * SR <= cut <= 2.05 * SR
    assert find_warmup_cut(_tone(1.0), SR, reference_start=0.0) is None


class _FakeRuntime:
    calls = []
    unloaded = 0
    audio = None

    def __init__(self, model_id, allow_downloads, settings=None):
        type(self).calls.append(("init", model_id, allow_downloads, settings))

    def synthesize_voice_design(self, text, *, instruct, language, max_new_tokens):
        type(self).calls.append(("design", text, instruct, language))
        return type(self).audio, SR

    def unload(self):
        type(self).unloaded += 1


@pytest.fixture
def fake_runtime(monkeypatch):
    _FakeRuntime.calls, _FakeRuntime.unloaded = [], 0
    monkeypatch.setattr("abstractvoice.qwen3_tts.runtime.Qwen3TTSRuntime", _FakeRuntime)
    return _FakeRuntime


def _whisper_hearing(seconds, warm, ref, *, clip_opens_with=None):
    """Fake transcriber: the full take has the warm-up at 0 s and the reference at
    `seconds`; the cut clip (shorter than the take) opens with `clip_opens_with`."""
    calls = []

    def transcribe(audio16k, language):
        calls.append(len(audio16k))
        if len(calls) % 2 == 1:
            return [(warm, 0.0), (ref, seconds)]
        return [(clip_opens_with if clip_opens_with is not None else ref, 0.0)]

    transcribe.calls = calls
    return transcribe


def test_render_cuts_where_the_reference_starts_and_keeps_the_settled_voice(fake_runtime):
    from abstractvoice.cloning.design import DESIGN_MODEL_ID, TEXTS, WARMUPS, render_designed_reference

    warm, ref = WARMUPS["fr"], TEXTS["fr"]
    fake_runtime.audio = np.concatenate([_tone(2.0), _gap(0.3), _tone(6.0)])
    settings = object()
    audio, sr, text = render_designed_reference(
        "voix féminine chaleureuse",
        language="fr",
        settings=settings,
        transcriber=_whisper_hearing(2.3, warm, ref),
    )
    assert sr == SR and text == ref
    assert abs(len(audio) / SR - 6.15) < 0.05
    assert audio[0] == 0.0  # faded in at the cut
    init, design = fake_runtime.calls
    assert init == ("init", DESIGN_MODEL_ID, True, settings)
    assert design[1] == f"{warm} {ref}" and design[2] == "voix féminine chaleureuse" and design[3] == "French"
    assert fake_runtime.unloaded == 1


def test_render_gives_up_after_the_attempts_and_still_unloads(fake_runtime):
    from abstractvoice.cloning.design import render_designed_reference

    fake_runtime.audio = _tone(8.0)  # no pause anywhere
    with pytest.raises(RuntimeError, match="warm-up"):
        render_designed_reference("a calm voice", attempts=2, transcriber=_whisper_hearing(2.0, "x", "y"))
    assert sum(1 for c in fake_runtime.calls if c[0] == "design") == 2
    assert fake_runtime.unloaded == 1


def test_render_says_when_the_text_is_too_short(fake_runtime):
    from abstractvoice.cloning.design import render_designed_reference

    fake_runtime.audio = np.concatenate([_tone(2.0), _gap(0.3), _tone(2.0)])
    with pytest.raises(ValueError, match="too short"):
        render_designed_reference("a calm voice", text="Hi there.", attempts=1, transcriber=_whisper_hearing(2.3, "warm up", "Hi there."))


def test_render_retries_when_the_kept_clip_opens_on_the_warmup(fake_runtime):
    from abstractvoice.cloning.design import TEXTS, WARMUPS, render_designed_reference

    warm, ref = WARMUPS["de"], TEXTS["de"]
    fake_runtime.audio = np.concatenate([_tone(2.0), _gap(0.3), _tone(6.0)])
    hearing = _whisper_hearing(2.3, warm, ref, clip_opens_with="es losgeht. " + ref)
    with pytest.raises(RuntimeError, match="warm-up"):
        render_designed_reference("eine tiefe männliche Stimme", language="de", attempts=2, transcriber=hearing)
    assert len(hearing.calls) == 4  # each attempt: the take, then the kept clip


def test_render_rejects_bad_input_before_loading(fake_runtime):
    from abstractvoice.cloning.design import render_designed_reference

    t = lambda a, l: []  # noqa: E731
    with pytest.raises(ValueError):
        render_designed_reference("   ", transcriber=t)
    with pytest.raises(ValueError, match="languages"):
        render_designed_reference("a voice", language="xx", transcriber=t)
    with pytest.raises(ValueError, match="too long"):
        render_designed_reference("a voice", text="word " * 250, transcriber=t)
    assert fake_runtime.calls == []


# ---------------------------------------------------------------------------
# VoiceManager.design_voice wiring
# ---------------------------------------------------------------------------


def _designer(language, **settings):
    from abstractvoice.vm.tts_mixin import TtsMixin

    class _VM(TtsMixin):
        allow_downloads = False

        def __init__(self):
            self.language = language
            self.cloned = None
            for key, value in settings.items():
                setattr(self, key, value)

        def clone_voice_from_wav_bytes(self, wav_bytes, name=None, **kwargs):
            self.cloned = (wav_bytes, name, kwargs)
            return "voice-123"

    return _VM()


@pytest.mark.parametrize("manager_language,asked,expected", [("fr", None, "fr"), ("xx", None, "en"), ("en", "de", "de")])
def test_design_voice_renders_then_clones_with_the_base_model(monkeypatch, manager_language, asked, expected):
    from abstractvoice.cloning.design import CLONING_MODEL_ID, DESIGN_MODEL_ID

    seen = {}

    def fake_render(description, *, language, text, allow_downloads, settings):
        seen.update(language=language, allow_downloads=allow_downloads, settings=settings)
        return _tone(5.0), SR, "the reference text"

    monkeypatch.setattr("abstractvoice.cloning.design.render_designed_reference", fake_render)
    vm = _designer(manager_language, qwen3_tts_backend="torch", qwen3_tts_mlx_quantization="4bit")
    assert vm.design_voice("  a deep calm voice ", name=" Bass ", language=asked) == "voice-123"
    assert seen["language"] == expected and seen["allow_downloads"] is False
    # The manager's Qwen3-TTS settings reach the design run.
    assert seen["settings"].backend == "torch" and seen["settings"].mlx_quantization == "4bit"
    wav_bytes, name, kwargs = vm.cloned
    assert name == "Bass"
    with wave.open(io.BytesIO(wav_bytes)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2 and w.getnframes() == 5 * SR
    assert kwargs["engine"] == "qwen3-tts" and kwargs["model"] == CLONING_MODEL_ID
    assert kwargs["reference_text"] == "the reference text"
    assert kwargs["meta"] == {
        "designed": True,
        "description": "a deep calm voice",
        "language_code": expected,
        "design_model": DESIGN_MODEL_ID,
    }


# ---------------------------------------------------------------------------
# AbstractCore plugin surface
# ---------------------------------------------------------------------------


def _cap(vm):
    from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

    class _Owner:
        config = {"voice_manager_instance": vm}

    return _VoiceCapability(_Owner())


class _DesignVM:
    cloning_engine = "qwen3-tts"
    cloning_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    def __init__(self):
        self.records = {}
        self.events = []

    def render_designed_voice(self, description, *, language=None, text=None):
        self.events.append("render")
        return {"wav_bytes": b"RIFF", "reference_text": "ref", "meta": {"designed": True, "description": description}}

    def store_designed_voice(self, rendered, *, name=None):
        self.events.append("store")
        self.records["d1"] = {"voice_id": "d1", "name": name or "designed", "engine": "qwen3-tts", "meta": rendered["meta"]}
        return "d1"

    def get_cloned_voice(self, voice_id):
        return dict(self.records[voice_id])


def test_plugin_design_voice_renders_outside_the_lock_and_returns_the_record():
    vm = _DesignVM()
    cap = _cap(vm)
    lock = cap._vm_lock(vm)
    original_render = vm.render_designed_voice

    def render_checking_lock(*args, **kwargs):
        assert not lock.locked(), "rendering must not hold the manager lock"
        return original_render(*args, **kwargs)

    vm.render_designed_voice = render_checking_lock
    out = cap.design_voice("a warm narrator", name="Narrator", language="en")
    assert out["ok"] is True and out["voice_id"] == "d1" and out["name"] == "Narrator"
    assert out["meta"]["designed"] is True and vm.events == ["render", "store"]
    with pytest.raises(ValueError):
        cap.design_voice("  ")
    with pytest.raises(TypeError):
        cap.design_voice("a voice", provider="omnivoice")  # not silently ignored


def test_plugin_tags_designed_voices():
    from abstractvoice.integrations.abstractcore_plugin import _clone_profile_tags

    tags = _clone_profile_tags({"meta": {"designed": True, "description": "a warm narrator", "language_code": "en"}})
    assert tags == {"builtin": False, "designed": True, "description": "a warm narrator", "language_code": "en"}
    assert "designed" not in _clone_profile_tags({"meta": {"builtin": True}})


def test_plugin_list_languages_is_static_and_filterable():
    cap = _cap(None)
    cap._get_vm = lambda: pytest.fail("listing languages must not build a VoiceManager")
    cat = cap.list_languages(kind="stt", provider="faster_whisper")
    assert list(cat) == ["stt"] and "en" in cat["stt"]["faster-whisper"]["languages"]
    assert set(cap.list_languages()) == {"tts", "stt", "cloning"}
    assert list(cap.list_languages(provider="piper")) == ["tts"]


def test_plugin_tts_stream_refuses_one_shot_instead_of_ignoring_it():
    cap = _cap(None)
    cap._get_vm = lambda: pytest.fail("the refusal must come before any engine is built")
    with pytest.raises(ValueError, match="one_shot"):
        next(iter(cap.tts_stream("hello", one_shot=True)))


# ---------------------------------------------------------------------------
# Voice navigation: one record shape, language filter
# ---------------------------------------------------------------------------


def _raw_catalog():
    piper_fr = {"provider": "piper", "profile_id": "siwis", "label": "Piper siwis", "params": {"language": "fr", "voice": "siwis"}}
    piper_en = {"provider": "piper", "profile_id": "amy", "label": "Piper amy", "params": {"language": "en", "voice": "amy"}}
    qwen_preset = {"provider": "qwen3-tts", "profile_id": "aiden", "label": "Aiden", "tags": {"kind": "profile"}}
    clone_fr = {
        "voice_id": "fr-female-camille",
        "profile_id": "fr-female-camille",
        "name": "Camille",
        "kind": "clone",
        "provider": "qwen3-tts",
        "tags": {"source": "abstractvoice_clone_store", "language_code": "fr"},
    }
    remote = {"provider": "openai", "profile_id": "alloy", "label": "alloy"}
    voices = [piper_en, piper_fr, qwen_preset, clone_fr, remote]
    return {
        "voices": voices,
        "cloned_voices": [clone_fr],
        "tts_catalog_by_provider": {"piper": {"voices": [piper_en, piper_fr], "voices_by_model": {"m": [piper_fr]}}},
        "tts_languages_by_provider": {"piper": ["en", "fr"]},
    }


def test_every_voice_record_gets_the_same_navigable_fields():
    from abstractvoice.integrations.abstractcore_plugin import _navigable_voice_catalog

    cat = _navigable_voice_catalog(_raw_catalog())
    by_id = {v["voice_id"]: v for v in cat["voices"]}
    assert set(by_id) == {"amy", "siwis", "aiden", "fr-female-camille", "alloy"}
    for v in cat["voices"]:
        assert {"voice_id", "name", "kind", "provider", "model", "language", "languages"} <= set(v)
    assert by_id["siwis"]["languages"] == ["fr"] and by_id["siwis"]["kind"] == "profile"
    assert by_id["aiden"]["language"] == "en" and "fr" in by_id["aiden"]["languages"]
    assert by_id["fr-female-camille"]["kind"] == "clone" and by_id["fr-female-camille"]["language"] == "fr"
    assert by_id["alloy"]["languages"] == [] and by_id["alloy"]["language"] is None
    assert cat["tts_catalog_by_provider"]["piper"]["voices_by_model"]["m"][0]["voice_id"] == "siwis"


def test_language_filter_keeps_voices_that_speak_it_native_first():
    from abstractvoice.integrations.abstractcore_plugin import _navigable_voice_catalog

    cat = _navigable_voice_catalog(_raw_catalog(), language="fr-FR")
    ids = [v["voice_id"] for v in cat["voices"]]
    assert "amy" not in ids  # an English-only Piper voice
    assert ids[:2] == ["siwis", "fr-female-camille"]  # native French first
    assert set(ids[2:]) == {"aiden", "alloy"}  # speak French; a remote provider is not restricted
    assert [v["voice_id"] for v in cat["tts_catalog_by_provider"]["piper"]["voices"]] == ["siwis"]
    assert cat["language"] == "fr"


def test_plugin_voice_catalog_and_listings_apply_navigation(monkeypatch):
    cap = _cap(None)
    monkeypatch.setattr(type(cap), "_voice_catalog", lambda self, **kw: _raw_catalog())
    catalog = cap.voice_catalog()
    assert all("languages" in v for v in catalog["voices"])
    assert [v["voice_id"] for v in cap.list_cloned_voices(language="fr")] == ["fr-female-camille"]
    assert [v["voice_id"] for v in cap.list_tts_voices(provider="piper", language="fr")] == ["siwis"]


def test_voice_catalog_carries_languages_by_provider(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setattr(plugin, "_local_tts_engines", lambda: ["piper"])
    monkeypatch.setattr(plugin, "_local_tts_engine_available", lambda engine, *_: plugin._norm_engine_id(engine) == "piper")
    monkeypatch.setattr(plugin, "_selectable_tts_model_ids_for_provider", lambda provider, *_: ["en_US-amy-medium"])
    cap = _cap(None)
    cap._get_vm = lambda: pytest.fail("a local provider filter must not build a VoiceManager")
    catalog = cap.voice_catalog(provider="piper")
    assert "fr" in catalog["tts_languages_by_provider"]["piper"]
    assert catalog["stt_languages_by_provider"] == {}


def test_render_without_custom_text_does_not_blame_the_text(fake_runtime):
    from abstractvoice.cloning.design import TEXTS, WARMUPS, render_designed_reference

    fake_runtime.audio = np.concatenate([_tone(2.0), _gap(0.3), _tone(2.0)])  # built-in text rendered short
    hearing = _whisper_hearing(2.3, WARMUPS["en"], TEXTS["en"])
    with pytest.raises(RuntimeError):
        render_designed_reference("a calm voice", attempts=1, transcriber=hearing)


def test_design_runs_are_serialized(fake_runtime, monkeypatch):
    import threading

    from abstractvoice.cloning import design

    inside, peak = [0], [0]
    gate = threading.Event()

    def slow_render(*args, **kwargs):
        inside[0] += 1
        peak[0] = max(peak[0], inside[0])
        gate.wait(0.2)
        inside[0] -= 1
        return np.zeros(1), SR, "x"

    monkeypatch.setattr(design, "_render", slow_render)
    threads = [threading.Thread(target=design.render_designed_reference, args=("a voice",)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_language_filter_also_narrows_the_id_maps_and_keeps_native_speakers():
    from abstractvoice.integrations.abstractcore_plugin import _navigable_voice_catalog

    raw = _raw_catalog()
    raw["tts_voices_by_provider"] = {"piper": ["amy", "siwis"], "qwen3-tts": ["aiden", "fr-female-camille"]}
    # A clone of an en/zh-only engine whose own language is French still speaks French.
    raw["voices"].append({"voice_id": "c-fr", "kind": "clone", "provider": "audiodit",
                          "tags": {"source": "abstractvoice_clone_store", "language_code": "fr"}})
    cat = _navigable_voice_catalog(raw, language="fr")
    assert cat["tts_voices_by_provider"]["piper"] == ["siwis"]
    assert "c-fr" in [v["voice_id"] for v in cat["voices"]]
