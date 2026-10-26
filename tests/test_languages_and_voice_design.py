"""Language catalog and description-designed voices, without model weights."""
import io
import subprocess
import sys
import wave

import numpy as np
import pytest

from abstractvoice.languages import language_catalog, language_entry, languages_by_provider


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
    assert len(cat["tts"]["omnivoice"]["languages"]) > 100


def test_remote_engines_are_open_and_undeclared_lists_say_so():
    assert language_entry("tts", "openai") == {"provider": "openai", "languages": [], "open": True, "declared": True}
    chroma = language_entry("cloning", "chroma")
    assert chroma["declared"] is False and chroma["languages"] == []


def test_provider_names_are_normalized_and_filters_apply():
    cat = language_catalog(kind="tts", provider="Qwen3_TTS")
    assert list(cat) == ["tts"] and list(cat["tts"]) == ["qwen3-tts"]


def test_unknown_provider_and_kind_fail_loudly():
    with pytest.raises(KeyError, match="known: supertonic"):
        language_catalog(kind="tts", provider="nope")
    with pytest.raises(ValueError, match="kind must be one of"):
        language_catalog(kind="music")
    with pytest.raises(KeyError):
        language_entry("stt", "nope")


def test_languages_by_provider_leaves_out_engines_it_does_not_know():
    out = languages_by_provider("tts", ["piper", "third-party"])
    assert list(out) == ["piper"] and "en" in out["piper"]


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


def _speech_with_pause(warmup_s=1.5, pause_s=0.3, rest_s=5.0):
    return np.concatenate([_tone(warmup_s), np.zeros(int(pause_s * SR), np.float32), _tone(rest_s)])


def test_warmup_cut_lands_in_the_pause_nearest_the_warmup_share():
    audio = _speech_with_pause()
    share = 1.65 / (len(audio) / SR)
    cut = find_cut(audio, share)
    assert 1.5 * SR <= cut <= 1.8 * SR


def test_warmup_cut_is_none_without_a_pause_or_with_a_far_pause():
    assert find_cut(_tone(6.0), 0.25) is None
    # The only pause sits at ~83% of the clip, far from a 20% warm-up share.
    audio = np.concatenate([_tone(5.0), np.zeros(int(0.3 * SR), np.float32), _tone(1.0)])
    assert find_cut(audio, 0.2) is None


def find_cut(audio, share):
    from abstractvoice.cloning.design import find_warmup_cut

    return find_warmup_cut(audio, SR, warmup_share=share)


class _FakeRuntime:
    calls = []
    unloaded = 0
    audio = None

    def __init__(self, model_id, allow_downloads):
        type(self).calls.append(("init", model_id, allow_downloads))

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


def test_render_cuts_the_warmup_and_keeps_the_settled_voice(fake_runtime):
    from abstractvoice.cloning.design import DESIGN_MODEL_ID, TEXTS, WARMUPS, render_designed_reference

    warm, ref = WARMUPS["fr"], TEXTS["fr"]
    total = 2.0 / (len(warm) / (len(warm) + 1 + len(ref)))  # warm-up spoken in 2 s
    fake_runtime.audio = _speech_with_pause(2.0, 0.3, total - 2.3)
    audio, sr, text = render_designed_reference("voix féminine chaleureuse", language="fr")
    assert sr == SR and text == ref
    assert abs(len(audio) / SR - (len(fake_runtime.audio) / SR - 2.15)) < 0.2
    assert audio[0] == 0.0  # faded in at the cut
    init, design = fake_runtime.calls
    assert init == ("init", DESIGN_MODEL_ID, True)
    assert design[1] == f"{warm} {ref}" and design[2] == "voix féminine chaleureuse" and design[3] == "French"
    assert fake_runtime.unloaded == 1


def test_render_gives_up_after_the_attempts_and_still_unloads(fake_runtime):
    from abstractvoice.cloning.design import render_designed_reference

    fake_runtime.audio = _tone(8.0)
    with pytest.raises(RuntimeError, match="warm-up"):
        render_designed_reference("a calm voice", attempts=2)
    assert sum(1 for c in fake_runtime.calls if c[0] == "design") == 2
    assert fake_runtime.unloaded == 1


def test_render_rejects_bad_input_before_loading(fake_runtime):
    from abstractvoice.cloning.design import render_designed_reference

    with pytest.raises(ValueError):
        render_designed_reference("   ")
    with pytest.raises(ValueError, match="languages"):
        render_designed_reference("a voice", language="xx")
    assert fake_runtime.calls == []


# ---------------------------------------------------------------------------
# VoiceManager.design_voice wiring
# ---------------------------------------------------------------------------


def _designer(language):
    from abstractvoice.vm.tts_mixin import TtsMixin

    class _VM(TtsMixin):
        allow_downloads = False

        def __init__(self):
            self.language = language
            self.cloned = None

        def clone_voice_from_wav_bytes(self, wav_bytes, name=None, **kwargs):
            self.cloned = (wav_bytes, name, kwargs)
            return "voice-123"

    return _VM()


@pytest.mark.parametrize("manager_language,asked,expected", [("fr", None, "fr"), ("xx", None, "en"), ("en", "de", "de")])
def test_design_voice_renders_then_clones_with_the_base_model(monkeypatch, manager_language, asked, expected):
    from abstractvoice.cloning.design import CLONING_MODEL_ID, DESIGN_MODEL_ID

    seen = {}

    def fake_render(description, *, language, text, allow_downloads):
        seen.update(description=description, language=language, text=text, allow_downloads=allow_downloads)
        return _tone(5.0), SR, "the reference text"

    monkeypatch.setattr("abstractvoice.cloning.design.render_designed_reference", fake_render)
    vm = _designer(manager_language)
    assert vm.design_voice("  a deep calm voice ", name=" Bass ", language=asked) == "voice-123"
    assert seen["language"] == expected and seen["allow_downloads"] is False
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

    def design_voice(self, description, *, name=None, language=None, text=None):
        self.records["d1"] = {
            "voice_id": "d1",
            "name": name or "designed",
            "engine": "qwen3-tts",
            "meta": {"designed": True, "description": description, "language_code": language or "en"},
        }
        return "d1"

    def get_cloned_voice(self, voice_id):
        return dict(self.records[voice_id])


def test_plugin_design_voice_returns_the_stored_record():
    cap = _cap(_DesignVM())
    out = cap.design_voice("a warm narrator", name="Narrator", language="en")
    assert out["ok"] is True and out["voice_id"] == "d1" and out["name"] == "Narrator"
    assert out["meta"]["designed"] is True
    with pytest.raises(ValueError):
        cap.design_voice("  ")


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
