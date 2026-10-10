"""Round 18 — the spoken-language setting at its source (AbstractVoice).

ONE list of supported languages (`abstractvoice.stt.languages`) advertised by every local
adapter; `language=None` is AUTO on every engine (never a manager's TTS language in disguise);
the engine's own report of the language reaches the caller (`Transcription.detected_language`)
through the adapters and the AbstractCore plugin (which reads it off the manager's adapter).

Each test goes red when its seam is deleted: the shared list (adapters listing their own codes),
`_note_detected_language` in an adapter, the `language=None` construction in the mixin, the
`transcribe_detailed` method of the plugin.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Optional

import numpy as np
import pytest

from abstractvoice.adapters.base import STTAdapter, Transcription
from abstractvoice.stt import languages as L


# ----------------------------------------------------------------------------- the list


def test_normalize_language_auto_codes_and_refusal():
    assert L.normalize_language(None) is None
    assert L.normalize_language("") is None
    assert L.normalize_language("auto") is None
    assert L.normalize_language("AUTO ") is None
    assert L.normalize_language("fr") == "fr"
    assert L.normalize_language("FR") == "fr"
    assert L.normalize_language("fr-FR") == "fr"
    assert L.normalize_language("en_US") == "en"
    with pytest.raises(ValueError) as exc:
        L.normalize_language("xx")
    sentence = str(exc.value)
    assert sentence.startswith("spoken_language = 'xx' refused: not a language the speech engines support. Choose auto or one of: ")
    assert sentence.endswith(", ".join(sorted(L.SUPPORTED_LANGUAGES)) + ".")
    with pytest.raises(ValueError):
        L.normalize_language(42)


def test_choices_start_with_auto_then_languages_by_label():
    rows = L.choices()
    assert rows[0] == {"value": "auto", "label": "Auto (detected)"}
    labels = [r["label"] for r in rows[1:]]
    assert labels == sorted(labels)
    assert {r["value"] for r in rows[1:]} == set(L.SUPPORTED_LANGUAGES)
    assert L.language_label(None) == "Auto (detected)" == L.language_label("auto")
    assert L.language_label("fr") == "French"
    assert all(code in L.LANGUAGE_LABELS for code in L.SUPPORTED_LANGUAGES)


def test_every_local_adapter_advertises_the_one_list():
    from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter
    from abstractvoice.adapters.stt_mlx_whisper import MLXWhisperAdapter
    from abstractvoice.adapters.stt_transformers_asr import TransformersASRAdapter

    for adapter in (FasterWhisperAdapter, MLXWhisperAdapter, TransformersASRAdapter):
        assert list(adapter.LANGUAGES) == list(L.SUPPORTED_LANGUAGES), adapter.__name__
    assert "fr" in L.SUPPORTED_LANGUAGES and "en" in L.SUPPORTED_LANGUAGES


# ------------------------------------------------------------------ the adapter contract


class _Fake(STTAdapter):
    """A minimal adapter: records what it was asked, reports a detected language when told to."""

    def __init__(self, detected: Optional[str] = None) -> None:
        self.detected = detected
        self.calls: list = []

    def transcribe(self, audio_path: str, language: Optional[str] = None, **kwargs: Any) -> str:
        self.calls.append(("file", audio_path, language))
        if self.detected:
            self._note_detected_language(self.detected)
        return "bonjour"

    def transcribe_from_bytes(self, audio_bytes: bytes, language: Optional[str] = None, **kwargs: Any) -> str:
        self.calls.append(("bytes", len(audio_bytes), language))
        if self.detected:
            self._note_detected_language(self.detected)
        return "hello"

    def transcribe_from_array(self, audio_array: np.ndarray, sample_rate: int, language: Optional[str] = None) -> str:
        return ""

    def set_language(self, language: str) -> bool:
        return True

    def get_supported_languages(self) -> list[str]:
        return list(L.SUPPORTED_LANGUAGES)

    def is_available(self) -> bool:
        return True


def test_transcribe_detailed_carries_the_told_and_the_reported_language():
    a = _Fake(detected="FR")
    out = a.transcribe_detailed("/tmp/clip.wav", language=None)
    assert out == Transcription(text="bonjour", language=None, detected_language="fr")
    assert a.pop_detected_language() is None, "the report is cleared on read"
    out = a.transcribe_from_bytes_detailed(b"abc", language="en")
    assert out == Transcription(text="hello", language="en", detected_language="fr")
    quiet = _Fake()
    assert quiet.transcribe_detailed("/tmp/clip.wav").detected_language is None


def test_mixin_never_seeds_local_or_remote_stt_adapters_with_the_tts_language(monkeypatch):
    """The manager's `language` is the voice's (TTS). Before round 18 the mlx-whisper and the
    OpenAI-compatible adapters were built with it, so auto was silently a fixed language."""
    from abstractvoice.vm.stt_mixin import SttMixin
    import abstractvoice.adapters.stt_mlx_whisper as mlx_mod
    import abstractvoice.adapters.stt_openai_compatible as oc_mod

    seen: dict = {}

    class _MLX:
        DEFAULT_MODEL = "large-v3"

        def __init__(self, **kwargs):
            seen["mlx"] = kwargs

        def is_available(self):
            return True

    class _OC:
        def __init__(self, **kwargs):
            seen["oc"] = kwargs

        def is_available(self):
            return True

    monkeypatch.setattr(mlx_mod, "MLXWhisperAdapter", _MLX)
    monkeypatch.setattr(oc_mod, "OpenAICompatibleSTTAdapter", _OC)
    monkeypatch.setattr("abstractvoice.vm.stt_mixin.remote_endpoint_kwargs", lambda vm, provider: {"base_url": "http://x", "api_key": "k"})

    class _VM(SttMixin):
        def __init__(self, pref: str) -> None:
            self.language = "en"
            self.stt_adapter = None
            self._stt_engine_preference = pref
            self.stt_model = None
            self.whisper_model = "base"
            self.allow_downloads = False
            self.debug_mode = False

    _VM("mlx_whisper")._get_stt_adapter()
    assert "language" in seen["mlx"] and seen["mlx"]["language"] is None
    _VM("openai")._get_stt_adapter()
    assert "language" in seen["oc"] and seen["oc"]["language"] is None


# ------------------------------------------------------------------------- the engines


def test_faster_whisper_notes_the_language_the_engine_reports(monkeypatch, tmp_path):
    from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter

    a = FasterWhisperAdapter.__new__(FasterWhisperAdapter)
    a._faster_whisper_available = True
    a._model = object()
    a._model_size = "tiny"
    a._current_language = None
    seg = types.SimpleNamespace(text=" bonjour tout le monde ")
    info = types.SimpleNamespace(language="fr", language_probability=0.97)
    asked: dict = {}

    def _run(audio, **kwargs):
        asked.update(kwargs)
        return [seg], info

    monkeypatch.setattr(a, "_run", _run)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    out = a.transcribe_detailed(str(clip))
    assert out == Transcription(text="bonjour tout le monde", language=None, detected_language="fr")
    assert asked["language"] is None, "auto reaches faster-whisper as None (its own detection)"
    out = a.transcribe_detailed(str(clip), language="en")
    assert asked["language"] == "en" and out.language == "en"
    out = a.transcribe_from_array(np.zeros(16000, dtype=np.float32), 16000)
    assert out == "bonjour tout le monde" and a.pop_detected_language() == "fr"


def test_mlx_whisper_auto_is_none_and_the_report_is_noted(monkeypatch):
    from abstractvoice.adapters.stt_mlx_whisper import MLXWhisperAdapter

    a = MLXWhisperAdapter.__new__(MLXWhisperAdapter)
    a._current_language = None
    a._loaded_path = "/models/whisper"
    a._unavailable_reason = None
    monkeypatch.setattr(a, "_ensure_loaded", lambda: True)
    asked: dict = {}
    fake = types.ModuleType("mlx_whisper")

    def transcribe(audio, **kwargs):
        asked.update(kwargs)
        return {"text": " salut ", "language": "fr"}

    fake.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    out = a.transcribe_from_array(np.zeros(16000, dtype=np.float32), 16000)
    assert out == "salut" and asked["language"] is None
    assert a.pop_detected_language() == "fr"
    a.set_language("de")
    a.transcribe_from_array(np.zeros(16000, dtype=np.float32), 16000)
    assert asked["language"] == "de", "only an explicit set_language installs a default"


def test_openai_compatible_language_comes_from_a_verbose_answer_only():
    from abstractvoice.adapters.openai_compatible_http import extract_transcription_language

    class _R:
        def __init__(self, payload, ctype="application/json"):
            self.headers = {"content-type": ctype}
            self._payload = payload

        def json(self):
            return self._payload

    assert extract_transcription_language(_R({"text": "hi", "language": "English"})) == "english"
    assert extract_transcription_language(_R({"text": "hi"})) is None
    assert extract_transcription_language(_R({"data": {"text": "hi", "language": "fr"}})) == "fr"
    assert extract_transcription_language(_R({"text": "hi"}, ctype="text/plain")) is None


# -------------------------------------------------------------------------- the plugin


def test_plugin_transcribe_detailed_returns_the_three_facts(monkeypatch, tmp_path):
    from abstractvoice.integrations.abstractcore_plugin import _AudioCapability

    class _Owner:
        config: dict = {}

    cap = _AudioCapability(_Owner())

    class _Adapter:
        def __init__(self):
            self.report = None

        def pop_detected_language(self):
            out, self.report = self.report, None
            return out

    class _VM:
        def __init__(self):
            self.calls = []
            self.stt_adapter = _Adapter()

        def transcribe_file(self, path, language=None):
            self.calls.append(("file", language))
            self.stt_adapter.report = "fr"
            return "bonjour"

        def transcribe_from_bytes(self, data, language=None):
            self.calls.append(("bytes", language))
            return "hello"

    vm = _VM()
    monkeypatch.setattr(cap, "_get_vm_for_provider", lambda **kwargs: vm)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    out = cap.transcribe_detailed(str(clip))
    assert out == {"text": "bonjour", "language": None, "detected_language": "fr"}
    assert vm.calls[-1] == ("file", None)
    assert cap.transcribe(str(clip), language="fr") == "bonjour" and vm.calls[-1] == ("file", "fr")
    out = cap.transcribe_detailed(b"RIFF....", language="en")
    assert out == {"text": "hello", "language": "en", "detected_language": None}
