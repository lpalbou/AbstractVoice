"""Piper on an NVIDIA GPU: the pip CUDA libraries are preloaded before a CUDA session, and
the info reports the providers the session ACTUALLY runs on (ONNX Runtime falls back to the
CPU, with a log line only, when the CUDA provider cannot load).

ONNX Runtime and piper are faked: these run anywhere, without onnxruntime-gpu or a GPU."""
from __future__ import annotations

import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def _fake_ort(available, *, loadable=None):
    ort = types.ModuleType("onnxruntime")
    ort.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
    ort.preload_calls = 0
    ort.sessions = []
    loads = set(available if loadable is None else loadable)

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            # Piper asks for [("CUDAExecutionProvider", {options})] or ["CPUExecutionProvider"].
            self.requested = [p[0] if isinstance(p, tuple) else p for p in providers or []]
            got = [p for p in self.requested if p in loads and p != CPU]
            self._providers = got + [CPU]  # ONNX Runtime always keeps the CPU provider
            ort.sessions.append(self)

        def get_providers(self):
            return list(self._providers)

    def preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None):
        ort.preload_calls += 1

    ort.get_available_providers = lambda: list(available)
    ort.InferenceSession = InferenceSession
    ort.preload_dlls = preload_dlls
    return ort


@pytest.fixture
def piper_adapter(monkeypatch, tmp_path):
    """A PiperTTSAdapter on a fake piper whose voices open sessions on the installed fake ONNX Runtime."""
    monkeypatch.delenv("ABSTRACTVOICE_PIPER_USE_CUDA", raising=False)

    class PiperVoice:
        @staticmethod
        def load(model_path, config_path, use_cuda=False, espeak_data_dir=None):
            providers = [(CUDA, {"cudnn_conv_algo_search": "HEURISTIC"})] if use_cuda else [CPU]
            session = sys.modules["onnxruntime"].InferenceSession(model_path, providers=providers)
            return SimpleNamespace(session=session, config=SimpleNamespace(sample_rate=22050))

    fake_piper = types.ModuleType("piper")
    fake_piper.PiperVoice = PiperVoice
    monkeypatch.setitem(sys.modules, "piper", fake_piper)

    def make(ort):
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter

        monkeypatch.setitem(sys.modules, "onnxruntime", ort)
        adapter = PiperTTSAdapter(language="en", model_dir=str(tmp_path / "models"), auto_load=False)
        assert adapter._piper_available
        model_path, config_path = adapter._get_model_path("en")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"onnx")
        config_path.write_text("{}")
        monkeypatch.setattr(adapter, "_espeak_load_kwargs", lambda: {})
        return adapter

    return make


def test_cuda_available_preloads_and_reports_the_cuda_session(piper_adapter) -> None:
    ort = _fake_ort([CUDA, CPU])
    adapter = piper_adapter(ort)
    assert adapter.get_info()["execution_providers"] is None  # nothing loaded yet

    assert adapter._load_voice("en") is True

    assert ort.preload_calls == 1
    assert ort.sessions[0].requested == [CUDA]
    info = adapter.get_info()
    assert info["use_cuda"] is True
    assert info["execution_providers"] == [CUDA, CPU]


def test_a_cuda_provider_that_fails_to_load_is_reported_as_the_cpu(piper_adapter) -> None:
    # onnxruntime-gpu without its CUDA libraries: CUDA requested, the session got the CPU.
    ort = _fake_ort([CUDA, CPU], loadable=[CPU])
    adapter = piper_adapter(ort)

    assert adapter._load_voice("en") is True

    info = adapter.get_info()
    assert info["use_cuda"] is True  # what was requested ...
    assert info["execution_providers"] == [CPU]  # ... and what runs


def test_cuda_absent_runs_on_the_cpu_without_preloading(piper_adapter) -> None:
    ort = _fake_ort(["AzureExecutionProvider", CPU])
    adapter = piper_adapter(ort)

    assert adapter._load_voice("en") is True

    assert ort.preload_calls == 0
    assert ort.sessions[0].requested == [CPU]
    assert adapter.get_info()["execution_providers"] == [CPU]


def test_cuda_switched_off_by_the_environment(piper_adapter, monkeypatch) -> None:
    ort = _fake_ort([CUDA, CPU])
    adapter = piper_adapter(ort)
    monkeypatch.setenv("ABSTRACTVOICE_PIPER_USE_CUDA", "0")

    assert adapter._load_voice("en") is True

    assert ort.preload_calls == 0
    assert ort.sessions[0].requested == [CPU]
    assert adapter.get_info()["execution_providers"] == [CPU]
