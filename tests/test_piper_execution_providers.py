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


def _wrapper_fallback_if_no_device(providers):
    """What ONNX Runtime's Python wrapper does when CUDA is requested without a device: an "EP Error"
    banner on STDOUT, then the CPU. The engines must never get here."""
    torch = sys.modules.get("torch")
    if CUDA in providers and not (torch is not None and torch.cuda.is_available()):
        print("*************** EP Error ***************\nEP Error cudaErrorNoDevice when using [CUDA]")
        return [CPU]
    return None


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
            self._providers = _wrapper_fallback_if_no_device(self.requested) or (got + [CPU])  # CPU always kept
            ort.sessions.append(self)

        def get_providers(self):
            return list(self._providers)

    def preload_dlls(cuda=True, cudnn=True, msvc=True, directory=None):
        ort.preload_calls += 1

    ort.get_available_providers = lambda: list(available)
    ort.InferenceSession = InferenceSession
    ort.preload_dlls = preload_dlls
    return ort


@pytest.fixture(autouse=True)
def cuda_device(monkeypatch):
    """A fresh process (CUDA libraries never preloaded, device never asked) with a usable CUDA
    device as torch sees it; `cuda_device(False)` hides it (CUDA_VISIBLE_DEVICES= , no driver)."""
    from abstractvoice.compute import onnx_cuda

    monkeypatch.setattr(onnx_cuda, "_preload_done", False)
    monkeypatch.setattr(onnx_cuda, "_preload_note", None)
    onnx_cuda.usable_cuda_device.cache_clear()

    def set_device(present: bool = True):
        torch = types.ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: present)
        monkeypatch.setitem(sys.modules, "torch", torch)
        onnx_cuda.usable_cuda_device.cache_clear()

    set_device(True)
    yield set_device
    onnx_cuda.usable_cuda_device.cache_clear()


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


def test_each_voice_load_does_not_preload_again(piper_adapter) -> None:
    ort = _fake_ort([CUDA, CPU])
    adapter = piper_adapter(ort)

    assert adapter._load_voice("en") is True
    assert adapter._load_voice("en") is True  # a language switch reloads the voice

    assert ort.preload_calls == 1
    assert [s.requested for s in ort.sessions] == [[CUDA], [CUDA]]


@pytest.mark.parametrize("forced", [None, "1"])
def test_provider_listed_but_no_device_runs_on_the_cpu_and_keeps_stdout_clean(
    piper_adapter, cuda_device, monkeypatch, capsys, forced
) -> None:
    ort = _fake_ort([CUDA, CPU])
    adapter = piper_adapter(ort)
    cuda_device(False)
    if forced:
        monkeypatch.setenv("ABSTRACTVOICE_PIPER_USE_CUDA", forced)

    assert adapter._load_voice("en") is True

    assert capsys.readouterr().out == ""  # the wrapper's fallback banner never ran
    assert ort.sessions[0].requested == [CPU]
    info = adapter.get_info()
    assert info["use_cuda"] is False
    assert info["execution_providers"] == [CPU]
    assert info["execution_reason"] == (
        "CPU: ONNX Runtime has the CUDA provider but no usable CUDA device: torch.cuda.is_available() is False"
    )


def test_provider_listed_and_a_device_runs_on_cuda(piper_adapter, capsys) -> None:
    ort = _fake_ort([CUDA, CPU])
    adapter = piper_adapter(ort)

    assert adapter._load_voice("en") is True

    assert capsys.readouterr().out == ""
    assert ort.sessions[0].requested == [CUDA]
    assert adapter.get_info()["execution_reason"] is None
