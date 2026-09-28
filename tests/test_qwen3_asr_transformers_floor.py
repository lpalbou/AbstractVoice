"""Qwen3-ASR needs Transformers 5.4+; older releases get a plain error.

The `stt-hf` extra resolves Transformers 4.x on Python 3.9, where the vendored
model failed deep inside generation (`create_causal_mask() got an unexpected
keyword argument 'inputs_embeds'`). The loader now refuses before any weights
load, and that reason is what discovery reports -- not "model not downloaded".
"""

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from abstractvoice.adapters import stt_transformers_asr as asr


def _message(version: str) -> str:
    with pytest.raises(RuntimeError) as err:
        asr._require_qwen3_asr_transformers(version)
    return str(err.value)


def test_transformers_4_is_refused_with_a_plain_message():
    message = _message("4.57.6")
    assert "Qwen3-ASR needs Transformers 5.4.0 or newer (Python 3.10+)" in message
    assert "transformers 4.57.6 is installed" in message
    assert 'pip install -U "transformers>=5.4.0"' in message


def test_transformers_5_below_the_floor_is_refused():
    assert "transformers 5.3.0 is installed" in _message("5.3.0")


@pytest.mark.parametrize("version", ["5.4.0", "5.8.1", "5.17.0", "6.0.0"])
def test_supported_transformers_pass(version):
    asr._require_qwen3_asr_transformers(version)


def test_loader_refuses_before_loading_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    fake_tf = types.ModuleType("transformers")
    fake_tf.__version__ = "4.57.6"
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setattr(asr.TransformersASRAdapter, "_ensure_loaded", lambda self: None)

    adapter = asr.TransformersASRAdapter(model_id="Qwen/Qwen3-ASR-0.6B", device="cpu", allow_downloads=False)
    with pytest.raises(RuntimeError, match="Qwen3-ASR needs Transformers 5.4.0 or newer"):
        adapter._ensure_loaded_qwen3_asr(torch_device="cpu", torch_dtype=None, local_only=True)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="needs abstractvoice[stt-hf] (torch + transformers)",
)
def test_unavailable_reason_names_the_transformers_floor(monkeypatch):
    """Offline, any load error used to become "not available locally"; this one
    must stay the Transformers reason, because a download would not fix it."""
    from transformers import pipeline  # noqa: F401 -- transformers swaps its module object on first use

    monkeypatch.setattr(sys.modules["transformers"], "__version__", "4.57.6")
    adapter = asr.TransformersASRAdapter(model_id="Qwen/Qwen3-ASR-0.6B", device="cpu", allow_downloads=False)
    assert not adapter.is_available()
    reason = adapter.get_unavailable_reason() or ""
    assert "Qwen3-ASR needs Transformers 5.4.0 or newer" in reason
    assert "not available locally" not in reason
