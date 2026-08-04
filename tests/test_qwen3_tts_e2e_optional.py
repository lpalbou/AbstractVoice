"""Qwen3-TTS end-to-end proofs on the real checkpoints.

Gated: set ``ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1`` and prefetch the snapshots
(`abstractvoice-prefetch --qwen3-tts`, and ``--qwen3-tts
Qwen/Qwen3-TTS-12Hz-0.6B-Base`` for cloning). Marked ``model_download`` so CI's
default run skips them.
"""

from __future__ import annotations

import io
import os
import wave

import numpy as np
import pytest

pytestmark = [
    pytest.mark.model_download,
    pytest.mark.skipif(
        os.environ.get("ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS", "0") != "1",
        reason="set ABSTRACTVOICE_RUN_QWEN3_TTS_TESTS=1 to run Qwen3-TTS end-to-end tests",
    ),
]

CUSTOM = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


@pytest.fixture(scope="module")
def custom_adapter():
    from abstractvoice.adapters.tts_registry import create_tts_adapter

    adapter, resolved = create_tts_adapter(
        engine="qwen3-tts", language="en", allow_downloads=False, auto_load=False, model_id=CUSTOM
    )
    assert resolved == "qwen3-tts"
    return adapter


def test_custom_voice_speaks(custom_adapter):
    data = custom_adapter.synthesize_to_bytes("The Qwen integration is speaking through AbstractVoice.")
    assert data[:4] == b"RIFF"
    assert _wav_seconds(data) > 1.0


def test_every_preset_speaker_is_selectable(custom_adapter):
    profiles = custom_adapter.get_profiles()
    assert len(profiles) == 9
    assert custom_adapter.set_profile(profiles[-1].profile_id) is True
    data = custom_adapter.synthesize_to_bytes("A different preset speaker.")
    assert _wav_seconds(data) > 0.5


def test_voice_manager_path_speaks():
    from abstractvoice import VoiceManager

    vm = VoiceManager(tts_engine="qwen3-tts", language="en", allow_downloads=False, tts_model=CUSTOM)
    data = vm.tts_adapter.synthesize_to_bytes("Spoken through the public VoiceManager surface.")
    assert _wav_seconds(data) > 1.0


def test_cloning_end_to_end(tmp_path):
    """Reference audio synthesized by CustomVoice, cloned by Base (ICL)."""
    from abstractvoice.adapters.tts_registry import create_tts_adapter
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    reference_text = "This reference voice belongs to the abstract framework test suite."
    ref_adapter, _ = create_tts_adapter(
        engine="qwen3-tts", language="en", allow_downloads=False, auto_load=False, model_id=CUSTOM
    )
    ref_path = tmp_path / "reference.wav"
    ref_path.write_bytes(ref_adapter.synthesize_to_bytes(reference_text))
    ref_adapter.unload()

    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False, model_id=BASE)
    data = engine.infer_to_wav_bytes(
        text="The cloned voice now says something entirely different.",
        reference_paths=[str(ref_path)],
        reference_text=reference_text,
    )
    assert data[:4] == b"RIFF"
    assert _wav_seconds(data) > 1.0
    engine.unload()


def test_x_vector_only_cloning_is_an_explicit_choice(tmp_path):
    """No transcript, caller opts in explicitly (ADR 0003: never inferred)."""
    from abstractvoice.adapters.tts_registry import create_tts_adapter
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    ref_adapter, _ = create_tts_adapter(
        engine="qwen3-tts", language="en", allow_downloads=False, auto_load=False, model_id=CUSTOM
    )
    ref_path = tmp_path / "reference.wav"
    ref_path.write_bytes(ref_adapter.synthesize_to_bytes("A short reference for the embedding-only mode."))
    ref_adapter.unload()

    runtime = Qwen3TTSRuntime(model_id=BASE, allow_downloads=False)
    with pytest.raises(ValueError):
        # ICL without a transcript must refuse, not degrade.
        runtime.build_clone_prompt(ref_audio=str(ref_path), ref_text=None, x_vector_only=False)

    prompt = runtime.build_clone_prompt(ref_audio=str(ref_path), ref_text=None, x_vector_only=True)
    audio, sr = runtime.synthesize_clone("Embedding only cloning, chosen on purpose.", clone_prompt=prompt)
    assert audio.shape[0] > sr  # > 1 second
    runtime.unload()
