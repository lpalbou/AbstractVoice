"""Qwen3-TTS integration: everything provable without downloading weights.

The real-model end-to-end proofs live in ``tests/test_qwen3_tts_e2e_optional.py``
(env-gated). These tests pin the seams: the transformers-compat shims, the
vendored construction path, the mel/resample DSP, the adapter contract, the
cloning-engine contract (ADR 0003), and plugin discovery surfacing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
needs_torch = pytest.mark.skipif(not (HAS_TORCH and HAS_TRANSFORMERS), reason="torch+transformers required")


# --------------------------------------------------------------------- compat


def test_hf_compat_rope_supports_default_key():
    pytest.importorskip("transformers")
    from abstractvoice._hf_compat import rope_init_fn

    assert callable(rope_init_fn("default"))
    with pytest.raises(KeyError):
        rope_init_fn("no-such-rope")


@needs_torch
def test_hf_compat_default_rope_matches_the_formula():
    import torch

    from abstractvoice._hf_compat import _default_rope_init

    class _Cfg:
        rope_theta = 10000.0
        hidden_size = 64
        num_attention_heads = 4

    inv_freq, scale = _default_rope_init(_Cfg())
    dim = 16
    expected = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    assert scale == 1.0
    assert torch.allclose(inv_freq, expected)


def test_hf_compat_check_model_inputs_accepts_both_shapes():
    pytest.importorskip("transformers")
    from abstractvoice._hf_compat import check_model_inputs

    def probe(self, x):
        return x

    assert callable(check_model_inputs(probe))      # 5.x plain decorator
    assert callable(check_model_inputs()(probe))    # 4.57 factory form


def test_hf_compat_pad_token_id_reads_attribute_or_none():
    from abstractvoice._hf_compat import pad_token_id_of

    class _With:
        pad_token_id = 7

    class _Without:
        pass

    assert pad_token_id_of(_With()) == 7
    assert pad_token_id_of(_Without()) is None


@needs_torch
def test_qwen3_asr_vendored_package_imports_and_registers():
    """The pre-existing vendored family must never rot silently again."""
    import abstractvoice.qwen3_asr.modeling_qwen3_asr  # noqa: F401
    from abstractvoice.qwen3_asr import register_transformers_qwen3_asr

    register_transformers_qwen3_asr()
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "qwen3_asr" in CONFIG_MAPPING


# ----------------------------------------------------------- vendored construct


@needs_torch
def test_vendored_qwen3_tts_constructs_on_a_tiny_config():
    """Import success is not load success: constructing exercises the rope
    'default' key, pad_token_id reads, and the tied-weights mapping."""
    from abstractvoice.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from abstractvoice.qwen3_tts.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    cfg = Qwen3TTSConfig(
        talker_config=dict(
            vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, num_code_groups=4,
            text_vocab_size=512, text_hidden_size=64, spk_id={"aiden": 1, "serena": 2},
            codec_language_id={"english": 0, "chinese": 1},
            rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "interleaved": True},
            code_predictor_config=dict(
                vocab_size=512, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            ),
        ),
        speaker_encoder_config=dict(
            enc_dim=32, mel_dim=16, enc_channels=[8, 8, 8, 8, 24],
            enc_kernel_sizes=[5, 3, 3, 3, 1], enc_dilations=[1, 2, 3, 4, 1],
            enc_attention_channels=8, enc_res2net_scale=2, enc_se_channels=8,
        ),
        tts_model_type="custom_voice", tts_model_size="0b6",
        tokenizer_type="qwen3_tts_tokenizer_12hz",
    )
    model = Qwen3TTSForConditionalGeneration(cfg)
    assert sum(p.numel() for p in model.parameters()) > 0
    assert sorted(model.get_supported_speakers()) == ["aiden", "serena"]
    assert len(model.state_dict()) > 0


@needs_torch
def test_vendored_import_emits_no_error_lines(capsys):
    for mod in list(sys.modules):
        if mod.startswith("abstractvoice.qwen3_tts"):
            del sys.modules[mod]
    import abstractvoice.qwen3_tts.modeling_qwen3_tts  # noqa: F401
    import abstractvoice.qwen3_tts.modeling_qwen3_tts_tokenizer_v2  # noqa: F401

    out = capsys.readouterr()
    assert "[ERROR]" not in out.out and "[ERROR]" not in out.err


# ------------------------------------------------------------------------ DSP


def test_mel_adapter_matches_librosa_bit_for_bit_enough():
    librosa = pytest.importorskip("librosa")
    pytest.importorskip("transformers")
    from abstractvoice.qwen3_tts._mel import librosa_mel_fn

    ours = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
    theirs = librosa.filters.mel(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
    assert ours.shape == theirs.shape == (128, 513)
    assert float(np.abs(ours - theirs).max()) < 1e-6


def test_mel_adapter_defaults_fmax_to_half_sr():
    pytest.importorskip("transformers")
    from abstractvoice.qwen3_tts._mel import librosa_mel_fn

    explicit = librosa_mel_fn(sr=16000, n_fft=400, n_mels=80, fmin=0, fmax=8000)
    default = librosa_mel_fn(sr=16000, n_fft=400, n_mels=80, fmin=0, fmax=None)
    assert np.allclose(explicit, default)


def test_sinc_resampler_rejects_aliases_and_keeps_length():
    from abstractvoice.audio.resample import linear_resample_mono, sinc_resample_mono

    sr_in, sr_out, seconds = 48000, 24000, 1.0
    t = np.arange(int(sr_in * seconds)) / sr_in
    tone = np.sin(2 * np.pi * 18000 * t).astype(np.float32)  # above the new Nyquist

    out = sinc_resample_mono(tone, sr_in, sr_out)
    assert out.shape[0] == int(sr_in * seconds * sr_out / sr_in)

    # An 18 kHz tone cannot exist at 24 kHz sampling; nearly all of its energy
    # must be gone. Linear interpolation keeps it (folded to 6 kHz).
    def energy(x):
        return float(np.sum(np.square(x[1000:-1000], dtype=np.float64)))

    tone_energy = energy(tone) / 2  # resampled length is half
    assert energy(out) < 1e-3 * tone_energy
    # Linear interpolation keeps a large aliased residual where the sinc filter
    # leaves essentially nothing; the ratio is the point, not the exact fraction.
    assert energy(linear_resample_mono(tone, sr_in, sr_out)) > 100.0 * max(energy(out), 1e-12)
    assert energy(linear_resample_mono(tone, sr_in, sr_out)) > 0.1 * tone_energy


def test_sinc_resampler_preserves_in_band_content():
    from abstractvoice.audio.resample import sinc_resample_mono

    sr_in, sr_out = 48000, 24000
    t = np.arange(sr_in) / sr_in
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = sinc_resample_mono(tone, sr_in, sr_out)
    t2 = np.arange(out.shape[0]) / sr_out
    expected = np.sin(2 * np.pi * 1000 * t2).astype(np.float32)
    # Ignore filter edges; in-band content must survive nearly unchanged.
    a, b = out[500:-500], expected[500:-500]
    assert float(np.abs(a - b).max()) < 5e-3

    same = sinc_resample_mono(tone, sr_in, sr_in)
    assert same.shape == tone.shape


# ----------------------------------------------------------------- token budget


def test_max_new_tokens_derives_from_text_length():
    from abstractvoice.qwen3_tts.runtime import estimate_max_new_tokens

    assert estimate_max_new_tokens("hi") == 96  # floor
    short, longer = estimate_max_new_tokens("a" * 200), estimate_max_new_tokens("a" * 800)
    assert short < longer <= 2048
    assert estimate_max_new_tokens("a" * 100000) == 2048  # ceiling


# ------------------------------------------------------------- adapter contract


class _FakeRuntime:
    """Duck-typed Qwen3TTSRuntime standing in for the loaded model."""

    def __init__(self, model_type="custom_voice", speakers=("serena", "vivian")):
        self.model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        self.is_loaded = True
        self._model_type = model_type
        self._speakers = list(speakers)
        self.calls = []
        from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings

        self.settings = Qwen3TTSSettings()

    def model_type(self):
        return self._model_type

    def speaker_names(self):
        return list(self._speakers)

    def language_names(self):
        return ["english", "chinese"]

    def runtime_info(self):
        return {"model_id": self.model_id, "loaded": True}

    def synthesize_custom_voice(self, text, *, speaker, language, instruct=None, **kw):
        self.calls.append(("custom", text, speaker, language, instruct))
        return np.zeros(2400, dtype=np.float32), 24000

    def synthesize_voice_design(self, text, *, instruct, language, **kw):
        if not str(instruct or "").strip():
            raise ValueError("requires a non-empty voice description")
        self.calls.append(("design", text, instruct, language))
        return np.zeros(2400, dtype=np.float32), 24000

    def unload(self):
        self.is_loaded = False


def _adapter(runtime):
    from abstractvoice.adapters.tts_qwen3_tts import Qwen3TTSAdapter

    return Qwen3TTSAdapter(language="en", runtime=runtime, auto_load=False)


def test_adapter_profiles_come_from_the_runtime_speakers():
    adapter = _adapter(_FakeRuntime())
    profiles = adapter.get_profiles()
    assert [p.profile_id for p in profiles] == ["serena", "vivian"]
    assert all(p.engine_id == "qwen3-tts" for p in profiles)

    assert adapter.set_profile("vivian") is True
    assert adapter.set_profile("nope") is False
    assert adapter.get_active_profile().profile_id == "vivian"


def test_adapter_synthesizes_with_speaker_and_instructions():
    runtime = _FakeRuntime()
    adapter = _adapter(runtime)
    adapter.set_profile("serena")
    adapter.set_instructions("speak warmly")

    data = adapter.synthesize_to_bytes("hello", format="wav")
    assert data[:4] == b"RIFF"
    assert runtime.calls[-1] == ("custom", "hello", "serena", "English", "speak warmly")


def test_adapter_voice_override_routes_and_validates():
    runtime = _FakeRuntime()
    adapter = _adapter(runtime)

    data = adapter.synthesize_to_bytes_with_voice("hi", voice="vivian", instructions="calm")
    assert data[:4] == b"RIFF"
    assert runtime.calls[-1][2] == "vivian"

    with pytest.raises(ValueError):
        adapter.synthesize_to_bytes_with_voice("hi", voice="not-a-speaker")


def test_adapter_refuses_voice_design_without_instructions():
    adapter = _adapter(_FakeRuntime(model_type="voice_design", speakers=()))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.synthesize("hello")

    adapter.set_instructions("a deep calm narrator")
    audio = adapter.synthesize("hello")
    assert audio.shape[0] > 0


def test_adapter_directs_base_checkpoints_to_cloning():
    adapter = _adapter(_FakeRuntime(model_type="base", speakers=()))
    with pytest.raises(RuntimeError, match="cloning"):
        adapter.synthesize("hello")


def test_runtime_voice_design_requires_instructions_before_loading():
    """The refusal must not depend on a loaded model (the whole point is to
    fail before an expensive load produces an arbitrary voice)."""
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    runtime = Qwen3TTSRuntime(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", allow_downloads=False)
    with pytest.raises(ValueError, match="non-empty voice description"):
        runtime.synthesize_voice_design("hello", instruct="")


# ---------------------------------------------------------------- cloning engine


def test_cloning_engine_requires_reference_text():
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    with pytest.raises(RuntimeError, match="reference_text"):
        list(engine.infer_to_audio_chunks(text="hi", reference_paths=["a.wav"], reference_text=""))


def test_cloning_engine_requires_exactly_one_reference():
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)
    with pytest.raises(RuntimeError, match="exactly one"):
        list(engine.infer_to_audio_chunks(text="hi", reference_paths=["a.wav", "b.wav"], reference_text="t"))


def test_cloning_engine_chunks_and_caches_the_prompt():
    from abstractvoice.cloning.engine_qwen3_tts import Qwen3TTSVoiceCloningEngine

    engine = Qwen3TTSVoiceCloningEngine(allow_downloads=False)

    class _RT:
        def __init__(self):
            self.prompt_builds = 0
            self.synth_calls = []
            from abstractvoice.qwen3_tts.runtime import Qwen3TTSSettings

            self.settings = Qwen3TTSSettings()

        def build_clone_prompt(self, *, ref_audio, ref_text, x_vector_only):
            assert x_vector_only is False  # managed path is ICL, explicitly
            self.prompt_builds += 1
            return ["prompt"]

        def synthesize_clone(self, text, *, clone_prompt, language=None, **kw):
            self.synth_calls.append(text)
            return np.zeros(240, dtype=np.float32), 24000

    engine._runtime = _RT()
    text = "First sentence here. " * 30  # forces several chunks at max_chars=400
    chunks = list(
        engine.infer_to_audio_chunks(text=text, reference_paths=["ref.wav"], reference_text="the transcript")
    )
    assert len(chunks) >= 2
    assert engine._runtime.prompt_builds == 1  # cached across chunks
    assert len(engine._runtime.synth_calls) == len(chunks)


def test_cloning_manager_accepts_the_engine(tmp_path, monkeypatch):
    from abstractvoice.cloning.manager import VoiceCloner
    from abstractvoice.cloning.store import VoiceCloneStore

    cloner = VoiceCloner(store=VoiceCloneStore(base_dir=str(tmp_path)), default_engine="qwen3-tts")
    ref = tmp_path / "ref.wav"
    import soundfile as sf

    sf.write(ref, np.zeros(2400, dtype=np.float32), 24000)
    voice_id = cloner.clone_voice(str(ref), name="test", reference_text="a transcript")
    record = cloner.store.get_voice(voice_id)
    assert record.engine == "qwen3-tts"
    assert record.reference_text == "a transcript"


# ------------------------------------------------------------ plugin surfacing


@pytest.fixture()
def fake_qwen_snapshot(tmp_path, monkeypatch):
    """A weights-bearing fake snapshot of the default CustomVoice repo.

    huggingface_hub ships with the engine extras, not the base install.
    """
    pytest.importorskip("huggingface_hub")
    hub = tmp_path / "hub"
    snap = hub / "models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"x")
    (snap / "config.json").write_text(json.dumps({
        "tts_model_type": "custom_voice",
        "tts_model_size": "0b6",
        "talker_config": {
            "spk_id": {"Serena": 1, "Vivian": 2},
            "codec_language_id": {"English": 0, "Chinese": 1},
        },
    }))
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    return snap


def test_local_models_discovers_cached_qwen3_snapshots(fake_qwen_snapshot):
    from abstractvoice.local_models import cached_tts_model_ids, hf_cached_snapshot_dir

    assert cached_tts_model_ids("qwen3-tts") == ["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
    assert hf_cached_snapshot_dir("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice") == fake_qwen_snapshot


def test_weight_free_profiles_read_the_snapshot_config(fake_qwen_snapshot):
    from abstractvoice.adapters.tts_qwen3_tts import cached_qwen3_tts_voice_profiles

    profiles = cached_qwen3_tts_voice_profiles()
    assert [p.profile_id for p in profiles] == ["serena", "vivian"]
    assert profiles[0].params["model"] == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@needs_torch
def test_plugin_lists_qwen3_provider_models_and_voices(fake_qwen_snapshot, monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    for key in ("ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTVOICE_TTS_MODEL"):
        monkeypatch.delenv(key, raising=False)

    class _Owner:
        config: dict = {}

    cap = plugin._VoiceCapability(_Owner())
    providers = cap.available_providers()
    assert "qwen3-tts" in providers["tts"]
    assert "qwen3-tts" in providers["known_tts_providers"]
    assert "qwen3-tts" in providers["known_cloning_providers"]

    assert cap.list_tts_models(provider="qwen3-tts") == ["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]

    catalog = cap.voice_catalog(provider="qwen3-tts")
    entry = catalog["tts_catalog_by_provider"]["qwen3-tts"]
    assert entry["models"] == ["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"]
    assert {v["profile_id"] for v in entry["profiles"]} == {"serena", "vivian"}

    voices = cap.list_tts_voices(provider="qwen3-tts")
    assert {v["profile_id"] for v in voices} == {"serena", "vivian"}


def test_capability_catalog_expresses_per_model_instruction_truth():
    from abstractvoice.compatibility import build_compatibility_catalog

    catalog = build_compatibility_catalog()
    def support(model, surface="bytes"):
        return catalog.support_for(kind="tts", provider="qwen3-tts", feature="instructions", model=model, surface=surface)

    assert support("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice").support == "unsupported"
    assert support("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice").support == "native"
    design = support("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    assert design.support == "native" and design.metadata.get("required") is True
    assert catalog.support_for(kind="cloning", provider="qwen3-tts", feature="reference_text", surface="create").support == "native"


@needs_torch
def test_registry_creates_the_adapter_without_loading(fake_qwen_snapshot):
    from abstractvoice.adapters.tts_registry import create_tts_adapter, get_supported_tts_engines

    assert "qwen3-tts" in get_supported_tts_engines()
    adapter, resolved = create_tts_adapter(
        engine="qwen3-tts", language="en", allow_downloads=False, auto_load=False
    )
    assert resolved == "qwen3-tts"
    assert adapter.engine_id == "qwen3-tts"
    assert adapter.is_engine_loaded() is False
    assert [p.profile_id for p in adapter.get_profiles()] == ["serena", "vivian"]


# ------------------------------------------------------- adversary-B regressions


@needs_torch
def test_prompt_strings_are_byte_exact():
    """The checkpoints were trained on these exact strings; a drifted
    <|im_start|> produces plausible-sounding wrong audio, not an error."""
    from abstractvoice.qwen3_tts.orchestration import Qwen3TTSModel

    build_assistant = Qwen3TTSModel._build_assistant_text
    build_ref = Qwen3TTSModel._build_ref_text
    build_instruct = Qwen3TTSModel._build_instruct_text

    class _Self:
        pass

    self = _Self()
    assert build_assistant(self, "T") == "<|im_start|>assistant\nT<|im_end|>\n<|im_start|>assistant\n"
    assert build_ref(self, "R") == "<|im_start|>assistant\nR<|im_end|>\n"
    assert build_instruct(self, "I") == "<|im_start|>user\nI<|im_end|>\n"


def test_voice_override_does_not_leak_into_session_state():
    runtime = _FakeRuntime()
    adapter = _adapter(runtime)
    adapter.set_profile("serena")

    adapter.synthesize_to_bytes_with_voice("hi", voice="vivian")
    assert runtime.calls[-1][2] == "vivian"

    adapter.synthesize_to_bytes("hi again")
    assert runtime.calls[-1][2] == "serena", "per-call override must not persist"


def test_token_budget_never_cuts_the_slowest_preset():
    """The cap is runaway protection. dylan renders English at 4.8 chars/s, so a
    68-char sentence is ~14.2s = ~178 codec frames; the budget must clear that
    with headroom, for Latin and CJK alike (one conservative rate serves both)."""
    from abstractvoice.qwen3_tts.runtime import estimate_max_new_tokens

    budget = estimate_max_new_tokens("a" * 68)
    assert budget > 178 * 1.5
    assert estimate_max_new_tokens("你" * 68) == budget  # unified rate


def test_adapter_exposes_model_id_for_capability_gating():
    """vm/tts_mixin resolves per-model capability entries from adapter.model_id;
    without it the 0.6B instructions-unsupported entry never fires."""
    adapter = _adapter(_FakeRuntime())
    assert adapter.model_id == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def test_codec_decode_normalizes_list_of_dicts():
    """The orchestration decodes via [{"audio_codes": t}] — the path must accept
    tensors, numpy arrays, and single-sample shapes."""
    torch = pytest.importorskip("torch")
    from abstractvoice.qwen3_tts.codec import Qwen3TTSCodec

    captured = {}

    class _FakeModel:
        dtype = torch.float32

        def decode(self, padded, return_dict=True):
            captured["shape"] = tuple(padded.shape)

            class _Out:
                audio_values = [torch.zeros(1920)]

            return _Out()

        def get_output_sample_rate(self):
            return 24000

    codec = Qwen3TTSCodec()
    codec.model = _FakeModel()
    codec.device = torch.device("cpu")

    wavs, sr = codec.decode([{"audio_codes": np.zeros((5, 4), dtype=np.int64)}])
    assert sr == 24000 and wavs[0].shape == (1920,)
    assert captured["shape"] == (1, 5, 4)

    wavs, sr = codec.decode({"audio_codes": torch.zeros(5, 4, dtype=torch.long)})
    assert captured["shape"] == (1, 5, 4)


def test_runtime_unload_race_raises_cleanly():
    from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime

    runtime = Qwen3TTSRuntime(allow_downloads=False)
    runtime._model = None  # simulate: unloaded between _ensure_loaded and the lock
    with pytest.raises(RuntimeError, match="unloaded"):
        with runtime._lock:
            runtime._model_or_raise()


def test_sinc_resampler_is_polyphase_fast_and_dc_exact():
    """The naive zero-stuff-and-convolve route cost 16 s / 750 MB for ten
    seconds of 44.1 kHz input — paid twice per clone prompt."""
    import time

    from abstractvoice.audio.resample import sinc_resample_mono

    x = np.random.default_rng(0).standard_normal(441000).astype(np.float32)
    started = time.perf_counter()
    y = sinc_resample_mono(x, 44100, 24000)
    elapsed = time.perf_counter() - started
    assert y.shape[0] == 240000
    assert elapsed < 3.0, f"polyphase resample regressed to {elapsed:.1f}s"

    dc = sinc_resample_mono(np.ones(48000, dtype=np.float32), 48000, 24000)
    assert abs(float(dc[2000:-2000].mean()) - 1.0) < 1e-6


# ------------------------------------------------- weights actually load (P0 guard)


@needs_torch
def test_strict_loader_actually_assigns_weights(tmp_path):
    """Regression guard for the transformers-5.8 silent no-op load.

    The stock ``from_pretrained`` reported success for this composite model
    while assigning NOTHING -- random weights produced speech-shaped babble.
    The explicit loader must (a) make the module bit-identical to the file and
    (b) hard-error on key drift, never lenient-skip.
    """
    import torch
    from safetensors.torch import load_file, save_file

    from abstractvoice.qwen3_tts.orchestration import load_strict_safetensors

    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    reference = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    save_file(reference, str(tmp_path / "model.safetensors"))

    load_strict_safetensors(model, str(tmp_path))
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key]), f"{key} was not assigned from the file"

    # Key drift must be a hard error (strict=True), not a silent skip.
    drifted = {("renamed_" + k if i == 0 else k): v for i, (k, v) in enumerate(reference.items())}
    save_file(drifted, str(tmp_path / "model.safetensors"))
    with pytest.raises(RuntimeError):
        load_strict_safetensors(model, str(tmp_path))

    # A snapshot without safetensors must refuse loudly, not construct random weights.
    with pytest.raises(FileNotFoundError):
        load_strict_safetensors(model, str(tmp_path / "empty"))


@needs_torch
def test_strict_loader_reads_sharded_index(tmp_path):
    import torch
    from safetensors.torch import save_file

    from abstractvoice.qwen3_tts.orchestration import load_strict_safetensors

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    reference = {k: torch.randn_like(v) for k, v in model.state_dict().items()}
    keys = sorted(reference)
    half = len(keys) // 2
    save_file({k: reference[k] for k in keys[:half]}, str(tmp_path / "model-00001.safetensors"))
    save_file({k: reference[k] for k in keys[half:]}, str(tmp_path / "model-00002.safetensors"))
    weight_map = {k: ("model-00001.safetensors" if k in keys[:half] else "model-00002.safetensors") for k in keys}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    load_strict_safetensors(model, str(tmp_path))
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key])
