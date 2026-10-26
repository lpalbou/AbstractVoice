# ADR 0008: Qwen3-TTS runs on MLX on Apple silicon and on PyTorch elsewhere

Status: Accepted.

## Context

Qwen3-TTS is AbstractVoice's expressive local TTS and cloning engine (preset
speakers, voice design, voice cloning). Its PyTorch implementation is a
sequential, kernel-launch-bound decoder: every 80 ms codec frame costs one
talker step plus fifteen code-predictor steps. On Apple silicon, eager PyTorch
on MPS pays a fixed launch cost per kernel, so the 0.6B checkpoint is no faster
than the 1.7B and a 63-word reply synthesizes at a real-time factor near 1.0,
with the first streamed audio nine seconds in (`docs/benchmark.md`).

MLX runs the same checkpoints on the Apple GPU with far less per-kernel
overhead. It is also a different framework: its own array type, its own model
code, its own weight layouts. ADR 0005 governs device and dtype choices
*within* PyTorch; it says nothing about choosing a framework.

Two further constraints shape the decision:

- the public surfaces (`Qwen3TTSRuntime`, `Qwen3TTSAdapter`, the cloning
  engine, CLI/REPL flags, the AbstractCore plugin settings) must not change for
  callers, and the PyTorch path must stay byte-for-byte behaviourally the same
  for CUDA and CPU users;
- the engine must keep loading the ORIGINAL Qwen bf16 safetensors from the same
  Hugging Face snapshot, so no second download and no dependency on a
  third-party conversion repository.

## Decision

- Qwen3-TTS has two compute backends behind one runtime: `torch` (the vendored
  PyTorch model, ADR 0005 device/dtype policy) and `mlx` (a vendored MLX port
  under `abstractvoice/qwen3_tts/mlx/`).
- Backend selection is a small, framework-free policy
  (`abstractvoice/qwen3_tts/backend.py`):
  - `auto` (the default) selects `mlx` when the interpreter runs on
    macOS/arm64 **and** `mlx` is importable, else `torch`;
  - `ABSTRACTVOICE_QWEN3_TTS_BACKEND=torch|mlx` overrides `auto`, as does
    `Qwen3TTSSettings(backend=...)` / `Qwen3TTSRuntime(backend=...)`;
  - asking for `mlx` where it cannot run is a plain error before any snapshot
    is resolved, never a silent fallback.
- The MLX backend loads the original safetensors by name, strictly (every
  checkpoint tensor must land on a parameter of the same shape), converts the
  layouts MLX needs at load, and quantizes the talker's linear layers on load
  (8-bit by default; `bf16` and `4bit` through
  `Qwen3TTSSettings(mlx_quantization=...)` or
  `ABSTRACTVOICE_QWEN3_TTS_MLX_QUANTIZATION`). The speech codec and speaker
  encoder stay float32. No converted weights are cached: loading plus
  quantization is a fraction of the priming time.
- Prompt formats and sampling on MLX follow the vendored PyTorch
  `generate()`, which is the contract the checkpoints were trained against:
  the same text/codec interleaving for the three modes, the Hugging Face
  logits-processor semantics (whole-history repetition penalty, two-frame EOS
  suppression, temperature then top-k then top-p, separate sub-talker
  sampling). Parity is checked numerically against the PyTorch model
  (prompt embeddings, prefill logits, codec decoder, speaker embedding,
  reference codes) and by transcription.
- The MLX backend streams **codec frames of one generation per reply**: the
  runtime's `stream_*` methods yield contiguous waveform chunks (a short first
  chunk, then larger ones), the adapter, the cloning engine and the
  `VoiceManager` streaming path treat such a reply as ONE segment (bounded only
  by `NATIVE_STREAM_MAX_CHARS`), and only the utterance edges are faded.
  Sentence segmentation with one independent sample per sentence remains the
  behaviour of engines without native streaming (including Qwen3-TTS on
  PyTorch).
- The PyTorch-only switches (`predictor`, `sampler`) are accepted on every
  backend; the MLX backend reports them as not applied in
  `runtime_info()["pytorch_switches"]`.
- The MLX backend runs a short priming generation when it loads, so Metal
  kernel compilation is paid in the loading phase, like every other local
  engine's warm-up, not on the first utterance.
- Vendored MLX model code is derived from mlx-audio (MIT) and the Kyutai Mimi
  and mlx-lm modules it redistributes (MIT); the notice lives in
  `third_party_licenses/mlx_audio_notice.txt`. Runtime dependencies of the MLX
  path are low level only: `mlx`, `tokenizers` (the Qwen2 tokenizer built from
  the snapshot's `vocab.json`/`merges.txt`), plus the extra's `huggingface_hub`,
  `safetensors`, `numpy`, `soundfile`. Neither `torch` nor `transformers` is
  imported on that path.

## Consequences

### Positive

- Apple silicon gets a conversational Qwen3-TTS: a fraction of the real-time
  factor and first audio in well under a second, with the same checkpoints,
  the same voices and the same public surfaces.
- A multi-sentence reply is one sample, so the voice no longer drifts between
  sentences in streamed delivery.
- CUDA/CPU users are unaffected: the PyTorch path is untouched and remains
  the default everywhere MLX cannot run.

### Negative

- Two model implementations to keep in step with upstream checkpoints; the
  strict loader and the parity checks are what make drift visible.
- Numerical results differ slightly between backends (bf16/8-bit MLX versus
  bf16 MPS or fp32 CPU); reproducibility across backends is not promised.
- LLM-delta streaming (`open_tts_text_stream`) still synthesizes per text
  segment as the text arrives: one generation per reply needs the complete
  text.

### Neutral

- ADR 0005 keeps governing device and dtype within the PyTorch backend; this
  ADR sits above it and chooses the framework.

## Enforcement

- New Qwen3-TTS behaviour must be implemented on both backends or declared
  backend-specific in `runtime_info()`; a switch silently ignored by one
  backend is drift.
- Weight loading on either backend stays strict; a lenient loader is a
  regression (see `orchestration.load_strict_safetensors` and
  `mlx/weights.load_weights_strict`).
- Streaming consumers decide by the adapter's `native_frame_streaming`
  capability, never by engine name.

## Validation

- `tests/test_qwen3_tts_backend_selection.py` (policy, settings mapping,
  adapter/mixin/cloning seams; torch- and mlx-free)
- `tests/test_qwen3_tts_mlx_tokenizer.py` (tokenizer identity with Transformers)
- `tests/test_qwen3_tts_mlx_e2e_optional.py` (real checkpoints on Apple silicon:
  `ABSTRACTVOICE_RUN_QWEN3_TTS_MLX_TESTS=1`)
- `tests/test_qwen3_tts_predictor.py` (the PyTorch switches, pinned to `torch`)

## Backlog links

- `docs/backlog/completed/0061_qwen3_tts_mlx_backend.md`

## Related

- ADR 0005 (torch device and dtype policy)
- `abstractvoice/qwen3_tts/backend.py`, `abstractvoice/qwen3_tts/runtime.py`,
  `abstractvoice/qwen3_tts/mlx/`
- [../benchmark.md](../benchmark.md)
