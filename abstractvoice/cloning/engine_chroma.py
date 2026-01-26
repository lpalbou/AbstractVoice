from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ChromaArtifacts:
    root: Path
    model_id: str
    revision: Optional[str]


class ChromaVoiceCloningEngine:
    """In-process Chroma voice cloning engine (optional; GPU-heavy).

    Design principles
    -----------------
    - Never download weights implicitly from speak()/speak_to_bytes().
    - Provide explicit prefetch hooks.
    - Keep imports lightweight until the engine is actually used.
    """

    DEFAULT_MODEL_ID = "FlashLabs/Chroma-4B"
    # Pin a known revision by default (safer than floating remote code).
    DEFAULT_REVISION = "864b4aea0c1359f91af62f1367df64657dc5e90f"

    def __init__(
        self,
        *,
        debug: bool = False,
        device: str = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = DEFAULT_REVISION,
        temperature: float = 0.7,
        top_k: int = 50,
        do_sample: bool = True,
        max_new_tokens_per_chunk: int = 512,
    ):
        self.debug = bool(debug)
        self._device_pref = str(device or "auto")
        self._model_id = str(model_id or self.DEFAULT_MODEL_ID)
        self._revision = revision if revision else None

        self._temperature = float(temperature)
        self._top_k = int(top_k)
        self._do_sample = bool(do_sample)
        self._max_new_tokens_per_chunk = int(max_new_tokens_per_chunk)

        self._model = None
        self._processor = None
        self._resolved_device = None

    def runtime_info(self) -> Dict[str, object]:
        info: Dict[str, object] = {
            "model_id": self._model_id,
            "revision": self._revision,
            "requested_device": self._device_pref,
            "resolved_device": self._resolved_device,
            "temperature": self._temperature,
            "top_k": self._top_k,
            "do_sample": self._do_sample,
            "max_new_tokens_per_chunk": self._max_new_tokens_per_chunk,
        }
        try:
            import torch

            info["torch_version"] = getattr(torch, "__version__", "?")
            info["cuda_available"] = bool(torch.cuda.is_available())
            try:
                info["mps_available"] = bool(torch.backends.mps.is_available())
            except Exception:
                info["mps_available"] = False
        except Exception:
            pass
        return info

    def set_quality_preset(self, preset: str) -> None:
        p = (preset or "").strip().lower()
        if p not in ("fast", "balanced", "high"):
            raise ValueError("preset must be one of: fast|balanced|high")
        if p == "fast":
            self._temperature = 0.6
            self._top_k = 30
            self._max_new_tokens_per_chunk = 384
        elif p == "balanced":
            self._temperature = 0.7
            self._top_k = 50
            self._max_new_tokens_per_chunk = 512
        else:
            self._temperature = 0.75
            self._top_k = 80
            self._max_new_tokens_per_chunk = 768

    def _artifact_root(self) -> Path:
        cache_dir = Path(os.path.expanduser("~/.cache/abstractvoice/chroma"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_chroma_artifacts_local(self) -> ChromaArtifacts:
        root = self._artifact_root()
        required = [
            "config.json",
            "processor_config.json",
            "model.safetensors.index.json",
            "modeling_chroma.py",
            "processing_chroma.py",
            "configuration_chroma.py",
        ]
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise RuntimeError(
                "Chroma artifacts are not present locally.\n"
                "Prefetch explicitly (outside the REPL):\n"
                "  abstractvoice-prefetch --chroma\n"
                "or:\n"
                "  python -m abstractvoice download --chroma\n"
                f"Looked under: {root}\n"
                f"Missing: {', '.join(missing)}"
            )
        return ChromaArtifacts(root=root, model_id=self._model_id, revision=self._revision)

    def ensure_chroma_artifacts_downloaded(self) -> ChromaArtifacts:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to download Chroma artifacts.\n"
                "Install with: pip install huggingface_hub"
            ) from e

        import warnings

        root = self._artifact_root()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=r".*local_dir_use_symlinks.*deprecated.*",
            )
            snapshot_download(
                repo_id=self._model_id,
                revision=self._revision,
                local_dir=str(root),
            )
        return self._resolve_chroma_artifacts_local()

    def are_chroma_artifacts_available(self) -> bool:
        root = self._artifact_root()
        return bool((root / "config.json").exists() and (root / "model.safetensors.index.json").exists())

    def _resolve_device(self) -> str:
        if self._device_pref and self._device_pref != "auto":
            return str(self._device_pref)
        try:
            from ..compute import best_torch_device

            return best_torch_device()
        except Exception:
            return "cpu"

    def _ensure_chroma_runtime(self) -> None:
        try:
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                raise ImportError("torch not installed")
            if importlib.util.find_spec("transformers") is None:
                raise ImportError("transformers not installed")
        except Exception as e:
            raise RuntimeError(
                "Chroma requires the optional dependency group.\n"
                "Install with:\n"
                "  pip install \"abstractvoice[chroma]\"\n"
                f"Original error: {e}"
            ) from e

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None and self._resolved_device is not None:
            return

        self._ensure_chroma_runtime()
        artifacts = self._resolve_chroma_artifacts_local()

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self._resolve_device()
        self._resolved_device = device

        torch_dtype = None
        if device == "cuda":
            torch_dtype = torch.bfloat16
        elif device == "mps":
            torch_dtype = torch.float16

        device_map = None
        # `device_map="auto"` is most reliable on CUDA; MPS/CPU should load normally then move.
        if self._device_pref == "auto" and device == "cuda":
            device_map = "auto"

        # transformers 5.0.0 deprecates `torch_dtype` in favor of `dtype`.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(artifacts.root),
                trust_remote_code=True,
                device_map=device_map,
                dtype=torch_dtype,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(artifacts.root),
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        model.eval()

        processor = AutoProcessor.from_pretrained(str(artifacts.root), trust_remote_code=True)

        if device != "cuda" and device_map is None:
            try:
                model.to(device)
            except Exception:
                pass

        # Ensure tied weights are applied (Chroma relies on tied audio embeddings).
        try:
            if hasattr(model, "tie_weights"):
                model.tie_weights()
        except Exception:
            pass
        # Some releases log this key as MISSING even when it is meant to be tied.
        # Ensure backbone audio embeddings match decoder embeddings (best-effort).
        try:
            if hasattr(model, "backbone") and hasattr(model, "decoder"):
                b = getattr(model.backbone, "audio_embedding", None)
                d = getattr(model.decoder, "audio_embedding", None)
                if b is not None and d is not None:
                    bw = getattr(b, "embed_audio_tokens", None)
                    dw = getattr(d, "embed_audio_tokens", None)
                    if bw is not None and dw is not None and hasattr(bw, "weight") and hasattr(dw, "weight"):
                        if getattr(bw.weight, "shape", None) == getattr(dw.weight, "shape", None):
                            bw.weight.data.copy_(dw.weight.data)
        except Exception:
            pass

        self._patch_generation_compat(model)

        self._model = model
        self._processor = processor

    def _patch_generation_compat(self, model) -> None:
        """Patch known transformers incompatibilities for Chroma remote code.

        Upstream Chroma code (pinned revision) expects a `use_model_defaults` positional arg
        in `GenerationMixin._prepare_generation_config`. transformers 5.0.0 removed it.
        """
        try:
            import inspect

            from transformers.generation import GenerationMode
            from transformers.generation.utils import GenerationMixin

            base_sig = inspect.signature(GenerationMixin._prepare_generation_config)
            if "use_model_defaults" in base_sig.parameters:
                return

            chroma_gen_cls = None
            for cls in model.__class__.mro():
                if cls.__name__ == "ChromaGenerationMixin":
                    chroma_gen_cls = cls
                    break
            if chroma_gen_cls is None:
                return

            current = getattr(chroma_gen_cls, "_prepare_generation_config", None)
            if current is None or getattr(current, "_abstractvoice_patched", False):
                return

            def _patched_prepare_generation_config(
                self, generation_config=None, use_model_defaults=None, **kwargs
            ):
                depth_decoder_kwargs = {k[len("decoder_") :]: v for k, v in kwargs.items() if k.startswith("decoder_")}
                kwargs = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}

                # transformers 5.0.0 removed the `use_model_defaults` positional arg; keep compatibility.
                try:
                    generation_config_out, model_kwargs = super(chroma_gen_cls, self)._prepare_generation_config(
                        generation_config, use_model_defaults, **kwargs
                    )
                except TypeError:
                    generation_config_out, model_kwargs = super(chroma_gen_cls, self)._prepare_generation_config(
                        generation_config, **kwargs
                    )

                try:
                    self.decoder.generation_config.update(**depth_decoder_kwargs)
                except Exception:
                    pass

                try:
                    decoder_min_new_tokens = getattr(self.decoder.generation_config, "min_new_tokens") or (
                        self.decoder.config.audio_num_codebooks - 1
                    )
                    decoder_max_new_tokens = getattr(self.decoder.generation_config, "max_new_tokens") or (
                        self.decoder.config.audio_num_codebooks - 1
                    )

                    if {decoder_min_new_tokens, decoder_max_new_tokens} != {self.decoder.config.audio_num_codebooks - 1}:
                        raise ValueError(
                            "decoder_generation_config's min_new_tokens "
                            f"({decoder_min_new_tokens}) and max_new_tokens ({decoder_max_new_tokens}) "
                            f"must be equal to self.config.num_codebooks - 1 ({self.decoder.config.audio_num_codebooks - 1})"
                        )
                    elif getattr(self.decoder.generation_config, "return_dict_in_generate", False):
                        self.decoder.generation_config.return_dict_in_generate = False
                except Exception:
                    pass

                # Monkey patch the get_generation_mode method to support Chroma model.
                try:
                    original_get_generation_mode = generation_config_out.get_generation_mode

                    def patched_get_generation_mode(assistant_model=None):
                        generation_mode = original_get_generation_mode(assistant_model)
                        if generation_mode not in (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE):
                            raise ValueError(
                                f"Generation mode {generation_mode} is not supported for Chroma model. "
                                "Please set generation parameters to use greedy or sampling generation."
                            )
                        return generation_mode

                    generation_config_out.get_generation_mode = patched_get_generation_mode
                except Exception:
                    pass

                return generation_config_out, model_kwargs

            _patched_prepare_generation_config._abstractvoice_patched = True  # type: ignore[attr-defined]
            chroma_gen_cls._prepare_generation_config = _patched_prepare_generation_config  # type: ignore[assignment]
        except Exception:
            # Best-effort only: do not break core flows if patching fails.
            return

    def _select_prompt_audio(self, reference_paths: Iterable[str | Path]) -> Path:
        paths = [Path(p) for p in reference_paths]
        if not paths:
            raise ValueError("reference_paths must contain at least one path")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(str(p))
        # Chroma prompt_audio currently expects a single audio file.
        return paths[0]

    def _build_conversation(self, text: str) -> List[List[dict]]:
        system_prompt = (
            "You are a text-to-speech engine. "
            "Speak the user's text aloud exactly as written, without adding, removing, or rephrasing words. "
            "Do not answer the text; read it verbatim."
        )
        return [[
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": str(text)}]},
        ]]

    def infer_to_wav_bytes(
        self,
        *,
        text: str,
        reference_paths: Iterable[str | Path],
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        chunks = []
        sr_out = 24000
        for chunk, sr in self.infer_to_audio_chunks(
            text=text,
            reference_paths=reference_paths,
            reference_text=reference_text,
            speed=speed,
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
            sr_out = int(sr)
        audio = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sr_out), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def infer_to_audio_chunks(
        self,
        *,
        text: str,
        reference_paths: Iterable[str | Path],
        reference_text: Optional[str] = None,
        speed: Optional[float] = None,
        max_chars: int = 240,
    ):
        self._ensure_model_loaded()

        if not reference_text or not str(reference_text).strip():
            raise RuntimeError(
                "Missing reference_text for Chroma cloning.\n"
                "Provide reference_text when cloning, or set it via the voice store."
            )

        prompt_audio = self._select_prompt_audio(reference_paths)
        prompt_text = str(reference_text).strip()

        import re

        def _split_batches(s: str, limit: int) -> List[str]:
            s = " ".join(str(s).replace("\n", " ").split()).strip()
            if not s:
                return []
            sentences = re.split(r"(?<=[\\.!\\?\\。])\\s+", s)
            out: List[str] = []
            cur_s = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > limit:
                    words = sent.split(" ")
                    tmp = ""
                    for w in words:
                        cand = (tmp + " " + w).strip()
                        if len(cand) <= limit:
                            tmp = cand
                        else:
                            if tmp:
                                out.append(tmp)
                            tmp = w
                    if tmp:
                        out.append(tmp)
                    continue
                cand = (cur_s + " " + sent).strip()
                if len(cand) <= limit:
                    cur_s = cand
                else:
                    if cur_s:
                        out.append(cur_s)
                    cur_s = sent
            if cur_s:
                out.append(cur_s)
            return out

        batches = _split_batches(text, int(max_chars)) or [" "]

        model = self._model
        processor = self._processor
        if model is None or processor is None:
            raise RuntimeError("Chroma model not loaded")

        import torch

        for batch_text in batches:
            conversation = self._build_conversation(batch_text)
            inputs = processor(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                prompt_audio=[str(prompt_audio)],
                prompt_text=[prompt_text],
            )
            device = getattr(model, "device", None) or torch.device("cpu")
            try:
                param_dtype = next(model.parameters()).dtype
            except Exception:
                param_dtype = None
            moved = {}
            for k, v in dict(inputs).items():
                try:
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and param_dtype is not None:
                        moved[k] = v.to(device=device, dtype=param_dtype)
                    else:
                        moved[k] = v.to(device)
                except Exception:
                    moved[k] = v

            out = model.generate(
                **moved,
                max_new_tokens=int(self._max_new_tokens_per_chunk),
                do_sample=bool(self._do_sample),
                temperature=float(self._temperature),
                top_k=int(self._top_k),
                use_cache=True,
                output_attentions=False,
                output_audio=True,
            )
            audio_list = out.audio if hasattr(out, "audio") else out
            if not audio_list:
                continue
            a = audio_list[0]
            if isinstance(a, (list, tuple)) and a:
                a = a[0]
            if isinstance(a, torch.Tensor):
                arr = a.detach().cpu().numpy()
            else:
                arr = np.asarray(a)

            mono = np.asarray(arr, dtype=np.float32).reshape(-1)
            sr = 24000

            if speed and speed != 1.0:
                try:
                    from ..tts.tts_engine import apply_speed_without_pitch_change

                    mono = apply_speed_without_pitch_change(mono, float(speed), sr=sr)
                except Exception:
                    pass

            # Chroma's codec output can exceed [-1, 1] slightly; normalize to avoid
            # hard clipping/distortion in playback and PCM encoders.
            try:
                peak = float(np.max(np.abs(mono))) if mono.size else 0.0
                if peak > 1.0:
                    mono = mono / peak
            except Exception:
                pass

            yield mono, sr
