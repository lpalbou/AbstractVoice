"""Build the built-in Qwen3-TTS voice pack from validated reference recordings.

Input: a folder with ``profiles.json`` (one entry per voice: id, file, name, nationality,
language, language_code, sex, style, description, reference_text) and the reference WAVs it
names. Each reference is encoded ONCE with the Qwen3-TTS 1.7B Base checkpoint (PyTorch
reference implementation) into a voice-clone prompt: its 12Hz codec codes and its speaker
embedding. Only those arrays and the profile are written into the package:

    abstractvoice/assets/voices/qwen3_tts/voices.json
    abstractvoice/assets/voices/qwen3_tts/prompts/<voice-id>.npz

Usage (from the repository root, with the qwen3-tts extra and the 1.7B Base cached):

    python tools/voice_pack/build.py path/to/candidates
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import abstractvoice  # noqa: E402

if Path(abstractvoice.__file__).resolve().parents[1] != REPO:
    raise SystemExit(f"refusing to run: imported {abstractvoice.__file__}, not the checkout at {REPO}")

from abstractvoice.qwen3_tts.runtime import Qwen3TTSRuntime  # noqa: E402

BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
PACK = REPO / "abstractvoice" / "assets" / "voices" / "qwen3_tts"
LOCALE = {"American": "en-us", "British": "en-gb"}


def voice_id(profile: dict) -> str:
    """Readable, stable id: <locale>-<sex>-<name>, e.g. fr-female-camille, en-gb-male-oliver."""
    locale = LOCALE.get(profile["nationality"], profile["language_code"])
    name = profile["id"].split("_", 2)[2]
    return f"{locale}-{profile['sex']}-{name}"


def main(src: Path) -> None:
    profiles = json.loads((src / "profiles.json").read_text(encoding="utf-8"))
    runtime = Qwen3TTSRuntime(model_id=BASE_MODEL, allow_downloads=False, backend="torch")
    t0 = time.time()
    runtime._ensure_loaded()
    print(f"loaded {BASE_MODEL} (torch) in {time.time() - t0:.1f}s")
    (PACK / "prompts").mkdir(parents=True, exist_ok=True)
    entries = []
    for p in profiles:
        vid = voice_id(p)
        prompt = runtime.build_clone_prompt(ref_audio=str(src / p["file"]), ref_text=p["reference_text"], x_vector_only=False)
        arrays = runtime.export_clone_prompt(prompt)
        np.savez_compressed(
            PACK / "prompts" / f"{vid}.npz",
            ref_codes=arrays["ref_codes"],
            speaker_embedding=arrays["speaker_embedding"],
        )
        entries.append({
            "id": vid,
            "name": p["name"],
            "nationality": p["nationality"],
            "language": p["language"],
            "language_code": p["language_code"],
            "sex": p["sex"],
            "style": "bass" if str(p["style"]).startswith("bass") else "standard",
            "description": p["description"],
            "reference_text": p["reference_text"],
            "prompt": f"prompts/{vid}.npz",
            "model": BASE_MODEL,
            "frames": int(arrays["ref_codes"].shape[0]),
        })
        print(f"{vid:28s} {arrays['ref_codes'].shape[0]:4d} frames  embedding {arrays['speaker_embedding'].shape[0]}")
    entries.sort(key=lambda e: e["id"])
    pack = {
        "format": "abstractvoice-qwen3-voice-pack/1",
        "engine": "qwen3-tts",
        "model": BASE_MODEL,
        "voices": entries,
    }
    (PACK / "voices.json").write_text(json.dumps(pack, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} voices to {PACK}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
