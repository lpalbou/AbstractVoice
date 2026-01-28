"""Explicit model/artifact prefetch (cross-platform).

Design rule: This must never run implicitly during normal library usage.
Users/integrators call it explicitly after installation.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abstractvoice-prefetch", description="AbstractVoice explicit prefetch")
    parser.add_argument(
        "--stt",
        dest="stt_model",
        default=None,
        help="Prefetch faster-whisper model weights (e.g. tiny/base/small/medium/large-v3)",
    )
    parser.add_argument(
        "--openf5",
        action="store_true",
        help="Prefetch OpenF5 artifacts for cloning (~5.4GB, requires abstractvoice[cloning])",
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Prefetch Chroma-4B artifacts (~14GB+, requires HF access; install abstractvoice[chroma] to run inference)",
    )
    parser.add_argument(
        "--piper",
        dest="piper_language",
        default=None,
        help="Prefetch Piper voice model for a language (e.g. en/fr/de).",
    )
    args = parser.parse_args(argv)

    if not args.stt_model and not args.openf5 and not args.chroma and not args.piper_language:
        parser.print_help()
        return 2

    if args.stt_model:
        from abstractvoice.adapters.stt_faster_whisper import FasterWhisperAdapter

        model = str(args.stt_model).strip()
        print(f"Downloading STT model (faster-whisper): {model}")
        stt = FasterWhisperAdapter(model_size=model, device="cpu", compute_type="int8", allow_downloads=True)
        if not stt.is_available():
            raise RuntimeError("STT model download/load failed.")
        print("✅ STT model ready.")

    if args.openf5:
        from abstractvoice.cloning.engine_f5 import F5TTSVoiceCloningEngine

        print("Downloading OpenF5 artifacts (cloning)…")
        engine = F5TTSVoiceCloningEngine(debug=True)
        engine.ensure_openf5_artifacts_downloaded()
        print("✅ OpenF5 artifacts ready.")

    if args.chroma:
        from abstractvoice.cloning.engine_chroma import ChromaVoiceCloningEngine

        print("Downloading Chroma artifacts (cloning)…")
        engine = ChromaVoiceCloningEngine(debug=True)
        engine.ensure_chroma_artifacts_downloaded()
        print("✅ Chroma artifacts ready.")

    if args.piper_language:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter

        lang = str(args.piper_language).strip().lower()
        print(f"Downloading Piper voice model: {lang}")
        piper = PiperTTSAdapter(language=lang, allow_downloads=True, auto_load=False)
        if not piper.ensure_model_downloaded(lang):
            raise RuntimeError("Piper model download failed.")
        print("✅ Piper model ready.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
