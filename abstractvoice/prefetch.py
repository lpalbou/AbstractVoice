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
    args = parser.parse_args(argv)

    if not args.stt_model and not args.openf5:
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

