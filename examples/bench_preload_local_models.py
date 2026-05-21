from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# Ensure `import abstractvoice` resolves to the repo checkout when running this script
# directly (Python prepends the script's directory, not the current working dir).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class TtsCase:
    provider: str
    model: str | None = None
    language: str | None = None
    profile: str | None = None


def _fmt_s(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean([float(v) for v in values]))


def _measure_tts(
    *,
    case: TtsCase,
    text: str,
    runs: int,
    fmt: str,
) -> dict:
    from abstractvoice import VoiceManager

    cold_synth: list[float] = []
    hot_synth: list[float] = []
    preload_s: list[float] = []

    for _i in range(int(runs)):
        vm = VoiceManager(
            language=str(case.language or "en"),
            tts_engine=str(case.provider),
            stt_engine="faster_whisper",
            allow_downloads=False,
        )
        if case.profile and hasattr(vm, "set_profile"):
            try:
                vm.set_profile(str(case.profile), kind="tts")
            except TypeError:
                vm.set_profile(str(case.profile))
            except Exception:
                pass

        # Ensure each trial starts cold.
        try:
            vm.unload_tts_engine()
        except Exception:
            pass

        t0 = time.monotonic()
        _ = vm.speak_to_bytes(str(text), format=str(fmt))
        metrics = vm.pop_last_tts_metrics() if hasattr(vm, "pop_last_tts_metrics") else None
        cold_synth.append(float(metrics.get("synth_s")) if isinstance(metrics, dict) and metrics.get("synth_s") else float(time.monotonic() - t0))

        try:
            vm.unload_tts_engine()
        except Exception:
            pass

        tp = time.monotonic()
        _ = vm.preload_tts_engine(warmup=True, warmup_text=str(text), warmup_format=str(fmt))
        preload_s.append(float(time.monotonic() - tp))

        t1 = time.monotonic()
        _ = vm.speak_to_bytes(str(text), format=str(fmt))
        metrics = vm.pop_last_tts_metrics() if hasattr(vm, "pop_last_tts_metrics") else None
        hot_synth.append(float(metrics.get("synth_s")) if isinstance(metrics, dict) and metrics.get("synth_s") else float(time.monotonic() - t1))

    return {
        "provider": case.provider,
        "model": case.model,
        "profile": case.profile,
        "runs": int(runs),
        "cold_synth_s": cold_synth,
        "cold_mean_s": _mean(cold_synth),
        "preload_s": preload_s,
        "preload_mean_s": _mean(preload_s),
        "hot_synth_s": hot_synth,
        "hot_mean_s": _mean(hot_synth),
    }


def _measure_stt(
    *,
    provider: str,
    model: str | None,
    audio_path: str,
    runs: int,
    warmup: bool,
) -> dict:
    from abstractvoice import VoiceManager

    cold_s: list[float] = []
    hot_s: list[float] = []
    preload_s: list[float] = []

    for _i in range(int(runs)):
        vm = VoiceManager(
            tts_engine="piper",
            stt_engine=str(provider),
            stt_model=model,
            whisper_model=str(model or "base"),
            allow_downloads=False,
        )

        try:
            vm.unload_stt_engine()
        except Exception:
            pass

        t0 = time.monotonic()
        _ = vm.transcribe_file(str(audio_path))
        cold_s.append(float(time.monotonic() - t0))

        try:
            vm.unload_stt_engine()
        except Exception:
            pass

        tp = time.monotonic()
        _ = vm.preload_stt_engine(warmup=bool(warmup), warmup_audio_path=str(audio_path))
        preload_s.append(float(time.monotonic() - tp))

        t1 = time.monotonic()
        _ = vm.transcribe_file(str(audio_path))
        hot_s.append(float(time.monotonic() - t1))

    return {
        "provider": str(provider),
        "model": model,
        "runs": int(runs),
        "cold_s": cold_s,
        "cold_mean_s": _mean(cold_s),
        "preload_s": preload_s,
        "preload_mean_s": _mean(preload_s),
        "hot_s": hot_s,
        "hot_mean_s": _mean(hot_s),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark preload vs cold start for local TTS/STT engines.")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--text", type=str, default="This is a preload benchmark.")
    ap.add_argument("--format", type=str, default="wav")
    ap.add_argument("--stt-audio", type=str, default="/Users/albou/Documents/patrick_voice_short2.wav")
    ap.add_argument("--stt-provider", type=str, default="faster_whisper", help="Local STT provider id (e.g. faster_whisper, transformers-asr).")
    ap.add_argument("--stt-model", type=str, default="base", help="STT model id (faster-whisper size or HF repo id for transformers-asr).")
    ap.add_argument("--no-stt", action="store_true")
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    runs = int(args.runs)
    fmt = str(args.format or "wav").strip().lower() or "wav"

    cases: list[TtsCase] = []
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter

        piper_models = [model_filename for (_hf, model_filename) in PiperTTSAdapter.PIPER_MODELS.values()]
        cases.extend([TtsCase(provider="piper", model=m, profile=m) for m in piper_models])
    except Exception:
        pass

    # Supertone/Supertonic: single model id, profiles are fixed styles.
    cases.append(TtsCase(provider="supertonic", model="supertonic-3", profile=None))
    cases.append(TtsCase(provider="omnivoice", model="default", profile="default"))
    cases.append(TtsCase(provider="audiodit", model="default", profile=None))

    if not bool(args.no_tts):
        print(f"TTS preload benchmark (runs={runs}, format={fmt})")
        print(f"Text: {args.text!r}")
        print("")

        tts_results: list[dict] = []
        for case in cases:
            try:
                res = _measure_tts(case=case, text=str(args.text), runs=runs, fmt=fmt)
            except Exception as e:
                res = {"provider": case.provider, "model": case.model, "error": str(e)}
            tts_results.append(res)

            if "error" in res:
                print(f"- {case.provider}/{case.model}: ERROR: {res['error']}")
                continue

            cold_mean = res.get("cold_mean_s")
            hot_mean = res.get("hot_mean_s")
            preload_mean = res.get("preload_mean_s")

            print(f"- {case.provider}/{case.model}")
            print(f"  cold_synth:  {', '.join(_fmt_s(v) for v in res['cold_synth_s'])}  avg={_fmt_s(cold_mean)}")
            print(f"  preload:     {', '.join(_fmt_s(v) for v in res['preload_s'])}  avg={_fmt_s(preload_mean)}")
            print(f"  hot_synth:   {', '.join(_fmt_s(v) for v in res['hot_synth_s'])}  avg={_fmt_s(hot_mean)}")
            if isinstance(cold_mean, (int, float)) and isinstance(hot_mean, (int, float)) and hot_mean > 0:
                print(f"  speedup:     {float(cold_mean)/float(hot_mean):.2f}x")
            print("")

    if not bool(args.no_stt):
        print("STT preload benchmark")
        audio_path = str(args.stt_audio)
        provider = str(args.stt_provider or "faster_whisper").strip() or "faster_whisper"
        model = str(args.stt_model or "").strip() or None
        try:
            res = _measure_stt(provider=provider, model=model, audio_path=audio_path, runs=runs, warmup=False)
        except Exception as e:
            print(f"- {provider}/{model}: ERROR: {e}")
        else:
            print(f"- {provider}/{model}")
            print(f"  cold:      {', '.join(_fmt_s(v) for v in res['cold_s'])}  avg={_fmt_s(res['cold_mean_s'])}")
            print(f"  preload:   {', '.join(_fmt_s(v) for v in res['preload_s'])}  avg={_fmt_s(res['preload_mean_s'])}")
            print(f"  hot:       {', '.join(_fmt_s(v) for v in res['hot_s'])}  avg={_fmt_s(res['hot_mean_s'])}")
            if isinstance(res.get('cold_mean_s'), (int, float)) and isinstance(res.get('hot_mean_s'), (int, float)) and float(res['hot_mean_s']) > 0:
                print(f"  speedup:   {float(res['cold_mean_s'])/float(res['hot_mean_s']):.2f}x")
            print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
