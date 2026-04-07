#!/usr/bin/env python3
"""Run the TTS benchmark multiple times and aggregate results.

Rationale: TTS performance can vary across runs due to OS scheduling, thermal
conditions, and accelerator state. This script runs `examples/bench_tts.py`
in separate Python processes to capture those effects, then aggregates results.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuiteConfig:
    out_dir: str
    experiments: int
    sleep_s: float
    bench_args: list[str]


def _parse_args() -> SuiteConfig:
    p = argparse.ArgumentParser(description="Run AbstractVoice TTS benchmark suite (multiple experiments).")
    p.add_argument("--out-dir", default="untracked/bench_tts_suite", help="Base output folder.")
    p.add_argument("--experiments", type=int, default=5, help="Number of full benchmark runs. Default: 5")
    p.add_argument("--sleep-s", type=float, default=2.0, help="Sleep between experiments. Default: 2.0")

    # We accept the bench flags and forward them to bench_tts.py.
    # Anything unknown is also forwarded (future-proof).
    args, unknown = p.parse_known_args()
    if int(args.experiments) < 1:
        raise SystemExit("--experiments must be >= 1")
    if float(args.sleep_s) < 0:
        raise SystemExit("--sleep-s must be >= 0")

    out_dir = str(args.out_dir)
    return SuiteConfig(
        out_dir=out_dir,
        experiments=int(args.experiments),
        sleep_s=float(args.sleep_s),
        bench_args=list(unknown),
    )


def _fmean(xs: list[float]) -> float | None:
    if not xs:
        return None
    try:
        return float(statistics.fmean(xs))
    except Exception:
        return float(sum(xs) / max(1, len(xs)))


def _stdev(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    try:
        return float(statistics.stdev(xs))
    except Exception:
        return None


def _aggregate(merged_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate across ALL synth runs, plus per-experiment means."""

    # Collect raw synth runs by key.
    synth = [r for r in merged_rows if r.get("phase") == "synth" and r.get("ok") is True]

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in synth:
        k = (str(r.get("engine", "")), str(r.get("language", "")), str(r.get("variant", "")))
        by_key[k].append(r)

    # Per-experiment mean is useful to see time-level variability.
    by_key_exp: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in synth:
        exp = str(r.get("experiment", ""))
        k = (str(r.get("engine", "")), str(r.get("language", "")), str(r.get("variant", "")))
        by_key_exp[k][exp].append(float(r.get("elapsed_s")))

    out: dict[str, Any] = {"by_engine_language_variant": []}
    for (engine, language, variant), rs in sorted(by_key.items()):
        elapsed = [float(r.get("elapsed_s")) for r in rs if isinstance(r.get("elapsed_s"), (int, float))]
        audio_s = [float(r.get("audio_s")) for r in rs if isinstance(r.get("audio_s"), (int, float))]
        rtf = [float(r.get("rtf")) for r in rs if isinstance(r.get("rtf"), (int, float))]
        declared = rs[0].get("declared_supported") if rs else None

        # Per-experiment means + between-experiment stdev.
        exp_means = []
        for exp_id, xs in sorted(by_key_exp[(engine, language, variant)].items()):
            if xs:
                exp_means.append(float(sum(xs) / len(xs)))

        out["by_engine_language_variant"].append(
            {
                "engine": engine,
                "language": language,
                "variant": variant,
                "declared_supported": declared,
                "n_total": int(len(rs)),
                "n_experiments": int(len(by_key_exp[(engine, language, variant)])),
                "mean_elapsed_s": _fmean(elapsed),
                "stdev_elapsed_s": _stdev(elapsed),
                "min_elapsed_s": float(min(elapsed)) if elapsed else None,
                "max_elapsed_s": float(max(elapsed)) if elapsed else None,
                "mean_audio_s": _fmean(audio_s),
                "mean_rtf": _fmean(rtf),
                "between_experiment_stdev_elapsed_s": _stdev(exp_means),
            }
        )

    return out


def _to_markdown(cfg: SuiteConfig, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# AbstractVoice TTS benchmark suite (aggregated)")
    lines.append("")
    lines.append(f"- experiments: `{cfg.experiments}`")
    lines.append(f"- sleep_s: `{cfg.sleep_s}`")
    lines.append(f"- bench_args: `{ ' '.join(cfg.bench_args) }`")
    lines.append("")
    lines.append("| engine | lang | text | declared | n_total | avg_wav_gen_s | sd_wav_gen_s | between_exp_sd_s | avg_speech_s | rtf |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in summary.get("by_engine_language_variant", []):
        declared = row.get("declared_supported")
        declared_txt = "" if declared is None else ("yes" if declared else "no")

        def f(x: Any) -> str:
            return "" if x is None else f"{float(x):.3f}"

        lines.append(
            "| {engine} | {lang} | {variant} | {declared} | {n} | {mean_e} | {sd_e} | {sd_b} | {mean_a} | {mean_r} |".format(
                engine=row.get("engine", ""),
                lang=row.get("language", ""),
                variant=row.get("variant", ""),
                declared=declared_txt,
                n=int(row.get("n_total", 0) or 0),
                mean_e=f(row.get("mean_elapsed_s")),
                sd_e=f(row.get("stdev_elapsed_s")),
                sd_b=f(row.get("between_experiment_stdev_elapsed_s")),
                mean_a=f(row.get("mean_audio_s")),
                mean_r=f(row.get("mean_rtf")),
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `between_exp_sd_s` is the standard deviation of per-experiment mean times (captures time/thermal variance).")
    lines.append("- `text`: `s2` = 2 short sentences, `s5` = 5 short sentences, `s6` = 6 short sentences.")
    lines.append("- `avg_wav_gen_s`: average wall time to write a WAV via `VoiceManager.speak_to_file(...)`.")
    lines.append("- `avg_speech_s`: average duration of the generated WAV.")
    lines.append("- `rtf`: `avg_wav_gen_s / avg_speech_s` (lower is faster; `<1` means faster than real time).")
    lines.append("- `declared` is adapter-reported language support; synthesis may still succeed when `declared=no` (e.g. AudioDiT+FR is not guaranteed).")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = _parse_args()
    base = Path(cfg.out_dir)
    base.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiments": int(cfg.experiments),
        "sleep_s": float(cfg.sleep_s),
        "bench_args": list(cfg.bench_args),
        "python": sys.executable,
    }
    (base / "suite_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    merged: list[dict[str, Any]] = []

    for i in range(1, int(cfg.experiments) + 1):
        exp_id = f"exp_{i:02d}"
        exp_dir = base / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, "examples/bench_tts.py", "--out-dir", str(exp_dir)]
        cmd.extend(list(cfg.bench_args))

        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.monotonic() - t0
        (exp_dir / "suite_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (exp_dir / "suite_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        (exp_dir / "suite_status.json").write_text(
            json.dumps({"returncode": proc.returncode, "elapsed_s": elapsed, "cmd": cmd}, indent=2) + "\n",
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise SystemExit(f"Experiment {exp_id} failed (returncode={proc.returncode}). See: {exp_dir}")

        # Merge rows, annotate with experiment id.
        rows = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
        for r in rows:
            r = dict(r)
            r["experiment"] = exp_id
            merged.append(r)

        if i < int(cfg.experiments) and float(cfg.sleep_s) > 0:
            time.sleep(float(cfg.sleep_s))

    (base / "merged_results.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    summary = _aggregate(merged)
    (base / "suite_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (base / "suite_summary.md").write_text(_to_markdown(cfg, summary), encoding="utf-8")

    synth_ok = sum(1 for r in merged if r.get("phase") == "synth" and r.get("ok") is True)
    synth_total = sum(1 for r in merged if r.get("phase") == "synth")
    print(f"Done. {synth_ok}/{synth_total} synth runs succeeded.")
    print(f"Suite results: {base / 'suite_summary.md'}")


if __name__ == "__main__":
    main()

