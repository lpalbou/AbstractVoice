#!/usr/bin/env python3
"""TTS benchmark harness (no voice cloning).

Measures:
- init time (VoiceManager construction; may include model load for torch engines)
- warmup time (first WAV generation; also seeds session-prompt caches for some engines)
- repeated synth time (N repetitions; used for averages)
- audio duration + simple real-time factor (RTF)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from abstractvoice import VoiceManager
from abstractvoice.adapters.tts_registry import get_supported_tts_engines


@dataclass(frozen=True)
class BenchConfig:
    out_dir: str
    engines: list[str]
    languages: list[str]
    allow_downloads: bool
    variants: list[str]
    repetitions: int
    # Voice stability knobs (engine-specific; best-effort).
    audiodit_seed: int
    omnivoice_seed: int
    omnivoice_instruct: str | None


@dataclass
class BenchRow:
    engine: str
    language: str
    variant: str  # s2|s5|-
    rep: int | None  # 1..N for synth reps; None for init; 0 for warmup
    phase: str  # init|warmup|synth
    ok: bool
    elapsed_s: float | None = None
    wav_path: str | None = None
    sample_rate: int | None = None
    audio_s: float | None = None
    rtf: float | None = None
    sha256: str | None = None
    declared_supported: bool | None = None
    error: str | None = None
    notes: dict[str, Any] | None = None


class TtsBenchmarker:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self._out = Path(cfg.out_dir)
        self._out.mkdir(parents=True, exist_ok=True)

        # Text variants:
        # - s2: 2 short sentences
        # - s5: 5 short sentences
        # - s6: 6 short sentences (French long prompt)
        #
        # Keep these semantically similar across languages so timing comparisons
        # are meaningful.
        self._texts: dict[str, dict[str, str]] = {
            "en": {
                "s2": "Hello. This is a benchmark sentence for AbstractVoice.",
                "s5": (
                    "Hello. This is a benchmark sentence for AbstractVoice. "
                    "We are measuring how long speech synthesis takes. "
                    "This extra sentence increases the length of the prompt. "
                    "Finally, this is the fifth sentence."
                ),
            },
            "fr": {
                "s2": "Bonjour. Ceci est une phrase de benchmark pour AbstractVoice.",
                "s6": (
                    "Bonjour. Ceci est une phrase de benchmark pour AbstractVoice. "
                    "Nous mesurons le temps nécessaire pour générer la parole. "
                    "Cette phrase supplémentaire augmente la longueur du texte. "
                    "Nous ajoutons encore une phrase pour arriver à cinq. "
                    "Enfin, voici la sixième phrase."
                ),
            },
        }

    def run(self) -> list[BenchRow]:
        rows: list[BenchRow] = []
        meta = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": asdict(self.cfg),
        }
        (self._out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        for engine in self.cfg.engines:
            for lang in self.cfg.languages:
                rows.extend(self._bench_engine_language(engine=engine, language=lang))

        # Persist results.
        payload = [asdict(r) for r in rows]
        (self._out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (self._out / "results.md").write_text(self._to_markdown(rows), encoding="utf-8")
        (self._out / "summary.json").write_text(json.dumps(self._summarize(rows), indent=2) + "\n", encoding="utf-8")
        (self._out / "summary.md").write_text(self._summary_markdown(rows), encoding="utf-8")
        return rows

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _bench_engine_language(self, *, engine: str, language: str) -> list[BenchRow]:
        rows: list[BenchRow] = []
        lang = str(language or "").strip().lower() or "en"
        engine_id = str(engine or "").strip().lower()

        if engine_id in ("auto", ""):
            # `auto` is a policy alias (deterministic default => piper).
            engine_id = "piper"

        # Warmup uses s2 (2 sentences) to (a) reduce cold-start variability and
        # (b) establish session-prompt caches for engines like AudioDiT.
        warmup_text = self._texts.get(lang, {}).get("s2") or self._texts["en"]["s2"]

        out_base = self._out / engine_id / lang
        out_base.mkdir(parents=True, exist_ok=True)

        vm: VoiceManager | None = None
        declared_supported: bool | None = None
        try:
            # INIT timing (includes eager model load for some engines).
            t0 = time.monotonic()
            vm = VoiceManager(
                language=str(lang),
                tts_engine=str(engine_id),
                allow_downloads=bool(self.cfg.allow_downloads),
                debug_mode=False,
            )
            init_s = time.monotonic() - t0
            rows.append(
                BenchRow(
                    engine=engine_id,
                    language=lang,
                    variant="-",
                    rep=None,
                    phase="init",
                    ok=True,
                    elapsed_s=float(init_s),
                    notes=self._stability_notes(vm, engine_id),
                )
            )
        except Exception as e:
            rows.append(
                BenchRow(
                    engine=engine_id,
                    language=lang,
                    variant="-",
                    rep=None,
                    phase="init",
                    ok=False,
                    error=str(e),
                    notes={"hint": self._install_hint(engine_id)},
                )
            )
            return rows

        assert vm is not None

        # Configure engine-specific voice-stability knobs (best-effort).
        self._configure_voice_stability(vm, engine_id)
        try:
            # Ensure language is applied to the adapter (important for Piper).
            vm.set_language(str(lang))
        except Exception:
            # Best-effort; if language isn't supported by the engine, we still want
            # to attempt synthesis and record failure or degraded behavior.
            pass

        # Compute "declared supported" once (adapter-reported language list).
        try:
            adapter = getattr(vm, "tts_adapter", None)
            if adapter is not None and hasattr(adapter, "get_supported_languages"):
                declared_supported = str(lang) in set(str(x) for x in adapter.get_supported_languages())
        except Exception:
            declared_supported = None

        # Warmup (not included in averages).
        rows.append(
            self._synthesize_row(
                vm,
                engine_id,
                lang,
                variant="s2",
                rep=0,
                phase="warmup",
                declared_supported=declared_supported,
                text=warmup_text,
                path=out_base / "warmup_s2.wav",
            )
        )

        # Main synth runs: N repetitions for each requested text variant.
        reps = int(self.cfg.repetitions)
        for variant in self.cfg.variants:
            v = str(variant or "").strip().lower()
            if not v:
                continue
            text = self._texts.get(lang, {}).get(v) or self._texts["en"].get(v)
            if not text:
                continue

            var_dir = out_base / v
            var_dir.mkdir(parents=True, exist_ok=True)

            for i in range(1, reps + 1):
                rows.append(
                    self._synthesize_row(
                        vm,
                        engine_id,
                        lang,
                        variant=v,
                        rep=i,
                        phase="synth",
                        declared_supported=declared_supported,
                        text=text,
                        path=var_dir / f"run_{i}.wav",
                    )
                )

        # Bit-identical signal (best-effort): if all synth runs for a variant have
        # the same sha256, that's a strong determinism indicator. We record the
        # result as a separate row so it stays close to the raw data.
        for variant in self.cfg.variants:
            v = str(variant or "").strip().lower()
            if not v:
                continue
            vals = [r.sha256 for r in rows if r.phase == "synth" and r.variant == v and r.ok and r.sha256]
            if not vals:
                continue
            rows.append(
                BenchRow(
                    engine=engine_id,
                    language=lang,
                    variant=v,
                    rep=None,
                    phase="repeat_equal",
                    ok=True,
                    notes={"bit_identical_all": bool(len(set(vals)) == 1), "n": int(len(vals))},
                )
            )

        # Best-effort cleanup (especially important for torch engines).
        try:
            vm.cleanup()
        except Exception:
            pass
        del vm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return rows

    def _synthesize_row(
        self,
        vm: VoiceManager,
        engine: str,
        language: str,
        *,
        variant: str,
        rep: int | None,
        phase: str,
        declared_supported: bool | None,
        text: str,
        path: Path,
    ) -> BenchRow:
        try:
            t0 = time.monotonic()
            out = vm.speak_to_file(str(text), str(path), format="wav", voice=None, sanitize_syntax=True)
            elapsed = time.monotonic() - t0

            # Read back WAV for duration + hash (use bytes on disk).
            b = Path(out).read_bytes()
            sha = hashlib.sha256(b).hexdigest()
            audio, sr = sf.read(str(out), always_2d=True, dtype="float32")
            mono = np.mean(audio, axis=1).reshape(-1)
            dur_s = float(mono.size) / float(sr) if int(sr) > 0 else None
            rtf = (float(elapsed) / float(dur_s)) if (dur_s and dur_s > 0) else None
            return BenchRow(
                engine=engine,
                language=language,
                variant=str(variant),
                rep=int(rep) if rep is not None else None,
                phase=phase,
                ok=True,
                elapsed_s=float(elapsed),
                wav_path=str(out),
                sample_rate=int(sr),
                audio_s=float(dur_s) if dur_s is not None else None,
                rtf=float(rtf) if rtf is not None else None,
                sha256=str(sha),
                declared_supported=declared_supported,
            )
        except Exception as e:
            return BenchRow(
                engine=engine,
                language=language,
                variant=str(variant),
                rep=int(rep) if rep is not None else None,
                phase=phase,
                ok=False,
                declared_supported=declared_supported,
                error=str(e),
                notes={"hint": self._prefetch_hint(engine)},
            )

    def _configure_voice_stability(self, vm: VoiceManager, engine: str) -> None:
        e = str(engine or "").strip().lower()
        adapter = getattr(vm, "tts_adapter", None)
        if adapter is None:
            return

        if e == "omnivoice":
            try:
                if self.cfg.omnivoice_instruct is not None:
                    adapter.set_param("instruct", str(self.cfg.omnivoice_instruct))
                adapter.set_param("seed", int(self.cfg.omnivoice_seed))
            except Exception:
                pass
            return

        if e == "audiodit":
            # No stable public knob yet; best-effort through internal settings.
            try:
                st = getattr(adapter, "_settings", None)
                if st is not None and hasattr(st, "seed"):
                    setattr(st, "seed", int(self.cfg.audiodit_seed))
            except Exception:
                pass
            return

        # Piper: deterministic by voice selection (no sampling).

    def _stability_notes(self, vm: VoiceManager, engine: str) -> dict[str, Any]:
        e = str(engine or "").strip().lower()
        notes: dict[str, Any] = {"engine": e}

        adapter = getattr(vm, "tts_adapter", None)
        try:
            notes["adapter"] = str(getattr(adapter, "__class__", type("x", (), {})).__name__)
        except Exception:
            pass
        try:
            notes["sample_rate"] = int(adapter.get_sample_rate()) if adapter is not None else None
        except Exception:
            pass
        try:
            notes["supported_languages"] = list(adapter.get_supported_languages()) if adapter is not None else None
        except Exception:
            notes["supported_languages"] = None

        if e == "piper":
            notes["voice_reuse"] = "deterministic (keep same language/voice)"
        elif e == "audiodit":
            notes["voice_reuse"] = "best-effort via seed + session prompt (session prompt is default)"
            notes["seed"] = int(self.cfg.audiodit_seed)
        elif e == "omnivoice":
            notes["voice_reuse"] = "best-effort via seed (+ instruct, temps)"
            notes["seed"] = int(self.cfg.omnivoice_seed)
            notes["instruct"] = self.cfg.omnivoice_instruct
        else:
            notes["voice_reuse"] = "unknown"
        return notes

    def _install_hint(self, engine: str) -> str:
        e = str(engine or "").strip().lower()
        if e == "audiodit":
            return 'pip install "abstractvoice[audiodit]"'
        if e == "omnivoice":
            return 'pip install "abstractvoice[omnivoice]"'
        if e == "piper":
            return 'pip install abstractvoice  # includes piper-tts'
        return "See docs/installation.md"

    def _prefetch_hint(self, engine: str) -> str:
        e = str(engine or "").strip().lower()
        if e == "audiodit":
            return "python -m abstractvoice download --audiodit"
        if e == "omnivoice":
            return "python -m abstractvoice download --omnivoice"
        if e == "piper":
            return "python -m abstractvoice download --piper en  # (and --piper fr for French)"
        return "See docs/model-management.md"

    def _to_markdown(self, rows: list[BenchRow]) -> str:
        lines: list[str] = []
        lines.append("# AbstractVoice TTS benchmark (no cloning)")
        lines.append("")
        lines.append(f"- out_dir: `{self.cfg.out_dir}`")
        lines.append(f"- engines: `{', '.join(self.cfg.engines)}`")
        lines.append(f"- languages: `{', '.join(self.cfg.languages)}`")
        lines.append(f"- variants: `{', '.join(self.cfg.variants)}`")
        lines.append(f"- repetitions: `{self.cfg.repetitions}`")
        lines.append(f"- allow_downloads: `{self.cfg.allow_downloads}`")
        lines.append("")

        hdr = "| engine | lang | variant | phase | rep | ok | declared | elapsed_s | audio_s | rtf | wav |"
        sep = "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"
        lines.append(hdr)
        lines.append(sep)
        for r in rows:
            if r.phase in ("init", "repeat_equal"):
                continue
            elapsed = f"{r.elapsed_s:.3f}" if isinstance(r.elapsed_s, float) else ""
            audio_s = f"{r.audio_s:.3f}" if isinstance(r.audio_s, float) else ""
            rtf = f"{r.rtf:.3f}" if isinstance(r.rtf, float) else ""
            wav = f"`{r.wav_path}`" if r.wav_path else ""
            declared = "" if r.declared_supported is None else ("yes" if r.declared_supported else "no")
            rep = "" if r.rep is None else str(int(r.rep))
            lines.append(
                f"| {r.engine} | {r.language} | {r.variant} | {r.phase} | {rep} | {str(r.ok).lower()} | {declared} | {elapsed} | {audio_s} | {rtf} | {wav} |"
            )

        lines.append("")
        lines.append("## Notes")
        lines.append("- `init` timings are stored in `results.json` (VoiceManager construction may eagerly load models).")
        lines.append("- `summary.md` contains averaged timings (mean/std).")
        lines.append("- `repeat_equal` rows in `results.json` record whether all synth repetitions were bit-identical for a variant.")
        lines.append("")
        return "\n".join(lines)

    def _summarize(self, rows: list[BenchRow]) -> dict[str, Any]:
        # Group by (engine, language, variant) for phase=synth.
        groups: dict[tuple[str, str, str], list[BenchRow]] = {}
        for r in rows:
            if r.phase != "synth":
                continue
            if not r.ok:
                continue
            k = (str(r.engine), str(r.language), str(r.variant))
            groups.setdefault(k, []).append(r)

        def _mean(xs: list[float]) -> float | None:
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

        out: dict[str, Any] = {"by_engine_language_variant": []}
        for (engine, language, variant), rs in sorted(groups.items()):
            elapsed = [float(r.elapsed_s) for r in rs if isinstance(r.elapsed_s, float)]
            audio_s = [float(r.audio_s) for r in rs if isinstance(r.audio_s, float)]
            rtf = [float(r.rtf) for r in rs if isinstance(r.rtf, float)]
            declared = rs[0].declared_supported if rs else None
            out["by_engine_language_variant"].append(
                {
                    "engine": engine,
                    "language": language,
                    "variant": variant,
                    "n": int(len(rs)),
                    "declared_supported": declared,
                    "mean_elapsed_s": _mean(elapsed),
                    "stdev_elapsed_s": _stdev(elapsed),
                    "mean_audio_s": _mean(audio_s),
                    "mean_rtf": _mean(rtf),
                }
            )
        return out

    def _summary_markdown(self, rows: list[BenchRow]) -> str:
        s = self._summarize(rows)
        lines: list[str] = []
        lines.append("# AbstractVoice TTS benchmark summary (averages)")
        lines.append("")
        lines.append(f"- out_dir: `{self.cfg.out_dir}`")
        lines.append(f"- repetitions: `{self.cfg.repetitions}` (excludes warmup)")
        lines.append("")
        lines.append("| engine | lang | text | declared | n | avg_wav_gen_s | sd_wav_gen_s | avg_speech_s | rtf |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in s.get("by_engine_language_variant", []):
            declared = row.get("declared_supported")
            declared_txt = "" if declared is None else ("yes" if declared else "no")

            def f(x: Any) -> str:
                return "" if x is None else f"{float(x):.3f}"

            lines.append(
                "| {engine} | {lang} | {variant} | {declared} | {n} | {mean_e} | {sd_e} | {mean_a} | {mean_r} |".format(
                    engine=row.get("engine", ""),
                    lang=row.get("language", ""),
                    variant=row.get("variant", ""),
                    declared=declared_txt,
                    n=int(row.get("n", 0) or 0),
                    mean_e=f(row.get("mean_elapsed_s")),
                    sd_e=f(row.get("stdev_elapsed_s")),
                    mean_a=f(row.get("mean_audio_s")),
                    mean_r=f(row.get("mean_rtf")),
                )
            )
        lines.append("")
        lines.append("Notes:")
        lines.append("- `text`: `s2` = 2 short sentences, `s5` = 5 short sentences, `s6` = 6 short sentences.")
        lines.append("- `avg_wav_gen_s`: average wall time to write a WAV via `VoiceManager.speak_to_file(...)`.")
        lines.append("- `avg_speech_s`: average duration of the generated WAV.")
        lines.append("- `rtf`: `avg_wav_gen_s / avg_speech_s` (lower is faster; `<1` means faster than real time).")
        lines.append("- `declared` is adapter-reported language support; synthesis may still succeed when `declared=no` (e.g. AudioDiT+FR is not guaranteed).")
        lines.append("")
        return "\n".join(lines)


def _parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Benchmark AbstractVoice TTS engines (no voice cloning).")
    p.add_argument("--out-dir", default="untracked/bench_tts", help="Output folder for WAVs + results.json")
    p.add_argument(
        "--engines",
        default="piper,audiodit,omnivoice",
        help="Comma-separated engine ids (subset of known engines). Default: piper,audiodit,omnivoice",
    )
    p.add_argument("--languages", default="en,fr", help="Comma-separated language codes. Default: en,fr")
    p.add_argument("--variants", default="s2,s5,s6", help="Comma-separated text variants. Default: s2,s5,s6")
    p.add_argument("--repetitions", type=int, default=4, help="Repetitions per variant (excludes warmup). Default: 4")
    p.add_argument("--allow-downloads", action="store_true", help="Allow on-demand downloads (not recommended for benchmarks).")
    p.add_argument("--audiodit-seed", type=int, default=1024, help="AudioDiT seed (best-effort). Default: 1024")
    p.add_argument("--omnivoice-seed", type=int, default=123, help="OmniVoice seed (stable voice design). Default: 123")
    p.add_argument(
        "--omnivoice-instruct",
        default="female, young adult, moderate pitch, american accent",
        help="OmniVoice voice design attributes (instruct). Default: a stable English preset.",
    )
    args = p.parse_args()

    engines = [s.strip().lower() for s in str(args.engines).split(",") if s.strip()]
    # Validate against known engines (keep the benchmark honest).
    known = set(get_supported_tts_engines())
    # Users often omit "auto"; we normalize it away above. Still validate here.
    bad = [e for e in engines if e not in known and e != "piper"]
    if bad:
        raise SystemExit(f"Unknown engines: {bad}. Known: {sorted(known)}")

    langs = [s.strip().lower() for s in str(args.languages).split(",") if s.strip()]
    variants = [s.strip().lower() for s in str(args.variants).split(",") if s.strip()]
    reps = int(args.repetitions)
    if reps < 1:
        raise SystemExit("--repetitions must be >= 1")

    instruct = str(args.omnivoice_instruct).strip()
    return BenchConfig(
        out_dir=str(args.out_dir),
        engines=engines,
        languages=langs,
        allow_downloads=bool(args.allow_downloads),
        variants=variants,
        repetitions=reps,
        audiodit_seed=int(args.audiodit_seed),
        omnivoice_seed=int(args.omnivoice_seed),
        omnivoice_instruct=instruct if instruct else None,
    )


def main() -> None:
    cfg = _parse_args()
    rows = TtsBenchmarker(cfg).run()
    ok = sum(1 for r in rows if r.ok and r.phase == "synth")
    total = sum(1 for r in rows if r.phase == "synth")
    print(f"Done. {ok}/{total} synth runs succeeded.")
    print(f"Results: {Path(cfg.out_dir) / 'results.json'}")


if __name__ == "__main__":
    main()

