#!/usr/bin/env python3
"""Regenerate automated tables, plots, and analysis artifacts for the workshop paper.

    uv sync
    uv run python src/generate_workshop_artifacts.py

The script validates the CSV evaluation data against the experimental design, drops
checkpoint identifiers, and computes the summaries used by the Parquet frames, LaTeX
tables, and PDF plots.

Two evaluation protocols are in play and are never mixed.  Protocol A is the unified
greedy protocol behind Phase 1, the 300B factorial, and the mixture ablations, with three
decoding seeds.  Protocol B is the reasoning post-training protocol behind every
OpenThoughts3-30K experiment, with a single seed and no per-sample data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wsdata as wd
import wsfigures as wf
import wstables as wt

LATEX = wd.ROOT / "latex"
TABLES = LATEX / "tables"
FIGURES = LATEX / "figures"
PARQUET = wd.ROOT / "artifacts"

ZOO_KEYS = ["model_group", "context_length", "stage"]

# Figures and tables that this script owns.  Anything the paper references must appear
# here, and anything left over from an earlier draft is removed so it cannot be cited.
STALE_FIGURES = [
    "phase1_reasoning.pdf", "phase1_reasoning_bars.pdf", "phase1_reasoning_bars_v2.pdf",
    "phase1_reasoning_bars_v3.pdf", "phase1_language_bars_v2.pdf",
    "phase2_language_bars_v2.pdf", "phase2_reasoning_bars.pdf",
    "phase2_reasoning_bars_v2.pdf", "phase1_ablation_reasoning_v2.pdf",
    "phase1_ablation_language_v2.pdf", "language_compact_v1.pdf", "factorial_300b.pdf",
    "factorial_frontloading_generated.pdf", "phase1_trajectory_generated.pdf",
]


def banner(text: str) -> None:
    print(f"\n=== {text} ===")


def main() -> int:
    PARQUET.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    banner("Loading and validating Protocol A data")
    reasoning = wd.load_reasoning()
    factorial = wd.load_factorial(reasoning)
    zoo = wd.load_zoo(reasoning)
    language = wd.load_language()
    print(f"factorial {factorial.shape[0]:,} rows, zoo {zoo.shape[0]:,} rows, "
          f"language {language.shape[0]} rows")

    banner("Computing benchmark and suite summaries")
    fac_scores = wd.benchmark_scores(factorial, wd.FACTOR_COLS)
    zoo_scores = wd.benchmark_scores(zoo, ZOO_KEYS)
    fac_summary = wd.benchmark_summary(fac_scores, wd.FACTOR_COLS)
    zoo_summary = wd.benchmark_summary(zoo_scores, ZOO_KEYS)

    banner("Bootstrap contrasts")
    import wsstats as ws
    contrasts = ws.all_contrasts(factorial)
    fl = contrasts[contrasts.family == "front-loading"]
    print(f"front-loading effect positive in {int((fl.estimate > 0).sum())}/{len(fl)} cells, "
          f"range {fl.estimate.min():+.1f} to {fl.estimate.max():+.1f} pp")
    perm = contrasts[(contrasts.family == "permissiveness") & (contrasts.stage != "base")]
    print(f"permissiveness contrast after post-training: "
          f"{perm.estimate.min():+.2f} to {perm.estimate.max():+.2f} pp")

    banner("Writing Parquet frames")
    lang_avg = wd.language_average(language)
    for name, frame in [
        ("factorial_per_sample", factorial), ("zoo_per_sample", zoo),
        ("factorial_benchmark_summary", fac_summary), ("zoo_benchmark_summary", zoo_summary),
        ("language", language), ("language_average", lang_avg), ("contrasts", contrasts),
        ("equal_domain", wd.equal_domain_average(fac_scores, wd.FACTOR_COLS)),
        ("leave_one_domain_out", wd.leave_one_domain_out(fac_scores, wd.FACTOR_COLS)),
    ]:
        path = PARQUET / f"{name}.parquet"
        assert "model_id" not in frame.columns, f"{name} still carries checkpoint identifiers"
        frame.to_parquet(path, index=False)
        print(f"  {path.relative_to(wd.ROOT)}")

    banner("Writing LaTeX tables")
    written = [
        wt.factorial_summary(fac_summary, lang_avg, TABLES),
        wt.factorial_perbenchmark(fac_summary, TABLES),
        wt.zoo_perbenchmark(zoo_summary, TABLES, headline_only=True),
        wt.zoo_perbenchmark(zoo_summary, TABLES),
        wt.factorial_language(language, TABLES),
        wt.contrasts_table(contrasts, TABLES),
        wt.domain_robustness(fac_scores, TABLES),
        wt.phase2_endpoints(TABLES),
        wt.ot3_pretrain_ablation(TABLES),
        wt.heldout_reasoning(TABLES),
        wt.heldout_mv_grid(TABLES),
    ]
    for path in written:
        print(f"  {path.relative_to(wd.ROOT)}")

    banner("Writing figures")
    fac_suite = wd.suite_average(fac_scores, wd.FACTOR_COLS)
    zoo_suite = wd.suite_average(zoo_scores, ZOO_KEYS)
    figures = [
        wf.factorial_main(fac_suite, lang_avg, FIGURES),
        wf.frontloading_trajectory(contrasts, FIGURES),
        wf.phase1_zoo(zoo_suite, FIGURES),
        wf.phase2_receptivity(FIGURES),
        wf.frontloading_domains(fac_scores, FIGURES),
    ]
    for path in figures:
        print(f"  {path.relative_to(wd.ROOT)}")

    removed = 0
    for name in STALE_FIGURES:
        stale = FIGURES / name
        if stale.exists():
            stale.unlink()
            removed += 1
    print(f"removed {removed} superseded figure(s) from earlier drafts")

    banner("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
