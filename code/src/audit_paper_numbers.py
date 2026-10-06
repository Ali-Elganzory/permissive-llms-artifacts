#!/usr/bin/env python3
"""Check selected numerical prose claims against CSV-derived display values.

    uv run python src/audit_paper_numbers.py

The paper is what a reader checks, so a claim in the text must be reproducible from the
values the tables actually print, not only from the underlying per-sample data. Those two
can disagree by a rounding unit, because a difference of two rounded means is not the
rounded difference of two exact means. Each claim below is recomputed from CSV aggregates at the tables' display
precision and matched against the literal sentence in the LaTeX source, so the audit fails if
either the data or the prose moves.

Confidence intervals are exempt: they come from the item-level bootstrap and are not
derivable from a table of means.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol_b as pb
import wsdata as wd

LATEX = wd.ROOT / "latex"
MAIN = LATEX / "main_neurips_2026.tex"
APPENDIX = LATEX / "appendix" / "additional_results.tex"

_REASONING = wd.load_reasoning()
_FAC = wd.benchmark_summary(
    wd.benchmark_scores(wd.load_factorial(_REASONING), wd.FACTOR_COLS), wd.FACTOR_COLS)
_ZOO_KEYS = ["model_group", "context_length", "stage"]
_ZOO = wd.benchmark_summary(
    wd.benchmark_scores(wd.load_zoo(_REASONING), _ZOO_KEYS), _ZOO_KEYS)
_LANG = wd.language_average(wd.load_language()).set_index(
    ["model_group", "front_loading", "context_length"])["language"]


def P(pool: str, fl: bool, ctx: int, stage: str) -> float:
    """Suite average exactly as Table 1 and the per-benchmark tables print it."""
    r = _FAC[(_FAC.model_group == pool) & (_FAC.front_loading == fl)
             & (_FAC.context_length == ctx) & (_FAC.stage == stage)
             & (_FAC.benchmark == "__suite__")]
    return round(float(r["mean"].iloc[0]), 1)


def Z(group: str, ctx: int, stage: str) -> float:
    r = _ZOO[(_ZOO.model_group == group) & (_ZOO.context_length == ctx)
             & (_ZOO.stage == stage) & (_ZOO.benchmark == "__suite__")]
    return round(float(r["mean"].iloc[0]), 1)


def L(pool: str, fl: bool, ctx: int) -> float:
    """Language average exactly as the Language columns of Table 1 print it."""
    return round(float(_LANG.loc[(pool, fl, ctx)]), 1)


def eff(pool: str, ctx: int, stage: str) -> float:
    return round(P(pool, True, ctx, stage) - P(pool, False, ctx, stage), 1)


def gap(ref: str, ctx: int, stage: str) -> float:
    return round(P("MV", True, ctx, stage) - P(ref, True, ctx, stage), 1)


def pb_delta(name: str) -> float:
    start, end = next((a, b) for n, _, a, b in pb.PHASE2_PAIRED if n == name)
    return round(end - start, 1)


def pb_row(label: str) -> list:
    return next(v for l, v in pb.OT3_PRETRAIN_ABLATION if l.startswith(label))


POOLS = ("MV", "Nemotron", "FineWeb-Edu")
ALL_EFFECTS = [eff(p, c, s) for c in (4096, 16384) for s in ("base", "sft", "dpo") for p in POOLS]

# (label, file, template, values) -- template.format(*values) must appear verbatim.
CLAIMS = [
    ("abstract: front-loading effect range", MAIN,
     "by ${:.1f}$ to ${:.1f}$ points, with gains of", (min(ALL_EFFECTS), max(ALL_EFFECTS))),
    ("C1: front-loading effect range", MAIN,
     "in all eighteen cells of the design, by ${:.1f}$ to ${:.1f}$ points", (min(ALL_EFFECTS), max(ALL_EFFECTS))),
    ("S3: front-loading effect range", MAIN,
     "in all eighteen cells, by ${:.1f}$ to ${:.1f}$ points", (min(ALL_EFFECTS), max(ALL_EFFECTS))),
    ("S3: MV 16K SFT effect", MAIN,
     "the effect is $+{:.1f}$ points for the permissive substrate (${:.1f} \\to {:.1f}$)",
     (eff("MV", 16384, "sft"), P("MV", False, 16384, "sft"), P("MV", True, 16384, "sft"))),
    ("S3: Nemotron 16K SFT effect", MAIN,
     "$+{:.1f}$ for Nemotron-CC-HQ (${:.1f} \\to {:.1f}$)",
     (eff("Nemotron", 16384, "sft"), P("Nemotron", False, 16384, "sft"), P("Nemotron", True, 16384, "sft"))),
    ("S3: FineWeb-Edu 16K SFT effect", MAIN,
     "$+{:.1f}$ for FineWeb-Edu (${:.1f} \\to {:.1f}$)",
     (eff("FineWeb-Edu", 16384, "sft"), P("FineWeb-Edu", False, 16384, "sft"), P("FineWeb-Edu", True, 16384, "sft"))),
    ("S3: 4K SFT contrast vs Nemotron", MAIN,
     "leads Nemotron-CC-HQ by $+{:.1f}$ points", (gap("Nemotron", 4096, "sft"),)),
    ("S3: 4K SFT contrast vs FineWeb-Edu", MAIN,
     "and FineWeb-Edu by $+{:.1f}$ points", (gap("FineWeb-Edu", 4096, "sft"),)),
    ("S3: 16K SFT contrasts", MAIN,
     "at 16K it leads by $+{:.1f}$ ($[+3.2, +7.0]$) and $+{:.1f}$ points",
     (gap("Nemotron", 16384, "sft"), gap("FineWeb-Edu", 16384, "sft"))),
    ("abstract: 16K contrast range", MAIN,
     "is ahead by ${:.1f}$ to ${:.1f}$ points at 16K",
     tuple(sorted((gap("Nemotron", 16384, "sft"), gap("FineWeb-Edu", 16384, "sft"))))),
    ("C2: 4K and 16K contrasts", MAIN,
     "$+{:.1f}$ and $+{:.1f}$ points with intervals spanning zero, and leads at 16K by $+{:.1f}$ and $+{:.1f}$ points",
     (gap("Nemotron", 4096, "sft"), gap("FineWeb-Edu", 4096, "sft"),
      gap("Nemotron", 16384, "sft"), gap("FineWeb-Edu", 16384, "sft"))),
    ("S3: 4K base substrate lead", MAIN,
     "where it leads by ${:.1f}$ to ${:.1f}$ points",
     tuple(sorted((-gap("Nemotron", 4096, "base"), -gap("FineWeb-Edu", 4096, "base"))))),
    ("S3: language front-loading effects", MAIN,
     "by $+{:.1f}$ points for the permissive substrate (${:.1f} \\to {:.1f}$ at 16K) and $+{:.1f}$ for FineWeb-Edu (${:.1f} \\to {:.1f}$)",
     (round(L("MV", True, 16384) - L("MV", False, 16384), 1), L("MV", False, 16384), L("MV", True, 16384),
      round(L("FineWeb-Edu", True, 16384) - L("FineWeb-Edu", False, 16384), 1),
      L("FineWeb-Edu", False, 16384), L("FineWeb-Edu", True, 16384))),
    ("S3: language substrate values", MAIN,
     "reaches ${:.1f}$ against ${:.1f}$ for Nemotron-CC-HQ and ${:.1f}$ for FineWeb-Edu",
     (L("MV", True, 16384), L("Nemotron", True, 16384), L("FineWeb-Edu", True, 16384))),
    ("S3: residual language gap", MAIN,
     "a residual ${:.1f}$ to ${:.1f}$ points that front-loading does not recover",
     tuple(sorted((round(L("Nemotron", True, 16384) - L("MV", True, 16384), 1),
                   round(L("FineWeb-Edu", True, 16384) - L("MV", True, 16384), 1))))),
    ("C3: residual language gap", MAIN,
     "which remains at ${:.1f}$ to ${:.1f}$ points",
     tuple(sorted((round(L("Nemotron", True, 16384) - L("MV", True, 16384), 1),
                   round(L("FineWeb-Edu", True, 16384) - L("MV", True, 16384), 1))))),
    ("S4: zoo endpoints", MAIN,
     "it reaches ${:.1f}$, against ${:.1f}$ for Comma0.1, ${:.1f}$ for Nemotron-CC-HQ, and ${:.1f}$ for FineWeb-Edu",
     (Z("MV", 16384, "dpo"), Z("Comma0.1", 16384, "dpo"), Z("Nemotron", 16384, "dpo"), Z("FineWeb-Edu", 16384, "dpo"))),
    ("S4: DCLM", MAIN, "and ${:.1f}$ for DCLM at 4K", (Z("DCLM", 4096, "dpo"),)),
    ("S4: external references", MAIN,
     "exceeds SmolLM2 at 16K (${:.1f}$) and Qwen2.5 (${:.1f}$) and trails Qwen3 (${:.1f}$)",
     (Z("SmolLM2", 16384, "dpo"), Z("Qwen2.5", 4096, "dpo"), Z("Qwen3", 4096, "dpo"))),
    ("S4: 16K Tulu3 drop", MAIN,
     "strongest before Phase 1, at ${:.1f}$, and Tulu3 SFT lowers it to ${:.1f}$, with DPO recovering to ${:.1f}$",
     (Z("MV", 16384, "base"), Z("MV", 16384, "sft"), Z("MV", 16384, "dpo"))),
    ("S5: Protocol B paired deltas", MAIN,
     "by $-{:.1f}$ for Qwen2.5, $-{:.1f}$ for FineWeb-Edu, $-{:.1f}$ for Nemotron-CC-HQ, $-{:.1f}$ for SmolLM2, $-{:.1f}$ for Comma0.1, and $-{:.1f}$ for Qwen3",
     tuple(-pb_delta(n) for n in ("Qwen2.5", "FineWeb-Edu", "Nemotron-CC-HQ", "SmolLM2", "Comma0.1", "Qwen3"))),
    ("S5: MV Protocol B delta", MAIN,
     "by $+{:.1f}$ points from its post-YaRN base", (pb_delta("MixtureVitae"),)),
    ("S5: OT3 ablation endpoints", MAIN,
     "the suite average reaches ${:.1f}$, and without it the same post-training reaches ${:.1f}$",
     (pb_row("OT3 in pre-training")[11], pb_row("OT3 removed")[11])),
    ("S5: OT3 ablation gaps", MAIN,
     "The gap is ${:.1f}$ points on MATH500, ${:.1f}$ on AMC23, and ${:.1f}$ on HumanEval",
     (round(pb_row("OT3 in pre-training")[3] - pb_row("OT3 removed")[3], 1),
      round(pb_row("OT3 in pre-training")[2] - pb_row("OT3 removed")[2], 1),
      round(pb_row("OT3 in pre-training")[5] - pb_row("OT3 removed")[5], 1))),
    ("C4: OT3 ablation cost", MAIN,
     "costs ${:.1f}$ points of suite average and ${:.1f}$ points of MATH500",
     (round(pb_row("OT3 in pre-training")[11] - pb_row("OT3 removed")[11], 1),
      round(pb_row("OT3 in pre-training")[3] - pb_row("OT3 removed")[3], 1))),
    ("abstract: OT3 ablation", MAIN,
     "drops the suite average from ${:.1f}$ to ${:.1f}$",
     (pb_row("OT3 in pre-training")[11], pb_row("OT3 removed")[11])),
    ("S5: held-out MATH500 gains", MAIN,
     "gains $+{:.1f}$ points of MATH500 under OpenThoughts3 and $+{:.1f}$, $+{:.1f}$, and $+{:.1f}$",
     tuple(v[0] for v in pb.HELDOUT_MATH500["MixtureVitae"])),
]


def main() -> int:
    text = {MAIN: MAIN.read_text(), APPENDIX: APPENDIX.read_text()}
    failures = []
    for label, path, template, values in CLAIMS:
        expected = template.format(*values)
        if expected not in text[path]:
            failures.append((label, expected))
    for label, expected in failures:
        print(f"  FAIL  {label}\n        expected in {path.name}: {expected!r}")
    print(f"\n{len(CLAIMS) - len(failures)}/{len(CLAIMS)} derived claims reconcile "
          f"with the printed tables")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
