"""Protocol B results, transcribed.

Protocol B is the reasoning post-training evaluation protocol: sampled decoding on
several benchmarks, longer generation budgets, LiveCodeBench v5, and prompt-level
IFEval.  Every OpenThoughts3-30K experiment uses it.  No per-sample data exists for
these runs, so they are single-seed point estimates and cannot carry seed error bars or
recomputed bootstrap intervals.  They are collected here, in one place, so that they are
versioned and checkable rather than scattered through the LaTeX.

Nothing in this module may be compared against a Protocol A number.
"""

from __future__ import annotations

BENCH_ORDER = ("A24", "A25", "AMC", "M500", "GSM8K", "HE", "LCB", "MBPP", "GPQA", "JEE", "IFE")

#: Phase 2 endpoints.  ``start`` names the checkpoint each OT30K run was initialised
#: from, and is ``None`` for the reference rows that are themselves starting points.
#: Values are transcribed from the submitted manuscript's Phase 2 table, which was
#: regenerated on fully decontaminated checkpoints.
PHASE2_ENDPOINTS = [
    # (group, row label, is_start, scores...)
    ("MV-16k", "base", True, [0.0, 10.0, 30.0, 33.6, 58.4, 48.2, 11.7, 41.0, 26.3, 8.3, 42.6, 28.2]),
    ("MV-16k", r"$\to$ SFT $\to$ OT30K", False, [20.0, 16.7, 29.8, 72.8, 52.5, 58.5, 10.0, 29.2, 36.4, 4.6, 25.6, 32.4]),
    ("MV-16k", r"$\to$ DPO $\to$ OT30K", False, [10.0, 6.7, 20.0, 36.4, 51.4, 46.3, 11.9, 40.8, 26.8, 11.5, 43.0, 27.7]),
    ("MV-16k", r"$\to$ OT30K \textbf{(no Phase 1)}", False, [20.0, 16.7, 25.0, 71.4, 58.9, 57.3, 10.0, 31.6, 26.3, 6.0, 26.6, 31.8]),
    ("FineWeb-Edu-16k", r"$\to$ SFT", True, [0.0, 0.0, 2.5, 1.4, 8.5, 2.4, 0.3, 2.2, 30.3, 3.0, 27.6, 7.1]),
    ("FineWeb-Edu-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 0.8, 9.2, 0.0, 0.0, 0.0, 23.7, 0.0, 17.8, 4.7]),
    ("Nemotron-16k", r"$\to$ SFT", True, [0.0, 0.0, 2.5, 1.4, 6.4, 1.5, 0.2, 0.2, 25.9, 3.3, 36.5, 7.1]),
    ("Nemotron-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 0.4, 5.3, 0.0, 0.0, 0.2, 21.2, 0.2, 20.0, 4.3]),
    ("Comma0.1-16k", r"$\to$ SFT", True, [0.0, 0.0, 5.0, 2.2, 5.8, 12.8, 0.8, 23.6, 30.3, 5.0, 35.4, 11.0]),
    ("Comma0.1-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 1.6, 14.0, 1.8, 0.0, 0.0, 28.8, 0.0, 20.0, 6.0]),
    ("SmolLM2-16k", r"$\to$ SFT", True, [0.0, 3.3, 2.5, 6.6, 31.9, 28.7, 3.3, 37.4, 25.8, 6.5, 45.0, 17.4]),
    ("SmolLM2-16k", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 14.0, 26.3, 23.2, 0.5, 15.8, 27.3, 3.3, 34.0, 13.1]),
    ("Qwen2.5", r"$\to$ SFT", True, [0.0, 0.0, 7.5, 19.6, 39.7, 48.2, 3.8, 44.6, 22.7, 7.5, 44.4, 21.6]),
    ("Qwen2.5", r"\quad $\to$ OT30K", False, [10.0, 6.7, 15.0, 49.2, 56.9, 20.7, 1.4, 8.4, 25.8, 2.1, 27.6, 20.3]),
    ("Qwen3", r"$\to$ SFT", True, [10.0, 0.0, 12.5, 23.0, 69.2, 64.6, 3.8, 55.8, 25.8, 11.7, 53.2, 30.0]),
    ("Qwen3", r"\quad $\to$ OT30K", False, [0.0, 0.0, 0.0, 10.6, 29.0, 7.3, 1.1, 10.6, 28.3, 0.5, 30.4, 10.7]),
]

#: Paired suite-average change under identical OT30K post-training, computed within
#: PHASE2_ENDPOINTS from each model's own Protocol B starting checkpoint.
#: (display label, starting point description, start avg, end avg)
PHASE2_PAIRED = [
    ("MixtureVitae", "post-YaRN base", 28.2, 31.8),
    ("FineWeb-Edu", "Tulu3 SFT", 7.1, 4.7),
    ("Nemotron-CC-HQ", "Tulu3 SFT", 7.1, 4.3),
    ("Comma0.1", "Tulu3 SFT", 11.0, 6.0),
    ("SmolLM2", "Tulu3 SFT", 17.4, 13.1),
    ("Qwen2.5", "Tulu3 SFT", 21.6, 20.3),
    ("Qwen3", "Tulu3 SFT", 30.0, 10.7),
]

#: Removing OpenThoughts3 from the 300B pre-training mixture, then applying the
#: identical OT30K reasoning post-training to both bases.  Single seed.  The matched
#: no-OT3 base evaluation was not complete, so only the two post-trained endpoints are
#: directly compared; the intact base is shown as a reference level.
OT3_PRETRAIN_ABLATION = [
    ("Base with OT3, before reasoning post-training",
     [3.3, 8.3, 25.6, 34.6, 54.1, 42.1, 8.4, 29.2, 27.3, 10.2, 45.2, 26.2]),
    (r"OT3 in pre-training, $\to$ OT30K",
     [11.7, 10.0, 47.5, 69.8, 56.4, 57.9, 12.7, 26.6, 28.3, 20.6, 27.6, 33.6]),
    (r"OT3 removed from pre-training, $\to$ OT30K",
     [0.0, 0.0, 17.5, 38.0, 53.3, 38.4, 6.2, 33.8, 28.8, 11.0, 36.4, 23.9]),
]

#: MATH500 change under four reasoning post-training corpora applied to the Tulu3 SFT
#: checkpoint of five bases.  95% paired bootstrap intervals over the same 500 problems.
#: ``sig`` is "+" for significantly positive and "-" for significantly negative after
#: Benjamini-Hochberg correction.
HELDOUT_CORPORA = ("OpenThoughts3", "AceReason", "Nemotron-Cascade", "Superior-Reasoning")
HELDOUT_MATH500 = {
    "MixtureVitae": [(35.4, 30.4, 40.4, "+"), (34.2, 29.4, 38.8, "+"),
                     (31.6, 27.0, 36.4, "+"), (16.4, 11.2, 21.4, "+")],
    "Comma0.1": [(-0.4, -2.2, 1.4, ""), (0.4, -1.4, 2.2, ""),
                 (0.8, -1.2, 2.8, ""), (0.0, -1.6, 1.6, "")],
    "FineWeb-Edu": [(-0.2, -1.6, 1.2, ""), (1.0, -0.4, 2.6, ""),
                    (0.6, -1.0, 2.2, ""), (-0.8, -2.0, 0.4, "")],
    "Nemotron-CC-HQ": [(-2.2, -3.8, -0.8, "-"), (0.0, -2.0, 2.0, ""),
                       (-0.8, -2.6, 1.0, ""), (-1.0, -2.8, 0.8, "")],
    "SmolLM2": [(-0.6, -3.4, 2.2, ""), (8.4, 5.0, 11.8, "+"),
                (6.6, 3.2, 10.2, "+"), (5.6, 2.2, 9.0, "+")],
}

#: The full MixtureVitae grid: four reasoning corpora applied from three starting
#: points.  ``None`` marks a benchmark that was not evaluated.
HELDOUT_MV_GRID = {
    "base": [
        ("starting checkpoint", [0.0, 10.0, 30.0, 33.6, 58.6, 48.2, 11.7, 41.0, 26.3, 8.3, 42.6, 28.2]),
        ("OpenThoughts3", [20.0, 16.7, 25.0, 70.2, 59.1, 57.3, 10.0, 31.6, 26.3, 6.0, 26.6, 31.7]),
        ("AceReason", [3.3, 13.3, 30.0, 69.2, 61.8, 52.4, 9.8, 35.6, 26.3, 10.4, 24.6, 30.6]),
        ("Nemotron-Cascade", [3.3, 6.7, 27.5, 70.0, 60.0, 49.4, 11.4, 35.8, 26.8, 10.2, 29.0, 30.0]),
        ("Superior-Reasoning", [3.3, 3.3, 20.0, 51.0, 62.5, 35.4, 6.5, 12.2, 28.3, 4.2, 26.4, 23.0]),
    ],
    "sft": [
        ("starting checkpoint", [0.0, 0.0, 5.0, 35.2, 51.0, 45.1, 8.1, 40.2, 28.3, 6.5, 41.8, 23.8]),
        ("OpenThoughts3", [6.7, 10.0, 20.0, 70.6, 50.7, 58.5, 10.0, 29.2, 36.4, 4.6, 25.6, 29.3]),
        ("AceReason", [3.3, 20.0, 25.0, 69.4, 54.4, 48.2, 10.8, 33.6, 27.3, 9.9, 25.6, 29.8]),
        ("Nemotron-Cascade", [3.3, 13.3, 27.5, 66.8, 52.6, 50.6, None, 32.8, 34.8, 7.3, 30.6, 32.0]),
        ("Superior-Reasoning", [6.7, 3.3, 20.0, 51.6, 55.6, 22.0, 6.2, 15.6, 25.8, 4.9, 27.0, 21.7]),
    ],
    "dpo": [
        ("starting checkpoint", [0.0, 0.0, 2.5, 33.2, 51.9, 47.6, 8.7, 40.4, 27.8, 8.9, 45.0, 24.2]),
        ("OpenThoughts3", [10.0, 6.7, 20.0, 36.4, 50.7, 44.5, 11.9, 40.8, 26.8, 11.5, 43.0, 27.5]),
        ("AceReason", [6.7, 16.7, 27.5, 70.8, 55.4, 48.2, 11.4, 32.6, 23.7, 9.8, 25.0, 29.8]),
        ("Nemotron-Cascade", [6.7, 10.0, 25.0, 68.2, 53.5, 50.0, 13.3, 34.8, 28.8, 9.7, 30.4, 30.0]),
        ("Superior-Reasoning", [3.3, 6.7, 20.0, 54.2, 55.1, 28.7, 5.7, 13.2, 24.7, 5.3, 28.0, 22.3]),
    ],
}

STARTING_POINT_LABEL = {
    "base": "post-YaRN base",
    "sft": "Tulu3 SFT",
    "dpo": "Tulu3 SFT + DPO",
}


def _self_check() -> None:
    """Row lengths and the suite averages that can be recomputed from the row itself."""
    for _, _, _, row in PHASE2_ENDPOINTS:
        assert len(row) == 12, row
        assert abs(sum(row[:11]) / 11 - row[11]) < 0.15, row
    for label, row in OT3_PRETRAIN_ABLATION:
        assert len(row) == 12, label
        assert abs(sum(row[:11]) / 11 - row[11]) < 0.15, (label, sum(row[:11]) / 11, row[11])
    for corpora in HELDOUT_MATH500.values():
        assert len(corpora) == len(HELDOUT_CORPORA)
        for est, lo, hi, _ in corpora:
            assert lo <= est <= hi, (est, lo, hi)
    for stage, rows in HELDOUT_MV_GRID.items():
        for label, row in rows:
            assert len(row) == 12, (stage, label)
            present = [v for v in row[:11] if v is not None]
            assert abs(sum(present) / len(present) - row[11]) < 0.15, (stage, label)


_self_check()
