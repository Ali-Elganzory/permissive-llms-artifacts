"""Loading, validation, and aggregation of the Protocol A per-sample evaluation data.

Protocol A is the unified all-greedy evaluation protocol used for Phase 1 across the
model zoo, for the 300B front-loading x source-pool factorial, and for the mixture
ablations.  Protocol B (reasoning post-training) numbers are transcribed constants and
live in ``protocol_b.py``: the two must never be mixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REASONING_CSV = ROOT / "artifacts" / "canonical_reasoning_per_sample.csv"
LANGUAGE_CSV = ROOT / "artifacts" / "ablation_frontloading_x_permissiveness_language_evaluations.csv"

SEEDS = (42, 1234, 2025)
STAGES = ("base", "sft", "dpo")
CONTEXTS = (4096, 16384)

# Canonical suite order and display names.
BENCHMARKS = (
    "AIME24",
    "AIME25",
    "AMC23",
    "MATH500",
    "gsm8k",
    "HumanEval",
    "LiveCodeBench",
    "MBPP",
    "GPQADiamond",
    "JEEBench",
    "IFEval",
)
BENCH_ABBR = {
    "AIME24": "A24",
    "AIME25": "A25",
    "AMC23": "AMC",
    "MATH500": "M500",
    "gsm8k": "GSM8K",
    "HumanEval": "HE",
    "LiveCodeBench": "LCB",
    "MBPP": "MBPP",
    "GPQADiamond": "GPQA",
    "JEEBench": "JEE",
    "IFEval": "IFE",
}
# Number of scored items per benchmark.  JEEBench carries partial credit
# (0.25/0.5/0.75), so `correct` is a score in [0, 1] and is not binary.
BENCH_ITEMS = {
    "AIME24": 30,
    "AIME25": 30,
    "AMC23": 40,
    "MATH500": 500,
    "gsm8k": 1319,
    "HumanEval": 164,
    "LiveCodeBench": 511,
    "MBPP": 500,
    "GPQADiamond": 198,
    "JEEBench": 515,
    "IFEval": 769,
}
DOMAIN = {
    "AIME24": "math",
    "AIME25": "math",
    "AMC23": "math",
    "MATH500": "math",
    "gsm8k": "math",
    "HumanEval": "code",
    "LiveCodeBench": "code",
    "MBPP": "code",
    "GPQADiamond": "science",
    "JEEBench": "science",
    "IFEval": "instruction",
}
DOMAINS = ("math", "code", "science", "instruction")

# The 11-task language-understanding suite.  social_iqa is present in the raw file but
# is not part of the suite reported in the paper and is dropped.
LANGUAGE_TASKS = (
    "mmlu",
    "hellaswag",
    "commonsense_qa",
    "arc_challenge",
    "arc_easy",
    "piqa",
    "boolq",
    "winogrande",
    "openbookqa",
    "copa",
    "lambada_openai",
)
LANGUAGE_ABBR = {
    "mmlu": "MMLU",
    "hellaswag": "HellaS",
    "commonsense_qa": "CSQA",
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "winogrande": "WinoG",
    "openbookqa": "OBQA",
    "copa": "COPA",
    "lambada_openai": "LAMBADA",
}

POOLS = ("MV", "Nemotron", "FineWeb-Edu")
POOL_LABEL = {
    "MV": "MixtureVitae",
    "Nemotron": "Nemotron-CC-HQ",
    "FineWeb-Edu": "FineWeb-Edu",
}
ZOO_ORDER = (
    "MV",
    "Nemotron",
    "FineWeb-Edu",
    "DCLM",
    "Comma0.1",
    "SmolLM2",
    "Qwen2.5",
    "Qwen3",
)
ZOO_LABEL = {
    "MV": "MixtureVitae",
    "Nemotron": "Nemotron-CC-HQ",
    "FineWeb-Edu": "FineWeb-Edu",
    "DCLM": "DCLM",
    "Comma0.1": "Comma0.1",
    "SmolLM2": "SmolLM2",
    "Qwen2.5": "Qwen2.5",
    "Qwen3": "Qwen3",
}
# Pretraining token budget in trillions, for the zoo table.
ZOO_TOKENS = {
    "MV": "0.3",
    "Nemotron": "0.3",
    "FineWeb-Edu": "0.3",
    "DCLM": "0.3",
    "Comma0.1": "0.3",
    "SmolLM2": "11",
    "Qwen2.5": "18",
    "Qwen3": "36",
}

FACTOR_COLS = ["model_group", "front_loading", "context_length", "stage"]
ITEM_KEY = ["benchmark", "sample_id"]


class ValidationError(AssertionError):
    """Raised when the raw evaluation data does not match the expected design."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_per_sample(df: pd.DataFrame, name: str, factors: list[str]) -> None:
    """Structural checks shared by both per-sample tables."""
    _check(not df.isna().any().any(), f"{name}: null values present")
    _check(
        df["correct"].between(0.0, 1.0).all(),
        f"{name}: `correct` outside [0, 1]",
    )
    _check(
        sorted(df["seed"].unique()) == sorted(SEEDS),
        f"{name}: expected seeds {SEEDS}, found {sorted(df['seed'].unique())}",
    )
    _check(
        set(df["benchmark"].unique()) == set(BENCHMARKS),
        f"{name}: benchmark set mismatch",
    )
    _check(
        sorted(df["stage"].unique()) == sorted(STAGES),
        f"{name}: stage set mismatch",
    )

    dup = df.duplicated(subset=factors + ["seed"] + ITEM_KEY).sum()
    _check(dup == 0, f"{name}: {dup} duplicate (factors, seed, benchmark, sample_id) keys")

    counts = df.groupby("benchmark")["sample_id"].nunique()
    for bench, expected in BENCH_ITEMS.items():
        _check(
            counts[bench] == expected,
            f"{name}: {bench} has {counts[bench]} items, expected {expected}",
        )

    # Every (arm, seed) cell must score every item of every benchmark exactly once.
    per_cell = df.groupby(factors + ["seed"], observed=True).size()
    expected_rows = sum(BENCH_ITEMS.values())
    bad = per_cell[per_cell != expected_rows]
    _check(bad.empty, f"{name}: {len(bad)} cells with wrong row count\n{bad.head()}")


def load_reasoning() -> pd.DataFrame:
    """Load and validate the Protocol A per-sample results."""
    df = pd.read_csv(REASONING_CSV)
    _validate_per_sample(df, "reasoning", FACTOR_COLS)
    for column in ("front_loading", "in_factorial", "in_zoo"):
        _check(df[column].dtype == bool, f"reasoning: {column} must be boolean")
    _check(df.groupby(FACTOR_COLS)["model_id"].nunique().eq(1).all(),
           "reasoning: multiple model IDs for one checkpoint")
    _check(df.groupby("model_id")[FACTOR_COLS].nunique().eq(1).all().all(),
           "reasoning: one model ID describes multiple checkpoints")
    _check(df.groupby(FACTOR_COLS)[["in_factorial", "in_zoo"]].nunique().eq(1).all().all(),
           "reasoning: inconsistent experiment membership")
    _check((df["in_factorial"] | df["in_zoo"]).all(),
           "reasoning: checkpoint belongs to no experiment")
    return df.drop(columns=["model_id"])


def load_factorial(reasoning: pd.DataFrame) -> pd.DataFrame:
    """Select the 3 x 2 factorial at 300B tokens."""
    df = reasoning[reasoning["in_factorial"]].drop(columns=["in_factorial", "in_zoo"])
    _validate_per_sample(df, "factorial", FACTOR_COLS)
    design = pd.MultiIndex.from_product([POOLS, (False, True), CONTEXTS, STAGES],
                                        names=FACTOR_COLS)
    cells = pd.MultiIndex.from_frame(df[FACTOR_COLS].drop_duplicates())
    _check(cells.sort_values().equals(design.sort_values()),
           "factorial: design not fully crossed")
    return df


def load_zoo(reasoning: pd.DataFrame) -> pd.DataFrame:
    """Select Phase 1 across the full model zoo."""
    df = reasoning[reasoning["in_zoo"]].drop(
        columns=["front_loading", "in_factorial", "in_zoo"])
    _validate_per_sample(df, "zoo", ["model_group", "context_length", "stage"])
    _check(set(df["model_group"].unique()) == set(ZOO_ORDER),
           "zoo: model group set mismatch")
    return df


def load_language() -> pd.DataFrame:
    """Language-understanding scores for the twelve factorial checkpoints.

    Single seed, and only the base and post-YaRN 16K checkpoints were evaluated.
    """
    df = pd.read_csv(LANGUAGE_CSV)
    df = df[df["task"].isin(LANGUAGE_TASKS)].copy()
    _check(len(df) == 12 * len(LANGUAGE_TASKS), f"language: unexpected row count {len(df)}")

    name = df["model_name"].str.replace("open-sci/open-sci-ref-v0.02-1.7b-", "", regex=False)
    df["context_length"] = np.where(name.str.contains("longsft_16k"), 16384, 4096)
    stem = name.str.replace("-longsft_16k", "", regex=False)
    df["model_group"] = np.select(
        [stem.str.startswith("mixturevitae"), stem.str.startswith("nemotron-hq"),
         stem.str.startswith("fineweb-edu")],
        ["MV", "Nemotron", "FineWeb-Edu"],
        default="?",
    )
    df["front_loading"] = ~stem.str.contains("noinstruct") & (
        stem.str.contains("mv_reasoning") | stem.str.startswith("mixturevitae")
    )
    _check((df["model_group"] != "?").all(), "language: unmapped model group")
    _check(
        df.groupby(["model_group", "front_loading", "context_length"]).ngroups == 12,
        "language: design not fully crossed",
    )
    df["performance"] = df["performance"] * 100.0
    return df.drop(columns=["model_name"])


def benchmark_scores(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Per-(arm, seed, benchmark) accuracy in percent."""
    out = (
        df.groupby(factors + ["seed", "benchmark"], observed=True)["correct"]
        .mean()
        .mul(100.0)
        .reset_index(name="score")
    )
    out["domain"] = out["benchmark"].map(DOMAIN)
    return out


def suite_average(scores: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Unweighted mean over the eleven benchmarks, per seed, then mean +/- SD over seeds."""
    per_seed = (
        scores.groupby(factors + ["seed"], observed=True)["score"].mean().reset_index(name="suite")
    )
    agg = (
        per_seed.groupby(factors, observed=True)["suite"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
    )
    return agg


def equal_domain_average(scores: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Mean over the four domain means, so that the five math benchmarks do not
    dominate the suite average."""
    dom = (
        scores.groupby(factors + ["seed", "domain"], observed=True)["score"]
        .mean()
        .reset_index(name="dom")
    )
    per_seed = dom.groupby(factors + ["seed"], observed=True)["dom"].mean().reset_index(name="suite")
    return (
        per_seed.groupby(factors, observed=True)["suite"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
    )


def leave_one_domain_out(scores: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Suite average recomputed with each domain removed in turn."""
    frames = []
    for dropped in DOMAINS:
        kept = scores[scores["domain"] != dropped]
        agg = suite_average(kept, factors)
        agg["dropped"] = dropped
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def language_average(lang: pd.DataFrame) -> pd.DataFrame:
    return (
        lang.groupby(["model_group", "front_loading", "context_length"], observed=True)["performance"]
        .mean()
        .reset_index(name="language")
    )


def benchmark_summary(scores: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """Per-benchmark mean +/- SD over seeds, plus a ``__suite__`` pseudo-benchmark row
    carrying the eleven-benchmark suite average."""
    per_bench = (
        scores.groupby(factors + ["benchmark"], observed=True)["score"]
        .agg(mean="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
    )
    suite = suite_average(scores, factors)
    suite["benchmark"] = "__suite__"
    return pd.concat([per_bench, suite], ignore_index=True)
