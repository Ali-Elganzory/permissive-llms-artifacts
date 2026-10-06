"""Paired item-level bootstrap for the Protocol A contrasts.

Scores are averaged over the three decoding seeds first, so the bootstrap quantifies
uncertainty due to the finite number of benchmark items, conditional on the trained
checkpoints.  Items are resampled within benchmark, which preserves the composition of
the eleven-benchmark suite average in every replicate.  JEEBench carries partial credit,
so items are treated as scores in [0, 1] rather than as Bernoulli outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wsdata import BENCHMARKS, CONTEXTS, POOLS, STAGES

N_BOOT = 2000
BOOT_SEED = 20260904


@dataclass(frozen=True)
class Contrast:
    family: str
    label: str
    context_length: int
    stage: str
    estimate: float
    lo: float
    hi: float
    p: float
    q: float = float("nan")


class ArmMatrix:
    """Items x arms matrix of seed-averaged scores for one (context, stage) cell."""

    def __init__(self, df: pd.DataFrame, context_length: int, stage: str):
        cell = df[(df["context_length"] == context_length) & (df["stage"] == stage)]
        item = (
            cell.groupby(
                ["model_group", "front_loading", "benchmark", "sample_id"], observed=True
            )["correct"]
            .mean()
            .reset_index()
        )
        item["arm"] = item["model_group"] + np.where(item["front_loading"], "|FL", "|noFL")
        wide = item.pivot_table(
            index=["benchmark", "sample_id"], columns="arm", values="correct"
        )
        if wide.isna().any().any():
            raise ValueError("arm matrix has missing item scores")
        self.arms = {name: i for i, name in enumerate(wide.columns)}
        self.values = wide.to_numpy() * 100.0
        benches = wide.index.get_level_values(0).to_numpy()
        self.blocks = [np.flatnonzero(benches == b) for b in BENCHMARKS]

    def _suite(self, index_blocks) -> np.ndarray:
        return np.mean([self.values[idx].mean(axis=0) for idx in index_blocks], axis=0)

    def observed(self) -> np.ndarray:
        return self._suite(self.blocks)

    def replicates(self, rng: np.random.Generator, n: int = N_BOOT) -> np.ndarray:
        out = np.empty((n, self.values.shape[1]))
        for i in range(n):
            drawn = [rng.choice(idx, size=idx.size, replace=True) for idx in self.blocks]
            out[i] = self._suite(drawn)
        return out


def _two_sided_p(draws: np.ndarray) -> float:
    """Bootstrap two-sided p-value for the null that the contrast is zero."""
    n = draws.size
    below = float(np.mean(draws <= 0.0))
    above = float(np.mean(draws >= 0.0))
    return float(min(1.0, 2.0 * min(below, above) + 1.0 / n))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    ranked = pvals[order] * pvals.size / np.arange(1, pvals.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def all_contrasts(factorial: pd.DataFrame) -> pd.DataFrame:
    """Front-loading effects, permissiveness contrasts, and the interaction.

    Each family is FDR-corrected separately with Benjamini-Hochberg.
    """
    rng = np.random.default_rng(BOOT_SEED)
    rows: list[Contrast] = []

    for context_length in CONTEXTS:
        for stage in STAGES:
            mat = ArmMatrix(factorial, context_length, stage)
            obs = mat.observed()
            rep = mat.replicates(rng)
            idx = mat.arms

            def record(family: str, label: str, o: float, d: np.ndarray) -> None:
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(
                    Contrast(family, label, context_length, stage, o, float(lo), float(hi),
                             _two_sided_p(d))
                )

            for pool in POOLS:
                record(
                    "front-loading",
                    pool,
                    obs[idx[f"{pool}|FL"]] - obs[idx[f"{pool}|noFL"]],
                    rep[:, idx[f"{pool}|FL"]] - rep[:, idx[f"{pool}|noFL"]],
                )

            for pool in ("Nemotron", "FineWeb-Edu"):
                record(
                    "permissiveness",
                    f"MV - {pool}",
                    obs[idx["MV|FL"]] - obs[idx[f"{pool}|FL"]],
                    rep[:, idx["MV|FL"]] - rep[:, idx[f"{pool}|FL"]],
                )

            mv_effect = rep[:, idx["MV|FL"]] - rep[:, idx["MV|noFL"]]
            np_effect = 0.5 * (
                (rep[:, idx["Nemotron|FL"]] - rep[:, idx["Nemotron|noFL"]])
                + (rep[:, idx["FineWeb-Edu|FL"]] - rep[:, idx["FineWeb-Edu|noFL"]])
            )
            mv_obs = obs[idx["MV|FL"]] - obs[idx["MV|noFL"]]
            np_obs = 0.5 * (
                (obs[idx["Nemotron|FL"]] - obs[idx["Nemotron|noFL"]])
                + (obs[idx["FineWeb-Edu|FL"]] - obs[idx["FineWeb-Edu|noFL"]])
            )
            record("interaction", "MV - non-permissive", mv_obs - np_obs, mv_effect - np_effect)

    out = pd.DataFrame([c.__dict__ for c in rows])
    out["q"] = np.nan
    for family, block in out.groupby("family", observed=True):
        out.loc[block.index, "q"] = benjamini_hochberg(block["p"].to_numpy())
    return out
