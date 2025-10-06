from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable

from .simulate import simulate_forward


@dataclass
class EnsembleResult:
    members: list[pd.DataFrame]
    quantiles: pd.DataFrame
    residuals_used: pd.Series


def build_daily_ensemble(
    deterministic_future: pd.Series,
    residuals: pd.Series,
    n_members: int = 50,
    block_size: int = 5,
    random_state: int | None = None,
) -> list[pd.Series]:
    """Bootstrap residuals (moving contiguous blocks) and add to deterministic path."""
    rng = np.random.default_rng(random_state)
    res = residuals.dropna()
    if res.empty:
        return [deterministic_future.copy() for _ in range(n_members)]
    arr = res.to_numpy()
    L = len(arr)
    blocks: list[pd.Series] = []
    for _ in range(n_members):
        needed = len(deterministic_future)
        out_vals = []
        while needed > 0:
            start = rng.integers(0, max(1, L - block_size))
            blk = arr[start : start + block_size]
            out_vals.append(blk)
            needed -= len(blk)
        seq = np.concatenate(out_vals)[: len(deterministic_future)]
        blocks.append(pd.Series(seq, index=deterministic_future.index))
    members = [deterministic_future + b for b in blocks]
    return members


def run_volume_ensemble(
    *,
    start_volume_mcm: float,
    vol_to_area: Callable[[float], float],
    p_clim: pd.Series,
    et_clim: pd.Series,
    deterministic_p: pd.Series,
    residual_sets: list[pd.Series],
    p_scale: float = 1.0,
    et_scale: float = 1.0,
) -> EnsembleResult:
    members: list[pd.DataFrame] = []
    for res in residual_sets:
        p_member = deterministic_p + res
        sim = simulate_forward(
            deterministic_p.index[0],
            deterministic_p.index[-1],
            start_volume_mcm,
            p_clim,
            et_clim,
            vol_to_area,
            p_scale=p_scale,
            et_scale=et_scale,
            p_daily=p_member,
        )
        members.append(sim)

    aligned = []
    for df in members:
        aligned.append(df.set_index("date")["volume_mcm"])
    all_vols = pd.concat(aligned, axis=1)
    qs = all_vols.quantile([0.05, 0.5, 0.95], axis=1).T
    qs.columns = ["vol_q5", "vol_q50", "vol_q95"]
    qs.reset_index(inplace=True)
    qs.rename(columns={"index": "date"}, inplace=True)
    return EnsembleResult(members=members, quantiles=qs, residuals_used=residual_sets[0])

__all__ = [
    "EnsembleResult",
    "build_daily_ensemble",
    "run_volume_ensemble",
]
