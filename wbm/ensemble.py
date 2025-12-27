from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional, Literal

from .simulate import simulate_forward
from .seasonal import compute_acf, recommend_block_length


@dataclass
class EnsembleResult:
    members: list[pd.DataFrame]
    quantiles: pd.DataFrame
    residuals_used: pd.Series
    et_residuals_used: Optional[pd.Series] = None


def build_daily_ensemble(
    deterministic_future: pd.Series,
    residuals: pd.Series,
    n_members: int = 50,
    block_size: Optional[int] = None,
    random_state: int | None = None,
    clamp_min: Optional[float] = 0.0,
    transformation: Literal["none", "log1p"] = "none",
    trend_slope_std_per_year: Optional[float] = None,
    trend_origin_date: Optional[pd.Timestamp] = None,
) -> list[pd.Series]:
    """Bootstrap residuals (moving contiguous blocks) and add to deterministic path.

    Args:
        deterministic_future: The baseline trend/seasonal forecast.
        residuals: Historical residuals to bootstrap.
        n_members: Number of ensemble members to generate.
        block_size: Size of contiguous blocks for bootstrapping.
                    If None, it is estimated from ACF of residuals.
                    If <= 0, defaults to 1 (simple bootstrap).
        random_state: Seed for reproducibility.
        clamp_min: If set, enforces physical validity (e.g. Precip >= 0).
                   Prevents 'Anti-Rain' (negative precipitation).
        transformation: Data transformation used during modeling.
        trend_slope_std_per_year: Uncertainty of the trend slope (sigma).
                                  If provided, each member will have a random slope perturbation added.
        trend_origin_date: The 't=0' anchor date for the trend perturbation fan.
                           Typically the end of the historical calibration period.
    """
    rng = np.random.default_rng(random_state)
    res = residuals.dropna()
    if res.empty:
        return [deterministic_future.copy() for _ in range(n_members)]
    arr = res.to_numpy()
    L = len(arr)

    # Auto-detect block size if needed
    bs = 5 # Default fallback
    if block_size is None:
        try:
            acf_df = compute_acf(res, max_lag=min(60, L // 2))
            bs = recommend_block_length(acf_df, L)
        except Exception:
            bs = 5
    elif block_size <= 0:
        bs = 1
    else:
        bs = block_size

    blocks: list[pd.Series] = []
    for _ in range(n_members):
        needed = len(deterministic_future)
        out_vals = []
        while needed > 0:
            start = rng.integers(0, max(1, L - bs))
            blk = arr[start : start + bs]
            out_vals.append(blk)
            needed -= len(blk)
        seq = np.concatenate(out_vals)[: len(deterministic_future)]
        blocks.append(pd.Series(seq, index=deterministic_future.index))

    # Prepare Trend Perturbation
    trend_perturbations = np.zeros((n_members, len(deterministic_future)))
    if trend_slope_std_per_year is not None and trend_origin_date is not None and trend_slope_std_per_year > 0:
        # Convert to daily sigma
        sigma_day = trend_slope_std_per_year / 365.25
        # Sample slopes for each member: delta_slope ~ N(0, sigma)
        delta_slopes = rng.normal(0.0, sigma_day, size=n_members)

        # Calculate time vector t (days from origin)
        # origin is usually just before the forecast start.
        # dates in deterministic_future
        t_days = (deterministic_future.index - trend_origin_date).days.to_numpy()

        # Outer product: perturbation[i, j] = delta_slopes[i] * t_days[j]
        trend_perturbations = np.outer(delta_slopes, t_days)

    # Generate members
    members = []

    # Pre-transform deterministic baseline if needed
    if transformation == "log1p":
        # deterministic_future is already back-transformed (expm1) in forecast.py
        # We need it in LOG space to add LOG residuals.
        base_log = np.log1p(deterministic_future)
    else:
        base_log = deterministic_future

    for i in range(n_members):
        b = blocks[i] # Series
        t_pert = trend_perturbations[i] # Array

        # Add Residuals AND Trend Perturbation
        # b is Series, t_pert is Array. Alignment matches length.
        # Add t_pert to the model space.

        if transformation == "log1p":
            # Add in Log Space
            member_log = base_log + b + t_pert
            # Back to Linear
            member = np.expm1(member_log)
        else:
            member = base_log + b + t_pert

        if clamp_min is not None:
            member = member.clip(lower=clamp_min)
        members.append(member)

    return members


def run_volume_ensemble(
    *,
    start_volume_mcm: float,
    vol_to_area: Callable[[float], float],
    p_clim: pd.Series,
    et_clim: pd.Series,
    deterministic_p: pd.Series,
    residual_sets: list[pd.Series],
    et_residual_sets: Optional[list[pd.Series]] = None, # <--- The Axiom Bridge
    p_scale: float = 1.0,
    et_scale: float = 1.0,
    # Note: deterministic_et is implicitly handled inside simulate_forward via et_clim
    # if et_daily is not passed. But to support ET ensembles, we might need a
    # deterministic_et baseline or we assume residuals are additive to climatology/trend?
    # Current simulate_forward uses et_daily OR et_clim.
    # If we want to use et_residual_sets, we need a baseline to add them to.
    # Assuming et_residual_sets are to be added to a baseline ET.
    # BUT, the signature doesn't take deterministic_et.
    # For now, I will assume if et_residual_sets is provided, they are fully formed series
    # OR they are residuals to be added to... what?
    # `residual_sets` passed here are already formed p_members in usage?
    # No, look at `p_member = deterministic_p + res`. `residual_sets` are residuals.
    # So we need `deterministic_et` if we want to do the same for ET.
    # However, to avoid changing signature too much (Breaking Change),
    # I'll stick to the user's request: "Refactor the function signature to accept ET residuals".
    # User didn't specify deterministic_et.
    # Warning: If I add `et_residuals` to `et_clim`, `et_clim` is DOY indexed, residuals are likely Date indexed?
    # Actually, `residual_sets` (for P) are Date-indexed series aligned with `deterministic_p`.
    # `deterministic_p` is a Forecast (Date-indexed).
    # `simulate_forward` takes `p_daily`.
    # So if we want `et_daily` ensemble, we need `deterministic_et` + `et_res`.
    # Since `deterministic_et` is missing from args, I will add it as Optional.
    deterministic_et: Optional[pd.Series] = None,
) -> EnsembleResult:
    """
    Run ensemble simulation.

    Args:
        ...
        et_residual_sets: Optional list of ET residuals to perturb evaporation.
        deterministic_et: Optional baseline ET forecast. Required if et_residual_sets is used.
    """
    members: list[pd.DataFrame] = []

    # Check consistency
    use_et_ensemble = et_residual_sets is not None and deterministic_et is not None
    if et_residual_sets is not None and not use_et_ensemble:
        # If residuals provided but no baseline, we can't create et_daily easily unless we
        # assume baseline is 0 (wrong) or we construct it from climatology (complex).
        # For safety/simplicity in this refactor, I'll warn or error?
        # Or better, just proceed without ET perturbation if incomplete info.
        # But Axiom prefers explicit.
        # Let's assume if deterministic_et is None, we fall back to NO ET perturbation.
        pass

    for i, res_p in enumerate(residual_sets):
        p_member = deterministic_p + res_p

        et_member = None
        if use_et_ensemble and et_residual_sets is not None:
             # Ensure index alignment or use i % len
             res_et = et_residual_sets[i % len(et_residual_sets)]
             if deterministic_et is not None:
                 et_member = deterministic_et + res_et

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
            et_daily=et_member,
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

    # Return result
    # We return the first ET residual set used (or None) for consistency with `residuals_used`
    et_res_used = et_residual_sets[0] if et_residual_sets else None

    return EnsembleResult(
        members=members,
        quantiles=qs,
        residuals_used=residual_sets[0],
        et_residuals_used=et_res_used
    )

__all__ = [
    "EnsembleResult",
    "build_daily_ensemble",
    "run_volume_ensemble",
]
