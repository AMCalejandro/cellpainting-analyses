"""Domain-agnostic engine shared by `imaging.reversion` and
`proteomics.reversion`: the reversion axis, per-well projection, and every
DMSO-bootstrap-calibrated gate that doesn't depend on a per-well cell count.

Axis, from each condition's DMSO-control centroid:

    u = (mu_B - mu_s) / L,  L = ||mu_B - mu_s||

Per-well scores (projection onto the axis, from mu_s):

    rho_w = <x_w - mu_s, u> / L   (fraction of the Baseline-Stress gap closed)
    a_w   = cos(x_w - mu_s, u)    (is the move actually along the axis?)

Gates (2) consistency, (2b) leave-one-out robustness, and (4) stress-specific
difference-in-differences all live here (`_consistency_table`,
`_add_specificity`), along with the promiscuity outlier rule
(`_promiscuity_gate`). Gate (3)'s cell-count viability check has no analog on
a proteomics plate, so `_cytotox_table`/`compute_reversion` stay in each
domain's own `reversion` module -- see `imaging.reversion` and
`proteomics.reversion` for the full gate spec, the assembled `compute_reversion`,
and `docs/reversion_pipeline_final.md`."""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

N_BOOT = 10000
SEED = 0
CTRL_PERCENTILE = 95
# BH level for the FDR-controlled gates (2 and 4). Matches utils.copairs's
# activity/distinctiveness convention.
FDR_Q = 0.10
# Promiscuity outlier rule: median + k*MAD of beta_perp over the screened
# pool -- adapts to the pool's actual spread instead of rejecting a fixed
# fraction of any pool by construction.
PROMISCUITY_MAD_K = 3.0
# Replicate-level bootstrap for the reported confidence intervals (resamples
# a compound's own replicates), separate from the DMSO nulls below.
N_CI = 2000
CI_ALPHA = 0.10


def _condition_mask(meta: pd.DataFrame, condition: str) -> np.ndarray:
    return (meta["Metadata_condition"] == condition).to_numpy()


def _control_mask(meta: pd.DataFrame) -> np.ndarray:
    return (meta["Metadata_pert_type"] == "negcon").to_numpy()


def _treated_mask(
    meta: pd.DataFrame, compound_allowlist: Optional[Iterable[str]] = None
) -> np.ndarray:
    mask = meta["Metadata_pert_type"] == "trt"
    if compound_allowlist is not None:
        mask &= meta["Metadata_broad_sample"].isin(set(compound_allowlist))
    return mask.to_numpy()


def compute_axis(
    meta: pd.DataFrame,
    feats: np.ndarray,
    baseline_condition: str,
    stress_condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """mu_B, mu_s, u, L from each condition's DMSO-control centroid."""
    base_ctrl = _condition_mask(meta, baseline_condition) & _control_mask(meta)
    stress_ctrl = _condition_mask(meta, stress_condition) & _control_mask(meta)
    if not base_ctrl.any():
        raise ValueError(f"no DMSO controls found for condition {baseline_condition!r}")
    if not stress_ctrl.any():
        raise ValueError(f"no DMSO controls found for condition {stress_condition!r}")

    mu_b = feats[base_ctrl].mean(axis=0)
    mu_s = feats[stress_ctrl].mean(axis=0)
    diff = mu_b - mu_s
    L = float(np.linalg.norm(diff))
    u = diff / L
    return mu_b, mu_s, u, L


def axis_scores(
    x: np.ndarray, mu_s: np.ndarray, u: np.ndarray, L: float
) -> tuple[np.ndarray, np.ndarray]:
    """rho_w = <x - mu_s, u> / L  and  a_w = cos(x - mu_s, u) for every row of x."""
    delta = x - mu_s
    rho = (delta @ u) / L
    norm_delta = np.linalg.norm(delta, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(norm_delta > 0, (delta @ u) / norm_delta, np.nan)
    return rho, a


def _sample_without_replacement(
    pool_size: int, sample_size: int, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorized equivalent of drawing `n_draws` independent size-`sample_size`
    subsets from `range(pool_size)`, without replacement within a draw."""
    rand_vals = rng.random((n_draws, pool_size))
    return np.argsort(rand_vals, axis=1)[:, :sample_size]


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values (q-values), NaN-safe.

    NaNs (compounds whose statistic is not estimable) are excluded from the
    test family rather than counted in `n` -- including them would inflate
    the correction with tests that were never performed."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    scaled = pv[order] * n / np.arange(1, n + 1)
    # Step-up: a q-value can never exceed that of a larger p-value.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(scaled, 0.0, 1.0)
    q[ok] = out
    return q


def _bootstrap_a_mean_null(
    a_ctrl: np.ndarray, n_reps: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """a_mean of `n_boot` fake compounds, each `n_reps` stress-DMSO wells.
    Calibrates the ALIGNMENT gate: rho alone is magnitude-dominated
    (rho_w = a_w * ||delta_w|| / L), so direction needs its own null."""
    idx = _sample_without_replacement(len(a_ctrl), n_reps, n_boot, rng)
    return a_ctrl[idx].mean(axis=1)


def _bootstrap_rho_mean_null(
    rho_ctrl: np.ndarray, n_reps: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """rho_mean of `n_boot` fake compounds, each `n_reps` stress-DMSO wells
    drawn without replacement in place of a compound's real replicates.
    Calibrates gate 2 (n_reps) and, reused at n_reps-1, the LOO gate."""
    idx = _sample_without_replacement(len(rho_ctrl), n_reps, n_boot, rng)
    return rho_ctrl[idx].mean(axis=1)


def _replicate_ci(
    values: np.ndarray,
    n_ci: int,
    alpha: float,
    rng: np.random.Generator,
    offset: float = 0.0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`, resampling the
    compound's own replicates WITH replacement. `offset` is subtracted from
    every resampled mean, so passing `beta_par` gives a CI for `rho_int`
    rather than for `rho_mean`."""
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    draws = values[rng.integers(0, n, size=(n_ci, n))]
    means = draws.mean(axis=1) - offset
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def _bootstrap_beta_par_null(
    ctrl_feats: np.ndarray,
    mu_b: np.ndarray,
    u: np.ndarray,
    L: float,
    n_reps: int,
    n_boot: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> np.ndarray:
    """beta_par of `n_boot` fake compounds, each the mean of `n_reps`
    Baseline-DMSO wells drawn without replacement, in place of a compound's
    real Baseline-arm replicates. Calibrates `tau_par_n`. Chunked to bound
    peak memory."""
    idx = _sample_without_replacement(len(ctrl_feats), n_reps, n_boot, rng)
    beta_par = np.empty(n_boot)
    for start in range(0, n_boot, chunk_size):
        chunk_idx = idx[start : start + chunk_size]
        disp = ctrl_feats[chunk_idx].mean(axis=1) - mu_b
        beta_par[start : start + chunk_size] = (disp @ u) / L
    return beta_par


def _bootstrap_rho_int_null(
    rho_ctrl_s: np.ndarray,
    rho_ctrl_b: np.ndarray,
    n_stress: int,
    n_baseline: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """rho_int of `n_boot` fake compounds: the mean of `n_stress` stress-DMSO
    wells' rho minus the mean of `n_baseline` Baseline-DMSO wells' on-axis
    displacement, both drawn without replacement. Both arms are centered on
    their own condition's control centroid by construction, so this null is
    centered on zero -- it measures only the replicate-count-matched sampling
    noise of the difference."""
    idx_s = _sample_without_replacement(len(rho_ctrl_s), n_stress, n_boot, rng)
    idx_b = _sample_without_replacement(len(rho_ctrl_b), n_baseline, n_boot, rng)
    return rho_ctrl_s[idx_s].mean(axis=1) - rho_ctrl_b[idx_b].mean(axis=1)


def _consistency_table(
    meta: pd.DataFrame,
    feats: np.ndarray,
    stress_condition: str,
    mu_s: np.ndarray,
    u: np.ndarray,
    L: float,
    n_boot: int,
    seed: int,
    compound_allowlist: Optional[Iterable[str]] = None,
    n_ci: int = N_CI,
    ci_alpha: float = CI_ALPHA,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Gates (2) and (2b, leave-one-out): rho_mean/a_mean against their
    DMSO nulls, and rho_loo_min against a mean-of-(n-1) null. Returns
    (per_compound, rho_ctrl, rho_by_compound) -- rho_ctrl and rho_by_compound
    are reused by the gate-4 null and CI in `_add_specificity`."""
    stress_ctrl = _condition_mask(meta, stress_condition) & _control_mask(meta)
    rho_ctrl, a_ctrl = axis_scores(feats[stress_ctrl], mu_s, u, L)
    a_ctrl = a_ctrl[~np.isnan(a_ctrl)]

    trt = _condition_mask(meta, stress_condition) & _treated_mask(meta, compound_allowlist)
    trt_meta = meta.loc[trt, ["Metadata_broad_sample"]].reset_index(drop=True)
    rho, a = axis_scores(feats[trt], mu_s, u, L)
    trt_meta["rho"] = rho
    trt_meta["a"] = a

    # rho_mean feeds both gate_consistency_mean below and gate (4)'s
    # difference-in-differences (the Baseline arm contributes a mean over
    # its own replicates, so the stress arm has to as well to cancel).
    per_compound = trt_meta.groupby("Metadata_broad_sample", observed=True).agg(
        n_reps_stress=("rho", "size"),
        rho_mean=("rho", "mean"),
        a_mean=("a", "mean"),
    )

    # Per-well rho, kept per compound for the LOO statistic and the
    # replicate CIs below.
    rho_by_compound = {
        c: g.to_numpy(dtype=float)
        for c, g in trt_meta.groupby("Metadata_broad_sample", observed=True)["rho"]
    }
    rho_lists = [rho_by_compound[c] for c in per_compound.index]

    # rho_loo_min: the mean with the single most favourable replicate
    # REMOVED -- "is this still a hit without its best well?"
    per_compound["rho_loo_min"] = [
        float((r.sum() - r.max()) / (len(r) - 1)) if len(r) > 1 else np.nan
        for r in rho_lists
    ]

    rng = np.random.default_rng(seed)
    # Separate stream for the LOO null, since it is drawn at n_reps-1 rather
    # than n_reps and must not perturb the n_reps draws above.
    rng_loo = np.random.default_rng(seed + 101)
    n_c = len(per_compound)
    tau_s_mean = np.empty(n_c)
    p_noise_mean = np.empty(n_c)
    tau_a = np.empty(n_c)
    p_noise_loo = np.full(n_c, np.nan)
    for n_reps, rows in per_compound.groupby("n_reps_stress", observed=True).groups.items():
        idxr = per_compound.index.get_indexer(rows)

        null_mean = _bootstrap_rho_mean_null(rho_ctrl, int(n_reps), n_boot, rng)
        obs_mean = per_compound.loc[rows, "rho_mean"].to_numpy()
        tau_s_mean[idxr] = float(np.percentile(null_mean, CTRL_PERCENTILE))
        # (k+1)/(n+1) rather than k/n: a bootstrap p-value of exactly 0 is
        # not defensible, and BH-correcting exact zeros is worse.
        p_noise_mean[idxr] = [
            (float(np.sum(null_mean >= o)) + 1) / (len(null_mean) + 1) for o in obs_mean
        ]

        null_a = _bootstrap_a_mean_null(a_ctrl, int(n_reps), n_boot, rng)
        tau_a[idxr] = float(np.percentile(null_a, CTRL_PERCENTILE))

        # Dropping a replicate widens the noise floor, so the LOO null is
        # drawn at n_reps-1, not n_reps.
        if int(n_reps) > 1:
            null_loo = _bootstrap_rho_mean_null(
                rho_ctrl, int(n_reps) - 1, n_boot, rng_loo
            )
            obs_loo = per_compound.loc[rows, "rho_loo_min"].to_numpy()
            p_noise_loo[idxr] = [
                (float(np.sum(null_loo >= o)) + 1) / (len(null_loo) + 1) for o in obs_loo
            ]

    per_compound["tau_s_mean"] = tau_s_mean
    per_compound["p_noise_mean"] = p_noise_mean
    per_compound["tau_a"] = tau_a
    per_compound["p_noise_loo"] = p_noise_loo

    # Replicate-level CI (resampling the compound's OWN wells).
    ci_rng = np.random.default_rng(seed + 1)
    ci = [_replicate_ci(r, n_ci, ci_alpha, ci_rng) for r in rho_lists]
    per_compound["rho_mean_lo"] = [c[0] for c in ci]
    per_compound["rho_mean_hi"] = [c[1] for c in ci]

    # Gate 2: FDR-controlled consistency AND alignment (direction, since rho
    # alone is magnitude-dominated).
    per_compound["q_noise_mean"] = _bh_adjust(per_compound["p_noise_mean"].to_numpy())
    per_compound["gate_alignment"] = per_compound["a_mean"] >= per_compound["tau_a"]
    per_compound["gate_consistency_mean"] = (
        per_compound["q_noise_mean"] <= FDR_Q
    ) & per_compound["gate_alignment"]

    # Gate 2b: not driven by one replicate.
    per_compound["q_noise_loo"] = _bh_adjust(per_compound["p_noise_loo"].to_numpy())
    per_compound["gate_loo_robust"] = (per_compound["q_noise_loo"] <= FDR_Q).fillna(False)

    return per_compound.reset_index(), rho_ctrl, rho_by_compound


def _promiscuity_gate(
    beta_perp: pd.Series, mad_k: float = PROMISCUITY_MAD_K
) -> pd.Series:
    """gate_promiscuity: beta_perp <= median(beta_perp) + mad_k * MAD(beta_perp)
    over the screened pool -- a robust outlier rule on off-axis Baseline
    activity, so it adapts to the pool's actual spread instead of rejecting
    a fixed fraction of any pool by construction."""
    if beta_perp.notna().sum() == 0:
        return pd.Series(False, index=beta_perp.index)
    med = beta_perp.median()
    mad = (beta_perp - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return beta_perp.notna()
    return beta_perp <= med + mad_k * mad


def _add_specificity(
    per_compound: pd.DataFrame,
    rho_ctrl_s: np.ndarray,
    rho_ctrl_b: np.ndarray,
    n_boot: int,
    seed: int,
    rho_by_compound: dict,
    base_par_by_compound: dict,
    n_ci: int = N_CI,
    ci_alpha: float = CI_ALPHA,
) -> pd.DataFrame:
    """Gate (4): the difference-in-differences columns rho_int / tau_int /
    p_int / gate_specificity, added onto the merged consistency+cytotox
    frame (which must already carry rho_mean, beta_par and both replicate
    counts). See `imaging.reversion`'s module docstring for why the
    interaction is the confound-robust estimand.

    The null is cached per (n_reps_stress, n_reps_baseline) pair, of which
    there are only a handful across the whole panel. Compounds with no
    Baseline-arm wells get NaN scores and fail the gate -- their interaction
    is simply not estimable.

    A CI for `rho_int` is added by resampling BOTH arms' replicates."""
    per_compound = per_compound.copy()
    per_compound["rho_int"] = per_compound["rho_mean"] - per_compound["beta_par"]

    rng = np.random.default_rng(seed)
    tau_int = np.full(len(per_compound), np.nan)
    p_int = np.full(len(per_compound), np.nan)
    null_cache: dict = {}

    estimable = per_compound["n_reps_baseline"].notna() & per_compound["rho_int"].notna()
    keys = list(
        zip(
            per_compound["n_reps_stress"].to_numpy(),
            per_compound["n_reps_baseline"].to_numpy(),
        )
    )
    for i, (n_s, n_b) in enumerate(keys):
        if not estimable.iloc[i]:
            continue
        key = (int(n_s), int(n_b))
        if key not in null_cache:
            null_cache[key] = _bootstrap_rho_int_null(
                rho_ctrl_s, rho_ctrl_b, key[0], key[1], n_boot, rng
            )
        null = null_cache[key]
        tau_int[i] = float(np.percentile(null, CTRL_PERCENTILE))
        p_int[i] = (
            float(np.sum(null >= per_compound["rho_int"].iloc[i])) + 1
        ) / (len(null) + 1)

    per_compound["tau_int"] = tau_int
    per_compound["p_int"] = p_int

    # CI on the interaction -- the primary estimand: resample the stress
    # arm's replicates and the Baseline arm's replicates independently, both
    # with replacement, and take the difference of the two resampled means.
    ci_rng = np.random.default_rng(seed + 2)
    bounds = {k: np.full(len(per_compound), np.nan) for k in ("lo", "hi")}
    for i, compound in enumerate(per_compound["Metadata_broad_sample"]):
        r_s = rho_by_compound.get(compound)
        r_b = base_par_by_compound.get(compound)
        if r_s is None or r_b is None or len(r_s) < 2 or len(r_b) < 2:
            continue
        draws_s = r_s[ci_rng.integers(0, len(r_s), size=(n_ci, len(r_s)))]
        draws_b = r_b[ci_rng.integers(0, len(r_b), size=(n_ci, len(r_b)))]
        pcts = [100 * ci_alpha / 2, 100 * (1 - ci_alpha / 2)]
        bounds["lo"][i], bounds["hi"][i] = np.percentile(
            draws_s.mean(axis=1) - draws_b.mean(axis=1), pcts
        )
    per_compound["rho_int_lo"] = bounds["lo"]
    per_compound["rho_int_hi"] = bounds["hi"]
    # Reported and available as an optional extra tier (`nominated_robust_ci`)
    # rather than folded into the main call: at n_s ~ 5 and n_b ~ 5 a
    # percentile bootstrap of a DIFFERENCE of two 5-well means is a blunt
    # instrument, and requiring it to exclude zero is a stricter demand than
    # the FDR-controlled null test, not a better-calibrated one.
    per_compound["gate_ci_positive"] = per_compound["rho_int_lo"] > 0
    # Gate 4 under FDR control, for the same reason as gate 2: a bare 95th
    # percentile fires on ~5% of compounds by construction, which on a
    # 75-compound pool is ~4 expected false positives before any biology.
    per_compound["q_int"] = _bh_adjust(p_int)
    per_compound["gate_specificity"] = (per_compound["q_int"] <= FDR_Q).fillna(False)
    return per_compound
