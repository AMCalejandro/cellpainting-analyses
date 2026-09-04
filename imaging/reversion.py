"""Compound-reversion scoring: does an active compound pull a stress
condition's profile back toward the unstressed Baseline condition?

Given a stress condition s (e.g. "FFA") and the "Baseline" condition, define
the reversion axis from the two conditions' DMSO-control centroids:

    u = (mu_B - mu_s) / L,  L = ||mu_B - mu_s||

For a compound c with treated wells W(c) in condition s, each well's
displacement from mu_s projects onto that axis as:

    delta_w = x_w - mu_s
    rho_w = <delta_w, u> / L        (fraction of the Baseline-Stress gap closed)
    a_w   = cos(delta_w, u)         (is the move actually along the axis?)

`NOMINATE(c, s)` requires two gates:

  (2) consistency -- every replicate moves along the axis, and the worst
      replicate (rho_min = min_w rho_w) still clears the DMSO noise floor
      tau_s (95th percentile of individual stress-DMSO wells' own rho):
          rho_min >= tau_s  and  min_w a_w > 0

  (3) not cytotoxic -- base(c) is compound c's OWN wells in the Baseline
      arm (this panel is tested at Baseline too, unstressed). Their mean
      shouldn't drift from mu_B any further than n-averaged Baseline-DMSO
      controls naturally do:
          beta_c = ||mean(x_w : w in base(c)) - mu_B|| / L  <=  tau^B_n

      beta is really an INERTNESS gate, not a toxicity one: it requires the
      compound to do essentially nothing to healthy adipocytes, so any real
      pharmacology fails it. `gate_cytotox_decomp` is the decomposed
      replacement that `nominated_specific` uses instead:
          beta_par  >= tau_par_n   (only NEGATIVE on-axis drift disqualifies,
                                    i.e. pushing healthy cells toward stress)
          beta_perp <= tau_perp_n  (off-axis drift: polypharmacology/toxicity)
      Both thresholds come from the same Baseline-DMSO bootstrap as tau^B_n,
      so neither is a hand-picked cutoff.

  (4) stress-specific -- a difference-in-differences gate. Decompose the
      Baseline-arm displacement into its on- and off-axis parts:
          beta_par  = <mean(x_w : w in base(c)) - mu_B, u> / L   (signed)
          beta_perp = ||orthogonal residual|| / L
      so that beta^2 = beta_par^2 + beta_perp^2. beta_par is the SAME
      projection as rho, measured in unstressed cells, so:
          rho_int(c) = rho_mean(c) - beta_par(c)  >=  tau_int
      asks how much of the apparent reversion is specific to the stress
      context rather than something the compound does to any adipocyte.

      This gate exists because of the structural confound: every plate
      carries exactly one condition, so the axis we estimate is really
      u_hat = u + t, with t whatever technical difference separates the
      Baseline plate-set from the stress plate-set. That contamination
      enters rho(c) as <delta(c,s), t>/||u_hat||^2 -- compounds whose
      response happens to align with the technical direction score as
      false reverters. But the same compound, at the same well position
      and dose, aligns with t about equally in the Baseline arm, so
      <delta(c,B), t> ~= <delta(c,s), t> and the spurious term largely
      cancels in the difference. The confound is a condition-level MAIN
      effect; rho_int is an interaction, and interactions are orthogonal
      to main effects. Genuine stress-engaging biology does not cancel:
      a compound that acts identically on stressed and unstressed cells
      is not engaging the stress biology by definition.

      tau_int is the 95th percentile of a DMSO-only null built the same
      way (mean of n_s stress-DMSO wells' rho minus the mean of n_b
      Baseline-DMSO wells' on-axis displacement), so it is matched to
      each compound's own replicate counts on BOTH arms.

Significance:
  p_noise(c) -- bootstrap n_s (= |W(c)|) stress-DMSO wells at a time, in
      place of compound c's replicates, and ask how often that noise-only
      draw's rho_min is at least as large as the compound's observed
      rho_min.
  p_spec -- a single run-level calibration number, not per compound: shuffle
      the well->compound labels among condition-s treated wells and count
      how many permuted label-groups clear gate (2) alone. Gate (3) can't be
      re-evaluated under this permutation -- it depends on each compound's
      OWN Baseline-arm wells, which the label shuffle (confined to the
      stress arm) never touches, so re-pairing a permuted group with some
      other compound's real Baseline-arm data would just be an arbitrary
      extra randomization, not a null for the stress-arm signal being
      tested. p_spec asks: given how many compounds we screened, how many
      gate-(2) passes would pure relabeling noise produce?

      p_spec = (#{permutations with N(pi) >= N(obs)} + 1) / (n_perm + 1)

Two ranking indices are reported:
  RI     = rho_min - beta      -- the original, ranks gate (2) & (3) passes.
  RI_int = rho_int - beta_perp -- the specificity-aware index, and the one
      to rank on. It subtracts only the OFF-axis Baseline activity
      (polypharmacology / non-specific perturbation), because the on-axis
      part is already netted out inside rho_int. `nominated_specific`
      (gates 2 & 3 & 4) is the corresponding call.

Both `mu_B` and `mu_s`, and every downstream quantity, must come from a
*jointly* processed feature matrix -- `load_joint_residualized` below loads
Baseline + the stress condition together and z-scores/PCA-reduces/
residualizes them as one matrix, so the axis isn't an artifact of fitting
normalization (or a PCA basis) separately per condition. When PCA-reducing
(`n_components` given), `imaging.features.preprocess` runs on the
concatenated matrix, not per condition, which is exactly what makes Baseline
and the stress condition land in the same basis instead of two incomparable
ones -- mirroring the load -> preprocess -> residualize order
`run_pipeline.py` uses for the copairs experiment.
"""

from typing import Iterable, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat
from . import load

N_BOOT = 10000
N_PERM = 10000
SEED = 0
CTRL_PERCENTILE = 95


def load_joint_residualized(
    feature_space: str,
    covariates: Union[str, list],
    baseline_condition: str = "Baseline",
    stress_condition: str = load.DEFAULT_CONDITION,
    n_components: Optional[int] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load Baseline + `stress_condition` together for `feature_space`, tag
    each row's `Metadata_condition`, and jointly preprocess + residualize the
    combined matrix so both conditions share one coordinate space.

    `covariates` is either a method name from
    `imaging.features.RESIDUALIZE_METHODS` or, for backward compatibility, a
    raw Ridge covariate list (e.g. `["count", "batch"]`) passed straight to
    `feat.ridge_residualize`.

    Method choice matters more here than anywhere else in the project,
    because this is the one place two conditions are pooled into a single
    matrix. Do NOT use the pooled `count_plate` / `count_batch_plate`: every
    cpg0014 plate carries exactly one condition, so their plate dummies span
    the condition direction and the fit removes the entire Baseline-stress
    offset -- the reversion axis itself -- leaving nothing to project onto
    (docs/batch_effect_conclusions.md). Use instead:

    - `nested_count_plate` (the default in run_reversion.py) -- the identical
      Ridge fit run separately within each condition, which removes exactly
      the same within-condition plate drift at full strength while
      preserving the between-condition offset by construction.
    - `control_centered` -- same per-condition anchoring, but the plate
      offset is estimated from that plate's ~30 DMSO wells and shrunk toward
      the condition mean, so it corrects more conservatively.
    - `count_batch` -- safe (batch is balanced across conditions, so batch
      dummies don't span the condition direction) but coarser than either.

    If `n_components` is given, PCA-reduce the concatenated matrix to that
    many components via `feat.preprocess` before residualizing -- the same
    load -> preprocess -> residualize recipe `run_pipeline.py --preprocess`
    uses for the copairs experiment, just fit jointly across both conditions
    instead of per condition. `feat.preprocess` ends on a PCA basis, not a
    z-scored one (see its docstring), so no separate `feat.zscore` call
    follows it here, matching `run_pipeline.py`. If omitted, the raw
    z-scored features are residualized directly (previous behavior).

    Returns (meta, feats), row-aligned, Baseline rows first."""
    meta_b, feats_b = load.load_feature_space(feature_space, baseline_condition)
    meta_s, feats_s = load.load_feature_space(feature_space, stress_condition)
    meta_b = meta_b.copy()
    meta_b["Metadata_condition"] = baseline_condition
    meta_s = meta_s.copy()
    meta_s["Metadata_condition"] = stress_condition

    meta = pd.concat([meta_b, meta_s], ignore_index=True)
    feats = np.vstack([feats_b, feats_s])

    if n_components is not None:
        feats = feat.preprocess(feats, n_components=n_components)
    else:
        feats = feat.zscore(feats)

    if isinstance(covariates, str):
        feats = feat.RESIDUALIZE_METHODS[covariates](feats, meta)
    else:
        feats = feat.ridge_residualize(feats, meta, covariates)
    return meta, feats


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


def _bootstrap_rho_min_null(
    rho_ctrl: np.ndarray, n_reps: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """rho_min of `n_boot` fake compounds, each `n_reps` stress-DMSO wells
    drawn without replacement in place of a compound's real replicates."""
    idx = _sample_without_replacement(len(rho_ctrl), n_reps, n_boot, rng)
    return rho_ctrl[idx].min(axis=1)


def _bootstrap_beta_null(
    ctrl_feats: np.ndarray,
    mu_b: np.ndarray,
    u: np.ndarray,
    L: float,
    n_reps: int,
    n_boot: int,
    rng: np.random.Generator,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(beta, beta_par, beta_perp) of `n_boot` fake compounds, each the mean
    of `n_reps` Baseline-DMSO wells drawn without replacement, in place of a
    compound's real Baseline-arm replicates. Chunked to bound peak memory.

    Returning the decomposition alongside the norm means the on-/off-axis
    gate thresholds come from exactly the same resampling as the original
    beta threshold, rather than from a hand-picked cutoff."""
    idx = _sample_without_replacement(len(ctrl_feats), n_reps, n_boot, rng)
    beta = np.empty(n_boot)
    beta_par = np.empty(n_boot)
    beta_perp = np.empty(n_boot)
    for start in range(0, n_boot, chunk_size):
        chunk_idx = idx[start : start + chunk_size]
        disp = ctrl_feats[chunk_idx].mean(axis=1) - mu_b
        on_axis = disp @ u
        sl = slice(start, start + chunk_size)
        beta[sl] = np.linalg.norm(disp, axis=1) / L
        beta_par[sl] = on_axis / L
        beta_perp[sl] = np.linalg.norm(disp - on_axis[:, None] * u, axis=1) / L
    return beta, beta_par, beta_perp


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
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float, np.ndarray]:
    """Gate (2) per compound, plus the raw per-well rho/a for the permuted
    calibration test, and tau_s / rho_ctrl for reuse."""
    stress_ctrl = _condition_mask(meta, stress_condition) & _control_mask(meta)
    rho_ctrl, _ = axis_scores(feats[stress_ctrl], mu_s, u, L)
    tau_s = float(np.percentile(rho_ctrl, CTRL_PERCENTILE))

    trt = _condition_mask(meta, stress_condition) & _treated_mask(meta, compound_allowlist)
    trt_meta = meta.loc[trt, ["Metadata_broad_sample"]].reset_index(drop=True)
    rho, a = axis_scores(feats[trt], mu_s, u, L)
    trt_meta["rho"] = rho
    trt_meta["a"] = a

    per_compound = trt_meta.groupby("Metadata_broad_sample", observed=True).agg(
        n_reps_stress=("rho", "size"),
        rho_min=("rho", "min"),
        # rho_mean is what the difference-in-differences gate (4) uses: the
        # Baseline arm contributes a mean over its replicates, so the stress
        # arm has to as well for the two to cancel like for like. rho_min
        # stays the basis of the consistency gate (2).
        rho_mean=("rho", "mean"),
        a_min=("a", "min"),
        a_mean=("a", "mean"),
    )
    per_compound["gate_consistency"] = (per_compound["rho_min"] >= tau_s) & (
        per_compound["a_min"] > 0
    )

    rng = np.random.default_rng(seed)
    p_noise = np.empty(len(per_compound))
    for n_reps, rows in per_compound.groupby("n_reps_stress", observed=True).groups.items():
        null = _bootstrap_rho_min_null(rho_ctrl, int(n_reps), n_boot, rng)
        obs = per_compound.loc[rows, "rho_min"].to_numpy()
        p_noise[per_compound.index.get_indexer(rows)] = [
            float(np.mean(null >= o)) for o in obs
        ]
    per_compound["p_noise"] = p_noise

    return (
        per_compound.reset_index(),
        trt_meta["rho"].to_numpy(),
        trt_meta["a"].to_numpy(),
        tau_s,
        rho_ctrl,
    )


def _cytotox_table(
    meta: pd.DataFrame,
    feats: np.ndarray,
    baseline_condition: str,
    mu_b: np.ndarray,
    u: np.ndarray,
    L: float,
    n_boot: int,
    seed: int,
    compound_allowlist: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Gate (3) per compound from each compound's own Baseline-arm wells,
    plus the on-/off-axis decomposition of that displacement (beta_par,
    beta_perp) that gate (4) and RI_int are built from."""
    base_ctrl_mask = _condition_mask(meta, baseline_condition) & _control_mask(meta)
    base_ctrl_feats = feats[base_ctrl_mask]

    base_trt_mask = _condition_mask(meta, baseline_condition) & _treated_mask(
        meta, compound_allowlist
    )
    base_trt_meta = meta.loc[base_trt_mask, ["Metadata_broad_sample"]].reset_index(drop=True)
    base_trt_feats = feats[base_trt_mask]

    rng = np.random.default_rng(seed)
    rows = []
    null_cache: dict = {}
    for compound, group in base_trt_meta.groupby("Metadata_broad_sample", observed=True):
        idx = group.index.to_numpy()
        n_reps = len(idx)
        mean_vec = base_trt_feats[idx].mean(axis=0)
        displacement = mean_vec - mu_b
        beta = float(np.linalg.norm(displacement) / L)
        # Signed on-axis component -- the SAME projection rho measures, but
        # in unstressed cells. Positive: pushes healthy adipocytes further
        # along the Baseline direction. Negative: pushes them toward the
        # stress phenotype, which is the genuinely disqualifying behaviour
        # that the unsigned beta gate cannot distinguish from a strong,
        # beneficial perturbation.
        on_axis = float(displacement @ u)
        beta_par = on_axis / L
        beta_perp = float(np.linalg.norm(displacement - on_axis * u) / L)
        if n_reps not in null_cache:
            b_null, par_null, perp_null = _bootstrap_beta_null(
                base_ctrl_feats, mu_b, u, L, n_reps, n_boot, rng
            )
            null_cache[n_reps] = (
                float(np.percentile(b_null, CTRL_PERCENTILE)),
                # Lower tail: beta_par only disqualifies when it is
                # NEGATIVE beyond noise, i.e. the compound pushes healthy
                # adipocytes toward the stress phenotype. Positive on-axis
                # activity at Baseline is not toxicity -- it is exactly what
                # rho_int nets out, so penalizing it here would double-count.
                float(np.percentile(par_null, 100 - CTRL_PERCENTILE)),
                float(np.percentile(perp_null, CTRL_PERCENTILE)),
            )
        tau_b_n, tau_par_n, tau_perp_n = null_cache[n_reps]
        rows.append(
            (
                compound,
                n_reps,
                beta,
                beta_par,
                beta_perp,
                tau_b_n,
                tau_par_n,
                tau_perp_n,
                beta <= tau_b_n,
                (beta_par >= tau_par_n) and (beta_perp <= tau_perp_n),
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Metadata_broad_sample",
            "n_reps_baseline",
            "beta",
            "beta_par",
            "beta_perp",
            "tau_B_n",
            "tau_par_n",
            "tau_perp_n",
            "gate_cytotox",
            "gate_cytotox_decomp",
        ],
    )


def _add_specificity(
    per_compound: pd.DataFrame,
    rho_ctrl_s: np.ndarray,
    rho_ctrl_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Gate (4): the difference-in-differences columns rho_int / tau_int /
    p_int / gate_specificity, added onto the merged consistency+cytotox
    frame (which must already carry rho_mean, beta_par and both replicate
    counts). See the module docstring for why the interaction is the
    confound-robust estimand.

    The null is cached per (n_reps_stress, n_reps_baseline) pair, of which
    there are only a handful across the whole panel. Compounds with no
    Baseline-arm wells get NaN scores and fail the gate -- their interaction
    is simply not estimable."""
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
        p_int[i] = float(np.mean(null >= per_compound["rho_int"].iloc[i]))

    per_compound["tau_int"] = tau_int
    per_compound["p_int"] = p_int
    per_compound["gate_specificity"] = (
        per_compound["rho_int"] >= per_compound["tau_int"]
    ).fillna(False)
    return per_compound


def _permutation_p_spec(
    rho: np.ndarray,
    a: np.ndarray,
    group_sizes: np.ndarray,
    tau_s: float,
    n_obs: int,
    n_perm: int,
    seed: int,
) -> float:
    """Shuffle which stress-condition treated well belongs to which compound,
    re-split the shuffled (rho, a) into groups of the SAME sizes as the real
    compound groups (a label permutation preserves the group-size multiset),
    and count how many of those fake groups clear gate (2) alone."""
    n = len(rho)
    starts = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])
    rng = np.random.default_rng(seed)
    counts = np.empty(n_perm, dtype=np.int64)
    for i in range(n_perm):
        order = rng.permutation(n)
        rho_perm = rho[order]
        a_perm = a[order]
        rho_mins = np.minimum.reduceat(rho_perm, starts)
        a_mins = np.minimum.reduceat(a_perm, starts)
        counts[i] = int(((rho_mins >= tau_s) & (a_mins > 0)).sum())
    return (int((counts >= n_obs).sum()) + 1) / (n_perm + 1)


def compute_reversion(
    meta: pd.DataFrame,
    feats: np.ndarray,
    baseline_condition: str = "Baseline",
    stress_condition: str = load.DEFAULT_CONDITION,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    seed: int = SEED,
    compound_allowlist: Optional[Iterable[str]] = None,
) -> dict:
    """Full reversion scoring for one (jointly residualized) feature space.

    `meta`/`feats` must come from `load_joint_residualized` (or an
    equivalent joint load covering both `baseline_condition` and
    `stress_condition`, row-aligned). Returns a dict with:

    - `per_compound`: one row per compound in `stress_condition`, with the
      consistency gate (rho_min, rho_mean, a_min, a_mean, p_noise), the
      cytotoxicity gate (beta, beta_par, beta_perp, tau_B_n, tau_par_n,
      tau_perp_n), the stress-specificity gate (rho_int, tau_int, p_int),
      both calls (`nominated` = gates 2 & 3 with the original unsigned beta;
      `nominated_specific` = gates 2 & 3-decomposed & 4)
      and both ranking indices (`RI`, `RI_int`). Sorted by `RI_int`, which
      is the index to rank on -- see the module docstring.
    - `tau_s`, `L`: the DMSO-noise floor and axis length.
    - `n_obs`, `p_spec`: run-level calibration (see module docstring).
    - `n_nominated`, `n_nominated_specific`: how many compounds each call
      returns. The gap between them is how much of the original hit list
      was context-independent activity pointing along the axis.

    `compound_allowlist`, if given, restricts every treated-well population
    (both the stress-arm consistency gate and the compound's own
    Baseline-arm cytotoxicity gate) to `Metadata_broad_sample` values in the
    list -- e.g. compounds already called active by `copairs_pipeline`'s
    `compute_activity`. DMSO-control wells (the axis and tau_s) are never
    filtered, since the allowlist only narrows which compounds get scored.
    """
    mu_b, mu_s, u, L = compute_axis(meta, feats, baseline_condition, stress_condition)

    consistency, rho, a, tau_s, rho_ctrl_s = _consistency_table(
        meta, feats, stress_condition, mu_s, u, L, n_boot, seed, compound_allowlist
    )
    cytotox = _cytotox_table(
        meta, feats, baseline_condition, mu_b, u, L, n_boot, seed, compound_allowlist
    )

    # On-axis displacement of each Baseline-DMSO well from its own centroid:
    # the gate-(4) null's Baseline arm, centered on zero by construction.
    base_ctrl = _condition_mask(meta, baseline_condition) & _control_mask(meta)
    rho_ctrl_b = ((feats[base_ctrl] - mu_b) @ u) / L

    per_compound = consistency.merge(cytotox, on="Metadata_broad_sample", how="left")
    per_compound = _add_specificity(per_compound, rho_ctrl_s, rho_ctrl_b, n_boot, seed)

    per_compound["nominated"] = (
        per_compound["gate_consistency"] & per_compound["gate_cytotox"].fillna(False)
    )
    # The new call uses the new gates throughout: the DECOMPOSED
    # cytotoxicity gate, not the original unsigned beta. beta is an
    # inertness gate -- it requires a compound to do essentially nothing to
    # healthy adipocytes, so it rejects any real pharmacology (a PPARgamma
    # agonist fails it). Splitting it lets gate (4) net out the on-axis part
    # while beta_perp still catches polypharmacology / genuine toxicity.
    per_compound["nominated_specific"] = (
        per_compound["gate_consistency"]
        & per_compound["gate_cytotox_decomp"].fillna(False)
        & per_compound["gate_specificity"]
    )
    per_compound["RI"] = per_compound["rho_min"] - per_compound["beta"]
    per_compound["RI_int"] = per_compound["rho_int"] - per_compound["beta_perp"]

    annot_cols = ["Metadata_broad_sample", "Metadata_target", "Metadata_moa"]
    annot_cols = [c for c in annot_cols if c in meta.columns]
    annot = (
        meta.loc[
            _condition_mask(meta, stress_condition)
            & _treated_mask(meta, compound_allowlist),
            annot_cols,
        ]
        .drop_duplicates("Metadata_broad_sample")
    )
    per_compound = per_compound.merge(annot, on="Metadata_broad_sample", how="left")
    per_compound = per_compound.sort_values("RI_int", ascending=False, na_position="last")

    # A label permutation just reassigns which well gets which compound
    # name, so the fake groups it produces have exactly the multiset of
    # real per-compound group sizes -- their order relative to `rho`/`a`
    # doesn't matter, since the permutation itself randomizes positions.
    group_sizes = consistency["n_reps_stress"].to_numpy()

    n_obs = int(per_compound["gate_consistency"].sum())
    p_spec = _permutation_p_spec(rho, a, group_sizes, tau_s, n_obs, n_perm, seed)

    return {
        "per_compound": per_compound.reset_index(drop=True),
        "tau_s": tau_s,
        "L": L,
        "n_obs": n_obs,
        "p_spec": p_spec,
        "n_nominated": int(per_compound["nominated"].sum()),
        "n_nominated_specific": int(per_compound["nominated_specific"].sum()),
        "n_boot": n_boot,
        "n_perm": n_perm,
        "seed": seed,
        "baseline_condition": baseline_condition,
        "stress_condition": stress_condition,
    }
