"""Compound-reversion scoring for proteomics -- shares its axis/gate engine
with `imaging.reversion` via `utils.reversion`, but keeps its own
`_cytotox_table`/`compute_reversion` because gate (3)'s viability/cytotoxicity
check needs a per-well cell count that has no analog on a proteomics plate
(a physically distinct plate from the one CellProfiler counted cells on --
see `proteomics.imaging_metadata`'s docstring). Every other gate is
unchanged from `imaging.reversion`; see that module's docstring for the
full gate spec and rationale. Differences from the imaging version:

  - `gate_viability` is dropped from gate (3): `viability`/`tau_viab_n` are
    not computed (no `Metadata_cell_count` to compute them from), so gate
    (3) here is `gate_beta_par & gate_promiscuity` only, not also
    `gate_viability`.
  - No `load_joint_residualized`: `proteomics.pipeline.load_corrected_proteomics`
    already returns every condition jointly z-scored and per-condition
    (nested) residualized in one call, so there is no separate two-condition
    joint-load step to do here -- see `proteomics.concordance.condition_hits`,
    which slices Baseline + one stress condition out of that pooled matrix
    before calling `compute_reversion` below.

Axis, from each condition's DMSO-control centroid:

    u = (mu_B - mu_s) / L,  L = ||mu_B - mu_s||

Per-well scores (projection onto the axis, from mu_s):

    rho_w = <x_w - mu_s, u> / L   (fraction of the Baseline-Stress gap closed)
    a_w   = cos(x_w - mu_s, u)    (is the move actually along the axis?)

`nominated_robust_ci` requires four gates, all DMSO-bootstrap-calibrated
(10,000 draws) and BH-adjusted at FDR_Q where noted:

  (2) consistency -- q_noise_mean <= FDR_Q (rho_mean against a
      mean-of-n_reps DMSO null) AND a_mean >= tau_a (alignment, against its
      own DMSO null -- rho alone is magnitude-dominated, so direction has to
      be tested separately). -> `gate_consistency_mean`.

  (2b) robustness -- leave-one-out: rho_loo_min (the mean with the single
      most favourable replicate deleted) against a matched mean-of-(n-1)
      null, so one wild well can't carry the call. -> `gate_loo_robust`.
      Plus a replicate-bootstrap CI on rho_int (below) excluding zero.
      -> `gate_ci_positive`.

  (3) not cytotoxic / not promiscuous -- from the compound's OWN
      Baseline-arm (unstressed) wells:
          gate_beta_par    beta_par >= tau_par_n (no negative on-axis
                            drift, i.e. doesn't push healthy cells toward
                            the stress phenotype)
          gate_promiscuity beta_perp (off-axis Baseline activity) below a
                            pool-relative median + 3*MAD outlier rule
      (no gate_viability here -- see module docstring)

  (4) stress-specific -- a difference-in-differences test against the
      structural confound that every plate carries exactly one condition
      (so the fitted axis is u_hat = u + t, t a technical Baseline-vs-stress
      difference). t is a condition-level main effect and rho_int is an
      interaction, so it cancels:
          rho_int(c) = rho_mean(c) - beta_par(c) >= tau_int
      tested against a matched DMSO null. -> `gate_specificity`.

`nominated_robust = gate_consistency_mean & gate_loo_robust & gate_beta_par &
gate_promiscuity & gate_specificity`;
`nominated_robust_ci = nominated_robust & gate_ci_positive` is the call to
act on. `RI_spec = rho_int / tau_int` is the sort key: stress-specific
reversion in units of its own replicate-matched noise floor.

See `imaging/reversion.py`, `utils/reversion.py` and
`docs/reversion_pipeline_final.md` for the full spec this forks from.
"""

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.reversion import (
    CTRL_PERCENTILE,
    FDR_Q,
    N_BOOT,
    SEED,
    _add_specificity,
    _bootstrap_beta_par_null,
    _condition_mask,
    _consistency_table,
    _control_mask,
    _promiscuity_gate,
    _treated_mask,
    compute_axis,
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
) -> tuple[pd.DataFrame, dict]:
    """Gate (3) per compound from each compound's own Baseline-arm wells:
    the on-/off-axis decomposition of that displacement (beta_par, needed
    for gate_beta_par and gate (4)'s rho_int; beta_perp, needed for
    gate_promiscuity). Unlike `imaging.reversion._cytotox_table`, there is
    no cell-count viability readout here -- see module docstring -- so this
    returns no `viability`/`gate_viability` columns at all.

    Also returns each compound's PER-WELL Baseline-arm on-axis displacement
    (over L), which `_add_specificity` resamples for the `rho_int` CI -- the
    interaction has a noisy replicate on both sides, so a CI that resampled
    only the stress arm would understate its spread."""
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
    base_par_by_compound: dict = {}
    for compound, group in base_trt_meta.groupby("Metadata_broad_sample", observed=True):
        idx = group.index.to_numpy()
        n_reps = len(idx)
        base_par_by_compound[compound] = (
            (base_trt_feats[idx] - mu_b) @ u
        ) / L
        mean_vec = base_trt_feats[idx].mean(axis=0)
        displacement = mean_vec - mu_b
        # Signed on-axis component -- the SAME projection rho measures, but
        # in unstressed cells. Negative means the compound pushes healthy
        # adipocytes toward the stress phenotype, which is the disqualifying
        # behaviour; positive is not penalized (rho_int nets it out).
        on_axis = float(displacement @ u)
        beta_par = on_axis / L
        beta_perp = float(np.linalg.norm(displacement - on_axis * u) / L)
        if n_reps not in null_cache:
            par_null = _bootstrap_beta_par_null(
                base_ctrl_feats, mu_b, u, L, n_reps, n_boot, rng
            )
            null_cache[n_reps] = float(
                # Lower tail: only a NEGATIVE beta_par beyond noise
                # disqualifies.
                np.percentile(par_null, 100 - CTRL_PERCENTILE)
            )
        tau_par_n = null_cache[n_reps]
        rows.append(
            (
                compound,
                n_reps,
                beta_par,
                beta_perp,
                tau_par_n,
                beta_par >= tau_par_n,
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Metadata_broad_sample",
            "n_reps_baseline",
            "beta_par",
            "beta_perp",
            "tau_par_n",
            "gate_beta_par",
        ],
    ), base_par_by_compound


def compute_reversion(
    meta: pd.DataFrame,
    feats: np.ndarray,
    baseline_condition: str = "Baseline",
    stress_condition: str = "FFA",
    n_boot: int = N_BOOT,
    seed: int = SEED,
    compound_allowlist: Optional[Iterable[str]] = None,
) -> dict:
    """Full reversion scoring for one jointly-residualized proteomic feature
    matrix. `meta`/`feats` must jointly cover `baseline_condition` and
    `stress_condition`, row-aligned, already z-scored/residualized in a
    shared coordinate space (see `proteomics.concordance.condition_hits`).

    Returns a dict with `per_compound` (one row per scored compound, sorted
    by `RI_spec` descending, with every gate column plus `nominated_robust`
    and `nominated_robust_ci`), `L` (axis length), `n_nominated_robust(_ci)`,
    `funnel` (per-gate survivor counts conditional on gate 2, for localising
    a hit-count change to the gate responsible), and run metadata. There is
    no `n_toxic` here -- see module docstring on the dropped viability gate.

    `compound_allowlist`, if given, restricts every treated-well population
    (both the stress-arm consistency gate and the compound's own
    Baseline-arm gates) to `Metadata_broad_sample` values in the list --
    e.g. compounds already called active by `utils.copairs`.
    DMSO controls are never filtered."""
    mu_b, mu_s, u, L = compute_axis(meta, feats, baseline_condition, stress_condition)

    # On-axis displacement of each Baseline-DMSO well from its own centroid:
    # gate (4)'s null Baseline arm, centered on zero by construction.
    base_ctrl = _condition_mask(meta, baseline_condition) & _control_mask(meta)
    rho_ctrl_b = ((feats[base_ctrl] - mu_b) @ u) / L

    consistency, rho_ctrl_s, rho_by_compound = _consistency_table(
        meta, feats, stress_condition, mu_s, u, L, n_boot, seed, compound_allowlist,
    )
    cytotox, base_par_by_compound = _cytotox_table(
        meta, feats, baseline_condition, mu_b, u, L, n_boot, seed, compound_allowlist
    )

    per_compound = consistency.merge(cytotox, on="Metadata_broad_sample", how="left")
    per_compound = _add_specificity(
        per_compound, rho_ctrl_s, rho_ctrl_b, n_boot, seed,
        rho_by_compound, base_par_by_compound,
    )
    per_compound["gate_promiscuity"] = _promiscuity_gate(per_compound["beta_perp"])

    # Consistency (2) + robustness (2b, LOO) + not toxic/promiscuous (3,
    # gate_viability dropped -- see module docstring) + stress-specific (4).
    gate3 = (
        per_compound["gate_beta_par"].fillna(False)
        & per_compound["gate_promiscuity"].fillna(False)
    )
    per_compound["nominated_robust"] = (
        per_compound["gate_consistency_mean"]
        & per_compound["gate_loo_robust"]
        & gate3
        & per_compound["gate_specificity"]
    )
    # Production tier: also requires the replicate-bootstrap CI on the
    # interaction to exclude zero.
    per_compound["nominated_robust_ci"] = (
        per_compound["nominated_robust"] & per_compound["gate_ci_positive"]
    )
    # Sort key: stress-specific reversion in units of its own
    # replicate-matched noise floor.
    per_compound["RI_spec"] = per_compound["rho_int"] / per_compound["tau_int"]

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
    per_compound = per_compound.sort_values("RI_spec", ascending=False, na_position="last")

    return {
        "per_compound": per_compound.reset_index(drop=True),
        "L": L,
        "n_nominated_robust": int(per_compound["nominated_robust"].sum()),
        "n_nominated_robust_ci": int(per_compound["nominated_robust_ci"].sum()),
        # Per-gate survivor counts conditional on gate 2 (mean), so a change
        # in the hit count can be localised to the gate responsible.
        "funnel": {
            k: int((per_compound["gate_consistency_mean"] & v.fillna(False)).sum())
            for k, v in {
                "beta_par": per_compound["gate_beta_par"],
                "promiscuity": per_compound["gate_promiscuity"],
                "specificity": per_compound["gate_specificity"],
            }.items()
        },
        "funnel_gate2": {
            "fdr_q_noise_mean": int((per_compound["q_noise_mean"] <= FDR_Q).sum()),
            "alignment": int(per_compound["gate_alignment"].sum()),
            "all_two": int(per_compound["gate_consistency_mean"].sum()),
        },
        "robustness": {
            "n_gate2_mean": int(per_compound["gate_consistency_mean"].sum()),
            "n_gate2_mean_failing_loo": int(
                (
                    per_compound["gate_consistency_mean"]
                    & ~per_compound["gate_loo_robust"]
                ).sum()
            ),
            "n_rho_int_ci_excludes_zero": int(per_compound["gate_ci_positive"].sum()),
            "n_nominated_robust_ci_excludes_zero": int(
                per_compound["nominated_robust_ci"].sum()
            ),
            "median_rho_int_ci_width": float(
                (per_compound["rho_int_hi"] - per_compound["rho_int_lo"]).median()
            ),
        },
        "fdr_q": FDR_Q,
        "n_boot": n_boot,
        "seed": seed,
        "baseline_condition": baseline_condition,
        "stress_condition": stress_condition,
    }
