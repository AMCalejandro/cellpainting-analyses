"""Preprocessing / batch-correction bake-off: explores 3 competing
preprocessing branches x 2 batch/plate-correction methods on the proteomic
matrix, scoring each of the 6 combinations before vs. after correction on:

- `score_technical_noise`      -- mean within-condition plate silhouette.
  Should drop after correction: less separable by plate means the
  plate-to-plate offset was actually removed. Computed per condition (not
  pooled) because every plate carries exactly one condition, so a pooled
  silhouette would be inflated by condition separability itself.
- `score_replicate_consistency` -- mean normalized AP from
  `utils.copairs.compute_activity` (same compound across
  batches vs. plate-matched DMSO). Should INCREASE: tighter biological
  replicates. `null_size` here defaults low relative to utils.copairs's
  production 10,000 -- fine for ranking combos, but re-run the winning
  combo at production null_size before reporting a p-value.
- `score_axis_stability`       -- bootstrap cosine stability of the
  reversion axis u = (mu_baseline - mu_stress)/||.|| (matches
  `proteomics.reversion.compute_axis`). Every reversion projection depends
  on this axis, so if it isn't reproducible under resampling, no
  downstream reversion call can be trusted.

Preprocessing branches (starting from the post-QC, pre-transform vendor
normalized-nELISA matrix: LLOD=0/ULOD=1, negatives are noise-floor, values
>1 are real saturating signal):

- "A_minimal_floor"  -- floor sub-LLOD noise at 0, nothing else.
- "B_arcsinh_mad"    -- arcsinh + per-analyte robust (median/MAD) scaling,
  reusing `proteomics.normalization.transform_and_scale` (production).
- "C_affine_whiten"  -- floor-clip, winsorize each analyte's upper tail,
  then per-analyte mean/std z-score. Every step is affine, so this branch
  tests whether B's nonlinear arcsinh distorts the vector-arithmetic
  (difference-of-means) geometry that reversion scoring depends on: for
  nonlinear f, f(a) - f(b) is not a fixed function of (a - b), so a
  "stress -> baseline" shift measured post-arcsinh isn't the same
  quantity as the untransformed shift. An affine map doesn't have this
  problem.

Batch-correction methods (applied to each branch's output):

- "nested_ridge"       -- production `proteomics.correction.correct`
  (nested-Ridge, one fit per condition, using every well).
- "plate_dmso_anchor"  -- challenger: estimates each plate's own DMSO
  mean/variance from ONLY its negative controls (shrunk toward the
  condition-pooled DMSO variance by `PLATE_ANCHOR_SHRINKAGE_N0` pseudo-obs)
  and re-expresses every well on that plate in the condition-pooled DMSO
  frame. Unlike nested_ridge, treated wells never enter the correction
  estimate, so it can't attribute treatment-effect variance to "plate".

Run via `python cli.py proteomics batch_report`.
"""

import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from utils import copairs as cp

from proteomics import correction as prote_correction
from proteomics import imaging_metadata as imgmeta
from proteomics import normalization as norm
from proteomics import paths as prote_paths
from proteomics.merge import _prepare_proteomics_dataset

DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_NULL_SIZE = 1000
DEFAULT_N_BOOT = 200
DEFAULT_SEED = 0
BASELINE_CONDITION = "Baseline"
WINSOR_PCT = 99.5
PLATE_ANCHOR_SHRINKAGE_N0 = 5.0


# --- inputs ------------------------------------------------------------


def load_bakeoff_inputs(
    csv_path: str = prote_paths.RAW_PROTE_PATH,
    n_meta_cols: int = 24,
    min_detection_rate: float = 0.20,
    outlier_z_thresh: float = 8.0,
    outlier_min_flagged: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared starting point for every branch: post-QC, pre-transform
    proteomic data joined with imaging metadata. Row set (which wells
    survive) has to be identical across branches, so QC/outlier-flagging
    happens once here rather than inside each branch -- outlier flagging in
    particular uses the reference arcsinh+MAD transform (branch B's) so all
    3 branches compete on the same samples rather than each flagging a
    different set under its own scale."""
    meta_df, X = norm.load_normalized(csv_path, n_meta_cols)
    print("raw:", X.shape)

    meta_df, X = norm.drop_curated_bad_samples(meta_df, X)
    print("after Step 0 (drop_curated_bad_samples):", X.shape)

    X, _dropped_analytes = norm.analyte_detection_qc(
        X, min_detection_rate=min_detection_rate, verbose=False
    )
    X = norm.impute_missing(X)
    print("after shared QC (detection-rate filter + impute):", X.shape)

    X_reference = norm.transform_and_scale(X)
    outlier_idx = norm.flag_outlier_samples(
        X_reference, z_thresh=outlier_z_thresh, min_flagged=outlier_min_flagged, verbose=False
    )
    keep = ~meta_df.index.isin(outlier_idx)
    meta_df = meta_df[keep].reset_index(drop=True)
    X = X[keep].reset_index(drop=True)
    print(f"after shared outlier flagging (reference transform): {X.shape} "
          f"({len(outlier_idx)} dropped)")

    imaging_meta = imgmeta.load_hwat_imaging_metadata()
    ds = _prepare_proteomics_dataset(meta_df, X, imaging_meta=imaging_meta, merge_mode="inner")

    meta = ds.joined.copy()
    meta["Metadata_Plate"] = meta["Metadata_nomic_barcode"]
    X_joined = ds.X_joined

    print(f"after joining imaging metadata: {X_joined.shape}, "
          f"conditions: {meta['Metadata_condition'].value_counts().to_dict()}")

    return meta, X_joined


# --- preprocessing branches ----------------------------------------------


def branch_a_floor_clip(X: pd.DataFrame) -> pd.DataFrame:
    """Minimalist: floor sub-LLOD noise at 0, leave everything else
    (including ULOD saturation and each analyte's native scale) untouched."""
    return X.clip(lower=0.0)


def branch_b_arcsinh_mad(X: pd.DataFrame) -> pd.DataFrame:
    """Standard/incumbent: production arcsinh + per-analyte robust
    (median/MAD) scaling, `proteomics.normalization.transform_and_scale`."""
    return norm.transform_and_scale(X)


def branch_c_affine_whiten(X: pd.DataFrame, winsor_pct: float = WINSOR_PCT) -> pd.DataFrame:
    """Distance-preserving: floor-clip, winsorize each analyte's extreme
    upper tail, then per-analyte mean/std z-score -- every step affine, so
    Euclidean distances/cosine angles are a fixed rescaling of the
    original (floor-clipped) ones, unlike branch B's arcsinh."""
    Xc = X.clip(lower=0.0)
    upper = Xc.quantile(winsor_pct / 100.0)
    Xc = Xc.clip(upper=upper, axis=1)
    mean = Xc.mean()
    std = Xc.std().replace(0, np.nan)
    return (Xc - mean) / std


PREPROCESSING_BRANCHES = {
    "A_minimal_floor": branch_a_floor_clip,
    "B_arcsinh_mad": branch_b_arcsinh_mad,
    "C_affine_whiten": branch_c_affine_whiten,
}


# --- batch-correction methods ---------------------------------------------


def method_nested_ridge(
    X: pd.DataFrame, meta: pd.DataFrame, covariates: Iterable[str] = ("plate",)
) -> pd.DataFrame:
    """Baseline: production per-condition nested-Ridge residualization."""
    return prote_correction.correct(X, meta, covariates=covariates, method="nested")


def method_plate_dmso_anchor(
    X: pd.DataFrame, meta: pd.DataFrame, shrinkage_n0: float = PLATE_ANCHOR_SHRINKAGE_N0
) -> pd.DataFrame:
    """Challenger: anchor each plate's DMSO mean/variance (shrunk toward the
    condition-pooled DMSO variance by `shrinkage_n0` pseudo-obs) and
    re-express every well on that plate in the condition-pooled DMSO frame:

        x_corrected = (x - mu_dmso,p) / sd_dmso,p * sd_dmso,c + mu_dmso,c

    A plate with fewer than 2 DMSO wells has no plate-specific estimate and
    falls back to the condition-pooled DMSO stats (no plate correction
    applied on that plate)."""
    feats = X.to_numpy(dtype=np.float64)
    plate = meta["Metadata_Plate"].to_numpy()
    condition = meta["Metadata_condition"].to_numpy()
    is_control = (meta["Metadata_pert_type"] == "negcon").to_numpy()

    out = np.empty_like(feats)
    for c in np.unique(condition):
        c_mask = condition == c
        c_ctrl = c_mask & is_control
        if not c_ctrl.any():
            raise ValueError(f"no DMSO controls found for condition {c!r}; cannot anchor")

        mu_c = feats[c_ctrl].mean(axis=0)
        var_c = feats[c_ctrl].var(axis=0)
        var_c[var_c == 0] = 1.0
        sd_c = np.sqrt(var_c)

        for p in np.unique(plate[c_mask]):
            p_mask = c_mask & (plate == p)
            p_ctrl = p_mask & is_control
            n_p = int(p_ctrl.sum())

            if n_p >= 2:
                mu_p = feats[p_ctrl].mean(axis=0)
                var_p = feats[p_ctrl].var(axis=0)
                w = n_p / (n_p + shrinkage_n0)
                var_shrunk = w * var_p + (1 - w) * var_c
                var_shrunk[var_shrunk == 0] = var_c[var_shrunk == 0]
                sd_p = np.sqrt(var_shrunk)
            else:
                mu_p = mu_c
                sd_p = sd_c

            out[p_mask] = (feats[p_mask] - mu_p) / sd_p * sd_c + mu_c

    return pd.DataFrame(out, columns=X.columns, index=X.index)


BATCH_CORRECTION_METHODS = {
    "nested_ridge": method_nested_ridge,
    "plate_dmso_anchor": method_plate_dmso_anchor,
}


# --- metrics ----------------------------------------------------------


def _subsample_index(n: int, sample_size: int, seed: int) -> np.ndarray:
    if sample_size is None or n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False)


def score_technical_noise(
    X: pd.DataFrame, meta: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> float:
    """Mean, across conditions, of the within-condition plate silhouette
    score. Skips conditions with only one plate."""
    feats = X.to_numpy(dtype=np.float64)
    scores = []
    for cond, sub in meta.groupby("Metadata_condition"):
        idx = sub.index.to_numpy()
        plates = sub["Metadata_Plate"].to_numpy()
        if len(np.unique(plates)) < 2:
            continue
        sel = _subsample_index(len(idx), sample_size, seed)
        scores.append(silhouette_score(feats[idx[sel]], plates[sel]))
    if not scores:
        raise ValueError("no condition had >1 plate; cannot score technical noise")
    return float(np.mean(scores))


def score_replicate_consistency(
    X: pd.DataFrame, meta: pd.DataFrame, null_size: int = DEFAULT_NULL_SIZE, seed: int = DEFAULT_SEED
) -> float:
    """Mean normalized AP (same compound across batches vs. plate-matched
    DMSO), averaged first within a condition then across conditions."""
    feats = X.to_numpy(dtype=np.float64)
    scores = []
    for cond in meta["Metadata_condition"].unique():
        cond_mask = (meta["Metadata_condition"] == cond).to_numpy()
        cond_meta = meta.loc[cond_mask].reset_index(drop=True)
        cond_feats = feats[cond_mask]

        annotated = cond_meta["Metadata_broad_sample"].notna().to_numpy()
        cond_meta = cond_meta.loc[annotated].reset_index(drop=True)
        cond_feats = cond_feats[annotated]
        if cond_meta["Metadata_pert_type"].eq("trt").sum() == 0:
            continue

        activity = cp.compute_activity(cond_meta, cond_feats, null_size=null_size, seed=seed)
        if len(activity):
            scores.append(activity["normalized_average_precision"].mean())
    if not scores:
        raise ValueError("no condition had scoreable treated wells")
    return float(np.mean(scores))


def score_axis_stability(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    baseline_condition: str = BASELINE_CONDITION,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> float:
    """Bootstrap stability of the reversion axis u = (mu_baseline -
    mu_stress)/||.|| (`proteomics.reversion.compute_axis`'s definition):
    resample each condition's DMSO wells with replacement, recompute u, and
    report the mean cosine similarity to the full-data axis, averaged over
    every non-baseline condition present."""
    feats = X.to_numpy(dtype=np.float64)
    condition = meta["Metadata_condition"].to_numpy()
    is_control = (meta["Metadata_pert_type"] == "negcon").to_numpy()
    rng = np.random.default_rng(seed)

    base_ctrl_idx = np.flatnonzero((condition == baseline_condition) & is_control)
    if len(base_ctrl_idx) == 0:
        raise ValueError(f"no DMSO controls found for baseline condition {baseline_condition!r}")

    stress_conditions = [c for c in pd.unique(condition) if c != baseline_condition]
    cond_means = []
    for stress in stress_conditions:
        stress_ctrl_idx = np.flatnonzero((condition == stress) & is_control)
        if len(stress_ctrl_idx) == 0:
            continue

        diff_full = feats[base_ctrl_idx].mean(axis=0) - feats[stress_ctrl_idx].mean(axis=0)
        norm_full = np.linalg.norm(diff_full)
        if norm_full == 0:
            continue
        u_full = diff_full / norm_full

        cosines = []
        for _ in range(n_boot):
            b_idx = rng.choice(base_ctrl_idx, size=len(base_ctrl_idx), replace=True)
            s_idx = rng.choice(stress_ctrl_idx, size=len(stress_ctrl_idx), replace=True)
            diff_boot = feats[b_idx].mean(axis=0) - feats[s_idx].mean(axis=0)
            norm_boot = np.linalg.norm(diff_boot)
            if norm_boot == 0:
                continue
            cosines.append(float(np.dot(diff_boot / norm_boot, u_full)))
        if cosines:
            cond_means.append(float(np.mean(cosines)))

    if not cond_means:
        raise ValueError("no stress condition had DMSO controls to build an axis against")
    return float(np.mean(cond_means))


def _score(X: pd.DataFrame, meta: pd.DataFrame, null_size: int, n_boot: int, seed: int) -> dict:
    return {
        "technical_noise": score_technical_noise(X, meta, seed=seed),
        "replicate_consistency": score_replicate_consistency(X, meta, null_size=null_size, seed=seed),
        "axis_stability": score_axis_stability(X, meta, n_boot=n_boot, seed=seed),
    }


# --- bake-off orchestration ------------------------------------------------


def run_bakeoff(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    branches: dict = PREPROCESSING_BRANCHES,
    methods: dict = BATCH_CORRECTION_METHODS,
    null_size: int = DEFAULT_NULL_SIZE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """For every (branch, method) combo: compute the 3 metrics once on the
    branch's uncorrected output (cost paid once, not once per correction
    method), then once more on the corrected output. Returns one row per
    combo with before/after/delta for all 3 metrics."""
    rows = []
    for branch_name, branch_fn in branches.items():
        t0 = time.time()
        X_pre = branch_fn(X)
        before = _score(X_pre, meta, null_size, n_boot, seed)
        print(f"[{branch_name}] before-correction: {before} ({time.time() - t0:.1f}s)", flush=True)

        for method_name, method_fn in methods.items():
            t1 = time.time()
            X_corr = method_fn(X_pre, meta)
            after = _score(X_corr, meta, null_size, n_boot, seed)
            print(f"[{branch_name}/{method_name}] after-correction: {after} "
                  f"({time.time() - t1:.1f}s)", flush=True)

            rows.append({
                "branch": branch_name,
                "method": method_name,
                "n_rows": int(len(meta)),
                "n_analytes": int(X.shape[1]),
                "technical_noise_before": before["technical_noise"],
                "technical_noise_after": after["technical_noise"],
                "technical_noise_delta": after["technical_noise"] - before["technical_noise"],
                "replicate_consistency_before": before["replicate_consistency"],
                "replicate_consistency_after": after["replicate_consistency"],
                "replicate_consistency_delta": after["replicate_consistency"] - before["replicate_consistency"],
                "axis_stability_before": before["axis_stability"],
                "axis_stability_after": after["axis_stability"],
                "axis_stability_delta": after["axis_stability"] - before["axis_stability"],
            })

    return pd.DataFrame(rows)
