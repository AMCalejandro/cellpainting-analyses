"""Feature engineering for the copairs-reproduction pipeline: optional
cleanup/dimensionality reduction (any feature space), followed by batch/
plate correction of a feature matrix. Two correction approaches live here:

- Ridge-regression residualization of nuisance covariates (cell count,
  batch, plate), mirroring the "Ridge model" variables shown in the target
  reproduction figure (Count / Count+batch / Count+plate /
  Count+batch+plate) -- `ridge_residualize`.
- Control-centered hierarchical correction (`control_centered_residualize`
  and the `center_*` functions it composes) -- estimates batch/plate
  offsets from DMSO CONTROL wells only, matched WITHIN each condition, then
  applies that offset to every well (control and treated alike) in the
  corresponding batch/plate. This sidesteps a confound Ridge residualization
  has on this dataset: cpg0014's plates are single-condition (every plate
  carries rows for exactly one Metadata_condition), so a Ridge fit with
  plate dummies as predictors removes real condition signal along with the
  plate effect -- Metadata_Plate is a near-perfect proxy for
  Metadata_condition. Estimating the plate offset from that plate's OWN
  controls, relative to its own condition's pooled control mean, instead of
  from a design matrix that also "explains" every other row's features by
  plate membership, only removes the part of plate-to-plate variation
  visible in unstressed DMSO wells -- not whatever part of that variation
  happens to coincide with the condition itself. See
  docs/batch_effect_conclusions.md for the evidence and comparison numbers.

Raw (non-preprocessed) features are z-scored once per feature space, via
`zscore`, before either correction so that the fit -- and any later cosine
similarity computed on the residuals -- treats all columns on a comparable
scale. `preprocess` already ends on a z-scored basis (right before its PCA
step), so its output is intentionally *not* re-z-scored afterwards: doing so
would flatten the PCA components back to equal variance and undo the
variance-ranked ordering PCA produces.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

COVARIATE_SETS = {
    "count": ["count"],
    "count_batch": ["count", "batch"],
    "count_plate": ["count", "plate"],
    "count_batch_plate": ["count", "batch", "plate"],
}
SHRINKAGE_N0 = 100.0

def zscore(feats: np.ndarray) -> np.ndarray:
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (feats - mean) / std


def drop_zero_variance(feats: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Drop columns whose variance is at or below `threshold`."""
    return feats[:, feats.var(axis=0) > threshold]


def drop_correlated(feats: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Greedily drop columns correlated above `threshold` with an earlier,
    already-kept column."""
    corr = np.abs(np.corrcoef(feats, rowvar=False))
    keep = np.ones(corr.shape[0], dtype=bool)
    for i in range(corr.shape[0]):
        if keep[i]:
            keep[i + 1 :] &= corr[i, i + 1 :] <= threshold
    return feats[:, keep]


def reduce_dimensionality(
    feats: np.ndarray, n_components: int = 100, seed: int = 0
) -> np.ndarray:
    """PCA the (already z-scored) features down to `n_components` components."""
    return PCA(n_components=n_components, random_state=seed).fit_transform(feats)


def preprocess(
    feats: np.ndarray,
    var_threshold: float = 1e-6,
    corr_threshold: float = 0.95,
    n_components: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Drop zero-variance and highly-correlated columns, then PCA-reduce
    what's left to `n_components` components."""
    feats = drop_zero_variance(feats, var_threshold)
    feats = drop_correlated(feats, corr_threshold)
    feats = zscore(feats)
    return reduce_dimensionality(feats, n_components, seed)


def _design_matrix(meta: pd.DataFrame, covariates: Iterable[str]) -> np.ndarray:
    parts = []
    if "count" in covariates:
        count = meta["Metadata_cell_count"].to_numpy(dtype=np.float64).reshape(-1, 1)
        std = count.std()
        count = (count - count.mean()) / (std if std > 0 else 1.0)
        parts.append(count)
    if "batch" in covariates:
        parts.append(pd.get_dummies(meta["Metadata_batch"]).to_numpy(dtype=np.float64))
    if "plate" in covariates:
        parts.append(pd.get_dummies(meta["Metadata_Plate"]).to_numpy(dtype=np.float64))
    return np.hstack(parts)


def ridge_residualize(
    feats: np.ndarray,
    meta: pd.DataFrame,
    covariates: Iterable[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """Regress `covariates` (any of "count", "batch", "plate") out of
    `feats` with a multi-output Ridge fit, returning the residuals.

    `feats` must already be z-scored (call `zscore` once per feature space,
    not once per covariate set, since it doesn't depend on `covariates`)."""
    design = _design_matrix(meta, covariates)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, feats)
    return feats - model.predict(design)


def condition_nested_residualize(
    feats: np.ndarray,
    meta: pd.DataFrame,
    covariates: Iterable[str] = ("count", "plate"),
    alpha: float = 1.0,
) -> np.ndarray:
    """`ridge_residualize` fit SEPARATELY within each `Metadata_condition`,
    with that condition's own mean added back afterwards.

    This is the same estimator as the pooled `ridge_residualize(...,
    ["count", "plate"])` -- same covariates, same alpha, same ~350 wells per
    plate behind each offset -- reparameterized so it stops destroying the
    condition signal. The pooled fit builds ONE design matrix over every
    condition's rows, and because cpg0014's plates are single-condition, each
    condition indicator is the sum of that condition's plate indicators and
    therefore lies inside the design's column space: the fit removes the
    between-condition offset entirely, as a side effect of removing the
    between-plate offset (see docs/batch_effect_conclusions.md). Fitting one
    Ridge per condition instead means a condition's plate dummies can only
    ever span deviations from that condition's own mean, so:

    - within-condition, between-plate variance is removed at full strength --
      identical to what the pooled `count_plate` fit removes, and the part
      archive/diagnose_overcorrection.py and
      archive/diagnose_holdout_generalization.py showed is genuine technical
      drift rather than a fitting artifact;
    - the between-condition offset (the reversion axis `mu_B - mu_s`) is
      preserved EXACTLY. sklearn's Ridge centers its inputs when
      `fit_intercept=True`, so each condition's residuals are exactly
      mean-zero, and re-adding that condition's raw mean restores its level.

    This is the correction `run_pipeline.py` already applies per condition for
    the copairs calls -- condition is constant there, so a `count_plate` fit
    is *already* condition-nested. The only change is to stop pooling
    conditions before fitting when Baseline and a stress arm are loaded
    together (`imaging.reversion.load_joint_residualized`).

    Note this preserves the axis; it does NOT de-confound it. Every plate
    carries exactly one condition, so whatever technical difference separates
    "Baseline plates" from "stress plates" survives here along with the
    biology. It makes the axis less noisy, not less biased.

    `feats` must already be z-scored; `meta` must carry `Metadata_condition`
    plus whatever columns `covariates` needs."""
    feats = np.asarray(feats, dtype=float)
    condition = meta["Metadata_condition"].to_numpy()
    out = np.empty_like(feats)
    for cond in np.unique(condition):
        mask = condition == cond
        sub_feats = feats[mask]
        residual = ridge_residualize(sub_feats, meta.loc[mask], covariates, alpha)
        out[mask] = residual + sub_feats.mean(axis=0)
    return out


def center_batches_within_condition(
    X: np.ndarray, batch: np.ndarray, condition: np.ndarray, is_control: np.ndarray
) -> np.ndarray:
    """Estimate and apply the batch offset SEPARATELY within each condition
    (a batch x condition interaction model): for every (condition, batch)
    group, shift that group's rows so its control-well mean matches the
    condition's pooled control mean."""
    X = np.asarray(X, dtype=float)
    batch = np.asarray(batch)
    condition = np.asarray(condition)
    is_control = np.asarray(is_control, dtype=bool)

    X_out = X.copy()
    for cond in np.unique(condition):
        cond_mask = condition == cond
        mu_global_cond = X[cond_mask & is_control].mean(axis=0)
        for b in np.unique(batch[cond_mask]):
            mask_b = cond_mask & (batch == b)
            mu_b = X[mask_b & is_control].mean(axis=0)
            X_out[mask_b] = X[mask_b] - mu_b + mu_global_cond
    return X_out


def center_plates_within_condition_shrunk(
    X: np.ndarray,
    plate: np.ndarray,
    condition: np.ndarray,
    is_control: np.ndarray,
    shrinkage_n0: float = SHRINKAGE_N0,
) -> np.ndarray:
    """Plate-level analogue of `center_batches_within_condition`, with
    empirical-Bayes-style shrinkage of each plate's estimated control offset
    toward its condition's pooled control mean, weighted by
    n_plate / (n_plate + shrinkage_n0)."""
    X = np.asarray(X, dtype=float)
    plate = np.asarray(plate)
    condition = np.asarray(condition)
    is_control = np.asarray(is_control, dtype=bool)

    X_out = X.copy()
    for cond in np.unique(condition):
        cond_mask = condition == cond
        mu_global_cond = X[cond_mask & is_control].mean(axis=0)
        for p in np.unique(plate[cond_mask]):
            mask_p = cond_mask & (plate == p)
            ctrl_p = mask_p & is_control
            n_p = int(ctrl_p.sum())
            if n_p == 0:
                continue
            mu_p = X[ctrl_p].mean(axis=0)
            weight = n_p / (n_p + shrinkage_n0)
            X_out[mask_p] = X[mask_p] + weight * (mu_global_cond - mu_p)
    return X_out


def center_plates_within_batch_shrunk(
    X: np.ndarray,
    plate: np.ndarray,
    batch: np.ndarray,
    condition: np.ndarray,
    is_control: np.ndarray,
    shrinkage_n0: float = SHRINKAGE_N0,
) -> np.ndarray:
    """Hierarchical (batch, THEN plate) correction: remove the batch x
    condition offset first via the exact, unshrunk
    `center_batches_within_condition`, then estimate and shrink-correct each
    plate's RESIDUAL deviation on the now-batch-corrected data -- the
    production correction."""
    X_batch_corrected = center_batches_within_condition(X, batch, condition, is_control)
    return center_plates_within_condition_shrunk(
        X_batch_corrected, plate, condition, is_control, shrinkage_n0
    )


def control_centered_residualize(
    feats: np.ndarray, meta: pd.DataFrame, shrinkage_n0: float = SHRINKAGE_N0
) -> np.ndarray:
    """Ridge-residualize cell count only (a continuous nuisance covariate,
    better handled by regression than by group-centering), then
    hierarchically control-center batch (exact) and plate (shrunk) within
    each condition via `center_plates_within_batch_shrunk`. `feats` must
    already be z-scored; `meta` must carry Metadata_batch, Metadata_Plate,
    Metadata_condition and Metadata_pert_type (DMSO controls == "negcon")."""
    feats = ridge_residualize(feats, meta, ["count"])
    is_control = (meta["Metadata_pert_type"] == "negcon").to_numpy()
    return center_plates_within_batch_shrunk(
        feats,
        meta["Metadata_Plate"].to_numpy(),
        meta["Metadata_batch"].to_numpy(),
        meta["Metadata_condition"].to_numpy(),
        is_control,
        shrinkage_n0,
    )


def _ridge_method(covariates: list):
    return lambda feats, meta: ridge_residualize(feats, meta, covariates)


def _nested_method(covariates: list):
    return lambda feats, meta: condition_nested_residualize(feats, meta, covariates)


# Every correction in this module, callable the same way:
# fn(feats_zscored, meta) -> feats_residualized. `imaging.batch_report`,
# run_pipeline.py and `imaging.reversion` all dispatch through this registry,
# so a method only has to be registered once to be available everywhere.
#
# The `nested_*` variants only exist for covariate sets containing "plate":
# without plate dummies the pooled fit's column space doesn't contain the
# condition direction, so nesting would be a no-op relative to the pooled fit.
RESIDUALIZE_METHODS = {key: _ridge_method(covs) for key, covs in COVARIATE_SETS.items()}
RESIDUALIZE_METHODS.update(
    {
        f"nested_{key}": _nested_method(COVARIATE_SETS[key])
        for key in ("count_plate", "count_batch_plate")
    }
)
RESIDUALIZE_METHODS["control_centered"] = control_centered_residualize

# The copairs pipeline runs ONE condition at a time, so Metadata_condition is
# constant there and every `nested_*` method is exactly equivalent to its
# pooled counterpart -- sweeping both would only double an already
# memory-bound runtime.
WITHIN_CONDITION_METHODS = [
    key for key in RESIDUALIZE_METHODS if not key.startswith("nested_")
]
# The cross-condition batch report pools conditions, which is exactly where
# the pooled plate fits over-correct and the nested variants are the point.
CROSS_CONDITION_METHODS = list(RESIDUALIZE_METHODS)
