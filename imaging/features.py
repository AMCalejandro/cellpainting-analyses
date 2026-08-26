"""Feature engineering for the copairs-reproduction pipeline: optional
cleanup/dimensionality reduction (any feature space), followed by
Ridge-regression residualization of nuisance covariates (cell count, batch,
plate) out of a feature matrix, mirroring the "Ridge model" variables shown
in the target reproduction figure (Count / Count+batch / Count+plate /
Count+batch+plate).

Raw (non-preprocessed) features are z-scored once per feature space, via
`zscore`, before Ridge residualization so that the fit -- and any later
cosine similarity computed on the residuals -- treats all columns on a
comparable scale. `preprocess` already ends on a z-scored basis (right
before its PCA step), so its output is intentionally *not* re-z-scored
afterwards: doing so would flatten the PCA components back to equal
variance and undo the variance-ranked ordering PCA produces.
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
        count = (count - count.mean()) / count.std()
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
