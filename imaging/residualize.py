"""Ridge-regression residualization of nuisance covariates (cell count, batch,
plate) out of a feature matrix, mirroring the "Ridge model" variables shown in
the target reproduction figure (Count / Count+batch / Count+plate /
Count+batch+plate).

Features are z-scored first so that the Ridge fit (and any later cosine
similarity computed on the residuals) treats all columns on a comparable
scale.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

COVARIATE_SETS = {
    "count": ["count"],
    "count_batch": ["count", "batch"],
    "count_plate": ["count", "plate"],
    "count_batch_plate": ["count", "batch", "plate"],
}


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


def zscore(feats: np.ndarray) -> np.ndarray:
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (feats - mean) / std


def ridge_residualize(
    feats: np.ndarray,
    meta: pd.DataFrame,
    covariates: Iterable[str],
    alpha: float = 1.0,
) -> np.ndarray:
    """Z-score `feats` then regress out `covariates` (any of "count", "batch",
    "plate") with a multi-output Ridge fit, returning the residuals."""
    feats = zscore(feats)
    design = _design_matrix(meta, covariates)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(design, feats)
    return feats - model.predict(design)
