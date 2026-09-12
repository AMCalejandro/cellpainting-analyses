"""Batch/plate correction for the merged, normalized proteomic matrix.

Every profiled nomic-barcode plate carries exactly one `Metadata_condition`
(same single-condition-per-plate structure `docs/batch_effect_conclusions.md`
documents for the imaging plates), so the same over-correction failure mode
applies here: a pooled Ridge fit with plate dummies as predictors would strip
the between-condition offset along with the between-plate one. `method=
"nested"` reuses `imaging.features.condition_nested_residualize` --
fit separately per `Metadata_condition`, that condition's mean added back --
which removes the same plate drift at full strength while preserving the
condition signal exactly. There is no per-well cell count for proteomics
wells (a physically distinct plate from the one CellProfiler counted cells
on), so unlike imaging's `nested_count_plate` there is no "count" covariate
to include -- `covariates` defaults to `("plate",)` only.

`method="control_centered"` is kept as an explicit, more conservative
alternative (estimates the plate offset from DMSO controls only, shrunk
toward the condition's pooled control mean) -- `docs/batch_effect_conclusions.md`
found this under-corrects relative to the nested-Ridge estimator, so it is
not the default.

Both methods expect `meta` to carry `Metadata_condition` and `Metadata_Plate`
(the caller aliases `Metadata_nomic_barcode` -> `Metadata_Plate` before
calling this, so this module stays agnostic of the proteomics join-key
naming) and, for `control_centered`, `Metadata_batch` and `Metadata_pert_type`
(`"negcon"` for DMSO controls).

`correct` re-standardizes `X` with `robust_zscore` (median/1.4826*MAD),
NOT `imaging.features.zscore` (plain mean/std) -- `X` arriving here has
already been through `proteomics.normalization.transform_and_scale`'s own
arcsinh + median/MAD scaling, specifically so a handful of genuinely
saturating wells on a heavily-secreted analyte (IL6, TNF, CCL20, CXCL10 --
exactly the readouts an IL6/FFA stress screen cares about) can't dominate
that analyte's apparent spread. A plain mean/std z-score undoes that: on
this dataset those analytes' plain std runs 10-15x every other analyte's
(measured on the merged hWAT matrix) purely from a few saturating-but-real
stimulated wells, which then divides the ENTIRE column -- including the
bulk of non-saturating replicate wells -- by an inflated denominator before
any batch correction or reversion-axis math runs on it. Re-applying the
same robust estimator Stage 1 already committed to keeps the two stages
consistent instead of one silently overriding the other's design choice."""

from typing import Iterable

import numpy as np
import pandas as pd

from utils import features as feat

DEFAULT_COVARIATES = ("plate",)


def robust_zscore(X: np.ndarray) -> np.ndarray:
    """Median/1.4826*MAD standardization -- same formula as
    `proteomics.normalization.transform_and_scale`, so re-standardizing
    already-robust-scaled input is close to a no-op rather than a silent
    override. A zero-MAD (degenerate) column is left centered but unscaled,
    matching `imaging.features.zscore`'s zero-std fallback.

    Public -- also used by `proteomics.pipeline.load_raw_proteomics` for the
    processed=False (uncorrected) copairs input, so that saturating analytes
    (IL6, TNF, CCL20, CXCL10 -- see this module's docstring) don't dominate
    copairs' pairwise similarity there either."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    scale = 1.4826 * mad
    scale[scale == 0] = 1.0
    return (X - med) / scale


def correct(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    covariates: Iterable[str] = DEFAULT_COVARIATES,
    method: str = "nested",
) -> pd.DataFrame:
    """Robust-re-standardize `X` per analyte, then batch/plate-correct via
    `method`. Returns a DataFrame with the same columns/index as `X`."""
    feats = robust_zscore(X.to_numpy(dtype=np.float64))

    if method == "nested":
        corrected = feat.condition_nested_residualize(feats, meta, covariates)
    elif method == "control_centered":
        is_control = (meta["Metadata_pert_type"] == "negcon").to_numpy()
        corrected = feat.center_plates_within_batch_shrunk(
            feats,
            meta["Metadata_Plate"].to_numpy(),
            meta["Metadata_batch"].to_numpy(),
            meta["Metadata_condition"].to_numpy(),
            is_control,
        )
    else:
        raise ValueError(f"unknown method {method!r}, expected 'nested' or 'control_centered'")

    return pd.DataFrame(corrected, columns=X.columns, index=X.index)
