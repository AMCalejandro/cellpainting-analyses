"""Batch-effect report: for one feature space, jointly load every hWAT
condition (Baseline + the stress conditions), z-score once, then measure
how a residualization method (a Ridge covariate set, or the control-centered
correction, both in `imaging.features`) changes two silhouette scores
computed on the same row sample:

- `silhouette_batch` (Metadata_batch) -- batch signal. Want this to DROP
  after residualization: less separable by batch means the covariate was
  actually absorbed.
- `silhouette_condition` (Metadata_condition) -- biological signal (which
  of Baseline/FFA/IL6/Low Gluc a well belongs to). Want this to survive
  residualization: still separable by condition means the correction didn't
  wash out real signal along with the batch effect.

`compute_report` is pure (no disk writes, no plotting), matching
`imaging.copairs_pipeline` and `imaging.reversion`'s convention: this
module is a library, and the driver script (run_pipeline.py) decides what
gets persisted and calls `imaging.plot` to render figures.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from . import features as feat
from . import load

ALL_CONDITIONS = ["Baseline", "FFA", "IL6", "Low Gluc"]
SAMPLE_SIZE = 5000
SEED = 0
BATCH_COL = "Metadata_batch"
CONDITION_COL = "Metadata_condition"


# Re-exported from `imaging.features`, which owns the registry so that
# run_pipeline.py, this module and `imaging.reversion` all dispatch method
# names through one place. Includes every Ridge covariate set, the
# `nested_*` (condition-nested Ridge) variants and the control-centered
# hierarchical correction, all callable as
# fn(feats_zscored, meta) -> feats_residualized.
RESIDUALIZE_METHODS = feat.RESIDUALIZE_METHODS


def load_all_conditions(
    feature_space: str,
    conditions: list = ALL_CONDITIONS,
    cell_line: str = load.DEFAULT_CELL_LINE,
    log_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load every condition in `conditions` for `feature_space` (same
    `cell_line` throughout), tag each row with its `Metadata_condition`, and
    concatenate. Row-aligned, not yet z-scored or residualized."""
    metas, featss = [], []
    for condition in conditions:
        m, f = load.load_feature_space(
            feature_space, condition, log_dir=log_dir, cell_line=cell_line
        )
        m = m.copy()
        m[CONDITION_COL] = condition
        metas.append(m)
        featss.append(f)
    meta = pd.concat(metas, ignore_index=True)
    feats = np.vstack(featss)
    return meta, feats


def _subsample_index(n: int, sample_size: Optional[int], seed: int) -> np.ndarray:
    if sample_size is None or n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False)


def compute_silhouette_scores(
    feats: np.ndarray, batch_labels: pd.Series, condition_labels: pd.Series
) -> dict:
    return {
        "n_samples": int(len(feats)),
        "silhouette_batch": float(silhouette_score(feats, batch_labels)),
        "silhouette_condition": float(silhouette_score(feats, condition_labels)),
    }


def compute_report(
    feature_space: str,
    method: str,
    conditions: list = ALL_CONDITIONS,
    cell_line: str = load.DEFAULT_CELL_LINE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    seed: int = SEED,
) -> dict:
    """Load+jointly z-score `feature_space` across `conditions`, residualize
    with `RESIDUALIZE_METHODS[method]`, and compute batch/condition
    silhouette before vs after on a shared row subsample.

    Pure: no disk writes, no plotting. Returns `{"metrics": ..., "meta_sample":
    ..., "before_sample": ..., "after_sample": ...}` -- the caller (e.g.
    run_pipeline.py's --batch-report branch) decides what to persist and
    calls `imaging.plot.make_batch_report_figures` on the samples itself."""
    meta, feats_raw = load_all_conditions(feature_space, conditions, cell_line)
    feats_before = feat.zscore(feats_raw)
    feats_after = RESIDUALIZE_METHODS[method](feats_before, meta)

    idx = _subsample_index(len(feats_before), sample_size, seed)
    meta_s = meta.iloc[idx].reset_index(drop=True)
    before_s, after_s = feats_before[idx], feats_after[idx]

    metrics = {
        "feature_space": feature_space,
        "method": method,
        "cell_line": cell_line,
        "conditions": conditions,
        "before": compute_silhouette_scores(before_s, meta_s[BATCH_COL], meta_s[CONDITION_COL]),
        "after": compute_silhouette_scores(after_s, meta_s[BATCH_COL], meta_s[CONDITION_COL]),
    }
    return {
        "metrics": metrics,
        "meta_sample": meta_s,
        "before_sample": before_s,
        "after_sample": after_s,
    }
