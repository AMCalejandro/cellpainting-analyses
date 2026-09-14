"""Batch-effect report: for one feature space, jointly load every hWAT
condition (Baseline + the stress conditions), z-score once, then measure
how a residualization method (a Ridge covariate set, or the control-centered
correction, both in `utils.features`) changes two silhouette scores
computed on the same row sample:

- `silhouette_batch` (Metadata_batch) -- batch signal. Want this to DROP
  after residualization: less separable by batch means the covariate was
  actually absorbed.
- `silhouette_condition` (Metadata_condition) -- biological signal (which
  of Baseline/FFA/IL6/Low Gluc a well belongs to). Want this to survive
  residualization: still separable by condition means the correction didn't
  wash out real signal along with the batch effect.
- `silhouette_plate` (Metadata_Plate) -- finer-grained batch signal. Every
  plate carries exactly one condition (see utils.features docstring), so
  this is inflated by condition separability itself; read it relative to
  `silhouette_condition` rather than against zero. If it stays well above
  `silhouette_condition` after residualization, plate structure survives
  *within* a condition (e.g. control_centered's shrunk plate offsets); if it
  converges toward `silhouette_condition`, within-condition plate drift was
  absorbed (e.g. nested_count_plate).

Global `silhouette_batch` above can read near zero -- "perfectly
integrated" -- even while UMAPs still show batches sub-clustering, once a
feature space (or a residualization method) creates a much larger
between-condition gap (Baseline vs. IL6, etc.) than the within-condition
batch distances: the global silhouette calculation only sees the dominant
biological axis and is blind to local batch structure sitting on top of it
(see docs/batch_effect_conclusions.md). Two more metric families only ever
compare a point to its own condition subset or its k nearest neighbors, so
a single dominant biological axis can't mask them:

- `silhouette_batch_stratified` -- silhouette_batch computed independently
  within each condition subset, then averaged, so cross-condition
  distances never enter the calculation at all.
- `ilisi` / `clisi` / `kbet_rejection_rate` -- single-cell kBET / iLISI /
  cLISI, each computed from every cell's k0 nearest neighbors only.

`compute_report` is pure (no disk writes, no plotting), matching
`utils.copairs` and `imaging.reversion`'s convention: this
module is a library, and the driver script (run_pipeline.py) decides what
gets persisted and calls `utils.plot` to render figures.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from utils import features as feat

from . import load

ALL_CONDITIONS = ["Baseline", "FFA", "IL6", "Low Gluc"]
SAMPLE_SIZE = 5000
SEED = 0
BATCH_COL = "Metadata_batch"
CONDITION_COL = "Metadata_condition"
PLATE_COL = "Metadata_Plate"
COUNT_COL = "Metadata_cell_count"
DEFAULT_K0 = 90
DEFAULT_KBET_ALPHA = 0.05


# Re-exported from `utils.features`, which owns the registry so that
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


def stratified_silhouette_batch(
    feats: np.ndarray,
    batch_labels: pd.Series,
    condition_labels: pd.Series,
    min_batches: int = 2,
) -> float:
    """silhouette_batch computed independently within each
    `condition_labels` subset, then averaged across conditions -- instead
    of pooled across all conditions at once. Pooling lets a large
    Baseline-vs-IL6 gap dominate the global silhouette calculation and
    drown out real within-condition batch clustering (the metric reads
    ~0, "perfectly integrated", even when UMAPs show batches still
    sub-clustering within each condition). Stratifying removes the
    cross-condition distances from the calculation entirely, so each
    condition's batch mixing is scored on its own scale.

    Conditions with fewer than `min_batches` distinct batches are skipped
    (silhouette is undefined for a single label) and excluded from the
    average rather than counted as perfect (0) or missing (NaN) mixing.
    """
    condition_labels = pd.Series(condition_labels).reset_index(drop=True)
    batch_labels = pd.Series(batch_labels).reset_index(drop=True)
    feats = np.asarray(feats)

    scores = []
    for condition in condition_labels.unique():
        idx = np.flatnonzero((condition_labels == condition).to_numpy())
        b = batch_labels.iloc[idx]
        if b.nunique() < min_batches:
            continue
        scores.append(float(silhouette_score(feats[idx], b)))

    return float(np.mean(scores)) if scores else float("nan")


def _neighbor_indices(feats: np.ndarray, k0: int) -> tuple[np.ndarray, int]:
    """Row `i` of the returned array holds the indices of cell i's k0
    nearest neighbors (self excluded)."""
    k0 = min(k0, len(feats) - 1)
    nn = NearestNeighbors(n_neighbors=k0 + 1).fit(feats)
    _, neighbor_idx = nn.kneighbors(feats)
    return neighbor_idx[:, 1:], k0


def _lisi(codes: np.ndarray, n_categories: int, neighbor_idx: np.ndarray) -> float:
    """Mean, over cells, of the inverse Simpson's index of `codes` among
    each cell's neighbors (LISI; Korsunsky et al. 2019), rescaled from
    [1, n_categories] to [0, 1] (scib's convention) so scores are
    comparable across feature spaces/methods with different
    batch/condition counts."""
    if n_categories <= 1:
        return float("nan")
    k0 = neighbor_idx.shape[1]
    simpson = np.empty(len(neighbor_idx))
    for i, neighbors in enumerate(neighbor_idx):
        counts = np.bincount(codes[neighbors], minlength=n_categories)
        p = counts / k0
        simpson[i] = np.sum(p**2)
    lisi = 1.0 / simpson
    return float(np.mean((lisi - 1.0) / (n_categories - 1)))


def _kbet_rejection_rate(
    codes: np.ndarray,
    n_categories: int,
    neighbor_idx: np.ndarray,
    alpha: float = DEFAULT_KBET_ALPHA,
) -> float:
    """Fraction of cells whose neighborhood's label composition rejects
    (chi-squared p < alpha) the null hypothesis that it matches the
    global label frequencies -- the kBET statistic (Buttner et al. 2019).
    Lower = better mixed."""
    k0 = neighbor_idx.shape[1]
    global_freq = np.bincount(codes, minlength=n_categories) / len(codes)
    expected = global_freq * k0
    mask = expected > 0
    rejections = 0
    for neighbors in neighbor_idx:
        observed = np.bincount(codes[neighbors], minlength=n_categories)
        _, p = chisquare(observed[mask], expected[mask])
        if p < alpha:
            rejections += 1
    return rejections / len(neighbor_idx)


def _scib_lisi(
    feats: np.ndarray,
    batch_labels: pd.Series,
    condition_labels: pd.Series,
    k0: int,
) -> tuple[Optional[float], Optional[float]]:
    """Best-effort iLISI/cLISI via `scib.metrics.{ilisi,clisi}_graph` (the
    community-standard, rpy2/R-free graph-LISI implementation). Returns
    `(None, None)` on any failure -- `scib`/`anndata` not installed, an
    API mismatch across scib versions, etc. -- so callers fall back to the
    pure-Python `_lisi` above rather than dying on an optional dependency.
    """
    try:
        import anndata as ad
        import scib.metrics as scib_metrics

        adata = ad.AnnData(
            X=np.zeros((len(feats), 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "batch": pd.Categorical(batch_labels.astype(str).to_numpy()),
                    "condition": pd.Categorical(condition_labels.astype(str).to_numpy()),
                }
            ),
        )
        adata.obsm["X_emb"] = np.asarray(feats, dtype=np.float32)
        ilisi = float(
            scib_metrics.ilisi_graph(
                adata, batch_key="batch", type_="embed", use_rep="X_emb", k0=k0, scale=True
            )
        )
        clisi = float(
            scib_metrics.clisi_graph(
                adata, label_key="condition", type_="embed", use_rep="X_emb", k0=k0, scale=True
            )
        )
        return ilisi, clisi
    except Exception:
        return None, None


def compute_local_mixing_metrics(
    feats: np.ndarray,
    batch_labels: pd.Series,
    condition_labels: pd.Series,
    k0: int = DEFAULT_K0,
    kbet_alpha: float = DEFAULT_KBET_ALPHA,
    use_scib: bool = True,
) -> dict:
    """Single-cell local-mixing metrics on `feats`: kBET rejection rate,
    iLISI (batch mixing) and cLISI (condition purity), each computed from
    every cell's k0 nearest neighbors only -- so, unlike a pooled
    silhouette score, a large between-condition gap can't mask local
    batch sub-clustering.

    - `ilisi`: higher = better batch mixing (batches interleaved within a
      condition).
    - `clisi`: lower = better -- neighborhoods stay condition-pure, i.e.
      the correction didn't wash out real biological signal along with
      batch effects. Read alongside `stratified_silhouette_batch`: a
      method that pushes both `ilisi` and `clisi` up removed condition
      signal, not just batch signal.
    - `kbet_rejection_rate`: fraction of neighborhoods whose batch
      composition significantly deviates from the global batch
      frequencies; lower = better mixed.

    If `use_scib` and `scib`+`anndata` are importable, `ilisi`/`clisi` use
    scib's graph-LISI implementation; otherwise (or on any failure) this
    transparently falls back to the pure-Python implementation below.
    kBET always uses the pure-Python reimplementation here, since
    `scib.metrics.kBET` requires R + the R `kBET` package via `rpy2`,
    which is often unavailable outside a curated R environment.
    """
    feats = np.asarray(feats, dtype=np.float64)
    batch_labels = pd.Series(batch_labels).reset_index(drop=True)
    condition_labels = pd.Series(condition_labels).reset_index(drop=True)

    batch_codes, batch_cats = pd.factorize(batch_labels)
    condition_codes, condition_cats = pd.factorize(condition_labels)
    neighbor_idx, k0_used = _neighbor_indices(feats, k0)

    ilisi = clisi = None
    if use_scib:
        ilisi, clisi = _scib_lisi(feats, batch_labels, condition_labels, k0_used)
    if ilisi is None:
        ilisi = _lisi(batch_codes, len(batch_cats), neighbor_idx)
    if clisi is None:
        clisi = _lisi(condition_codes, len(condition_cats), neighbor_idx)

    return {
        "k0": k0_used,
        "ilisi": ilisi,
        "clisi": clisi,
        "kbet_rejection_rate": _kbet_rejection_rate(
            batch_codes, len(batch_cats), neighbor_idx, kbet_alpha
        ),
    }


def compute_silhouette_scores(
    feats: np.ndarray,
    batch_labels: pd.Series,
    condition_labels: pd.Series,
    plate_labels: pd.Series,
) -> dict:
    """Global batch/condition/plate silhouette, plus the local-mixing
    metrics above: `silhouette_batch_stratified` (silhouette_batch
    averaged within each condition instead of pooled across all of them)
    and single-cell `ilisi`/`clisi`/`kbet_rejection_rate`."""
    local_mixing = compute_local_mixing_metrics(feats, batch_labels, condition_labels)
    return {
        "n_samples": int(len(feats)),
        "silhouette_batch": float(silhouette_score(feats, batch_labels)),
        "silhouette_condition": float(silhouette_score(feats, condition_labels)),
        "silhouette_plate": float(silhouette_score(feats, plate_labels)),
        "silhouette_batch_stratified": stratified_silhouette_batch(
            feats, batch_labels, condition_labels
        ),
        "ilisi": local_mixing["ilisi"],
        "clisi": local_mixing["clisi"],
        "kbet_rejection_rate": local_mixing["kbet_rejection_rate"],
    }


def compute_raw_pca_sample(
    feature_space: str,
    conditions: list = ALL_CONDITIONS,
    cell_line: str = load.DEFAULT_CELL_LINE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    seed: int = SEED,
) -> dict:
    """Load+jointly z-score `feature_space` across `conditions` (no
    residualization) and return a row subsample, for a raw batch-effect PCA
    snapshot (`utils.plot.make_batch_effect_pca_figure`). Independent of
    any `RESIDUALIZE_METHODS` entry, so callers compute this once per
    feature space rather than once per (feature_space, method) pair like
    `compute_report`.

    Pure: no disk writes, no plotting. Returns `{"meta_sample": ...,
    "feats_sample": ...}`."""
    meta, feats_raw = load_all_conditions(feature_space, conditions, cell_line)
    feats = feat.zscore(feats_raw)
    idx = _subsample_index(len(feats), sample_size, seed)
    return {
        "meta_sample": meta.iloc[idx].reset_index(drop=True),
        "feats_sample": feats[idx],
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
    calls `utils.plot.make_batch_report_figures` on the samples itself."""
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
        "before": compute_silhouette_scores(
            before_s, meta_s[BATCH_COL], meta_s[CONDITION_COL], meta_s[PLATE_COL]
        ),
        "after": compute_silhouette_scores(
            after_s, meta_s[BATCH_COL], meta_s[CONDITION_COL], meta_s[PLATE_COL]
        ),
    }
    return {
        "metrics": metrics,
        "meta_sample": meta_s,
        "before_sample": before_s,
        "after_sample": after_s,
    }
