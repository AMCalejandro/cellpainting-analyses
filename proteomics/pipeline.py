"""End-to-end proteomics entry point: normalize (stage 1, `.normalization`)
-> merge with imaging metadata (stage 2a, `.merge`) -> batch/plate-correct
(stage 2b, `.correction`). `load_corrected_proteomics` is what downstream
code (compound validation, Tier D of experiments/benchmark_feature_representation.md)
should call; `load_raw_proteomics` is its processed=False counterpart, used
by `run_proteomics_copairs.py` to run copairs on the merged-but-uncorrected
matrix instead."""

import pickle
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import correction
from . import imaging_metadata as imgmeta
from . import merge
from . import paths

DEFAULT_COVARIATES = correction.DEFAULT_COVARIATES
DEFAULT_METHOD = "nested"


def _log(lines: list, path: Path = paths.RESULTS_DIR / "loading_log.txt") -> None:
    with open(path, "a") as fh:
        fh.write("\n".join(lines) + "\n\n")


def _load_joined() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize + join against the curated cpg0014 hWAT metadata, shared by
    both `load_corrected_proteomics` and `load_raw_proteomics` so the two
    only differ in whether `.correction.correct` runs afterwards. Returns
    `(meta, X)`, row-aligned and pre-correction."""
    imaging_meta = imgmeta.load_hwat_imaging_metadata()
    ds = merge.load_proteomics_normalized(imaging_meta=imaging_meta, merge_mode="inner")
    meta, X = ds.joined, ds.X_joined

    lines = [
        f"normalized proteomics rows (post stage-1 QC): {len(ds.meta)}",
        f"hWAT imaging metadata rows (profiled plates, QC-compatible): {len(imaging_meta)}",
        f"rows after inner join on (Metadata_nomic_barcode, Metadata_well_position): {len(meta)}",
        f"conditions: {meta['Metadata_condition'].value_counts().to_dict()}",
    ]
    _log(lines)

    meta = meta.copy()
    meta["Metadata_Plate"] = meta["Metadata_nomic_barcode"]
    return meta, X


def load_corrected_proteomics(
    covariates: Iterable[str] = DEFAULT_COVARIATES,
    method: str = DEFAULT_METHOD,
    use_cache: bool = True,
    cache_path: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalized + batch/plate-corrected proteomic feature matrix, joined
    against the curated cpg0014 hWAT metadata. Returns `(meta, X_corrected)`,
    row-aligned, `Metadata_broad_sample`/`Metadata_pert_type` conventions
    matching `imaging.load.load_feature_space`'s output.

    Caches the full result (`meta`, pre-correction `X`, and `X_corrected`) to
    `cache_path` so repeated calls (e.g. a fresh Python process) skip
    re-running normalization/correction; delete the cache file (or pass
    `use_cache=False`) to force a rebuild."""
    covariates = tuple(covariates)
    cache_path = cache_path or paths.corrected_interim_path(method, covariates)
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        return cached["meta"], cached["X_corrected"]

    meta, X = _load_joined()
    X_corrected = correction.correct(X, meta, covariates=covariates, method=method)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump({"meta": meta, "X": X, "X_corrected": X_corrected}, fh)

    return meta, X_corrected


def load_raw_proteomics(
    use_cache: bool = True,
    cache_path: Path = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalized + merged proteomic feature matrix WITHOUT batch/plate
    correction -- the processed=False counterpart to
    `load_corrected_proteomics`. Re-standardizes with
    `correction.robust_zscore` (same estimator `.correction.correct` re-applies
    before residualizing, see that module's docstring) rather than leaving
    the already arcsinh+MAD-scaled stage-1 output as-is, so copairs' cosine
    similarity is computed on the same per-analyte scale either way. Returns
    `(meta, X_raw)`, row-aligned, same conventions as
    `load_corrected_proteomics`."""
    cache_path = cache_path or paths.RAW_INTERIM_PATH
    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        return cached["meta"], cached["X_raw"]

    meta, X = _load_joined()
    X_raw = pd.DataFrame(
        correction.robust_zscore(X.to_numpy(dtype="float64")),
        columns=X.columns, index=X.index,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump({"meta": meta, "X_raw": X_raw}, fh)

    return meta, X_raw
