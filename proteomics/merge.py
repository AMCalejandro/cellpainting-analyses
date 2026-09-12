"""
Stage 2a: merge normalized/QC'd proteomics data with imaging metadata.

merge_metadata() joins on (Metadata_nomic_barcode, Metadata_well_position).
load_proteomics_normalized() is the main entry point the rest of the
codebase should use: it runs proteomics_pipeline.normalization's full
pipeline (via a cache, see below), then annotates/splits/merges.

See proteomics.correction for the batch/plate correction that runs
on this merged output.
"""

import dataclasses
import os
import pickle
from typing import Any, Literal

import numpy as np
import pandas as pd

from .normalization import (
    load_normalized,
    drop_curated_bad_samples,
    analyte_detection_qc,
    impute_missing,
    transform_and_scale,
    flag_outlier_samples,
)
from .paths import NORMALIZED_INTERIM_PATH, RAW_PROTE_PATH

# --------------------------------------------------------------------------
# Merge utility
# --------------------------------------------------------------------------

MergeMode = Literal["inner", "imaging", "proteomics", "outer"]
_JOIN_KEYS = ["Metadata_nomic_barcode", "Metadata_well_position"]


def merge_metadata(
    imaging_meta: pd.DataFrame,
    prote_meta: pd.DataFrame,
    mode: MergeMode = "inner",
    join_keys: list[str] | None = None,
    suffixes: tuple[str, str] = ("_imaging", "_prote"),
) -> pd.DataFrame:
    """Merge imaging metadata with proteomics metadata.

    mode: "inner" (wells in both) / "imaging" (keep all imaging wells) /
    "proteomics" (keep all proteomics wells) / "outer" (union).
    """
    keys = join_keys if join_keys is not None else _JOIN_KEYS

    _how_map: dict[MergeMode, str] = {
        "inner": "inner",
        "imaging": "left",
        "proteomics": "right",
        "outer": "outer",
    }
    if mode not in _how_map:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {list(_how_map)}")

    if mode == "proteomics":
        merged = prote_meta.merge(imaging_meta, on=keys, how="left", suffixes=(suffixes[1], suffixes[0]))
    else:
        merged = imaging_meta.merge(prote_meta, on=keys, how=_how_map[mode], suffixes=suffixes)

    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# Proteomics loading / merging
# --------------------------------------------------------------------------

_DEFAULT_PROTE_META_COLS = [
    "plate_barcode",
    "well_id",
    "group",
    "target_gene",
    "sample_passed_qc",
    "hit",
    "broad_sample",
    "user_sample_batch",
    "pert_iname",
    "moa",
]


@dataclasses.dataclass
class ProteomicsDataset:
    """Everything needed after loading + (optionally) merging proteomics data.

    meta: metadata slab (original columns + ``hit``), row-aligned with ``X``.
    X: analyte feature matrix, row-aligned with ``meta``.
    prote_meta: deduplicated per-well metadata with join-key column names.
    joined / X_joined: result of merging with imaging metadata, when supplied.
    """

    meta: pd.DataFrame
    X: pd.DataFrame
    prote_meta: pd.DataFrame
    joined: pd.DataFrame | None = None
    X_joined: pd.DataFrame | None = None


def _prepare_proteomics_dataset(
    meta_df: pd.DataFrame,
    X: pd.DataFrame,
    *,
    hit_keys: set[tuple] | None = None,
    prote_meta_cols: list[str] | None = None,
    add_prefix: str | None = None,
    imaging_meta: pd.DataFrame | None = None,
    merge_mode: MergeMode = "imaging",
    join_keys: list[str] | None = None,
    fill_missing: dict[str, Any] | None = None,
) -> ProteomicsDataset:
    """Shared logic for annotating / splitting / merging, given an
    already-loaded (and, ideally, already-normalized) ``meta_df``/``X`` pair
    that are row-aligned with a shared 0..N-1 index."""
    keys = join_keys if join_keys is not None else _JOIN_KEYS
    keep_cols = prote_meta_cols if prote_meta_cols is not None else _DEFAULT_PROTE_META_COLS

    meta_df = meta_df.reset_index(drop=True).copy()
    X = X.reset_index(drop=True).copy()

    if hit_keys is not None:
        meta_df["hit"] = [
            (b, w) in hit_keys for b, w in zip(meta_df["plate_barcode"], meta_df["well_id"])
        ]
    else:
        meta_df["hit"] = False

    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = impute_missing(X)

    available_cols = [c for c in keep_cols if c in meta_df.columns]
    prote_meta = (
        meta_df[available_cols]
        .rename(columns={"plate_barcode": "Metadata_nomic_barcode", "well_id": "Metadata_well_position"})
        .reset_index(drop=True)
    )
    if add_prefix:
        non_key = [c for c in prote_meta.columns if c not in keys]
        prote_meta = prote_meta.rename(columns={c: f"{add_prefix}{c}" for c in non_key})

    joined: pd.DataFrame | None = None
    X_joined: pd.DataFrame | None = None
    if imaging_meta is not None:
        joined = merge_metadata(
            imaging_meta=imaging_meta.reset_index(drop=True),
            prote_meta=prote_meta,
            mode=merge_mode,
            join_keys=keys,
        )
        if fill_missing:
            for col, fill_val in fill_missing.items():
                if col in joined.columns:
                    joined[col] = joined[col].fillna(fill_val).astype(str)

        # Re-key X by the join columns and reindex to `joined`'s row order,
        # since a non-"inner" merge may include wells with no proteomics signal.
        keyed_signals = X.copy()
        keyed_signals[keys[0]] = meta_df["plate_barcode"].to_numpy()
        keyed_signals[keys[1]] = meta_df["well_id"].to_numpy()
        keyed_signals = keyed_signals.drop_duplicates(subset=keys)

        X_joined = (
            joined[keys]
            .merge(keyed_signals, on=keys, how="left")
            .drop(columns=keys)
            .reset_index(drop=True)
        )

    return ProteomicsDataset(meta=meta_df, X=X, prote_meta=prote_meta, joined=joined, X_joined=X_joined)


def load_proteomics(
    csv_path: str,
    *,
    n_meta_cols: int = 24,
    read_csv_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ProteomicsDataset:
    """Load proteomics data straight from a CSV, with no QC/normalization.
    Prefer `load_proteomics_normalized` - this is kept for quick, raw-data
    exploration only (we no longer feed raw proteomics data into the
    merge/correction pipeline)."""
    df = pd.read_csv(csv_path, **(read_csv_kwargs or {}))
    meta_df = df.iloc[:, :n_meta_cols].reset_index(drop=True)
    X = df.iloc[:, n_meta_cols:].reset_index(drop=True)
    return _prepare_proteomics_dataset(meta_df, X, **kwargs)


def load_proteomics_normalized(
    csv_path: str = RAW_PROTE_PATH,
    *,
    n_meta_cols: int = 24,
    min_detection_rate: float = 0.20,
    outlier_z_thresh: float = 8.0,
    outlier_min_flagged: int = 15,
    use_cache: bool = True,
    cache_path: str = NORMALIZED_INTERIM_PATH,
    **kwargs: Any,
) -> ProteomicsDataset:
    """Run the full proteomics_pipeline.normalization pipeline, then
    annotate / split / (optionally) merge with imaging metadata. This is the
    entry point the rest of the codebase should use going forward.

    When `use_cache` and `cache_path` (proteomics_pipeline.normalization's
    saved stage-1 output) exists, reads normalized meta/X from there instead
    of recomputing - so re-running the merge/correction stage doesn't
    silently redo normalization with whatever defaults happen to be passed
    here. If the cache is missing, this computes it fresh AND writes it, so
    the interim artifact always exists after the first run regardless of
    which script ran first.
    """
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        meta_df, X = cached["meta"], cached["X"]
    else:
        meta_df, X = load_normalized(csv_path, n_meta_cols)
        meta_df, X = drop_curated_bad_samples(meta_df, X)
        X, dropped_analytes = analyte_detection_qc(X, min_detection_rate=min_detection_rate, verbose=False)
        X = impute_missing(X)
        X = transform_and_scale(X)

        outlier_idx = flag_outlier_samples(
            X, z_thresh=outlier_z_thresh, min_flagged=outlier_min_flagged, verbose=False
        )
        meta_df = meta_df[~meta_df.index.isin(outlier_idx)].reset_index(drop=True)
        X = X[~X.index.isin(outlier_idx)].reset_index(drop=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({
                "meta": meta_df, "X": X,
                "dropped_analytes": dropped_analytes, "outlier_idx": outlier_idx,
            }, f)

    return _prepare_proteomics_dataset(meta_df, X, **kwargs)
