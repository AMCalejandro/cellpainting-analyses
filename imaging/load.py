"""Load the three raw feature spaces (CellProfiler, CPCNN, UniDino) for a
given condition (e.g. "FFA", "IL6", "Low Gluc"), join them against the
cpg0014 plate metadata, attach a shared cell-count covariate, and drop known
outlier wells.

Each `load_*` function returns a `(meta, feats)` pair with a standardized
metadata schema:

    Metadata_Plate, Metadata_Well, Metadata_batch, Metadata_broad_sample,
    Metadata_target, Metadata_moa, Metadata_label, Metadata_label2,
    Metadata_pert_type, Metadata_cell_count, Metadata_profile_id

`Metadata_broad_sample` is unique-per-well for control (DMSO/Empty) wells so
that they never form spurious positive pairs with each other.
`Metadata_profile_id` is a stable `Plate::Well` key kept only for debugging.
`meta` and `feats` are always row-aligned (same length, same order).

Every call also appends a shape trace -- row/column counts at each loading
and filtering step, and why rows were dropped -- to
`imaging/results/loading_log_<condition>.log`.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

DEFAULT_CONDITION = "FFA"

_METADATA_COLS = [
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_batch",
    "Metadata_broad_sample",
    "Metadata_target",
    "Metadata_moa",
    "Metadata_label",
    "Metadata_label2",
    "Metadata_qc_incompatible",
]


def loading_log_path(condition: str, log_dir: Path = None) -> Path:
    return (log_dir or paths.RESULTS_DIR) / f"loading_log_{condition}.log"


def _write_loading_log(
    feature_space: str, condition: str, lines: list, log_dir: Path = None
) -> None:
    with open(loading_log_path(condition, log_dir), "a") as fh:
        fh.write(f"=== {feature_space} (condition={condition}) ===\n")
        for line in lines:
            fh.write(f"{line}\n")
        fh.write("\n")


def load_metadata(condition: str = DEFAULT_CONDITION) -> pd.DataFrame:
    meta = pd.read_csv(paths.METADATA_TSV, sep="\t", dtype=str)
    meta["Metadata_qc_incompatible"] = pd.to_numeric(
        meta["Metadata_qc_incompatible"], errors="coerce"
    )
    meta = meta.rename(
        columns={
            "Metadata_assay_plate_barcode": "Metadata_Plate",
            "Metadata_well_position": "Metadata_Well",
        }
    )
    return meta[meta["Metadata_condition"] == condition][_METADATA_COLS].copy()


def load_batch_outliers() -> set:
    outliers = pd.read_csv(paths.BATCH_OUTLIERS_TSV, sep="\t")
    key_col = outliers.columns[0]
    return set(outliers[key_col].astype(str))


def load_cell_counts() -> pd.DataFrame:
    """Well-level cell counts, derived once from the CellProfiler QC columns
    so that all three feature spaces share the exact same `Count` covariate.
    """
    df = pd.read_parquet(
        paths.CELLPROFILER_PARQUET,
        columns=["Metadata_Plate", "Metadata_Well", "Metadata_Count_Cells"],
    )
    df = df.rename(columns={"Metadata_Count_Cells": "Metadata_cell_count"})
    return df.drop_duplicates(["Metadata_Plate", "Metadata_Well"])


def _finalize(
    meta: pd.DataFrame, feats: np.ndarray, lines: list
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach cell counts, drop outlier/QC-flagged/countless wells, and mint
    the control-safe compound id. `meta` and `feats` must already be
    row-aligned with a shared default RangeIndex."""
    assert len(meta) == len(feats)
    lines.append(f"rows before well-level filtering: {len(meta)}")

    meta = meta.merge(
        load_cell_counts(), on=["Metadata_Plate", "Metadata_Well"], how="left"
    )

    profile_id = meta["Metadata_Plate"] + "::" + meta["Metadata_Well"]
    outliers = load_batch_outliers()

    is_outlier = profile_id.isin(outliers)
    missing_count = meta["Metadata_cell_count"].isna()
    is_qc_incompatible = meta["Metadata_qc_incompatible"] == 1
    # "Empty" wells carry no compound and no DMSO vehicle - they are not a
    # valid comparator for any of the three call types, so they are dropped
    # rather than folded into the DMSO control pool.
    is_empty = meta["Metadata_label"] == "Empty"
    has_nan_feats = np.isnan(feats).any(axis=1)

    lines.append(f"  flagged as batch outlier: {int(is_outlier.sum())}")
    lines.append(f"  missing cell count: {int(missing_count.sum())}")
    lines.append(f"  QC-incompatible: {int(is_qc_incompatible.sum())}")
    lines.append(f"  Empty-labeled wells: {int(is_empty.sum())}")
    lines.append(f"  rows with a NaN feature: {int(has_nan_feats.sum())}")
    lines.append("  (these can overlap; a row may match more than one of the above)")

    keep = (
        ~is_outlier
        & ~missing_count
        & ~is_qc_incompatible
        & ~is_empty
        & ~has_nan_feats
    )
    meta = meta.loc[keep].copy()
    feats = feats[keep.to_numpy()]
    lines.append(
        f"rows kept after combined filter: {len(meta)} "
        f"(dropped {int((~keep).sum())})"
    )

    meta["Metadata_profile_id"] = profile_id.loc[keep]
    is_control = meta["Metadata_label"] == "DMSO"
    meta["Metadata_broad_sample"] = np.where(
        is_control, meta["Metadata_profile_id"], meta["Metadata_broad_sample"]
    )
    meta["Metadata_pert_type"] = np.where(is_control, "negcon", "trt")

    meta = meta.reset_index(drop=True)
    return meta, feats


def _load_feats_with_metadata(
    df: pd.DataFrame, feature_cols: list, condition: str, lines: list
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge a raw (Plate, Well, *features*) table against the authoritative
    cpg0014 metadata, keeping only the feature columns from `df` so that any
    metadata columns it happens to carry (e.g. its own stale broad_sample)
    never collide with the curated metadata."""
    lines.append(f"raw rows (all conditions): {len(df)}")
    df = df[["Metadata_Plate", "Metadata_Well"] + feature_cols]
    metadata = load_metadata(condition)
    lines.append(f"condition-filtered metadata rows: {len(metadata)}")
    merged = metadata.merge(df, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    lines.append(f"after inner merge on (Plate, Well): {len(merged)}")
    meta = merged[_METADATA_COLS].reset_index(drop=True)
    feats = merged[feature_cols].to_numpy(dtype=np.float64)
    return _finalize(meta, feats, lines)


def load_cellprofiler(
    condition: str = DEFAULT_CONDITION, log_dir: Path = None
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(paths.CELLPROFILER_PARQUET)
    feature_cols = [c for c in df.columns if not c.startswith("Metadata")]
    lines = [f"loaded raw parquet: {df.shape}"]
    meta, feats = _load_feats_with_metadata(df, feature_cols, condition, lines)
    lines.append(f"final: {feats.shape}")
    _write_loading_log("CellProfiler", condition, lines, log_dir)
    return meta, feats


def load_cpcnn(
    condition: str = DEFAULT_CONDITION, log_dir: Path = None
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(paths.CPCNN_TSV_GZ, sep="\t")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    lines = [f"loaded raw tsv: {df.shape}"]
    meta, feats = _load_feats_with_metadata(df, feature_cols, condition, lines)
    lines.append(f"final: {feats.shape}")
    _write_loading_log("CPCNN", condition, lines, log_dir)
    return meta, feats


def load_unidino(
    condition: str = DEFAULT_CONDITION, log_dir: Path = None
) -> tuple[pd.DataFrame, np.ndarray]:
    with open(paths.UNIDINO_PKL, "rb") as fh:
        obj = pickle.load(fh)
    meta_raw, feats_raw = obj["meta"], obj["features"]

    meta_raw = meta_raw.rename(
        columns={
            "Metadata_assay_plate_barcode": "Metadata_Plate",
            "Metadata_well_position": "Metadata_Well",
        }
    )
    meta_raw["Metadata_qc_incompatible"] = pd.to_numeric(
        meta_raw["Metadata_qc_incompatible"], errors="coerce"
    )
    lines = [f"loaded raw pickle: {feats_raw.shape}"]
    is_condition = (meta_raw["Metadata_condition"] == condition).to_numpy()
    lines.append(f"condition-filtered rows: {int(is_condition.sum())}")
    meta = meta_raw.loc[is_condition, _METADATA_COLS].reset_index(drop=True)
    feats = feats_raw[is_condition]
    meta, feats = _finalize(meta, feats, lines)
    lines.append(f"final: {feats.shape}")
    _write_loading_log("UniDino", condition, lines, log_dir)
    return meta, feats


FEATURE_LOADERS = {
    "CellProfiler": load_cellprofiler,
    "CPCNN": load_cpcnn,
    "UniDino": load_unidino,
}


def load_feature_space(
    name: str, condition: str = DEFAULT_CONDITION, log_dir: Path = None
) -> tuple[pd.DataFrame, np.ndarray]:
    return FEATURE_LOADERS[name](condition, log_dir=log_dir)
