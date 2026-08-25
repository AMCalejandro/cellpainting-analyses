"""Load the three raw feature spaces (CellProfiler, CPCNN, UniDino) for the
FFA condition, join them against the cpg0014 plate metadata, attach a shared
cell-count covariate, and drop known outlier wells.

Each `load_*` function returns a `(meta, feats)` pair with a standardized
metadata schema:

    Metadata_Plate, Metadata_Well, Metadata_batch, Metadata_broad_sample,
    Metadata_target, Metadata_moa, Metadata_label, Metadata_label2,
    Metadata_pert_type, Metadata_cell_count, Metadata_profile_id

`Metadata_broad_sample` is unique-per-well for control (DMSO/Empty) wells so
that they never form spurious positive pairs with each other.
`Metadata_profile_id` is a stable `Plate::Well` key kept only for debugging.
`meta` and `feats` are always row-aligned (same length, same order).
"""

import pickle

import numpy as np
import pandas as pd

from . import paths

FFA_CONDITION = "FFA"

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


def load_metadata() -> pd.DataFrame:
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
    return meta[meta["Metadata_condition"] == FFA_CONDITION][_METADATA_COLS].copy()


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


def _finalize(meta: pd.DataFrame, feats: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach cell counts, drop outlier/QC-flagged/countless wells, and mint
    the control-safe compound id. `meta` and `feats` must already be
    row-aligned with a shared default RangeIndex."""
    assert len(meta) == len(feats)
    meta = meta.merge(
        load_cell_counts(), on=["Metadata_Plate", "Metadata_Well"], how="left"
    )

    profile_id = meta["Metadata_Plate"] + "::" + meta["Metadata_Well"]
    outliers = load_batch_outliers()

    # "Empty" wells carry no compound and no DMSO vehicle - they are not a
    # valid comparator for any of the three call types, so they are dropped
    # rather than folded into the DMSO control pool.
    keep = (
        ~profile_id.isin(outliers)
        & meta["Metadata_cell_count"].notna()
        & (meta["Metadata_qc_incompatible"] != 1)
        & (meta["Metadata_label"] != "Empty")
        & (~np.isnan(feats).any(axis=1))
    )
    meta = meta.loc[keep].copy()
    feats = feats[keep.to_numpy()]

    meta["Metadata_profile_id"] = profile_id.loc[keep]
    is_control = meta["Metadata_label"] == "DMSO"
    meta["Metadata_broad_sample"] = np.where(
        is_control, meta["Metadata_profile_id"], meta["Metadata_broad_sample"]
    )
    meta["Metadata_pert_type"] = np.where(is_control, "negcon", "trt")

    meta = meta.reset_index(drop=True)
    return meta, feats


def _load_feats_with_metadata(
    df: pd.DataFrame, feature_cols: list
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge a raw (Plate, Well, *features*) table against the authoritative
    cpg0014 metadata, keeping only the feature columns from `df` so that any
    metadata columns it happens to carry (e.g. its own stale broad_sample)
    never collide with the curated metadata."""
    df = df[["Metadata_Plate", "Metadata_Well"] + feature_cols]
    merged = load_metadata().merge(df, on=["Metadata_Plate", "Metadata_Well"], how="inner")
    meta = merged[_METADATA_COLS].reset_index(drop=True)
    feats = merged[feature_cols].to_numpy(dtype=np.float64)
    return _finalize(meta, feats)


def load_cellprofiler() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(paths.CELLPROFILER_PARQUET)
    feature_cols = [c for c in df.columns if not c.startswith("Metadata")]
    return _load_feats_with_metadata(df, feature_cols)


def load_cpcnn() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(paths.CPCNN_TSV_GZ, sep="\t")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    return _load_feats_with_metadata(df, feature_cols)


def load_unidino() -> tuple[pd.DataFrame, np.ndarray]:
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
    is_ffa = (meta_raw["Metadata_condition"] == FFA_CONDITION).to_numpy()
    meta = meta_raw.loc[is_ffa, _METADATA_COLS].reset_index(drop=True)
    feats = feats_raw[is_ffa]
    return _finalize(meta, feats)


FEATURE_LOADERS = {
    "CellProfiler": load_cellprofiler,
    "CPCNN": load_cpcnn,
    "UniDino": load_unidino,
}


def load_feature_space(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    return FEATURE_LOADERS[name]()
