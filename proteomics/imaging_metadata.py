"""Bridge to the curated cpg0014 metadata (`utils.paths.METADATA_TSV`) that
`proteomics.merge.merge_metadata` needs as its `imaging_meta` side.

Proteomics wells live on physically distinct plates from the imaging ones
(`Metadata_nomic_barcode` is never equal to `Metadata_assay_plate_barcode`,
see `docs/batch_effect_conclusions.md`-style investigation in this module's
tests/plan) -- `Metadata_nomic_barcode` + `Metadata_well_position` is the join
key that lines up with the proteomics CSVs' own `plate_barcode` + `well_id`.

`load_hwat_imaging_metadata` restricts to hWAT rows whose plate was actually
sent for nELISA profiling (`Metadata_nomic_barcode_profiled == True`) and
derives the same `Metadata_pert_type`/control-safe-`Metadata_broad_sample`
convention `imaging.load._finalize` uses, so a proteomics artifact merged
against this table has the same semantics as the three imaging
representations."""

import numpy as np
import pandas as pd

from utils import paths as utils_paths

CELL_LINE = "hWAT"
_KEEP_COLS = [
    "Metadata_nomic_barcode",
    "Metadata_well_position",
    "Metadata_batch",
    "Metadata_condition",
    "Metadata_broad_sample",
    "Metadata_label",
    "Metadata_label2",
    "Metadata_moa",
    "Metadata_target",
]


def load_hwat_imaging_metadata(cell_line: str = CELL_LINE) -> pd.DataFrame:
    """One row per (nomic-profiled, `cell_line`) well: join-key columns plus
    condition/batch/compound annotation and a proteomics-safe
    `Metadata_pert_type`/`Metadata_broad_sample`."""
    meta = pd.read_csv(utils_paths.METADATA_TSV, sep="\t", dtype=str)
    meta["Metadata_qc_incompatible"] = pd.to_numeric(
        meta["Metadata_qc_incompatible"], errors="coerce"
    )

    is_profiled = meta["Metadata_nomic_barcode_profiled"] == "True"
    keep = (
        (meta["Metadata_cell_line"] == cell_line)
        & meta["Metadata_nomic_barcode"].notna()
        & is_profiled
        & (meta["Metadata_qc_incompatible"] != 1)
    )
    meta = meta.loc[keep, _KEEP_COLS].reset_index(drop=True)

    is_control = meta["Metadata_label"] == "DMSO"
    profile_id = meta["Metadata_nomic_barcode"] + "::" + meta["Metadata_well_position"]
    meta["Metadata_broad_sample"] = np.where(
        is_control, profile_id, meta["Metadata_broad_sample"]
    )
    meta["Metadata_pert_type"] = np.where(is_control, "negcon", "trt")
    return meta
