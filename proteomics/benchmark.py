"""Proteomics-side Tier E: copairs-level cross-processed-tag/cross-condition
agreement and biological plausibility (experiments/benchmark_feature_representation.md).

This is proteomics's OWN implementation, not a reuse of `imaging.benchmark` --
this repo's module-independence rule (see `utils/__init__.py`) means imaging
and proteomics may not import each other, only `utils`. Where imaging's Tier
E groups by "representation" (CellProfiler/CPCNN/UniDino), proteomics has a
single feature space, so `processed_tag` (raw vs. a batch/plate-corrected
variant, e.g. "nested_plate_batch_batch_plate") plays that role instead --
the same swap `utils.plot.make_proteomics_copairs_summary_figure` already
makes for the copairs summary figure. `commands.proteomics.tier_e_main`
drives this module; `utils.plot.make_proteomics_tier_e_figure` renders its
output.

Reuses `utils.copairs`'s hit-set (`hit_overlap`/`mean_pairwise_jaccard`/
`consistency_called_terms`/`combine_allowlist_ranks`) and
`utils.bio_enrichment.moa_enrichment` unmodified -- only the parquet-loading
and compound-annotation conventions here are proteomics-specific.
"""

from pathlib import Path

import pandas as pd

from utils import bio_enrichment
from utils import copairs as cp


def _copairs_parquet_path(
    out_dir: Path, condition: str, processed_tag: str, call_name: str
) -> Path:
    """`commands.proteomics.copairs_main`'s file_stub convention:
    `<out_dir>/parquet/proteomics_<condition>_<processed_tag>_<call_name>.parquet`."""
    return out_dir / "parquet" / f"proteomics_{condition}_{processed_tag}_{call_name}.parquet"


def load_existing_copairs_call(
    out_dir: Path, condition: str, processed_tag: str, call_name: str
) -> pd.DataFrame:
    """Load an already-computed `proteomics copairs` call (`call_name` one of
    "activity", "distinctiveness", "consistency") instead of rerunning
    copairs -- mirrors `imaging.benchmark.load_existing_copairs_call`."""
    path = _copairs_parquet_path(out_dir, condition, processed_tag, call_name)
    if not path.exists():
        raise FileNotFoundError(
            f"no existing {call_name!r} result at {path} -- run "
            f'.venv/bin/python cli.py proteomics copairs --conditions "{condition}" first'
        )
    return pd.read_parquet(path)


def activity_and_distinctiveness(out_dir: Path, condition: str, processed_tag: str) -> dict:
    """The reversion-style compound_allowlist (active ∩ distinctive) for one
    (condition, processed_tag), from already-saved calls -- mirrors
    `imaging.benchmark.activity_and_distinctiveness`."""
    activity_df = load_existing_copairs_call(out_dir, condition, processed_tag, "activity")
    distinct_df = load_existing_copairs_call(out_dir, condition, processed_tag, "distinctiveness")
    active = set(activity_df.loc[activity_df["below_corrected_p"], "Metadata_broad_sample"])
    distinct = set(distinct_df.loc[distinct_df["below_corrected_p"], "Metadata_broad_sample"])
    return {
        "n_compounds": int(len(activity_df)),
        "n_active": int(len(active)),
        "n_distinct": int(len(distinct)),
        "active_compounds": active,
        "distinct_compounds": distinct,
        "allowlist": active & distinct,
        "activity_table": activity_df,
        "distinctiveness_table": distinct_df,
    }


def copairs_call_enrichment(
    out_dir: Path,
    condition: str,
    processed_tag: str,
    call_name: str,
    annotation: pd.DataFrame,
    moa_col: str = "Metadata_moa",
    score_col: str = "mean_average_precision",
    n_perm: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    """MoA/target preranked-GSEA enrichment among a (condition, processed_tag)'s
    copairs-called compounds -- mirrors `imaging.benchmark.copairs_call_enrichment`,
    with `annotation` (a Metadata_broad_sample -> Metadata_moa/Metadata_target
    lookup, e.g. from `proteomics.imaging_metadata.load_hwat_imaging_metadata`)
    supplied by the caller instead of being loaded here, since proteomics has
    no equivalent of imaging's per-condition `load.load_metadata`."""
    if call_name == "allowlist":
        act = activity_and_distinctiveness(out_dir, condition, processed_tag)
        df = cp.combine_allowlist_ranks(act["activity_table"], act["distinctiveness_table"], score_col)
    else:
        df = load_existing_copairs_call(out_dir, condition, processed_tag, call_name)
    annotated = df.merge(annotation, on="Metadata_broad_sample", how="left")
    return bio_enrichment.moa_enrichment(
        annotated, score_col=score_col, moa_col=moa_col, n_perm=n_perm, seed=seed
    )
