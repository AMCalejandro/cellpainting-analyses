"""Activity / distinctiveness / consistency mAP calls, computed with copairs.

Definitions follow the target reproduction figure's captions:

- Activity: same compound in different batches vs. plate-matched FFA DMSO
  controls. One nMAP value per compound (Metadata_broad_sample).
- Distinctiveness: same positive pairs as activity, compared against other
  FFA compounds instead of DMSO. One nMAP value per compound.
- Consistency: same target vs. different targets. One nMAP value per target
  group (Metadata_target).

Significance ("calls") comes straight from copairs' own permutation-test
p-values (`below_corrected_p`), per the figure footnote: activity and
distinctiveness use corrected p < 0.10, consistency uses corrected p < 0.05.
copairs doesn't expose a normalized AP, so `_add_normalized_ap` derives one
(0 = random retrieval, 1 = perfect, negative = worse than random) from the
same per-config null distributions copairs uses internally for its p-values.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from copairs import compute
from copairs.map import average_precision, mean_average_precision

NULL_SIZE = 10000
SEED = 0
ACTIVITY_THRESHOLD = 0.10
DISTINCTIVENESS_THRESHOLD = 0.10
CONSISTENCY_THRESHOLD = 0.05


def _add_normalized_ap(
    map_df: pd.DataFrame,
    ap_scores: pd.DataFrame,
    sameby: list,
    null_size: int,
    seed: int,
    cache_dir: Optional[Union[str, Path]],
) -> pd.DataFrame:
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    ap_scores = ap_scores.reset_index(drop=True).copy()
    null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
    null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = compute.get_null_dists(null_confs, null_size, seed=seed, cache_dir=cache_dir)
    ap_scores["null_ix"] = rev_ix

    def group_null_mean(ix):
        return null_dists[ix.to_numpy()].mean(axis=0).mean()

    null_means = ap_scores.groupby(sameby, observed=True)["null_ix"].apply(group_null_mean)
    null_means.name = "null_mean"

    map_df = map_df.join(null_means)
    map_df["normalized_average_precision"] = (
        map_df["mean_average_precision"] - map_df["null_mean"]
    ) / (1 - map_df["null_mean"])
    return map_df


def compute_activity(
    meta: pd.DataFrame,
    feats: np.ndarray,
    null_size: int = NULL_SIZE,
    seed: int = SEED,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    ap_scores = average_precision(
        meta,
        feats,
        pos_sameby=["Metadata_broad_sample"],
        pos_diffby=["Metadata_batch"],
        neg_sameby=["Metadata_Plate"],
        neg_diffby=["Metadata_broad_sample", "Metadata_pert_type"],
    )
    ap_scores = ap_scores[ap_scores["Metadata_pert_type"] == "trt"]
    map_df = mean_average_precision(
        ap_scores,
        sameby=["Metadata_broad_sample"],
        null_size=null_size,
        threshold=ACTIVITY_THRESHOLD,
        seed=seed,
        cache_dir=cache_dir,
    )
    return _add_normalized_ap(
        map_df, ap_scores, ["Metadata_broad_sample"], null_size, seed, cache_dir
    )


def compute_distinctiveness(
    meta: pd.DataFrame,
    feats: np.ndarray,
    null_size: int = NULL_SIZE,
    seed: int = SEED,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    trt = (meta["Metadata_pert_type"] == "trt").to_numpy()
    trt_meta, trt_feats = meta.loc[trt], feats[trt]
    ap_scores = average_precision(
        trt_meta,
        trt_feats,
        pos_sameby=["Metadata_broad_sample"],
        pos_diffby=["Metadata_batch"],
        neg_sameby=[],
        neg_diffby=["Metadata_broad_sample"],
    )
    map_df = mean_average_precision(
        ap_scores,
        sameby=["Metadata_broad_sample"],
        null_size=null_size,
        threshold=DISTINCTIVENESS_THRESHOLD,
        seed=seed,
        cache_dir=cache_dir,
    )
    return _add_normalized_ap(
        map_df, ap_scores, ["Metadata_broad_sample"], null_size, seed, cache_dir
    )


def compute_consistency(
    meta: pd.DataFrame,
    feats: np.ndarray,
    null_size: int = NULL_SIZE,
    seed: int = SEED,
    cache_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    has_target = meta["Metadata_target"].replace("", np.nan).notna()
    mask = ((meta["Metadata_pert_type"] == "trt") & has_target).to_numpy()
    c_meta, c_feats = meta.loc[mask], feats[mask]
    ap_scores = average_precision(
        c_meta,
        c_feats,
        pos_sameby=["Metadata_target"],
        pos_diffby=["Metadata_broad_sample"],
        neg_sameby=[],
        neg_diffby=["Metadata_target"],
    )
    map_df = mean_average_precision(
        ap_scores,
        sameby=["Metadata_target"],
        null_size=null_size,
        threshold=CONSISTENCY_THRESHOLD,
        seed=seed,
        cache_dir=cache_dir,
    )
    return _add_normalized_ap(map_df, ap_scores, ["Metadata_target"], null_size, seed, cache_dir)
