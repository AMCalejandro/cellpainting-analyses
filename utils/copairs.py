"""Activity / distinctiveness / consistency mAP calls, computed with copairs.

Definitions follow the target reproduction figure's captions:

- Activity: same compound in different batches vs. plate-matched FFA DMSO
  controls. One nMAP value per compound (Metadata_broad_sample).
- Distinctiveness: same positive pairs as activity, compared against other
  FFA compounds instead of DMSO. One nMAP value per compound.
- Consistency: same target vs. different targets. One nMAP value per target
  (Metadata_target by default; `compute_consistency`'s `groupby` can switch
  this to Metadata_moa). Both columns are "|"-delimited multi-label
  annotations, so "same target" is computed with copairs' multilabel matcher
  (shared-label intersection) rather than exact-string equality -- see
  `compute_consistency`'s docstring.

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
from copairs.map import multilabel as cp_multilabel

NULL_SIZE = 10000
SEED = 0
ACTIVITY_THRESHOLD = 0.10
DISTINCTIVENESS_THRESHOLD = 0.10
DEFAULT_CONSISTENCY_GROUPBY = "Metadata_target"
CONSISTENCY_THRESHOLD = 0.05


def _add_normalized_ap(
    map_df: pd.DataFrame,
    ap_scores: pd.DataFrame,
    sameby: list,
    null_size: int,
    seed: int,
    cache_dir: Optional[Union[str, Path]],
) -> pd.DataFrame:
    """Normalize per query first, then average within each group -- matching
    `normalized AP = (AP - null AP) / (1 - null AP)` applied query-by-query
    before the group mean, not the group mean normalized once using an
    averaged null. Those two orders only agree when every query in a group
    shares the same (n_pos_pairs, n_total_pairs) config; consistency groups
    (compounds with very different numbers of same-target partners) violate
    that badly, so normalizing the aggregate would silently bias the score."""
    ap_scores = ap_scores.query("~average_precision.isna() and n_pos_pairs > 0")
    ap_scores = ap_scores.reset_index(drop=True).copy()
    null_confs = ap_scores[["n_pos_pairs", "n_total_pairs"]].values
    null_confs, rev_ix = np.unique(null_confs, axis=0, return_inverse=True)
    null_dists = compute.get_null_dists(null_confs, null_size, seed=seed, cache_dir=cache_dir)
    null_ap = null_dists.mean(axis=1)[rev_ix]
    ap_scores["normalized_ap"] = (ap_scores["average_precision"] - null_ap) / (1 - null_ap)

    nmap = ap_scores.groupby(sameby, observed=True)["normalized_ap"].mean()
    nmap = nmap.rename("normalized_average_precision").reset_index()

    map_df = map_df.merge(nmap, on=sameby, how="left")
    return map_df


def _cached_ap_scores(ap_cache_path: Optional[Union[str, Path]], compute_fn) -> pd.DataFrame:
    """Compute (or load) the expensive per-profile `average_precision` output.

    `average_precision`'s pairwise-similarity pass dominates runtime; caching
    its raw output lets `_add_normalized_ap` (or other downstream tweaks) be
    iterated on without re-running that pass.
    """
    if ap_cache_path is not None and Path(ap_cache_path).exists():
        return pd.read_parquet(ap_cache_path)
    ap_scores = compute_fn()
    if ap_cache_path is not None:
        Path(ap_cache_path).parent.mkdir(parents=True, exist_ok=True)
        ap_scores.to_parquet(ap_cache_path)
    return ap_scores


def compute_activity(
    meta: pd.DataFrame,
    feats: np.ndarray,
    null_size: int = NULL_SIZE,
    seed: int = SEED,
    cache_dir: Optional[Union[str, Path]] = None,
    ap_cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    def _compute():
        ap_scores = average_precision(
            meta,
            feats,
            pos_sameby=["Metadata_broad_sample"],
            pos_diffby=["Metadata_batch"],
            neg_sameby=["Metadata_Plate"],
            neg_diffby=["Metadata_broad_sample", "Metadata_pert_type"],
        )
        return ap_scores[ap_scores["Metadata_pert_type"] == "trt"]

    ap_scores = _cached_ap_scores(ap_cache_path, _compute)
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
    ap_cache_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    def _compute():
        trt = (meta["Metadata_pert_type"] == "trt").to_numpy()
        trt_meta, trt_feats = meta.loc[trt], feats[trt]
        return average_precision(
            trt_meta,
            trt_feats,
            pos_sameby=["Metadata_broad_sample"],
            pos_diffby=["Metadata_batch"],
            neg_sameby=[],
            neg_diffby=["Metadata_broad_sample"],
        )

    ap_scores = _cached_ap_scores(ap_cache_path, _compute)
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
    ap_cache_path: Optional[Union[str, Path]] = None,
    groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> pd.DataFrame:
    """`groupby` is the grouping column for "same X vs different X" -- either
    "Metadata_target" (default) or "Metadata_moa". MoA groups are coarser
    (multiple targets can share a mechanism), giving more compounds per
    group and thus more power than the target grouping.

    Both columns are "|"-delimited MULTI-label annotations -- a compound can
    hit several targets (or share several mechanisms) -- so this uses
    copairs' multilabel matcher (`copairs.map.multilabel.average_precision`,
    with `groupby` split into a list first) instead of plain
    `average_precision`'s exact-string-equality grouping. Under string
    equality, a compound annotated "ADRB1" and one annotated
    "ADRB1|ADRB2|ADRB3" are treated as different groups even though they
    share a target -- fragmenting almost every real target-sharing
    relationship in this panel into near-singleton groups and leaving
    corrected p-values nowhere near the threshold. The multilabel matcher
    instead counts two compounds as "same target" whenever their label sets
    intersect."""

    def _compute():
        has_group = meta[groupby].replace("", np.nan).notna()
        mask = ((meta["Metadata_pert_type"] == "trt") & has_group).to_numpy()
        c_meta, c_feats = meta.loc[mask].copy(), feats[mask]
        c_meta[groupby] = c_meta[groupby].str.split("|")
        return cp_multilabel.average_precision(
            c_meta,
            c_feats,
            pos_sameby=[groupby],
            pos_diffby=["Metadata_broad_sample"],
            neg_sameby=[],
            neg_diffby=[groupby],
            multilabel_col=groupby,
        )

    ap_scores = _cached_ap_scores(ap_cache_path, _compute)
    map_df = mean_average_precision(
        ap_scores,
        sameby=[groupby],
        null_size=null_size,
        threshold=CONSISTENCY_THRESHOLD,
        seed=seed,
        cache_dir=cache_dir,
    )
    return _add_normalized_ap(map_df, ap_scores, [groupby], null_size, seed, cache_dir)


# --- hit-set agreement + allowlist enrichment ---------------------------
# Shared by `imaging.benchmark` and the proteomics Tier E figure pipeline --
# domain-agnostic over hit sets keyed by representation, processed/covariate
# tag, or condition, so it lives here rather than in either domain package.


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def hit_overlap(hit_sets: dict) -> pd.DataFrame:
    """Pairwise Jaccard index of hit sets, one set per key (e.g. one
    representation, one processed/covariate tag, or one condition)."""
    names = list(hit_sets)
    rows = [
        {"a": a, "b": b, "jaccard": jaccard(hit_sets[a], hit_sets[b])}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return pd.DataFrame(rows, columns=["a", "b", "jaccard"])


def mean_pairwise_jaccard(overlap_df: pd.DataFrame, name: str) -> float:
    """Mean Jaccard of `name` against every other key in a `hit_overlap`
    pairwise table -- a single per-key agreement summary pulled from the
    same table `hit_overlap` already returns, instead of a separate
    all-vs-one computation. NaN if `name` doesn't appear (e.g. a
    single-key run)."""
    rows = overlap_df[(overlap_df["a"] == name) | (overlap_df["b"] == name)]
    return float(rows["jaccard"].mean()) if len(rows) else float("nan")


def consistency_called_terms(consistency_df: pd.DataFrame) -> pd.DataFrame:
    """The consistency call's own called groups (below_corrected_p), sorted
    by normalized AP descending -- unlike activity/distinctiveness, a
    consistency call is already a per-term (Metadata_target or
    Metadata_moa, whichever `compute_consistency` was grouped by) test, so
    "which terms are enriched" is just this call's hit list rather than a
    separate permutation enrichment over a compound pool."""
    groupby_col = "Metadata_target" if "Metadata_target" in consistency_df.columns else "Metadata_moa"
    return (
        consistency_df.loc[consistency_df["below_corrected_p"]]
        .sort_values("normalized_average_precision", ascending=False)
        .reset_index(drop=True)
        .rename(columns={groupby_col: "term"})
    )


def combine_allowlist_ranks(
    activity_df: pd.DataFrame,
    distinctiveness_df: pd.DataFrame,
    score_col: str = "mean_average_precision",
) -> pd.DataFrame:
    """Combine an activity call and a distinctiveness call into one ranked
    table over their shared compound pool (the activity ∩ distinctiveness
    "allowlist"), feeding a downstream preranked-GSEA enrichment
    (`utils.bio_enrichment.moa_enrichment`) of the allowlist as a whole.

    Ranked by the per-compound min of the two calls' PERCENTILE RANK on
    `score_col` -- the continuous relaxation of "both criteria hold" (a
    compound only ranks as high as its weaker call). Percentile rank, not
    the raw `score_col` value: activity's and distinctiveness's
    `mean_average_precision` typically live on very different natural
    scales (distinctiveness's negative set is the whole treated population
    vs. activity's same-plate wells, a much harder discrimination task, so
    its scores sit far closer to 0). A raw `min()` across that scale
    mismatch just reduces to "whichever call is smaller", almost always
    distinctiveness, silently discarding the activity signal; ranking each
    call within its own pool first makes the two comparable before
    combining.

    Both percentile ranks are drawn from the same size-n grid (both tables
    score every compound in the pool), so a plain `min()` of the two often
    collides on ties. GSEA needs a strictly ordered list (ties make its
    null/leading-edge computation degrade), so the *other* call's
    percentile is folded in at a scale far below the 1/n grid spacing as a
    tiebreaker -- it can't reorder any two compounds whose min()s actually
    differ, only disambiguate exact ties."""
    df = activity_df.merge(
        distinctiveness_df[["Metadata_broad_sample", score_col]],
        on="Metadata_broad_sample",
        suffixes=("_activity", "_distinctiveness"),
    )
    pct_activity = df[f"{score_col}_activity"].rank(pct=True)
    pct_distinctiveness = df[f"{score_col}_distinctiveness"].rank(pct=True)
    pct = pd.concat([pct_activity, pct_distinctiveness], axis=1)
    tiebreak = (pct_activity + pct_distinctiveness) / (2 * (len(df) + 1) ** 2)
    df[score_col] = pct.min(axis=1) + tiebreak
    return df
