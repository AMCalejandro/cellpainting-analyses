"""Representation-space benchmark (experiments/benchmark_feature_representation.md):
matched-dimension comparison of CellProfiler / CPCNN / UniDino on the
existing residualize -> copairs -> reversion pipeline, unmodified except for
the input feature matrix (section 8 of the doc).

Dimensionality control (doc section 3): one PCA basis per representation,
fit on DMSO-control wells only, pooled across every hWAT condition, on
already-residualized features. The residualization methods this composes
with are exactly `utils.features.CROSS_CONDITION_METHODS` (nested_* and
control_centered): each fits its covariates SEPARATELY per
`Metadata_condition`, so residualizing one condition's wells in isolation
(`matched_features`) gives bit-identical output to residualizing them as
part of the pooled load `fit_pca_basis` uses -- the PCA basis and any
per-condition/per-pair feature matrix built from it are safe to combine.
Pooled Ridge covariate sets (count_plate, count_batch_plate) do NOT have
this property (see utils.features docstring) and are not supported here.

Tiers implemented: A (technical quality -- A1 via `imaging.batch_report`
directly, A2 via `activity_and_distinctiveness`), B1
(`condition_separability`), C1-C3 (`hit_overlap`,
`replicate_split_stability`, `effect_size_separation`), E (copairs-level
cross-representation/cross-condition agreement and biological
plausibility -- see below). B2 and Tier D are stubbed
(`known_effect_validation`, `proteomic_concordance`) pending a biology-team
reference-compound list and a proteomics preprocessing pipeline,
respectively -- see their docstrings.

Tier E runs entirely on run_pipeline.py's existing copairs calls, one step
upstream of reversion, so it's available for every scored compound/
condition regardless of whether reversion ever ran on it:
  - `hit_overlap` (reused, not new) on activity-only, distinctiveness-only,
    allowlist and consistency-called-term sets, within one condition across
    representations (E1) or, via `run_cellrep_benchmark.py`'s cross-
    condition loop, for one representation across stress conditions (E2) --
    `mean_pairwise_jaccard` pulls a single per-key agreement summary out of
    either table.
  - `copairs_call_enrichment`: MoA/target preranked-GSEA enrichment
    (`imaging.bio_enrichment.moa_enrichment`) among a condition's
    activity/distinctiveness/allowlist compounds (E3) -- e.g. are IL6
    hits enriched for anti-inflammatory MoAs? `consistency_called_terms`
    is the equivalent readout for the consistency call, which is already a
    per-term test rather than a compound pool to permute.

A2 (and the reversion compound_allowlist) reuse run_pipeline.py's ALREADY
COMPUTED activity/distinctiveness calls (see cli.sh) instead of rerunning
copairs: activity and distinctiveness are single-condition calls by
construction (same-compound-different-batch vs. that condition's own DMSO,
or vs. other compounds in that condition), so a fresh run on the
matched-dimension features would just reproduce the same expensive
permutation-null computation `run_pipeline.py` already did at native
dimensionality. `replicate_split_stability` (C2) is the one exception: it
needs copairs on a random replicate-well subset that was never precomputed,
so it calls `utils.copairs.compute_activity` itself.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import copairs as cp
from utils import features as feat

from . import batch_report as br
from . import bio_enrichment
from . import load
from . import paths
from . import reversion as rev

SEED = 0
DEFAULT_COVARIATE_SET = "nested_count_plate"
# The single-condition equivalent of nested_count_plate: run_pipeline.py
# never computed a nested_* variant (nested_count_plate IS count_plate when
# Metadata_condition is constant, i.e. every single-condition copairs run --
# see utils.features's WITHIN_CONDITION_METHODS comment), so this is the
# covariate-set family whose existing activity/distinctiveness parquets
# `activity_and_distinctiveness` loads.
EXISTING_COPAIRS_COVARIATE_SET = "count_plate"
BASELINE_CONDITION = "Baseline"
DEFAULT_STRESS_CONDITIONS = ["FFA", "IL6", "Low Gluc"]
K_MIN = 50
K_MAX = 150
N_SPLITS = 5


def choose_k(explained_variance_ratio: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX) -> int:
    """Elbow of the cumulative variance-explained curve: the component where
    the curve is farthest above the straight line joining its first and last
    point (standard "max distance from the chord" elbow rule for a concave
    increasing curve), clipped to [k_min, k_max] per the doc's guidance."""
    cum = np.cumsum(explained_variance_ratio)
    n = len(cum)
    x = (np.arange(1, n + 1) - 1) / (n - 1)
    y = (cum - cum[0]) / (cum[-1] - cum[0])
    k = int(np.argmax(y - x)) + 1
    return int(np.clip(k, k_min, k_max))


@dataclass
class RepresentationBasis:
    """A single, shared coordinate system for one feature space: the z-score
    mean/std fit once on every pooled condition's RAW features, plus a PCA
    fit on the pooled, z-scored + residualized DMSO-control rows. Reused by
    every subsequent single- or joint-condition load for that representation
    so every load lands in the same coordinate system -- see module
    docstring and `matched_features`."""

    mean: np.ndarray
    std: np.ndarray
    pca: PCA
    k: int


def fit_pca_basis(
    feature_space: str,
    covariate_set: str = DEFAULT_COVARIATE_SET,
    k: Optional[int] = None,
    seed: int = SEED,
    cell_line: str = load.DEFAULT_CELL_LINE,
    conditions: list = br.ALL_CONDITIONS,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
) -> RepresentationBasis:
    """One shared z-score + PCA basis for `feature_space`: load every
    condition, z-score ONCE on the pooled matrix (so every condition's mean
    is expressed relative to the same origin -- z-scoring each condition
    separately would recenter every condition to zero and destroy the very
    Baseline-vs-stress offset the reversion axis is built from, exactly the
    failure mode `imaging.reversion.load_joint_residualized`'s docstring
    warns about), residualize with `covariate_set` (must be in
    `utils.features.CROSS_CONDITION_METHODS`), and fit PCA on the
    DMSO-control rows only. If `k` is None, it is chosen via `choose_k` on
    this representation's own control wells; pass a fixed `k` (e.g.
    CellProfiler's elbow) to match dimensionality across representations,
    per doc section 3."""
    if covariate_set not in feat.CROSS_CONDITION_METHODS:
        raise ValueError(
            f"{covariate_set!r} is not composable across independently-loaded "
            "conditions; use a nested_* method or control_centered"
        )
    meta, feats_raw = br.load_all_conditions(feature_space, conditions, cell_line)
    mean = feats_raw.mean(axis=0, keepdims=True)
    std = feats_raw.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats_raw - mean) / std
    feats = feat.RESIDUALIZE_METHODS[covariate_set](feats, meta)
    ctrl_feats = feats[(meta["Metadata_pert_type"] == "negcon").to_numpy()]

    if k is None:
        max_k = min(k_max, ctrl_feats.shape[0] - 1, ctrl_feats.shape[1])
        full = PCA(n_components=max_k, random_state=seed).fit(ctrl_feats)
        k = choose_k(full.explained_variance_ratio_, k_min, k_max)
    pca = PCA(n_components=k, random_state=seed).fit(ctrl_feats)
    return RepresentationBasis(mean=mean, std=std, pca=pca, k=k)


def matched_features(
    feature_space: str,
    condition: str,
    covariate_set: str,
    basis: RepresentationBasis,
    cell_line: str = load.DEFAULT_CELL_LINE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load one condition, z-score with `basis`'s pooled mean/std (NOT this
    condition's own -- see `fit_pca_basis`), residualize it in isolation,
    and project onto `basis.pca`. See module docstring for why residualizing
    in isolation matches the pooled fit `basis` came from."""
    meta, feats_raw = load.load_feature_space(feature_space, condition, cell_line=cell_line)
    meta = meta.copy()
    meta["Metadata_condition"] = condition
    feats = (feats_raw - basis.mean) / basis.std
    feats = feat.RESIDUALIZE_METHODS[covariate_set](feats, meta)
    return meta, basis.pca.transform(feats)


def matched_joint_features(
    feature_space: str,
    stress_condition: str,
    covariate_set: str,
    basis: RepresentationBasis,
    baseline_condition: str = BASELINE_CONDITION,
    cell_line: str = load.DEFAULT_CELL_LINE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """`matched_features` for Baseline and `stress_condition`, concatenated
    (Baseline rows first) so `imaging.reversion.compute_reversion` can score
    reversion on the matched-dimension representation."""
    meta_b, feats_b = matched_features(feature_space, baseline_condition, covariate_set, basis, cell_line)
    meta_s, feats_s = matched_features(feature_space, stress_condition, covariate_set, basis, cell_line)
    return pd.concat([meta_b, meta_s], ignore_index=True), np.vstack([feats_b, feats_s])


def _condition_results_dir(condition: str, results_dir: Path = paths.RESULTS_DIR) -> Path:
    """cli.sh's per-condition copairs output layout: results/imaging/copairs/
    <condition, spaces replaced with underscores>/ (e.g. "Low Gluc" ->
    .../Low_Gluc/); the parquet filenames themselves keep the space."""
    return results_dir / condition.replace(" ", "_")


def _copairs_parquet_path(
    feature_space: str, condition: str, call_name: str, covariate_set: str,
    results_dir: Path = paths.RESULTS_DIR,
) -> Path:
    """run_pipeline.py's file_stub convention: no condition tag for
    `load.DEFAULT_CONDITION` ("FFA"), `_{condition}` otherwise."""
    condition_tag = "" if condition == load.DEFAULT_CONDITION else f"_{condition}"
    stub = f"{feature_space}{condition_tag}_{covariate_set}_{call_name}"
    return _condition_results_dir(condition, results_dir) / "parquet" / f"{stub}.parquet"


def load_existing_copairs_call(
    feature_space: str,
    condition: str,
    call_name: str,
    covariate_set: str = EXISTING_COPAIRS_COVARIATE_SET,
    results_dir: Path = paths.RESULTS_DIR,
) -> pd.DataFrame:
    """Load an already-computed run_pipeline.py copairs call (`call_name` one
    of "activity", "distinctiveness", "consistency") instead of rerunning
    copairs -- see module docstring for why that's safe for a single-
    condition call."""
    path = _copairs_parquet_path(feature_space, condition, call_name, covariate_set, results_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no existing {call_name!r} result at {path} -- run "
            f'.venv/bin/python cli.py imaging copairs --condition "{condition}" '
            f"--out-dir results/imaging/copairs/{condition.replace(' ', '_')} first"
        )
    return pd.read_parquet(path)


def activity_and_distinctiveness(
    feature_space: str,
    condition: str,
    covariate_set: str = EXISTING_COPAIRS_COVARIATE_SET,
    results_dir: Path = paths.RESULTS_DIR,
) -> dict:
    """A2 (activity_rate, from the existing activity call) plus the
    reversion compound_allowlist: activity ∩ distinctiveness, matching
    run_reversion.py's --activity-parquet + --distinctiveness-parquet
    convention. Both calls are loaded from disk, not recomputed -- see
    module docstring."""
    activity_df = load_existing_copairs_call(feature_space, condition, "activity", covariate_set, results_dir)
    distinct_df = load_existing_copairs_call(feature_space, condition, "distinctiveness", covariate_set, results_dir)
    active = set(activity_df.loc[activity_df["below_corrected_p"], "Metadata_broad_sample"])
    distinct = set(distinct_df.loc[distinct_df["below_corrected_p"], "Metadata_broad_sample"])
    return {
        "activity_rate": float(activity_df["below_corrected_p"].mean()),
        "n_compounds": int(len(activity_df)),
        "n_active": int(len(active)),
        "n_distinct": int(len(distinct)),
        "active_compounds": active,
        "distinct_compounds": distinct,
        "allowlist": active & distinct,
        "activity_table": activity_df,
        "distinctiveness_table": distinct_df,
    }


def annotate_compound_table(
    df: pd.DataFrame,
    condition: str,
    cell_line: str = load.DEFAULT_CELL_LINE,
) -> pd.DataFrame:
    """Merge Metadata_target/Metadata_moa onto a Metadata_broad_sample-keyed
    copairs result table -- the activity/distinctiveness parquets carry no
    annotation columns of their own (see `load_existing_copairs_call`'s
    caller, `utils.copairs`'s compute_activity/distinctiveness).
    Same drop_duplicates-on-compound convention as
    `imaging.reversion.compute_reversion`'s own annot merge."""
    meta = load.load_metadata(condition, cell_line)
    annot_cols = [c for c in ("Metadata_broad_sample", "Metadata_target", "Metadata_moa") if c in meta.columns]
    annot = meta[annot_cols].drop_duplicates("Metadata_broad_sample")
    return df.merge(annot, on="Metadata_broad_sample", how="left")


def copairs_call_enrichment(
    feature_space: str,
    condition: str,
    call_name: str,
    covariate_set: str = EXISTING_COPAIRS_COVARIATE_SET,
    moa_col: str = "Metadata_moa",
    score_col: str = "mean_average_precision",
    n_perm: int = 20000,
    seed: int = SEED,
    results_dir: Path = paths.RESULTS_DIR,
) -> pd.DataFrame:
    """MoA/target preranked-GSEA enrichment (`imaging.bio_enrichment.moa_enrichment`)
    among a condition's copairs-called compounds, run directly on
    run_pipeline.py's activity/distinctiveness calls instead of waiting for
    a reversion nominee list -- a biological-plausibility read on Tier A2
    itself ("are the compounds this representation calls active/distinct
    enriched for a condition-relevant mechanism, e.g. anti-inflammatory
    MoAs among IL6 hits?"), available even for compounds/conditions
    `imaging.reversion` never scores.

    `call_name` is "activity" or "distinctiveness" (ranked by that call's
    own `score_col`), or "allowlist" for the activity ∩ distinctiveness set,
    ranked by the per-compound min of the two calls' PERCENTILE RANK on
    `score_col` -- the continuous relaxation of "both criteria hold" (a
    compound only ranks as high as its weaker call). Percentile rank, not
    the raw `score_col` value: activity's and distinctiveness's
    `mean_average_precision` live on very different natural scales here
    (distinctiveness's negative set is the whole treated population vs.
    activity's same-plate wells, a much harder discrimination task, so its
    scores sit far closer to 0 -- e.g. median ~0.003 vs. activity's ~0.29 on
    this panel). A raw `min()` across that scale mismatch just reduces to
    "whichever call is smaller", almost always distinctiveness, silently
    discarding the activity signal; ranking each call within its own pool
    first makes the two comparable before combining.

    Both percentile ranks are drawn from the same size-n grid (both tables
    score every compound in the pool), so a plain `min()` of the two often
    collides -- e.g. compound A's min(0.30, 0.05) and compound B's
    min(0.05, 0.90) both land on 0.05 even though only B is really capped
    by its weaker call at that value. GSEA needs a strictly ordered list
    (ties make its null/leading-edge computation degrade), so the *other*
    call's percentile is folded in at a scale far below the 1/n grid
    spacing as a tiebreaker -- it can't reorder any two compounds whose
    min()s actually differ, only disambiguate exact ties."""
    if call_name == "allowlist":
        act = activity_and_distinctiveness(feature_space, condition, covariate_set, results_dir)
        df = act["activity_table"].merge(
            act["distinctiveness_table"][["Metadata_broad_sample", score_col]],
            on="Metadata_broad_sample",
            suffixes=("_activity", "_distinctiveness"),
        )
        pct_activity = df[f"{score_col}_activity"].rank(pct=True)
        pct_distinctiveness = df[f"{score_col}_distinctiveness"].rank(pct=True)
        pct = pd.concat([pct_activity, pct_distinctiveness], axis=1)
        tiebreak = (pct_activity + pct_distinctiveness) / (2 * (len(df) + 1) ** 2)
        df[score_col] = pct.min(axis=1) + tiebreak
    else:
        df = load_existing_copairs_call(feature_space, condition, call_name, covariate_set, results_dir)
    annotated = annotate_compound_table(df, condition)
    return bio_enrichment.moa_enrichment(
        annotated, score_col=score_col, moa_col=moa_col, n_perm=n_perm, seed=seed
    )


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


def condition_separability(
    meta: pd.DataFrame,
    feats: np.ndarray,
    condition_col: str = "Metadata_condition",
    n_splits: int = 5,
    seed: int = SEED,
) -> dict:
    """B1: cross-validated linear-probe AUROC separating Baseline from a
    stress condition on the matched-dimension features. A representation
    that fails this can't produce a meaningful reversion signal regardless
    of downstream (Tier C) results."""
    labels = meta[condition_col].to_numpy()
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError(f"expected exactly 2 conditions, got {list(classes)}")
    y = (labels == classes[1]).astype(int)
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = cross_val_score(clf, feats, y, cv=cv, scoring="roc_auc")
    return {
        "auroc_mean": float(aurocs.mean()),
        "auroc_std": float(aurocs.std()),
        "positive_class": str(classes[1]),
    }


def known_effect_validation(*_args, **_kwargs):
    """TODO(B2): known-effect validation against a literature reference-
    compound set (e.g. anti-inflammatory agents expected to revert IL6,
    insulin-sensitizing agents expected to revert FFA lipotoxicity) is not
    implemented -- it needs a curated table of Metadata_broad_sample ->
    (condition, expected_direction) from the biology team, which doesn't
    exist in this repo yet. Once available (e.g. a CSV with those columns),
    wire it in here: for each reference compound, check the sign of
    `rho_int` from `imaging.reversion.compute_reversion`'s per_compound
    table against `expected_direction`."""
    raise NotImplementedError("B2 needs a biology-team reference-compound list; see docstring")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def hit_overlap(hit_sets: dict) -> pd.DataFrame:
    """C1: pairwise Jaccard index of hit sets (one set per representation)."""
    names = list(hit_sets)
    rows = [
        {"a": a, "b": b, "jaccard": jaccard(hit_sets[a], hit_sets[b])}
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return pd.DataFrame(rows, columns=["a", "b", "jaccard"])


def mean_pairwise_jaccard(overlap_df: pd.DataFrame, name: str) -> float:
    """Mean Jaccard of `name` (a representation, from a within-condition
    `hit_overlap`, or a condition, from a cross-condition one) against every
    other key in a `hit_overlap` pairwise table -- a single per-key
    agreement summary pulled from the same table `hit_overlap` already
    returns, instead of a separate all-vs-one computation. NaN if `name`
    doesn't appear (e.g. a single-representation/single-condition run)."""
    rows = overlap_df[(overlap_df["a"] == name) | (overlap_df["b"] == name)]
    return float(rows["jaccard"].mean()) if len(rows) else float("nan")


def split_replicate_wells_loo(
    meta: pd.DataFrame, condition_col: str, target_condition: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out perturbation, not a disjoint 50/50 split: for each of
    `target_condition`'s treated compounds, half1/half2 each independently
    drop ONE randomly chosen replicate well (drawn separately per half, so
    they usually differ but may coincide), retaining n_reps-1 of n_reps
    wells per side instead of ~half. At this panel's typical n_reps=5, a
    disjoint split leaves as few as 2 wells on one side -- diluting signal
    and inflating bootstrap CI width/threshold instability well beyond what
    the full n_reps=5 pipeline sees, which understates stability rather than
    measuring it (see docs/pca_resisidualization_decisions.md-style
    decision notes on this file). LOO keeps 4 of 5, much closer to the full
    run's power, at the cost of the two draws no longer being independent
    (they share n_reps-2 wells) -- an accepted trade given N=5's ceiling. A
    compound with only one replicate can't be perturbed at all and is kept
    whole in both halves, rather than dropped from one entirely (the old
    split's behavior for n_reps=1). Every other row (the other condition's
    rows, and this condition's own DMSO controls) is included in BOTH
    halves untouched."""
    rng = np.random.default_rng(seed)
    is_target_trt = (
        (meta[condition_col] == target_condition).to_numpy()
        & (meta["Metadata_pert_type"] == "trt").to_numpy()
    )
    half1, half2 = ~is_target_trt, ~is_target_trt
    half1, half2 = half1.copy(), half2.copy()
    for _, idx in meta.loc[is_target_trt].groupby("Metadata_broad_sample", observed=True).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            half1[idx] = True
            half2[idx] = True
            continue
        drop1, drop2 = rng.choice(idx), rng.choice(idx)
        half1[idx[idx != drop1]] = True
        half2[idx[idx != drop2]] = True
    return half1, half2


def replicate_split_stability(
    meta: pd.DataFrame,
    feats: np.ndarray,
    stress_condition: str,
    baseline_condition: str = BASELINE_CONDITION,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
    n_boot: int = 1000,
    activity_null_size: int = 1000,
    min_spearman_pairs: int = 3,
) -> dict:
    """C2: over `n_splits` leave-one-out replicate-well perturbations
    (`split_replicate_wells_loo` -- see its docstring for why this isn't a
    disjoint 50/50 split), a full activity + reversion rerun per half per
    split (Tier's heaviest step). Uses reduced `n_boot`/`activity_null_size`
    relative to the main run: a stability estimate only needs consistent
    hit CALLS/scores, not full-power statistics.

    Reports three things per split, averaged across splits:
      - activity_jaccard / reversion_jaccard: binary hit-list agreement.
        Read these as a strict lower-bound / high-confidence-floor stress
        test, not the pipeline's overall stability -- a bare threshold
        crossing that flips between two nearly-identical LOO folds
        (a "near miss") is scored as complete disagreement here even though
        the underlying signal barely moved.
      - reversion_score_spearman: Spearman rank correlation of RI_spec
        between the two folds, over compounds BOTH folds scored with a
        non-null RI_spec. Continuous, so it isn't sensitive to the
        threshold-cliff effect above -- it asks whether compounds' relative
        reversion-signal ORDERING survives the perturbation, independent of
        where nominated_robust_ci's gates happen to draw the line. NaN for
        a split with fewer than `min_spearman_pairs` overlapping compounds
        (too few to estimate a correlation).

    Each half's reversion call is restricted to THAT half's own freshly
    copairs-active compounds (matching the main run's
    activity-restricted reversion -- restricting the BH family to
    already-active compounds is what makes the FDR gate clearable at all;
    see run_cellrep_benchmark.py), not a shared allowlist across halves."""
    activity_jaccards, reversion_jaccards, score_spearmans = [], [], []
    for split in range(n_splits):
        split_seed = seed + split
        half1, half2 = split_replicate_wells_loo(meta, "Metadata_condition", stress_condition, split_seed)
        halves = []
        for half_mask in (half1, half2):
            sub_meta = meta.loc[half_mask].reset_index(drop=True)
            sub_feats = feats[half_mask]
            stress_mask = (sub_meta["Metadata_condition"] == stress_condition).to_numpy()
            act = cp.compute_activity(
                sub_meta.loc[stress_mask].reset_index(drop=True),
                sub_feats[stress_mask],
                null_size=activity_null_size,
                seed=split_seed,
            )
            active = set(act.loc[act["below_corrected_p"], "Metadata_broad_sample"])
            result = rev.compute_reversion(
                sub_meta, sub_feats, baseline_condition, stress_condition,
                n_boot=n_boot, seed=split_seed, compound_allowlist=active,
            )
            per_compound = result["per_compound"]
            reverted = set(
                per_compound.loc[per_compound["nominated_robust_ci"], "Metadata_broad_sample"]
            )
            halves.append((active, reverted, per_compound))
        activity_jaccards.append(jaccard(halves[0][0], halves[1][0]))
        reversion_jaccards.append(jaccard(halves[0][1], halves[1][1]))

        scores = halves[0][2][["Metadata_broad_sample", "RI_spec"]].merge(
            halves[1][2][["Metadata_broad_sample", "RI_spec"]],
            on="Metadata_broad_sample", suffixes=("_1", "_2"),
        ).dropna()
        if len(scores) >= min_spearman_pairs:
            score_spearmans.append(float(spearmanr(scores["RI_spec_1"], scores["RI_spec_2"]).statistic))
        else:
            score_spearmans.append(float("nan"))
    with warnings.catch_warnings():
        # nan is the expected jaccard() value when both halves have zero
        # hits (e.g. tiny --n-boot smoke runs); not an error condition.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        activity_jaccard_mean = float(np.nanmean(activity_jaccards))
        reversion_jaccard_mean = float(np.nanmean(reversion_jaccards))
        reversion_score_spearman_mean = float(np.nanmean(score_spearmans))
    return {
        "n_splits": n_splits,
        "activity_jaccard_mean": activity_jaccard_mean,
        "reversion_jaccard_mean": reversion_jaccard_mean,
        "reversion_score_spearman_mean": reversion_score_spearman_mean,
        "activity_jaccards": activity_jaccards,
        "reversion_jaccards": reversion_jaccards,
        "reversion_score_spearmans": score_spearmans,
    }


def effect_size_separation(
    per_compound: pd.DataFrame,
    score_col: str = "RI_spec",
    hit_col: str = "nominated_robust_ci",
) -> dict:
    """Cohen's d and AUROC of `score_col` as a classifier for `hit_col`.
    Generic utility -- safe when `hit_col` was NOT derived from `score_col`
    (e.g. Tier D3's `discordance_detail`, or `cross_representation_effect_size`
    below, comparing one modality's/representation's label against a
    DIFFERENT one's independently-computed score).

    NOT safe as a same-representation Tier C3 metric with the defaults:
    `nominated_robust_ci`'s gate_specificity and gate_ci_positive gates are
    both thresholds on `rho_int` -- the exact numerator of `RI_spec`
    (`imaging.reversion`'s docstring) -- so "hits have high RI_spec" is
    close to true by construction (empirically, min RI_spec among IL6/
    CellProfiler hits is 1.07, right at gate_specificity's ~1.0 cutoff),
    not evidence of a clean, generalizable threshold. Use
    `cross_representation_effect_size` for that comparison instead."""
    scored = per_compound.dropna(subset=[score_col])
    is_hit = scored[hit_col].fillna(False).to_numpy()
    hits, non_hits = scored.loc[is_hit, score_col].to_numpy(), scored.loc[~is_hit, score_col].to_numpy()
    if len(hits) == 0 or len(non_hits) == 0:
        return {"cohens_d": float("nan"), "auroc": float("nan"), "n_hits": len(hits), "n_non_hits": len(non_hits)}
    pooled_std = np.sqrt((hits.var(ddof=1) + non_hits.var(ddof=1)) / 2)
    cohens_d = float((hits.mean() - non_hits.mean()) / pooled_std) if pooled_std > 0 else float("nan")
    y = np.concatenate([np.ones(len(hits)), np.zeros(len(non_hits))])
    auroc = float(roc_auc_score(y, np.concatenate([hits, non_hits])))
    return {"cohens_d": cohens_d, "auroc": auroc, "n_hits": int(len(hits)), "n_non_hits": int(len(non_hits))}


def cross_representation_effect_size(
    hit_per_compound: pd.DataFrame,
    score_per_compound: pd.DataFrame,
    hit_col: str = "nominated_robust_ci",
    score_col: str = "RI_spec",
) -> dict:
    """C3, non-circular: are compounds THIS representation calls hits
    (`hit_per_compound`'s `hit_col`) ALSO ranked highly by a DIFFERENT
    representation's independently-computed reversion score
    (`score_per_compound`'s `score_col`)? See `effect_size_separation`'s
    docstring for why running that function on a single representation's
    own table is circular and not a valid Tier C3 metric.

    Merges the two representations' per_compound tables on
    Metadata_broad_sample (inner -- only compounds BOTH representations
    scored, since each is independently restricted to its own
    active∩distinctive allowlist) and evaluates one's label against the
    other's score, via `effect_size_separation`."""
    merged = hit_per_compound[["Metadata_broad_sample", hit_col]].merge(
        score_per_compound[["Metadata_broad_sample", score_col]],
        on="Metadata_broad_sample", how="inner",
    )
    return effect_size_separation(merged, score_col=score_col, hit_col=hit_col)


def _pair_stats(a: set, b: set) -> dict:
    """Jaccard plus both-directions recall for one hit-set pair -- Jaccard
    alone can hide a big size mismatch (e.g. a=1/b=50 with 1 shared member
    scores the same 0.02 as a=25/b=25 with 1 shared member)."""
    return {
        "jaccard": jaccard(a, b),
        "n_a": len(a),
        "n_b": len(b),
        "n_both": len(a & b),
        "recall_a_in_b": (len(a & b) / len(a)) if a else float("nan"),
        "recall_b_in_a": (len(a & b) / len(b)) if b else float("nan"),
    }


def discordance_detail(
    only_in_one: set, other_per_compound: pd.DataFrame, score_col: str = "RI_spec"
) -> pd.DataFrame:
    """D3: for compounds reversion-nominated in one modality but not the
    other, pull that compound's own score/gate status FROM the other
    modality's per_compound table (imaging.reversion.compute_reversion's or
    proteomics.reversion.compute_reversion's output) -- lets a human tell a
    genuinely complementary call (clearly negative or not estimable
    elsewhere) apart from a borderline one, rather than scoring the
    disagreement against either representation (see the doc's Tier D3)."""
    cols = [c for c in ("Metadata_broad_sample", score_col, "nominated_robust_ci") if c in other_per_compound.columns]
    return (
        other_per_compound.loc[other_per_compound["Metadata_broad_sample"].isin(only_in_one), cols]
        .rename(columns={score_col: f"other_modality_{score_col}"})
        .reset_index(drop=True)
    )


def proteomic_concordance(
    imaging_active: set,
    imaging_distinct: set,
    imaging_reverted: set,
    imaging_per_compound: pd.DataFrame,
    proteomic_active: set,
    proteomic_distinct: set,
    proteomic_reverted: set,
    proteomic_per_compound: pd.DataFrame,
    score_col: str = "RI_spec",
) -> dict:
    """D2 (concordance rate) + D3 (discordance detail) between one imaging
    representation's hit lists and the independent proteomic call from
    `proteomics.concordance.condition_hits`. `imaging_active`/
    `imaging_distinct`/`imaging_reverted` are this representation's
    active/distinctiveness/reversion-nominated compound sets (see
    `activity_and_distinctiveness` and a `{space}_{condition}_reversion.parquet`
    from `run_cellrep_benchmark.py`); the `proteomic_*` arguments are
    `condition_hits`'s matching fields. Per the doc's aggregation guidance
    (section 6): use this as a moderate-weight tie-breaker among
    representations that score similarly on Tiers A-C, not a gate."""
    imaging_allowlist = imaging_active & imaging_distinct
    proteomic_allowlist = proteomic_active & proteomic_distinct
    return {
        "active_concordance": _pair_stats(imaging_active, proteomic_active),
        "allowlist_concordance": _pair_stats(imaging_allowlist, proteomic_allowlist),
        "reversion_concordance": _pair_stats(imaging_reverted, proteomic_reverted),
        "imaging_only_reversion_detail": discordance_detail(
            imaging_reverted - proteomic_reverted, proteomic_per_compound, score_col
        ),
        "proteomic_only_reversion_detail": discordance_detail(
            proteomic_reverted - imaging_reverted, imaging_per_compound, score_col
        ),
    }


def build_scorecard(rows: list) -> pd.DataFrame:
    """Rows (one dict per representation, one key per metric, all assumed
    higher-is-better -- flip the sign beforehand for a lower-is-better
    metric) -> a representation x metric scorecard, each column min-max
    normalized to [0, 1] within itself so differently-scaled metrics are
    comparable."""
    df = pd.DataFrame(rows).set_index("representation")
    normalized = df.copy()
    for col in df.columns:
        lo, hi = df[col].min(), df[col].max()
        normalized[col] = 0.5 if hi == lo else (df[col] - lo) / (hi - lo)
    return normalized
