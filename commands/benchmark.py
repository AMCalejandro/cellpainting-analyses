"""`benchmark` command group: run, figures.

Moved from the former root `run_cellrep_benchmark.py` /
`make_benchmark_figures.py` scripts -- logic unchanged, see each `*_main`
docstring below for what it does. Full tier documentation lives in
experiments/benchmark_feature_representation.md.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import plot

from imaging import batch_report as br
from imaging import benchmark as bm
from imaging import bio_enrichment
from imaging import paths
from imaging import reversion as rev

FEATURE_SPACES = ["CellProfiler", "CPCNN", "UniDino"]
MOA_FDR_Q = 0.10

FIGURES_FEATURE_SPACES = ["CellProfiler", "CPCNN", "UniDino"]
FIGURES_STRESS_CONDITIONS = ["FFA", "IL6", "Low Gluc"]
FIGURES_PROTEOMICS_CONDITIONS = ["FFA", "IL6"]


# --- run (formerly run_cellrep_benchmark.py) --------------------------------


def _run_tier_a1(feature_spaces: list, covariate_set: str, out_dir: Path) -> dict:
    """Runs A1 per feature space and returns {space: metrics}, so `run_main`
    can fold a representation-level (not per-condition) technical-quality
    signal into every scorecard row for that representation -- see module
    docstring."""
    metrics_by_space = {}
    for space in feature_spaces:
        t0 = time.time()
        result = br.compute_report(space, covariate_set)
        metrics = result["metrics"]
        metrics_by_space[space] = metrics
        (out_dir / f"{space}_A1_silhouette.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"[{space}/A1] silhouette_plate {metrics['before']['silhouette_plate']:.3f} -> "
            f"{metrics['after']['silhouette_plate']:.3f}, silhouette_condition "
            f"{metrics['before']['silhouette_condition']:.3f} -> "
            f"{metrics['after']['silhouette_condition']:.3f} in {time.time() - t0:.1f}s",
            flush=True,
        )
    return metrics_by_space


def run_main(
    feature_spaces: list,
    stress_conditions: list,
    covariate_set: str,
    copairs_covariate_set: str,
    n_components: int,
    n_boot: int,
    n_splits: int,
    split_n_boot: int,
    split_null_size: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Matched-dimension benchmark of CellProfiler vs. CPCNN vs. UniDino on
    the existing residualize -> copairs -> reversion pipeline
    (imaging.benchmark), unmodified except for the input feature matrix. See
    experiments/benchmark_feature_representation.md and the module docstring
    of the former run_cellrep_benchmark.py for the full Tier A-E
    description.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    a1_metrics = _run_tier_a1(feature_spaces, covariate_set, out_dir)

    k_reference = "CellProfiler" if "CellProfiler" in feature_spaces else feature_spaces[0]
    k = bm.fit_pca_basis(k_reference, covariate_set, k=n_components, seed=seed).k
    print(f"[dimensionality] k={k} (elbow of {k_reference}'s control-well variance)", flush=True)

    bases = {}
    for space in feature_spaces:
        t0 = time.time()
        bases[space] = bm.fit_pca_basis(space, covariate_set, k=k, seed=seed)
        print(f"[{space}] fit PCA basis (k={k}) in {time.time() - t0:.1f}s", flush=True)

    scorecard_rows = []
    # E2: cross-condition hit-set agreement per representation, accumulated
    # across the whole loop below and only summarized once every condition
    # has run (see module docstring), into pooled_scorecard.csv.
    cross_condition_hit_sets = {
        call: {space: {} for space in feature_spaces}
        for call in ("activity", "distinctiveness", "allowlist", "consistency")
    }
    for condition in stress_conditions:
        hit_sets_active, hit_sets_reversion = {}, {}
        hit_sets_activity, hit_sets_distinctiveness, hit_sets_consistency = {}, {}, {}
        per_compound_by_space, space_metrics = {}, {}

        for space in feature_spaces:
            t0 = time.time()
            meta, feats = bm.matched_joint_features(space, condition, covariate_set, bases[space])
            print(
                f"[{space}/{condition}] matched features {feats.shape} in {time.time() - t0:.1f}s",
                flush=True,
            )

            b1 = bm.condition_separability(meta, feats, seed=seed)

            act = bm.activity_and_distinctiveness(space, condition, copairs_covariate_set)
            hit_sets_active[space] = act["active_compounds"] & act["distinct_compounds"]
            hit_sets_activity[space] = act["active_compounds"]
            hit_sets_distinctiveness[space] = act["distinct_compounds"]
            cross_condition_hit_sets["activity"][space][condition] = act["active_compounds"]
            cross_condition_hit_sets["distinctiveness"][space][condition] = act["distinct_compounds"]
            cross_condition_hit_sets["allowlist"][space][condition] = act["allowlist"]

            # E3: consistency's own called terms (already a per-term test --
            # see imaging.benchmark.consistency_called_terms's docstring)
            # plus MoA/target permutation enrichment of the copairs
            # allowlist, both computed upstream of (and independent from)
            # reversion.
            consistency_df = bm.load_existing_copairs_call(space, condition, "consistency", copairs_covariate_set)
            called_terms = bm.consistency_called_terms(consistency_df)
            called_terms.to_csv(out_dir / f"{space}_{condition}_consistency_called_terms.csv", index=False)
            hit_sets_consistency[space] = set(called_terms["term"])
            cross_condition_hit_sets["consistency"][space][condition] = hit_sets_consistency[space]

            copairs_moa = bm.copairs_call_enrichment(
                space, condition, "allowlist", copairs_covariate_set, moa_col="Metadata_moa"
            )
            copairs_moa.to_csv(out_dir / f"{space}_{condition}_copairs_allowlist_moa_enrichment.csv", index=False)
            n_copairs_moa_sig = int((copairs_moa["q_perm"] <= MOA_FDR_Q).sum()) if len(copairs_moa) else 0

            copairs_target = bm.copairs_call_enrichment(
                space, condition, "allowlist", copairs_covariate_set, moa_col="Metadata_target"
            )
            copairs_target.to_csv(out_dir / f"{space}_{condition}_copairs_allowlist_target_enrichment.csv", index=False)
            n_copairs_target_sig = int((copairs_target["q_perm"] <= MOA_FDR_Q).sum()) if len(copairs_target) else 0

            print(
                f"[{space}/{condition}] consistency: {len(hit_sets_consistency[space])}/{len(consistency_df)} "
                f"terms called; copairs allowlist enrichment: {n_copairs_moa_sig}/{len(copairs_moa)} MoA terms, "
                f"{n_copairs_target_sig}/{len(copairs_target)} target terms significant at q<={MOA_FDR_Q}",
                flush=True,
            )

            t0 = time.time()
            rev_result = rev.compute_reversion(
                meta, feats, bm.BASELINE_CONDITION, condition, n_boot=n_boot, seed=seed,
                compound_allowlist=act["allowlist"],
            )
            per_compound = rev_result["per_compound"]
            per_compound.to_parquet(out_dir / f"{space}_{condition}_reversion.parquet")
            per_compound_by_space[space] = per_compound
            hit_sets_reversion[space] = set(
                per_compound.loc[per_compound["nominated_robust_ci"], "Metadata_broad_sample"]
            )
            print(
                f"[{space}/{condition}] reversion (restricted to {len(act['allowlist'])} "
                f"active∩distinctive compounds): {rev_result['n_nominated_robust_ci']} "
                f"nominated_robust_ci / {len(per_compound)} scored in {time.time() - t0:.1f}s",
                flush=True,
            )

            moa = bio_enrichment.moa_enrichment(per_compound)
            moa.to_csv(out_dir / f"{space}_{condition}_moa_enrichment.csv", index=False)
            n_moa_sig = int((moa["q_perm"] <= MOA_FDR_Q).sum()) if len(moa) else 0
            if len(moa):
                top = moa.iloc[0]
                print(
                    f"[{space}/{condition}] MoA enrichment: {len(moa)} terms tested, "
                    f"{n_moa_sig} significant at q<={MOA_FDR_Q}, top {top['moa_term']!r} "
                    f"(q_perm={top['q_perm']:.3f}, fold={top['fold_enrichment']:.2f})",
                    flush=True,
                )
            else:
                print(
                    f"[{space}/{condition}] MoA enrichment: no scored compounds carry Metadata_moa",
                    flush=True,
                )

            t0 = time.time()
            c2 = bm.replicate_split_stability(
                meta, feats, condition, n_splits=n_splits, seed=seed,
                n_boot=split_n_boot, activity_null_size=split_null_size,
            )
            print(
                f"[{space}/{condition}] C2 stability ({n_splits} leave-one-out splits): "
                f"activity_jaccard={c2['activity_jaccard_mean']:.2f}, "
                f"reversion_jaccard={c2['reversion_jaccard_mean']:.2f}, "
                f"reversion_score_spearman={c2['reversion_score_spearman_mean']:.2f} "
                f"in {time.time() - t0:.1f}s",
                flush=True,
            )

            space_metrics[space] = {
                "b1": b1, "act": act, "rev_result": rev_result,
                "per_compound": per_compound, "n_moa_sig": n_moa_sig, "c2": c2,
                "n_copairs_moa_sig": n_copairs_moa_sig,
                "n_copairs_target_sig": n_copairs_target_sig,
                "n_consistency_called": len(hit_sets_consistency[space]),
            }

        # E1: within-condition, cross-representation hit overlap -- computed
        # here (rather than after the C3 loop below, where it used to live)
        # so `mean_pairwise_jaccard` can fold a per-representation summary
        # into that representation's `summary` dict.
        overlap_active = bm.hit_overlap(hit_sets_active)
        overlap_reversion = bm.hit_overlap(hit_sets_reversion)
        overlap_activity = bm.hit_overlap(hit_sets_activity)
        overlap_distinctiveness = bm.hit_overlap(hit_sets_distinctiveness)
        overlap_consistency = bm.hit_overlap(hit_sets_consistency)
        overlap_active.to_csv(out_dir / f"{condition}_hit_overlap_active.csv", index=False)
        overlap_reversion.to_csv(out_dir / f"{condition}_hit_overlap_reversion.csv", index=False)
        overlap_activity.to_csv(out_dir / f"{condition}_hit_overlap_activity.csv", index=False)
        overlap_distinctiveness.to_csv(out_dir / f"{condition}_hit_overlap_distinctiveness.csv", index=False)
        overlap_consistency.to_csv(out_dir / f"{condition}_hit_overlap_consistency.csv", index=False)
        print(
            f"[{condition}] hit overlap (active): {overlap_active.to_dict('records')}\n"
            f"[{condition}] hit overlap (reversion): {overlap_reversion.to_dict('records')}\n"
            f"[{condition}] hit overlap (activity): {overlap_activity.to_dict('records')}\n"
            f"[{condition}] hit overlap (distinctiveness): {overlap_distinctiveness.to_dict('records')}\n"
            f"[{condition}] hit overlap (consistency): {overlap_consistency.to_dict('records')}",
            flush=True,
        )

        # C3, non-circular: run only now that every representation's
        # per_compound table for this condition is available -- see module
        # docstring and imaging.benchmark.cross_representation_effect_size.
        for space in feature_spaces:
            others = [s for s in feature_spaces if s != space]
            cross = [
                bm.cross_representation_effect_size(per_compound_by_space[space], per_compound_by_space[other])
                for other in others
            ]
            cohens_ds = [c["cohens_d"] for c in cross if not np.isnan(c["cohens_d"])]
            aurocs = [c["auroc"] for c in cross if not np.isnan(c["auroc"])]
            cross_cohens_d = float(np.mean(cohens_ds)) if cohens_ds else float("nan")
            cross_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
            print(
                f"[{space}/{condition}] C3 cross-representation effect size "
                f"(avg vs. {others}): cohens_d={cross_cohens_d:.2f}, auroc={cross_auroc:.2f}",
                flush=True,
            )

            m = space_metrics[space]
            b1, act, rev_result, per_compound = m["b1"], m["act"], m["rev_result"], m["per_compound"]
            n_moa_sig, c2 = m["n_moa_sig"], m["c2"]
            summary = {
                "representation": space,
                "condition": condition,
                "k": k,
                "covariate_set": covariate_set,
                "activity_rate": act["activity_rate"],
                "n_active": act["n_active"],
                "n_distinct": act["n_distinct"],
                "n_allowlist": len(act["allowlist"]),
                "condition_separability_auroc": b1["auroc_mean"],
                "n_nominated_robust_ci": rev_result["n_nominated_robust_ci"],
                "n_scored_compounds": len(per_compound),
                "cross_rep_effect_size_cohens_d": cross_cohens_d,
                "cross_rep_effect_size_auroc": cross_auroc,
                "c2_activity_jaccard_mean": c2["activity_jaccard_mean"],
                "c2_reversion_jaccard_mean": c2["reversion_jaccard_mean"],
                "c2_reversion_score_spearman_mean": c2["reversion_score_spearman_mean"],
                "moa_n_significant_q10": n_moa_sig,
                "copairs_allowlist_moa_n_significant_q10": m["n_copairs_moa_sig"],
                "copairs_allowlist_target_n_significant_q10": m["n_copairs_target_sig"],
                "n_consistency_called_terms": m["n_consistency_called"],
                "copairs_cross_rep_activity_jaccard": bm.mean_pairwise_jaccard(overlap_activity, space),
                "copairs_cross_rep_distinctiveness_jaccard": bm.mean_pairwise_jaccard(overlap_distinctiveness, space),
                "copairs_cross_rep_allowlist_jaccard": bm.mean_pairwise_jaccard(overlap_active, space),
                "copairs_cross_rep_consistency_jaccard": bm.mean_pairwise_jaccard(overlap_consistency, space),
                "a1_batch_reduction": (
                    a1_metrics[space]["before"]["silhouette_batch"]
                    - a1_metrics[space]["after"]["silhouette_batch"]
                ),
                "a1_plate_reduction": (
                    a1_metrics[space]["before"]["silhouette_plate"]
                    - a1_metrics[space]["after"]["silhouette_plate"]
                ),
                "a1_condition_retention": a1_metrics[space]["after"]["silhouette_condition"],
            }
            (out_dir / f"{space}_{condition}_summary.json").write_text(json.dumps(summary, indent=2))
            scorecard_rows.append(summary)

        cond_rows = [{k2: v for k2, v in r.items() if k2 != "condition"} for r in scorecard_rows if r["condition"] == condition]
        scorecard = bm.build_scorecard(cond_rows)
        scorecard.to_csv(out_dir / f"{condition}_scorecard.csv")

    # E2: cross-condition agreement, one representation at a time -- only
    # computable once every condition has been scored, so it lands in
    # pooled_scorecard.csv rather than any single condition's (see module
    # docstring).
    cross_condition_jaccard = {space: {} for space in feature_spaces}
    for call, by_space in cross_condition_hit_sets.items():
        for space in feature_spaces:
            overlap = bm.hit_overlap(by_space[space])
            overlap.to_csv(out_dir / f"{space}_cross_condition_hit_overlap_{call}.csv", index=False)
            cross_condition_jaccard[space][call] = (
                float(overlap["jaccard"].mean()) if len(overlap) else float("nan")
            )
            print(
                f"[{space}] cross-condition hit overlap ({call}): {overlap.to_dict('records')}",
                flush=True,
            )

    pooled = (
        pd.DataFrame(scorecard_rows)
        .drop(columns=["condition", "k", "covariate_set"])
        .groupby("representation", as_index=False)
        .mean(numeric_only=True)
    )
    for call in cross_condition_hit_sets:
        pooled[f"copairs_cross_condition_{call}_jaccard"] = pooled["representation"].map(
            lambda space, call=call: cross_condition_jaccard[space][call]
        )
    pooled_scorecard = bm.build_scorecard(pooled.to_dict("records"))
    pooled_scorecard.to_csv(out_dir / "pooled_scorecard.csv")
    print(f"Saved pooled scorecard -> {out_dir / 'pooled_scorecard.csv'}", flush=True)

    # Raw (not min-max-normalized) per-representation E2 values, mirroring
    # {space}_A1_silhouette.json, so the figures command's Tier E figure can
    # plot them without re-deriving from pooled_scorecard.csv's normalized
    # columns.
    for space in feature_spaces:
        (out_dir / f"{space}_cross_condition_jaccard.json").write_text(
            json.dumps(cross_condition_jaccard[space], indent=2)
        )


def add_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Matched-dimension CellProfiler vs. CPCNN vs. UniDino benchmark "
        "(Tiers A1/A2/B1/C1-C3/E) -- see "
        "experiments/benchmark_feature_representation.md."
    )
    parser.add_argument("--feature-spaces", type=str, default=",".join(FEATURE_SPACES))
    parser.add_argument("--stress-conditions", type=str, default=",".join(bm.DEFAULT_STRESS_CONDITIONS))
    parser.add_argument(
        "--covariate-set",
        type=str,
        default=bm.DEFAULT_COVARIATE_SET,
        help="A key from imaging.features.CROSS_CONDITION_METHODS (nested_* or control_centered).",
    )
    parser.add_argument(
        "--copairs-covariate-set",
        type=str,
        default=bm.EXISTING_COPAIRS_COVARIATE_SET,
        help=(
            "Which existing `imaging copairs` covariate-set family's activity/"
            "distinctiveness parquets to reuse for A2 and the reversion "
            "allowlist (results/imaging/copairs/<condition>/parquet/)."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Shared PCA dimensionality k. Default: elbow of CellProfiler's control-well variance (50-150).",
    )
    parser.add_argument("--n-boot", type=int, default=rev.N_BOOT)
    parser.add_argument("--n-splits", type=int, default=bm.N_SPLITS)
    parser.add_argument("--split-n-boot", type=int, default=1000)
    parser.add_argument("--split-null-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=bm.SEED)
    parser.add_argument("--out-dir", type=str, default=str(paths.BENCHMARK_DIR))
    parser.set_defaults(func=_run_run)


def _run_run(args: argparse.Namespace) -> None:
    run_main(
        args.feature_spaces.split(","),
        args.stress_conditions.split(","),
        args.covariate_set,
        args.copairs_covariate_set,
        args.n_components,
        args.n_boot,
        args.n_splits,
        args.split_n_boot,
        args.split_null_size,
        args.seed,
        Path(args.out_dir),
    )


# --- figures (formerly make_benchmark_figures.py) ---------------------------


def _load_summary_rows(benchmark_dir: Path, feature_spaces: list, stress_conditions: list) -> pd.DataFrame:
    rows = []
    for space in feature_spaces:
        for condition in stress_conditions:
            path = benchmark_dir / f"{space}_{condition}_summary.json"
            if not path.exists():
                print(f"skipping {path} (not found)")
                continue
            rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def _load_a1_metrics(benchmark_dir: Path, feature_spaces: list) -> dict:
    metrics = {}
    for space in feature_spaces:
        path = benchmark_dir / f"{space}_A1_silhouette.json"
        if not path.exists():
            print(f"skipping {path} (not found)")
            continue
        metrics[space] = json.loads(path.read_text())
    return metrics


def _load_cross_condition_jaccard(benchmark_dir: Path, feature_spaces: list) -> dict:
    metrics = {}
    for space in feature_spaces:
        path = benchmark_dir / f"{space}_cross_condition_jaccard.json"
        if not path.exists():
            print(f"skipping {path} (not found)")
            continue
        metrics[space] = json.loads(path.read_text())
    return metrics


def _load_concordance_rows(concordance_dir: Path, feature_spaces: list, proteomics_conditions: list) -> pd.DataFrame:
    rows = []
    for space in feature_spaces:
        for condition in proteomics_conditions:
            path = concordance_dir / f"{space}_{condition}_concordance_summary.json"
            if not path.exists():
                print(f"skipping {path} (not found)")
                continue
            summary = json.loads(path.read_text())
            rows.append(
                {
                    "representation": space,
                    "condition": condition,
                    "active_jaccard": summary["active_concordance"]["jaccard"],
                    "allowlist_jaccard": summary["allowlist_concordance"]["jaccard"],
                    "reversion_jaccard": summary["reversion_concordance"]["jaccard"],
                }
            )
    return pd.DataFrame(rows)


def figures_main(
    benchmark_dir: Path,
    concordance_dir: Path,
    feature_spaces: list,
    stress_conditions: list,
    proteomics_conditions: list,
) -> None:
    """One figure per tier of experiments/benchmark_feature_representation.md,
    assembled from the artifacts the benchmark `run` command and the
    proteomics `concordance` command already wrote to disk (no
    recomputation) -- see utils.plot's make_tier_a/b/c/d/e_figure for what
    each shows.
    """
    summary_rows = _load_summary_rows(benchmark_dir, feature_spaces, stress_conditions)
    a1_metrics = _load_a1_metrics(benchmark_dir, feature_spaces)
    cross_condition_jaccard = _load_cross_condition_jaccard(benchmark_dir, feature_spaces)
    concordance_rows = _load_concordance_rows(concordance_dir, feature_spaces, proteomics_conditions)

    if a1_metrics:
        out = plot.make_tier_a_figure(a1_metrics, benchmark_dir)
        print(f"Saved {out}")
    if not summary_rows.empty:
        out = plot.make_tier_b_figure(summary_rows, benchmark_dir)
        print(f"Saved {out}")
        out = plot.make_tier_c_figure(summary_rows, benchmark_dir)
        print(f"Saved {out}")
        if cross_condition_jaccard:
            out = plot.make_tier_e_figure(summary_rows, cross_condition_jaccard, benchmark_dir)
            print(f"Saved {out}")
        else:
            print(
                "skipping Tier E figure -- no {space}_cross_condition_jaccard.json found, "
                "run the benchmark `run` command first"
            )
    if not concordance_rows.empty:
        out = plot.make_tier_d_figure(concordance_rows, benchmark_dir)
        print(f"Saved {out}")
    else:
        print("skipping Tier D figure -- no concordance summaries found, run the proteomics `concordance` command first")


def add_figures_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Assemble Tier A-E figures from artifacts already written by "
        "`benchmark run` and `proteomics concordance` (no recomputation)."
    )
    parser.add_argument("--benchmark-dir", type=str, default=str(paths.BENCHMARK_DIR))
    parser.add_argument("--concordance-dir", type=str, default="results/proteomics/concordance")
    parser.add_argument("--feature-spaces", type=str, default=",".join(FIGURES_FEATURE_SPACES))
    parser.add_argument("--stress-conditions", type=str, default=",".join(FIGURES_STRESS_CONDITIONS))
    parser.add_argument("--proteomics-conditions", type=str, default=",".join(FIGURES_PROTEOMICS_CONDITIONS))
    parser.set_defaults(func=_run_figures)


def _run_figures(args: argparse.Namespace) -> None:
    figures_main(
        Path(args.benchmark_dir),
        Path(args.concordance_dir),
        args.feature_spaces.split(","),
        args.stress_conditions.split(","),
        args.proteomics_conditions.split(","),
    )
