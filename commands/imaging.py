"""`imaging` command group: copairs, batch_report, and reversion.

Moved from the former root `run_pipeline.py` / `run_reversion.py` scripts --
logic unchanged, see each `*_main` docstring below for what it does.
`copairs` and `batch_report` were originally a single `run_pipeline.py`
script toggled by a `--batch-report` flag; they're now separate commands
(`batch_report_main` used to be the `if batch_report:` branch of
`copairs_main`, and `copairs` was originally named `pipeline`).
"""

import argparse
import functools
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import bio_enrichment
from utils import copairs as cp
from utils import features as feat
from utils import plot

from imaging import batch_report as br
from imaging import load, paths
from imaging import reversion as rev

FEATURE_SPACES = ["CellProfiler", "CPCNN", "UniDino"]


# --- copairs (formerly run_pipeline.py's copairs branch, CLI name "pipeline") -


def copairs_main(
    null_size: int,
    feature_spaces: list,
    covariate_sets: list,
    preprocess: bool,
    out_dir: Path,
    condition: str,
    consistency_groupby: str = cp.DEFAULT_CONSISTENCY_GROUPBY,
) -> None:
    """Load -> preprocess (optional) -> residualize -> copairs for a set of
    feature spaces and Ridge covariate sets, saving one parquet per
    (feature_space, covariate_set, call_type) under <out-dir>/parquet/
    (default results/imaging/copairs/parquet/), then plotting a
    call-count/nMAP summary figure into <out-dir>/figures/ from those
    results.
    """
    parquet_dir = out_dir / "parquet"
    ap_cache_dir = out_dir / "ap_cache"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    ap_cache_dir.mkdir(parents=True, exist_ok=True)

    # Start each run with a clean loading log for this condition, rather
    # than appending onto a possibly stale one from a previous run.
    load.loading_log_path(condition, log_dir=out_dir).unlink(missing_ok=True)

    # Result/ap_cache filenames only carry the condition for non-default
    # conditions, so existing FFA-condition results and their filenames
    # (also assumed by plot.py) are left untouched.
    condition_tag = "" if condition == load.DEFAULT_CONDITION else f"_{condition}"
    # Same idea for consistency: only tag the filename when grouping by
    # something other than the default Metadata_target, so a plain rerun
    # can't collide with (or be shadowed by) an MoA-grouped one.
    consistency_tag = "" if consistency_groupby == cp.DEFAULT_CONSISTENCY_GROUPBY else "_moa"

    call_fns = {
        "activity": cp.compute_activity,
        "distinctiveness": cp.compute_distinctiveness,
        "consistency": functools.partial(
            cp.compute_consistency, groupby=consistency_groupby
        ),
    }
    call_tags = {"consistency": consistency_tag}

    for space in feature_spaces:
        t0 = time.time()
        meta, feats = load.load_feature_space(space, condition, log_dir=out_dir)
        # Needed by the "control_centered" residualize method, which groups
        # its control-centering by condition; every row here already shares
        # the same --condition, so this is just a constant column.
        meta["Metadata_condition"] = condition
        print(f"[{space}] loaded {feats.shape} in {time.time() - t0:.1f}s", flush=True)

        if preprocess:
            t0 = time.time()
            feats = feat.preprocess(feats)
            print(
                f"[{space}] preprocessed to {feats.shape} in {time.time() - t0:.1f}s",
                flush=True,
            )
        else:
            feats = feat.zscore(feats)

        for cov_key in covariate_sets:
            t0 = time.time()
            residual_feats = br.RESIDUALIZE_METHODS[cov_key](feats, meta)
            print(
                f"[{space}/{cov_key}] residualized in {time.time() - t0:.1f}s",
                flush=True,
            )

            for call_name, fn in call_fns.items():
                file_stub = (
                    f"{space}{condition_tag}_{cov_key}_{call_name}"
                    f"{call_tags.get(call_name, '')}"
                )
                out_path = parquet_dir / f"{file_stub}.parquet"
                if out_path.exists():
                    print(f"[{space}/{cov_key}/{call_name}] cached, skipping", flush=True)
                    continue
                t0 = time.time()
                df = fn(
                    meta,
                    residual_feats,
                    null_size=null_size,
                    cache_dir=paths.CACHE_DIR,
                    ap_cache_path=ap_cache_dir / f"{file_stub}.parquet",
                )
                df.to_parquet(out_path)
                n_calls = int(df["below_corrected_p"].sum())
                print(
                    f"[{space}/{cov_key}/{call_name}] {n_calls}/{len(df)} calls "
                    f"in {time.time() - t0:.1f}s -> {out_path.name}",
                    flush=True,
                )

    fig_path = plot.make_copairs_summary_figure(
        out_dir, feature_spaces, covariate_sets, condition, consistency_groupby
    )
    print(f"Saved figure -> {fig_path}", flush=True)


def add_copairs_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Copairs activity/distinctiveness/consistency pipeline for a set of "
        "feature spaces and Ridge covariate sets."
    )
    parser.add_argument("--null-size", type=int, default=cp.NULL_SIZE)
    parser.add_argument("--feature-spaces", type=str, default=",".join(FEATURE_SPACES))
    parser.add_argument(
        "--covariate-sets",
        type=str,
        default=None,
        help=(
            "Comma-separated imaging.features.RESIDUALIZE_METHODS keys. "
            "Defaults to features.WITHIN_CONDITION_METHODS (one condition at "
            "a time, so the nested_* variants would be exact duplicates of "
            "their pooled counterparts)."
        ),
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "Apply imaging.features.preprocess (drop dead/redundant columns, "
            "PCA-reduce) to every feature space before residualizing, instead "
            "of residualizing the raw features."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.RESULTS_DIR),
        help="Directory to write results into: parquet/, ap_cache/, and figures/ subdirs.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=load.DEFAULT_CONDITION,
        help='cpg0014 Metadata_condition to load, e.g. "FFA", "IL6", "Low Gluc".',
    )
    parser.add_argument(
        "--consistency-groupby",
        type=str,
        default=cp.DEFAULT_CONSISTENCY_GROUPBY,
        choices=["Metadata_target", "Metadata_moa"],
        help="Grouping column for the consistency call's same-X-vs-different-X pairs.",
    )
    parser.set_defaults(func=_run_copairs)


def _run_copairs(args: argparse.Namespace) -> None:
    covariate_sets = (
        args.covariate_sets.split(",")
        if args.covariate_sets is not None
        else list(feat.WITHIN_CONDITION_METHODS)
    )
    copairs_main(
        args.null_size,
        args.feature_spaces.split(","),
        covariate_sets,
        args.preprocess,
        Path(args.out_dir),
        args.condition,
        args.consistency_groupby,
    )


# --- batch_report (formerly run_pipeline.py's --batch-report branch) ------


def batch_report_main(
    feature_spaces: list,
    covariate_sets: list,
    out_dir: Path,
) -> None:
    """Jointly load all hWAT conditions per feature space, z-score,
    Ridge-residualize with each covariate_sets entry, and report
    batch/condition silhouette scores + PCA/UMAP figures
    (imaging.batch_report) into out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Split the requested methods so the comparison figure can shade the
    # pooled Ridge covariate sets (the ones whose plate dummies span the
    # condition direction) and draw everything else -- nested_*,
    # control_centered -- to the right of the divider.
    ridge_sets = [m for m in covariate_sets if m in feat.COVARIATE_SETS]
    extra_methods = [m for m in covariate_sets if m not in feat.COVARIATE_SETS]
    metrics_by_space = {}
    for space in feature_spaces:
        metrics_by_space[space] = {}
        for method in ridge_sets + extra_methods:
            t0 = time.time()
            result = br.compute_report(space, method)
            metrics = result["metrics"]
            metrics_by_space[space][method] = metrics

            file_stub = f"{space}_{method}"
            plot.make_batch_report_figures(
                result["before_sample"], result["after_sample"], result["meta_sample"],
                out_dir, file_stub,
                title_prefix=f"{space} / {method}",
                batch_col=br.BATCH_COL, condition_col=br.CONDITION_COL, seed=br.SEED,
            )
            (out_dir / f"{file_stub}_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )

            print(
                f"[{space}/{method}] silhouette_batch "
                f"{metrics['before']['silhouette_batch']:.3f} -> "
                f"{metrics['after']['silhouette_batch']:.3f}, "
                f"silhouette_condition {metrics['before']['silhouette_condition']:.3f} -> "
                f"{metrics['after']['silhouette_condition']:.3f}, "
                f"silhouette_plate {metrics['before']['silhouette_plate']:.3f} -> "
                f"{metrics['after']['silhouette_plate']:.3f} "
                f"in {time.time() - t0:.1f}s",
                flush=True,
            )
    comparison_path = plot.make_covariate_comparison_figure(
        metrics_by_space, out_dir, ridge_sets, extra_methods
    )
    print(f"Saved covariate-set comparison figure -> {comparison_path}", flush=True)


def add_batch_report_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Batch/condition silhouette report: jointly load all hWAT "
        "conditions per feature space, z-score, Ridge-residualize with each "
        "--covariate-sets entry, and report batch/condition silhouette "
        "scores + PCA/UMAP figures."
    )
    parser.add_argument("--feature-spaces", type=str, default=",".join(FEATURE_SPACES))
    parser.add_argument(
        "--covariate-sets",
        type=str,
        default=None,
        help=(
            "Comma-separated imaging.features.RESIDUALIZE_METHODS keys. "
            "Defaults to features.CROSS_CONDITION_METHODS."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(paths.BATCH_REPORT_DIR),
        help="Directory to write batch-report metrics/figures into.",
    )
    parser.set_defaults(func=_run_batch_report)


def _run_batch_report(args: argparse.Namespace) -> None:
    covariate_sets = (
        args.covariate_sets.split(",")
        if args.covariate_sets is not None
        else list(feat.CROSS_CONDITION_METHODS)
    )
    batch_report_main(args.feature_spaces.split(","), covariate_sets, Path(args.out_dir))


# --- reversion (formerly run_reversion.py) ----------------------------------


def _load_significant_compounds(parquet_path: Path) -> set:
    """Compounds a copairs_pipeline per-compound call (compute_activity or
    compute_distinctiveness) flagged significant, i.e. cleared
    `below_corrected_p`. Both share the same `Metadata_broad_sample`-keyed
    schema, so this works for either."""
    df = pd.read_parquet(parquet_path)
    return set(df.loc[df["below_corrected_p"], "Metadata_broad_sample"])


def reversion_main(
    feature_space: str,
    covariate_set: str,
    stress_condition: str,
    baseline_condition: str,
    n_boot: int,
    seed: int,
    out_dir: Path,
    activity_parquet: Optional[Path] = None,
    n_components: Optional[int] = None,
    distinctiveness_parquet: Optional[Path] = None,
    moa_enrichment: bool = False,
    make_figures: bool = True,
) -> None:
    """Jointly load Baseline + a stress condition, residualize them, and
    score compound reversion for one feature space / covariate set. Saves
    one parquet (per-compound) and one JSON (run-level summary) under
    out_dir, plus (unless make_figures=False) PCA/UMAP diagnostic figures of
    the joint Baseline+stress space before vs after residualization under
    out_dir/figures/.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if covariate_set not in feat.RESIDUALIZE_METHODS:
        raise SystemExit(
            f"unknown --covariate-set {covariate_set!r}; "
            f"choose one of {', '.join(feat.RESIDUALIZE_METHODS)}"
        )
    if covariate_set in ("count_plate", "count_batch_plate"):
        print(
            f"WARNING: --covariate-set {covariate_set} pools Baseline + "
            f"{stress_condition} into one Ridge fit whose plate dummies span "
            "the condition direction, so it removes the reversion axis itself "
            "(docs/batch_effect_conclusions.md). Use nested_" + covariate_set,
            flush=True,
        )

    compound_allowlist = None
    if activity_parquet is not None:
        allowlist = _load_significant_compounds(activity_parquet)
        source = f"{len(allowlist)} copairs-active compounds from {activity_parquet}"
        if distinctiveness_parquet is not None:
            distinct = _load_significant_compounds(distinctiveness_parquet)
            allowlist &= distinct
            source += (
                f", intersected with {len(distinct)} copairs-distinctive "
                f"compounds from {distinctiveness_parquet} -> {len(allowlist)} "
                "in both"
            )
        compound_allowlist = sorted(allowlist)
        print(f"[{feature_space}/{covariate_set}] restricting to {source}", flush=True)

    t0 = time.time()
    meta, feats_before, feats = rev.load_joint_residualized(
        feature_space, covariate_set, baseline_condition, stress_condition, n_components
    )
    print(
        f"[{feature_space}/{covariate_set}] loaded + jointly residualized"
        f"{' + PCA-reduced (n_components=' + str(n_components) + ')' if n_components else ''} "
        f"{feats.shape} ({baseline_condition} + {stress_condition}) in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    pca_tag = f"_pca{n_components}" if n_components else ""
    if make_figures:
        t0 = time.time()
        figures_dir = out_dir / "figures"
        file_stub = f"{feature_space}_{baseline_condition}_to_{stress_condition}_{covariate_set}{pca_tag}"
        plot.make_reversion_diagnostic_figures(
            feats_before,
            feats,
            meta,
            figures_dir,
            file_stub,
            title_prefix=f"{feature_space} / {covariate_set}: {baseline_condition} -> {stress_condition}",
        )
        print(
            f"[{feature_space}/{covariate_set}] saved joint-space PCA/UMAP "
            f"diagnostics -> {figures_dir}/{file_stub}_{{pca,umap}}.png "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

    t0 = time.time()
    result = rev.compute_reversion(
        meta,
        feats,
        baseline_condition=baseline_condition,
        stress_condition=stress_condition,
        n_boot=n_boot,
        seed=seed,
        compound_allowlist=compound_allowlist,
    )
    print(
        f"[{feature_space}/{covariate_set}] scored reversion in "
        f"{time.time() - t0:.1f}s -- {result['n_nominated_robust']} nominated_robust "
        f"(ci={result['n_nominated_robust_ci']}) "
        f"[funnel on gate2-mean: {result['funnel']}, n_toxic={result['n_toxic']}, "
        f"robustness={result['robustness']}]",
        flush=True,
    )

    activity_tag = "_copairs_active" if activity_parquet is not None else ""
    if distinctiveness_parquet is not None:
        activity_tag += "_distinct"
    file_stub = (
        f"{feature_space}_{stress_condition}_{covariate_set}{pca_tag}_reversion{activity_tag}"
    )
    pc_path = out_dir / f"{file_stub}.parquet"
    result["per_compound"].to_parquet(pc_path)

    summary_path = out_dir / f"{file_stub}_summary.json"
    summary = {k: v for k, v in result.items() if k != "per_compound"}
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved {pc_path.name} and {summary_path.name}", flush=True)

    if moa_enrichment:
        enrichment = bio_enrichment.moa_enrichment(result["per_compound"])
        enrichment_path = out_dir / f"{file_stub}_moa_enrichment.csv"
        enrichment.to_csv(enrichment_path, index=False)
        if len(enrichment):
            top = enrichment.iloc[0]
            print(
                f"[{feature_space}/{covariate_set}] MoA enrichment: "
                f"{len(enrichment)} terms tested, top {top['moa_term']!r} "
                f"(q_perm={top['q_perm']:.3f}). Saved {enrichment_path.name}",
                flush=True,
            )
        else:
            print(
                f"[{feature_space}/{covariate_set}] MoA enrichment: no MoA term "
                f"had >=3 scored compounds. Saved {enrichment_path.name}",
                flush=True,
            )


def add_reversion_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Score compound reversion for one feature space / covariate set "
        "(jointly loads Baseline + a stress condition)."
    )
    parser.add_argument("--feature-space", type=str, default="CellProfiler")
    parser.add_argument(
        "--covariate-set",
        type=str,
        default="nested_count_plate",
        help=(
            "One of imaging.features.RESIDUALIZE_METHODS. Defaults to "
            "nested_count_plate: the same Ridge count+plate fit as "
            "count_plate, but fit separately within each condition so it "
            "removes plate drift at full strength without removing the "
            "Baseline-stress offset the reversion axis IS. The pooled "
            "count_plate / count_batch_plate destroy that axis here and will "
            "warn if selected."
        ),
    )
    parser.add_argument("--stress-condition", type=str, default="FFA")
    parser.add_argument("--baseline-condition", type=str, default="Baseline")
    parser.add_argument("--n-boot", type=int, default=rev.N_BOOT)
    parser.add_argument("--seed", type=int, default=rev.SEED)
    parser.add_argument("--out-dir", type=str, default=str(paths.RESULTS_DIR))
    parser.add_argument(
        "--activity-parquet",
        type=str,
        default=None,
        help=(
            "Path to a copairs_pipeline.compute_activity output parquet "
            "(e.g. results/imaging/copairs/FFA/parquet/"
            "CellProfiler_count_batch_plate_activity.parquet). If given, "
            "restrict scoring to compounds with below_corrected_p == True."
        ),
    )
    parser.add_argument(
        "--distinctiveness-parquet",
        type=str,
        default=None,
        help=(
            "Path to a copairs_pipeline.compute_distinctiveness output "
            "parquet. If given (requires --activity-parquet too), restrict "
            "scoring to compounds significant in BOTH: the activity call "
            "and this distinctiveness call."
        ),
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help=(
            "If given, PCA-reduce the jointly residualized Baseline+stress "
            "feature matrix to this many components (via "
            "imaging.features.reduce_dimensionality), AFTER residualizing, "
            "for speed. See docs/pca_resisidualization_decisions.md: "
            "residualize-then-PCA is a measured near-no-op for the axis and "
            "the nominated-compound set; PCA-ing the raw matrix first (the "
            "prior behavior of this flag) was measured NOT to be a no-op. "
            "Omit to residualize the full-dimension z-scored features."
        ),
    )
    parser.add_argument(
        "--moa-enrichment",
        action="store_true",
        help=(
            "Also run utils.bio_enrichment.moa_enrichment on the scored "
            "per_compound table (post-hoc, not part of selection) and save "
            "it as {file_stub}_moa_enrichment.csv."
        ),
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Skip the PCA/UMAP before-vs-after-residualization diagnostic "
            "figures (utils.plot.make_reversion_diagnostic_figures), saved "
            "by default to <out-dir>/figures/."
        ),
    )
    parser.set_defaults(func=_run_reversion)


def _run_reversion(args: argparse.Namespace) -> None:
    reversion_main(
        args.feature_space,
        args.covariate_set,
        args.stress_condition,
        args.baseline_condition,
        args.n_boot,
        args.seed,
        Path(args.out_dir),
        Path(args.activity_parquet) if args.activity_parquet else None,
        args.n_components,
        Path(args.distinctiveness_parquet) if args.distinctiveness_parquet else None,
        args.moa_enrichment,
        not args.skip_figures,
    )
