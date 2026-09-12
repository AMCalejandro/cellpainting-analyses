"""`proteomics` command group: processing, copairs, tier_e, concordance,
batch_report.

Moved from the former root `run_proteomics_pipeline.py` /
`run_proteomics_copairs.py` / `run_proteomics_concordance.py` /
`run_proteomic_normalization_bakeoff.py` scripts -- logic unchanged, see each
`*_main` docstring below for what it does. `processing` and `batch_report`
here were originally named `pipeline` and `normalization-bakeoff`,
respectively. `tier_e` is new: the proteomics analogue of `benchmark run`'s
Tier E, driving `proteomics.benchmark` (see its docstring for why that's a
separate implementation from `imaging.benchmark`, not a shared one).
"""

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import pandas as pd

from utils import copairs as cp
from utils import features as feat
from utils import plot

from imaging import batch_report as br
from imaging import benchmark as bm
from proteomics import batch_report as pbr
from proteomics import benchmark as pbm
from proteomics import concordance as pconc
from proteomics import correction as pcorr
from proteomics import imaging_metadata as pim
from proteomics import paths as prote_paths
from proteomics import pipeline

FEATURE_SPACES = ["CellProfiler", "CPCNN", "UniDino"]
COPAIRS_CONDITIONS = ["FFA", "IL6"]
CONCORDANCE_STRESS_CONDITIONS = ["FFA", "IL6"]
TIER_E_MOA_FDR_Q = 0.10


# --- processing (formerly run_proteomics_pipeline.py, CLI name "pipeline") --


def processing_main(covariates: list, method: str, use_cache: bool, out_dir: Path) -> None:
    """Normalize + batch/plate-correct the proteomic matrix
    (proteomics.pipeline.load_corrected_proteomics), then report before/after
    batch(=plate)/condition silhouette scores + a PCA (and UMAP, if
    installed) figure -- the same read as docs/batch_effect_conclusions.md:
    batch/plate silhouette should drop after correction while condition
    silhouette survives.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Start each run with a clean loading log, rather than appending onto a
    # possibly stale one from a previous run (matches imaging copairs).
    (prote_paths.RESULTS_DIR / "loading_log.txt").unlink(missing_ok=True)

    covariates = tuple(covariates)
    meta, X_corrected = pipeline.load_corrected_proteomics(
        covariates=covariates, method=method, use_cache=use_cache
    )
    with open(prote_paths.corrected_interim_path(method, covariates), "rb") as fh:
        cached = pickle.load(fh)
    X_before = feat.zscore(cached["X"].to_numpy())
    X_after = X_corrected.to_numpy()

    metrics = {
        "method": method,
        "covariates": list(covariates),
        "n_rows": int(len(meta)),
        "n_analytes": int(X_after.shape[1]),
        "before": br.compute_silhouette_scores(
            X_before, meta[br.BATCH_COL], meta[br.CONDITION_COL], meta["Metadata_Plate"]
        ),
        "after": br.compute_silhouette_scores(
            X_after, meta[br.BATCH_COL], meta[br.CONDITION_COL], meta["Metadata_Plate"]
        ),
    }
    (out_dir / f"proteomics_{method}_metrics.json").write_text(json.dumps(metrics, indent=2))

    plot.make_batch_report_figures(
        X_before, X_after, meta, out_dir, f"proteomics_{method}",
        title_prefix=f"Proteomics / {method}",
        batch_col=br.BATCH_COL, condition_col=br.CONDITION_COL, seed=br.SEED,
    )

    print(
        f"[{method}] silhouette_batch {metrics['before']['silhouette_batch']:.3f} -> "
        f"{metrics['after']['silhouette_batch']:.3f}, "
        f"silhouette_condition {metrics['before']['silhouette_condition']:.3f} -> "
        f"{metrics['after']['silhouette_condition']:.3f}, "
        f"silhouette_plate {metrics['before']['silhouette_plate']:.3f} -> "
        f"{metrics['after']['silhouette_plate']:.3f}",
        flush=True,
    )
    print(f"Saved metrics + figures -> {out_dir}", flush=True)


def add_processing_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Normalize + batch/plate-correct the proteomic matrix and report "
        "before/after silhouette scores + PCA/UMAP figures."
    )
    parser.add_argument("--covariates", default="plate", help="comma-separated: plate,batch")
    parser.add_argument("--method", default=pipeline.DEFAULT_METHOD, choices=["nested", "control_centered"])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=prote_paths.RESULTS_DIR / "batch_correction_report")
    parser.set_defaults(func=_run_processing)


def _run_processing(args: argparse.Namespace) -> None:
    processing_main(args.covariates.split(","), args.method, not args.no_cache, args.out_dir)


# --- copairs (formerly run_proteomics_copairs.py) ---------------------------


def _processed_tag(processed: bool, method: str, covariate_set: str) -> str:
    return f"{method}_{covariate_set}" if processed else "raw"


# Default tag sweep for `tier_e` (and the copairs comparison figure): raw
# plus both correction methods at each of hail_batch/config.json's covariate
# sets -- matches the `proteomics copairs` runs already available under
# results/proteomics/copairs/.
DEFAULT_TIER_E_PROCESSED_TAGS = [_processed_tag(False, "", "")] + [
    _processed_tag(True, method, covariate_set)
    for method in ("nested", "control_centered")
    for covariate_set in pcorr.COVARIATE_SETS
]


def copairs_main(
    conditions: list,
    processed: bool,
    method: str,
    covariate_sets: list,
    null_size: int,
    consistency_groupby: str,
    use_cache: bool,
    out_dir: Path,
) -> None:
    """Load proteomic data (raw/uncorrected, or batch/plate-corrected) ->
    activity/distinctiveness/consistency for one or more stress conditions,
    via `utils.copairs` UNMODIFIED. Saves one parquet per
    (condition, processed_tag, call_type) under <out-dir>/parquet/ (default
    results/proteomics/copairs/parquet/), then plots a call-count/nMAP
    summary figure into <out-dir>/figures/.

    This is the proteomics analogue of the `imaging copairs` command:
    `processed` plays the role of `preprocess` there, and -- when
    `processed` is true -- `covariate_sets` (`proteomics.correction.
    COVARIATE_SETS` keys) is swept one full run at a time exactly like
    `imaging copairs`' `--covariate-sets`. `processed=False` skips
    `method`/`covariate_sets` entirely (see `proteomics.pipeline.
    load_raw_proteomics`) and runs a single "raw" pass.
    """
    parquet_dir = out_dir / "parquet"
    ap_cache_dir = out_dir / "ap_cache"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    ap_cache_dir.mkdir(parents=True, exist_ok=True)

    consistency_tag = "" if consistency_groupby == cp.DEFAULT_CONSISTENCY_GROUPBY else "_moa"
    call_fns = {
        "activity": cp.compute_activity,
        "distinctiveness": cp.compute_distinctiveness,
        "consistency": functools.partial(
            cp.compute_consistency, groupby=consistency_groupby
        ),
    }
    call_tags = {"consistency": consistency_tag}

    covariate_sets_to_run = covariate_sets if processed else [None]
    processed_tags = []

    for covariate_set in covariate_sets_to_run:
        if processed:
            meta, X = pipeline.load_corrected_proteomics(
                covariates=pcorr.COVARIATE_SETS[covariate_set], method=method, use_cache=use_cache
            )
        else:
            meta, X = pipeline.load_raw_proteomics(use_cache=use_cache)
        feats = X.to_numpy(dtype="float64")
        processed_tag = _processed_tag(processed, method, covariate_set)
        processed_tags.append(processed_tag)

        for condition in conditions:
            cond_mask = (meta["Metadata_condition"] == condition).to_numpy()
            cond_meta = meta.loc[cond_mask].reset_index(drop=True)
            cond_feats = feats[cond_mask]
            cond_meta, cond_feats = pconc.drop_unannotated(cond_meta, cond_feats, condition)

            for call_name, fn in call_fns.items():
                file_stub = f"proteomics_{condition}_{processed_tag}_{call_name}{call_tags.get(call_name, '')}"
                out_path = parquet_dir / f"{file_stub}.parquet"
                if out_path.exists():
                    print(f"[{condition}/{processed_tag}/{call_name}] cached, skipping", flush=True)
                    continue
                t0 = time.time()
                df = fn(
                    cond_meta,
                    cond_feats,
                    null_size=null_size,
                    cache_dir=prote_paths.NULL_CACHE_DIR,
                    ap_cache_path=ap_cache_dir / f"{file_stub}.parquet",
                )
                df.to_parquet(out_path)
                n_calls = int(df["below_corrected_p"].sum())
                print(
                    f"[{condition}/{processed_tag}/{call_name}] {n_calls}/{len(df)} calls "
                    f"in {time.time() - t0:.1f}s -> {out_path.name}",
                    flush=True,
                )

    fig_path = plot.make_proteomics_copairs_summary_figure(
        out_dir, conditions, processed_tags, consistency_groupby
    )
    print(f"Saved figure -> {fig_path}", flush=True)


def add_copairs_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Activity/distinctiveness/consistency copairs calls on the "
        "proteomic matrix (raw or batch/plate-corrected)."
    )
    parser.add_argument("--conditions", type=str, default=",".join(COPAIRS_CONDITIONS))
    parser.add_argument(
        "--processed",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "true: run copairs on the batch/plate-corrected proteomic matrix "
            "(proteomics.pipeline.load_corrected_proteomics). false: run it "
            "on the raw normalized-but-uncorrected matrix "
            "(proteomics.pipeline.load_raw_proteomics)."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default=pipeline.DEFAULT_METHOD,
        choices=["nested", "control_centered"],
        help="Correction method, only used when --processed true.",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        default=",".join(pipeline.DEFAULT_COVARIATES),
        help=(
            "Comma-separated proteomics.correction.COVARIATE_SETS keys "
            "(plate, batch, batch_plate), only used when --processed true. "
            "One full copairs run (own processed_tag/parquet files) per "
            "entry, mirroring `imaging copairs`'s --covariate-sets sweep."
        ),
    )
    parser.add_argument("--null-size", type=int, default=cp.NULL_SIZE)
    parser.add_argument(
        "--consistency-groupby",
        type=str,
        default=cp.DEFAULT_CONSISTENCY_GROUPBY,
        choices=["Metadata_target", "Metadata_moa"],
        help="Grouping column for the consistency call's same-X-vs-different-X pairs.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(prote_paths.COPAIRS_RESULTS_DIR),
        help="Directory to write results into: parquet/, ap_cache/, and figures/ subdirs.",
    )
    parser.set_defaults(func=_run_copairs)


def _run_copairs(args: argparse.Namespace) -> None:
    copairs_main(
        args.conditions.split(","),
        args.processed == "true",
        args.method,
        args.covariates.split(","),
        args.null_size,
        args.consistency_groupby,
        not args.no_cache,
        Path(args.out_dir),
    )


# --- tier_e (new: proteomics analogue of `benchmark run`'s Tier E) ---------


def tier_e_main(
    conditions: list,
    processed_tags: list,
    copairs_dir: Path,
    moa_fdr_q: float,
    n_perm: int,
    seed: int,
) -> None:
    """E3 only: MoA/target preranked-GSEA enrichment of the copairs
    allowlist (active ∩ distinctive), per (condition, processed_tag) -- the
    proteomics analogue of `benchmark run`'s Tier E E3 panel
    (`imaging.benchmark.copairs_call_enrichment`), computed by
    `proteomics.benchmark` instead (see that module's docstring for why
    it's a separate implementation, not a shared one, and for the
    representation -> processed_tag swap).

    Reads already-computed `proteomics copairs` parquets from
    `<copairs_dir>/parquet/` -- run that command once per `processed_tags`
    entry first (`_processed_tag` derives the tag `proteomics copairs`
    itself used, from `--processed`/`--method`/`--covariates`).

    Saves one `{tag}_{condition}_tier_e_summary.json` per (tag, condition)
    and the `proteomics_tier_e_copairs_agreement.png` figure under
    `copairs_dir`."""
    annotation = (
        pim.load_hwat_imaging_metadata()[["Metadata_broad_sample", "Metadata_moa", "Metadata_target"]]
        .drop_duplicates("Metadata_broad_sample")
    )

    summary_rows = []
    for condition in conditions:
        for tag in processed_tags:
            act = pbm.activity_and_distinctiveness(copairs_dir, condition, tag)
            moa = pbm.copairs_call_enrichment(
                copairs_dir, condition, tag, "allowlist", annotation,
                moa_col="Metadata_moa", n_perm=n_perm, seed=seed,
            )
            target = pbm.copairs_call_enrichment(
                copairs_dir, condition, tag, "allowlist", annotation,
                moa_col="Metadata_target", n_perm=n_perm, seed=seed,
            )
            n_moa_sig = int((moa["q_perm"] <= moa_fdr_q).sum()) if len(moa) else 0
            n_target_sig = int((target["q_perm"] <= moa_fdr_q).sum()) if len(target) else 0
            print(
                f"[{tag}/{condition}] allowlist={len(act['allowlist'])} compounds; "
                f"MoA enrichment {n_moa_sig}/{len(moa)}, target enrichment "
                f"{n_target_sig}/{len(target)} significant at q<={moa_fdr_q}",
                flush=True,
            )
            summary = {
                "processed_tag": tag,
                "condition": condition,
                "copairs_allowlist_moa_n_significant_q10": n_moa_sig,
                "copairs_allowlist_target_n_significant_q10": n_target_sig,
            }
            (copairs_dir / f"{tag}_{condition}_tier_e_summary.json").write_text(json.dumps(summary, indent=2))
            summary_rows.append(summary)

    fig_path = plot.make_proteomics_tier_e_figure(pd.DataFrame(summary_rows), copairs_dir / "figures")
    print(f"Saved {fig_path}", flush=True)


def add_tier_e_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Proteomics Tier E3: MoA/target enrichment of the copairs allowlist, "
        "from already-saved `proteomics copairs` results."
    )
    parser.add_argument("--conditions", type=str, default=",".join(COPAIRS_CONDITIONS))
    parser.add_argument(
        "--processed-tags", type=str, default=",".join(DEFAULT_TIER_E_PROCESSED_TAGS),
        help="Comma-separated processed_tag values (as `proteomics copairs` derived them via _processed_tag).",
    )
    parser.add_argument("--copairs-dir", type=str, default=str(prote_paths.COPAIRS_RESULTS_DIR))
    parser.add_argument("--moa-fdr-q", type=float, default=TIER_E_MOA_FDR_Q)
    parser.add_argument("--n-perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.set_defaults(func=_run_tier_e)


def _run_tier_e(args: argparse.Namespace) -> None:
    tier_e_main(
        args.conditions.split(","),
        args.processed_tags.split(","),
        Path(args.copairs_dir),
        args.moa_fdr_q,
        args.n_perm,
        args.seed,
    )


# --- concordance (formerly run_proteomics_concordance.py) -------------------


def concordance_main(
    feature_spaces: list,
    stress_conditions: list,
    benchmark_dir: Path,
    copairs_covariate_set: str,
    n_boot: int,
    null_size: int,
    seed: int,
    out_dir: Path,
) -> None:
    """Independent proteomic hit calling (proteomics.concordance.condition_hits)
    plus concordance against each already-benchmarked imaging
    representation's hit lists (imaging.benchmark.proteomic_concordance).

    Requires the benchmark `run` command to have already been run for the
    same stress_conditions: this reads its saved
    results/imaging/benchmark/{space}_{condition}_reversion.parquet, plus
    the existing `imaging copairs` command's activity/distinctiveness
    parquets via imaging.benchmark.activity_and_distinctiveness.

    Proteomics only has Baseline/FFA/IL6 profiled (no Low Gluc), so
    stress_conditions defaults to FFA,IL6 here, not benchmark's
    DEFAULT_STRESS_CONDITIONS.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, X = pipeline.load_corrected_proteomics()

    for condition in stress_conditions:
        t0 = time.time()
        prote = pconc.condition_hits(
            meta, X, condition, n_boot=n_boot, null_size=null_size, seed=seed
        )
        prote["per_compound"].to_parquet(out_dir / f"proteomics_{condition}_reversion.parquet")
        print(
            f"[proteomics/{condition}] n_active={len(prote['active_compounds'])}, "
            f"n_distinct={len(prote['distinct_compounds'])}, "
            f"n_allowlist={len(prote['allowlist'])}, "
            f"n_reverted={len(prote['reverted_compounds'])} "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

        for space in feature_spaces:
            rev_path = benchmark_dir / f"{space}_{condition}_reversion.parquet"
            if not rev_path.exists():
                print(
                    f"[{space}/{condition}] skipping -- no {rev_path}; "
                    "run the benchmark `run` command for this condition first",
                    flush=True,
                )
                continue

            imaging_hits = bm.activity_and_distinctiveness(space, condition, copairs_covariate_set)
            imaging_per_compound = pd.read_parquet(rev_path)
            imaging_reverted = set(
                imaging_per_compound.loc[
                    imaging_per_compound["nominated_robust_ci"], "Metadata_broad_sample"
                ]
            )

            result = bm.proteomic_concordance(
                imaging_hits["active_compounds"],
                imaging_hits["distinct_compounds"],
                imaging_reverted,
                imaging_per_compound,
                prote["active_compounds"],
                prote["distinct_compounds"],
                prote["reverted_compounds"],
                prote["per_compound"],
            )

            summary = {k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)}
            (out_dir / f"{space}_{condition}_concordance_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            result["imaging_only_reversion_detail"].to_csv(
                out_dir / f"{space}_{condition}_imaging_only.csv", index=False
            )
            result["proteomic_only_reversion_detail"].to_csv(
                out_dir / f"{space}_{condition}_proteomic_only.csv", index=False
            )
            print(
                f"[{space}/{condition}] active jaccard="
                f"{result['active_concordance']['jaccard']:.2f}, "
                f"allowlist jaccard={result['allowlist_concordance']['jaccard']:.2f}, "
                f"reversion jaccard={result['reversion_concordance']['jaccard']:.2f}",
                flush=True,
            )


def add_concordance_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Proteomic hit calling + concordance against imaging benchmark "
        "hit lists (Tier D)."
    )
    parser.add_argument("--feature-spaces", type=str, default=",".join(FEATURE_SPACES))
    parser.add_argument("--stress-conditions", type=str, default=",".join(CONCORDANCE_STRESS_CONDITIONS))
    parser.add_argument("--benchmark-dir", type=str, default=str(bm.paths.BENCHMARK_DIR))
    parser.add_argument(
        "--copairs-covariate-set", type=str, default=bm.EXISTING_COPAIRS_COVARIATE_SET
    )
    parser.add_argument("--n-boot", type=int, default=pconc.N_BOOT)
    parser.add_argument("--null-size", type=int, default=pconc.NULL_SIZE)
    parser.add_argument("--seed", type=int, default=pconc.SEED)
    parser.add_argument("--out-dir", type=str, default=str(prote_paths.RESULTS_DIR / "concordance"))
    parser.set_defaults(func=_run_concordance)


def _run_concordance(args: argparse.Namespace) -> None:
    concordance_main(
        args.feature_spaces.split(","),
        args.stress_conditions.split(","),
        Path(args.benchmark_dir),
        args.copairs_covariate_set,
        args.n_boot,
        args.null_size,
        args.seed,
        Path(args.out_dir),
    )


# --- batch_report (formerly run_proteomic_normalization_bakeoff.py, CLI name
# "normalization-bakeoff") -----------------------------------------------


def batch_report_main(null_size: int, n_boot: int, seed: int, out_dir: Path) -> None:
    """Proteomic preprocessing / batch-correction bake-off
    (`proteomics.batch_report`): 3 preprocessing branches x 2
    batch-correction methods, scored on technical-noise removal,
    replicate-consistency tightening, and reversion-axis (vector-arithmetic)
    stability -- each before vs. after correction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, X = pbr.load_bakeoff_inputs()
    results = pbr.run_bakeoff(meta, X, null_size=null_size, n_boot=n_boot, seed=seed)

    results_path = out_dir / "bakeoff_results.csv"
    results.to_csv(results_path, index=False)

    pd.set_option("display.width", 160)
    print("\n=== Bake-off results (before -> after correction) ===")
    print(results[[
        "branch", "method",
        "technical_noise_before", "technical_noise_after",
        "replicate_consistency_before", "replicate_consistency_after",
        "axis_stability_before", "axis_stability_after",
    ]].to_string(index=False))
    print(f"\nSaved full results (with deltas) -> {results_path}")


def add_batch_report_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Proteomic preprocessing / batch-correction bake-off "
        "(technical-noise removal, replicate consistency, reversion-axis "
        "stability, before vs. after correction)."
    )
    parser.add_argument("--null-size", type=int, default=pbr.DEFAULT_NULL_SIZE)
    parser.add_argument("--n-boot", type=int, default=pbr.DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=pbr.DEFAULT_SEED)
    parser.add_argument(
        "--out-dir", type=str, default=str(prote_paths.RESULTS_DIR / "normalization_bakeoff")
    )
    parser.set_defaults(func=_run_batch_report)


def _run_batch_report(args: argparse.Namespace) -> None:
    batch_report_main(args.null_size, args.n_boot, args.seed, Path(args.out_dir))
