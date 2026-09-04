"""Driver: jointly load Baseline + a stress condition, jointly z-score and
residualize them (Ridge covariate set, or control_centered_residualize),
then score compound reversion for one feature space / covariate set. Saves
one parquet (per-compound) and one JSON (run-level summary: tau_s, L, n_obs,
p_spec) under imaging/results/.

Usage: .venv/bin/python run_reversion.py [--feature-space CellProfiler]
           [--covariate-set nested_count_plate|control_centered|count_batch]
           [--stress-condition FFA]
           [--baseline-condition Baseline] [--n-boot 10000] [--n-perm 10000]
           [--out-dir imaging/results]
           [--activity-parquet results/imaging/copairs/FFA/parquet/CellProfiler_control_centered_activity.parquet]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from imaging import features as feat
from imaging import paths, reversion as rev


def _load_significant_compounds(parquet_path: Path) -> set:
    """Compounds a copairs_pipeline per-compound call (compute_activity or
    compute_distinctiveness) flagged significant, i.e. cleared
    `below_corrected_p`. Both share the same `Metadata_broad_sample`-keyed
    schema, so this works for either."""
    df = pd.read_parquet(parquet_path)
    return set(df.loc[df["below_corrected_p"], "Metadata_broad_sample"])


def main(
    feature_space: str,
    covariate_set: str,
    stress_condition: str,
    baseline_condition: str,
    n_boot: int,
    n_perm: int,
    seed: int,
    out_dir: Path,
    activity_parquet: Optional[Path] = None,
    n_components: Optional[int] = None,
    distinctiveness_parquet: Optional[Path] = None,
) -> None:
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
    meta, feats = rev.load_joint_residualized(
        feature_space, covariate_set, baseline_condition, stress_condition, n_components
    )
    print(
        f"[{feature_space}/{covariate_set}] loaded + jointly "
        f"{'preprocessed (n_components=' + str(n_components) + ') + ' if n_components else ''}"
        f"residualized {feats.shape} ({baseline_condition} + {stress_condition}) in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    t0 = time.time()
    result = rev.compute_reversion(
        meta,
        feats,
        baseline_condition=baseline_condition,
        stress_condition=stress_condition,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
        compound_allowlist=compound_allowlist,
    )
    print(
        f"[{feature_space}/{covariate_set}] scored reversion in "
        f"{time.time() - t0:.1f}s -- {result['n_obs']} pass gate (2), "
        f"{result['n_nominated']} NOMINATE(c,s), "
        f"{result['n_nominated_specific']} also stress-specific (gate 4), "
        f"p_spec={result['p_spec']:.4f}",
        flush=True,
    )

    activity_tag = "_copairs_active" if activity_parquet is not None else ""
    if distinctiveness_parquet is not None:
        activity_tag += "_distinct"
    pca_tag = f"_pca{n_components}" if n_components else ""
    file_stub = (
        f"{feature_space}_{stress_condition}_{covariate_set}{pca_tag}_reversion{activity_tag}"
    )
    pc_path = out_dir / f"{file_stub}.parquet"
    result["per_compound"].to_parquet(pc_path)

    summary_path = out_dir / f"{file_stub}_summary.json"
    summary = {k: v for k, v in result.items() if k != "per_compound"}
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved {pc_path.name} and {summary_path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--n-perm", type=int, default=rev.N_PERM)
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
            "If given, PCA-reduce the jointly z-scored Baseline+stress "
            "feature matrix to this many components (via "
            "imaging.features.preprocess) before residualizing -- mirrors "
            "run_pipeline.py's --preprocess step for the copairs experiment, "
            "fit jointly across both conditions. Omit to residualize raw "
            "z-scored features (previous behavior)."
        ),
    )
    args = parser.parse_args()
    main(
        args.feature_space,
        args.covariate_set,
        args.stress_condition,
        args.baseline_condition,
        args.n_boot,
        args.n_perm,
        args.seed,
        Path(args.out_dir),
        Path(args.activity_parquet) if args.activity_parquet else None,
        args.n_components,
        Path(args.distinctiveness_parquet) if args.distinctiveness_parquet else None,
    )
