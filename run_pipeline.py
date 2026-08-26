"""Driver: load -> preprocess (optional) -> residualize -> copairs for a set
of feature spaces and Ridge covariate sets, saving one parquet per
(feature_space, covariate_set, call_type) under imaging/results/.

Usage: .venv/bin/python run_pipeline.py [--null-size N] [--condition FFA]
           [--feature-spaces CellProfiler,CPCNN,UniDino]
           [--covariate-sets count,count_batch,count_plate,count_batch_plate]
"""

import argparse
import time
from pathlib import Path

from imaging import copairs_pipeline as cp
from imaging import features as feat
from imaging import load, paths

FEATURE_SPACES = ["CellProfiler", "CPCNN", "UniDino"]

CALL_FNS = {
    "activity": cp.compute_activity,
    "distinctiveness": cp.compute_distinctiveness,
    "consistency": cp.compute_consistency,
}


def main(
    null_size: int,
    feature_spaces: list,
    covariate_sets: list,
    preprocess: bool,
    out_dir: Path,
    condition: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ap_cache_dir = out_dir / "ap_cache"

    # Start each run with a clean loading log for this condition, rather
    # than appending onto a possibly stale one from a previous run.
    load.loading_log_path(condition).unlink(missing_ok=True)

    # Result/ap_cache filenames only carry the condition for non-default
    # conditions, so existing FFA-condition results and their filenames
    # (also assumed by make_figure.py) are left untouched.
    condition_tag = "" if condition == load.DEFAULT_CONDITION else f"_{condition}"

    for space in feature_spaces:
        t0 = time.time()
        meta, feats = load.load_feature_space(space, condition)
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
            covariates = feat.COVARIATE_SETS[cov_key]
            t0 = time.time()
            residual_feats = feat.ridge_residualize(feats, meta, covariates)
            print(
                f"[{space}/{cov_key}] residualized in {time.time() - t0:.1f}s",
                flush=True,
            )

            for call_name, fn in CALL_FNS.items():
                file_stub = f"{space}{condition_tag}_{cov_key}_{call_name}"
                out_path = out_dir / f"{file_stub}.parquet"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--null-size", type=int, default=cp.NULL_SIZE)
    parser.add_argument("--feature-spaces", type=str, default=",".join(FEATURE_SPACES))
    parser.add_argument(
        "--covariate-sets", type=str, default=",".join(feat.COVARIATE_SETS)
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
        help="Directory to write result parquets (and its ap_cache/ subdir) to.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=load.DEFAULT_CONDITION,
        help='cpg0014 Metadata_condition to load, e.g. "FFA", "IL6", "Low Gluc".',
    )
    args = parser.parse_args()
    main(
        args.null_size,
        args.feature_spaces.split(","),
        args.covariate_sets.split(","),
        args.preprocess,
        Path(args.out_dir),
        args.condition,
    )
