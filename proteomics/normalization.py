"""
Stage 1: proteomic normalization.

1. drop_curated_bad_samples  -- metadata-driven exclusion (Bad_IL6)
2. analyte_detection_qc      -- drop low-detection-rate analytes
3. impute_missing            -- per-analyte median imputation
4. transform_and_scale       -- arcsinh + per-analyte robust (median/MAD) scaling
5. flag_outlier_samples      -- MAD-based global-outlier detection

See project root README.md section 1 for the full rationale behind each step.
"""

import os
import pickle

import numpy as np
import pandas as pd

from .paths import NORMALIZED_INTERIM_PATH, RAW_PROTE_PATH


def load_normalized(path: str, n_meta_cols: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the normalized nELISA export into metadata and the 187-analyte matrix."""
    df = pd.read_csv(path)
    meta_df = df.iloc[:, :n_meta_cols].reset_index(drop=True)
    X = df.iloc[:, n_meta_cols:].reset_index(drop=True)
    return meta_df, X


def drop_curated_bad_samples(meta_df: pd.DataFrame, X: pd.DataFrame,
                              bad_groups=("Bad_IL6",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop samples from experimental runs the data generators already flagged as
    failed (e.g. `group == "Bad_IL6"` is two whole plates of a failed IL-6/TNFa
    stimulation) - a curation-level exclusion, distinct from statistical
    outlier detection below.
    """
    keep = ~meta_df["group"].isin(bad_groups)
    return meta_df[keep].reset_index(drop=True), X[keep.values].reset_index(drop=True)


def analyte_detection_qc(X: pd.DataFrame, min_detection_rate: float = 0.20,
                          verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop analytes rarely measured above their LLOD (normalized signal < 0
    means below that protein's LLOD). Analytes near-always below LLOD
    (e.g. CYCS, CSF1R, IFNG - detected in <20% of samples here) are mostly
    reading assay noise floor rather than adipocyte biology; keeping them
    inflates the effective dimensionality of any PCA/distance metric without
    adding signal. Values above ULOD (>1) are left untouched - those are
    real, just saturating, high-abundance measurements.
    """
    detection_rate = 1 - (X < 0).mean()
    keep_cols = detection_rate[detection_rate >= min_detection_rate].index
    dropped = detection_rate[detection_rate < min_detection_rate].sort_values()
    if verbose:
        print(f"analyte QC: dropping {len(dropped)}/{X.shape[1]} analytes "
              f"below {min_detection_rate:.0%} detection rate")
        if len(dropped):
            print((dropped * 100).round(1).to_string())
    return X[keep_cols], dropped


def impute_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Per-analyte median imputation. Missingness here is sparse (~0.03% of cells,
    scattered across wells/plates) and not systematic, so this is a safe default."""
    return X.fillna(X.median())


def transform_and_scale(X: pd.DataFrame) -> pd.DataFrame:
    """
    arcsinh + per-analyte robust (median/MAD) scaling.

    arcsinh instead of log: the normalized signal legitimately goes negative
    (below-LLOD noise) and log can't handle that without an arbitrary shift.
    arcsinh behaves linearly near 0 and like log for large values, so it
    compresses the heavy right tail of highly-secreted analytes (IL6, CCL2,
    CXCL8, TNF, ...) without distorting the below-LLOD region.

    Robust-scaling every analyte to comparable spread afterwards is what
    stops any one high-range/skewed analyte from dominating PCA or distance
    metrics downstream. This is only safe to do *after* analyte_detection_qc
    - equalizing the scale of a noise-floor analyte would give pure assay
    noise the same weight as real biological signal.
    """
    Xt = np.arcsinh(X)
    med = Xt.median()
    mad = (Xt - med).abs().median().replace(0, np.nan)
    return (Xt - med) / (1.4826 * mad)


def sample_level_outlier_summary(df: pd.DataFrame, z_thresh: float = 8.0) -> pd.Series:
    """
    Counts, per SAMPLE (row), how many analytes flag that sample as an
    extreme value (robust MAD-based z-score beyond z_thresh).
    """
    med = df.median()
    mad = (df - med).abs().median().replace(0, np.nan)
    z = 0.6745 * (df - med) / mad
    counts = (z.abs() > z_thresh).sum(axis=1).sort_values(ascending=False)
    counts.name = "n_analytes_flagged"
    return counts


def flag_outlier_samples(X: pd.DataFrame, z_thresh: float = 8.0,
                          min_flagged: int = 15, verbose: bool = True) -> pd.Index:
    """
    Flag samples that are extreme on many analytes *simultaneously* - a
    handful of unrelated proteins spiking together looks like a handling/
    assay failure, not biology. `min_flagged` should sit in the natural gap
    of the flagged-count distribution (inspect `sample_level_outlier_summary`
    before trusting the default): in this dataset the bulk of samples flag
    6-10 analytes just from multiple-testing noise at this z-threshold, then
    there's a clear gap before a long tail of samples flagging 15-98
    analytes at once, which are the true global outliers. A fixed top-N cut
    (e.g. always drop the worst 10) doesn't adapt to that gap and risks
    either dropping real, strongly-responding samples or leaving true
    failures in.
    """
    counts = sample_level_outlier_summary(X, z_thresh=z_thresh)
    flagged = counts[counts >= min_flagged].index
    if verbose:
        print(f"sample QC: flagging {len(flagged)}/{len(X)} globally-aberrant samples "
              f"(>= {min_flagged} of {X.shape[1]} analytes each beyond {z_thresh} MAD z)")
    return flagged


def run_normalization(
    csv_path: str = RAW_PROTE_PATH,
    n_meta_cols: int = 24,
    min_detection_rate: float = 0.20,
    outlier_z_thresh: float = 8.0,
    outlier_min_flagged: int = 15,
    save_path: str = NORMALIZED_INTERIM_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full stage-1 pipeline end to end and saves the interim
    artifact. This is what scripts/run_normalization.py and main.py's
    `correct` subcommand call."""
    meta_df, X = load_normalized(csv_path, n_meta_cols)
    print("raw:", X.shape)

    meta_df, X = drop_curated_bad_samples(meta_df, X)
    print("after dropping curated-bad group(s):", X.shape)

    X, dropped_analytes = analyte_detection_qc(X, min_detection_rate=min_detection_rate)
    print("after analyte detection-rate QC:", X.shape)

    X = impute_missing(X)
    X = transform_and_scale(X)

    outlier_idx = flag_outlier_samples(X, z_thresh=outlier_z_thresh, min_flagged=outlier_min_flagged)
    meta_df = meta_df[~meta_df.index.isin(outlier_idx)].reset_index(drop=True)
    X = X[~X.index.isin(outlier_idx)].reset_index(drop=True)

    print("final:", X.shape, meta_df.shape)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "meta": meta_df,
            "X": X,
            "dropped_analytes": dropped_analytes,
            "outlier_idx": outlier_idx,
        }, f)
    print(f"Saved normalized proteomics data to {save_path}")

    return meta_df, X
