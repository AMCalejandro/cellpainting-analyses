"""D1 of experiments/benchmark_feature_representation.md Tier D: independent
proteomic hit calling (activity, distinctiveness, reversion), entirely
separate from any imaging representation.

Activity/distinctiveness reuse `utils.copairs` UNMODIFIED -- both
calls are generic over any `meta`/`feats` carrying the
Metadata_broad_sample/Metadata_batch/Metadata_Plate/Metadata_pert_type
convention, which `proteomics.pipeline.load_corrected_proteomics` already
produces (see that function's docstring). Reversion uses `proteomics.reversion`,
a fork of `imaging.reversion` with the cell-count viability gate dropped --
see that module's docstring for why.

D2 (concordance rate) and D3 (discordance detail) against each imaging
representation's hit lists live in `imaging.benchmark.proteomic_concordance`,
which only needs the hit sets/tables this module returns -- it has no
proteomics-specific logic of its own.
"""

import numpy as np
import pandas as pd

from utils import copairs as cp
from . import reversion as prev

BASELINE_CONDITION = "Baseline"
N_BOOT = prev.N_BOOT
NULL_SIZE = cp.NULL_SIZE
SEED = 0


def drop_unannotated(meta: pd.DataFrame, feats: np.ndarray, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop treated wells with no `Metadata_broad_sample` -- a compound-
    identity join gap in the curated imaging metadata (~9% of proteomics trt
    wells overall, across all three conditions), not something copairs or
    reversion can score without an identity to group replicates by. DMSO
    controls always have one (their `profile_id`, per
    `proteomics.imaging_metadata`), so this only ever drops `trt` rows.

    Public -- also used by `run_proteomics_copairs.py`, which needs the same
    per-condition cleanup ahead of the activity/distinctiveness/consistency
    calls."""
    keep = meta["Metadata_broad_sample"].notna().to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(
            f"[proteomics/{label}] dropping {n_dropped}/{len(meta)} wells with no "
            "Metadata_broad_sample (compound-identity join gap)",
            flush=True,
        )
    return meta.loc[keep].reset_index(drop=True), feats[keep]


def condition_hits(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    condition: str,
    baseline_condition: str = BASELINE_CONDITION,
    n_boot: int = N_BOOT,
    null_size: int = NULL_SIZE,
    seed: int = SEED,
) -> dict:
    """Independent proteomic activity/distinctiveness/reversion calling for
    one stress `condition`, on the pooled, corrected proteomic matrix
    `proteomics.pipeline.load_corrected_proteomics` returns.

    That matrix already has every condition jointly z-scored (once, across
    all conditions) and then per-condition (nested) residualized -- see
    `proteomics.correction`'s docstring -- so Baseline and `condition`
    already share one coordinate space and there is no separate joint-load
    step to do here, unlike `imaging.reversion.load_joint_residualized`.

    Returns activity/distinctiveness tables + hit sets, the reversion
    per_compound table, and `allowlist`/`reverted_compounds` hit sets.
    """
    feats = X.to_numpy(dtype=np.float64)

    cond_mask = (meta["Metadata_condition"] == condition).to_numpy()
    cond_meta = meta.loc[cond_mask].reset_index(drop=True)
    cond_feats = feats[cond_mask]
    cond_meta, cond_feats = drop_unannotated(cond_meta, cond_feats, condition)

    activity = cp.compute_activity(cond_meta, cond_feats, null_size=null_size, seed=seed)
    distinct = cp.compute_distinctiveness(cond_meta, cond_feats, null_size=null_size, seed=seed)
    active = set(activity.loc[activity["below_corrected_p"], "Metadata_broad_sample"])
    distinct_set = set(distinct.loc[distinct["below_corrected_p"], "Metadata_broad_sample"])
    allowlist = active & distinct_set

    joint_mask = cond_mask | (meta["Metadata_condition"] == baseline_condition).to_numpy()
    joint_meta = meta.loc[joint_mask].reset_index(drop=True)
    joint_feats = feats[joint_mask]
    joint_meta, joint_feats = drop_unannotated(joint_meta, joint_feats, f"{baseline_condition}+{condition}")
    rev_result = prev.compute_reversion(
        joint_meta, joint_feats, baseline_condition, condition,
        n_boot=n_boot, seed=seed, compound_allowlist=allowlist,
    )
    per_compound = rev_result["per_compound"]
    reverted = set(per_compound.loc[per_compound["nominated_robust_ci"], "Metadata_broad_sample"])

    return {
        "condition": condition,
        "activity_table": activity,
        "distinctiveness_table": distinct,
        "active_compounds": active,
        "distinct_compounds": distinct_set,
        "allowlist": allowlist,
        "reversion_result": rev_result,
        "per_compound": per_compound,
        "reverted_compounds": reverted,
        "n_nominated_robust_ci": rev_result["n_nominated_robust_ci"],
    }
