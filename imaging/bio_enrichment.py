"""Post-hoc biological-plausibility checks on a reversion run's nominees --
not part of compound selection (see `imaging.reversion`), run afterward on
its `per_compound` output."""

import gseapy as gp
import pandas as pd

from .reversion import SEED

_EMPTY_RESULT_COLUMNS = ["moa_term", "n_pool", "n_nominees", "fold_enrichment", "p_perm", "q_perm"]


def moa_enrichment(
    per_compound: pd.DataFrame,
    score_col: str = "RI_spec",
    moa_col: str = "Metadata_moa",
    compound_col: str = "Metadata_broad_sample",
    min_size: int = 3,
    max_size: int = 500,
    n_perm: int = 20000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Preranked GSEA (`gseapy.prerank`) of MoA terms against `score_col`.

    Replaces the old binary-nominee permutation test (draw-and-count against
    an allowlist mask) with a continuous one: every scored compound
    contributes its rank by `score_col` instead of a thresholded in/out
    call, so a MoA term can register as enriched by clustering toward the
    top of the ranking without any individual compound having to clear a
    significance cutoff first. This also removes the old test's singleton-
    MoA-term pathology -- terms with too few members to say anything about
    (`min_size`) are dropped before testing instead of surviving to inflate
    the multiple-testing correction burden.

    `moa_col` is '|'-delimited (one compound can carry several MoA terms).
    Compounds missing `score_col` or `moa_col` are dropped before ranking.

    Returns one row per MoA term that clears `min_size`: `n_pool` (term
    size within the ranked list), `n_nominees` (leading-edge count, i.e.
    how many of those compounds drove the enrichment score), `fold_enrichment`
    (the run's normalized enrichment score, NES -- signed, not a ratio, but
    kept under the legacy column name so downstream consumers are
    unaffected), `p_perm` (NOM p-val), `q_perm` (FDR q-val) -- sorted by
    `p_perm`. Empty (with the same columns) if no MoA term clears
    `min_size`."""
    scored = per_compound.dropna(subset=[score_col, moa_col]).drop_duplicates(compound_col, keep="first")
    empty = pd.DataFrame(columns=_EMPTY_RESULT_COLUMNS)
    if scored.empty:
        return empty

    ranked = scored.set_index(compound_col)[score_col].sort_values(ascending=False)

    moa_dict: dict[str, list[str]] = {}
    for compound, moa in zip(scored[compound_col], scored[moa_col]):
        for term in moa.split("|"):
            moa_dict.setdefault(term, []).append(compound)
    moa_dict = {term: ids for term, ids in moa_dict.items() if len(ids) >= min_size}
    if not moa_dict:
        return empty

    try:
        result = gp.prerank(
            rnk=ranked,
            gene_sets=moa_dict,
            min_size=min_size,
            max_size=max_size,
            permutation_num=n_perm,
            outdir=None,
            seed=seed,
            threads=4,
        )
    except LookupError:
        return empty

    res2d = result.res2d
    if res2d.empty:
        return empty
    tag = res2d["Tag %"].str.split("/", expand=True).astype(int)

    return pd.DataFrame(
        {
            "moa_term": res2d["Term"].to_numpy(),
            "n_pool": tag[1].to_numpy(),
            "n_nominees": tag[0].to_numpy(),
            "fold_enrichment": res2d["NES"].to_numpy(),
            "p_perm": res2d["NOM p-val"].to_numpy(),
            "q_perm": res2d["FDR q-val"].to_numpy(),
        }
    ).sort_values("p_perm").reset_index(drop=True)
