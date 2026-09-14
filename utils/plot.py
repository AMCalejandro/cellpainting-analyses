"""All matplotlib figure-building shared by the imaging and proteomics
pipelines, so plotting code lives in one place rather than being scattered
across driver modules.

- `make_copairs_summary_figure`: recreates the imaging_results.png layout from the copairs
  pipeline outputs saved by run_pipeline.py -- a 3x2 grid of (activity,
  distinctiveness, consistency) x (call-count bars, nMAP violins), one group
  of covariate-set bars/violins per feature space.
- `make_copairs_cross_condition_figure`: cross-condition companion to
  `make_copairs_summary_figure` -- ONE figure spanning every condition, for
  a single (the "best") covariate set, feature spaces on the x-axis and
  conditions as the grouped/colored bars.
- `make_copairs_feature_space_figure`: the mirror image -- ONE figure
  spanning every condition, for a single feature space, conditions on the
  x-axis and covariate sets as the grouped/colored bars.
- `make_proteomics_copairs_summary_figure`: `run_proteomics_copairs.py`'s
  analogue of `make_copairs_summary_figure` for the single-feature-space
  proteomics pipeline.
- `make_batch_report_figures`: PCA/UMAP 2x2 grids (before/after
  residualization x colored-by-batch/colored-by-condition) for
  imaging.batch_report.
- `make_batch_effect_pca_figure`: single-PCA-fit, 3-panel snapshot (colored
  by cell count/plate/batch) of the batch effects present in the raw,
  pre-residualization feature space -- one per feature space, independent
  of residualization method, for imaging.batch_report.
- `make_reversion_diagnostic_figures`: same PCA/UMAP 2x2 grids, for one
  imaging.reversion.load_joint_residualized run (a single Baseline+stress
  joint space) -- a thin wrapper that subsamples then delegates to
  `make_batch_report_figures`.
- `make_copairs_pc_figure`: same PCA/UMAP 2x2 grids again, for one
  commands.imaging.copairs_main (feature_space, covariate_set) run --
  colored by batch/plate instead of batch/condition, since copairs runs a
  single condition at a time. Also a thin wrapper over
  `make_batch_report_figures`.
- `make_covariate_comparison_figure`: one panel per feature space plotting
  post-residualization silhouette_batch/silhouette_condition/silhouette_plate
  across covariate sets, shading covariate sets that include "plate" --
  makes it visually obvious when adding plate over-corrects (condition
  silhouette drops below zero alongside batch silhouette, instead of batch
  dropping while condition is preserved).
- `make_local_mixing_comparison_figure`: companion to
  `make_covariate_comparison_figure` -- two rows (silhouette-scale metrics;
  0-1-scale local-mixing metrics) x one column per feature space, across
  covariate sets. Plots `silhouette_batch_stratified`/`silhouette_condition`
  and kBET acceptance/`ilisi` (imaging.batch_report), which stay sensitive
  to local batch sub-clustering that a pooled `silhouette_batch` can miss
  when a feature space separates biological conditions strongly (see
  imaging.batch_report's module docstring).
- `make_feature_space_comparison_figures`: one figure per feature space,
  combining `make_covariate_comparison_figure`'s panel (left) with
  `make_local_mixing_comparison_figure`'s two panels (right, stacked) for
  that space -- no plate-shading/legend/descriptive suptitle, since only one
  feature space's covariate-set axis is shown per figure.
- `make_tier_a_figure` / `make_tier_b_figure` / `make_tier_c_figure` /
  `make_tier_d_figure` / `make_tier_e_figure`: one figure per tier of
  experiments/benchmark_feature_representation.md, for
  make_benchmark_figures.py -- grouped bars, one color per representation
  (`REPRESENTATION_COLORS`, fixed order/hue across all four so the same
  representation reads as the same color everywhere).
- `make_proteomics_tier_e_figure`: `commands.proteomics.tier_e_main`'s
  E3-only analogue of `make_tier_e_figure` for the single-feature-space
  proteomics pipeline -- `processed_tag` (raw/nested/control_centered) plays
  the representation role, colored with a `tab10` palette rather than the
  fixed `REPRESENTATION_COLORS` mapping.

All are meant for direct use (e.g. called from run_pipeline.py right after
loading/residualizing) rather than as `__main__`-guarded scripts.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils.copairs import DEFAULT_CONSISTENCY_GROUPBY

try:
    import umap
except ImportError:
    umap = None

# Fixed categorical hue per representation, shared across every tier figure
# so the same representation reads as the same color everywhere (never
# reassigned/cycled by which representations happen to be present in a
# given run) -- slots 1-3 of the validated default palette (dataviz skill,
# references/palette.md), the only three that clear the CVD/normal-vision
# floors under all-pairs comparison (bars/small multiples, not just
# adjacent).
REPRESENTATION_COLORS = {
    "CellProfiler": "#2a78d6",  # blue
    "CPCNN": "#eb6834",  # orange
    "UniDino": "#1baf7a",  # aqua
}

# Shared text sizes, used both by the PCA/UMAP diagnostic panels
# (_categorical_panel, _continuous_panel, _scatter_grid,
# make_batch_effect_pca_figure) and by the copairs call-count/nMAP summary
# figures below (make_copairs_summary_figure, _draw_copairs_calls_grid,
# make_proteomics_copairs_summary_figure) -- kept as named constants, rather
# than inlined magic numbers, so every figure in the imaging/proteomics
# pipelines stays legible and consistent with each other when tuned.
PANEL_TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 10
SUPTITLE_FONTSIZE = 17

# Suptitle/legend vertical placement for the copairs summary figures (all
# figsize=(20, 13)): both sit just above the tight_layout-reserved band
# (GRID_TOP) instead of floating well above the figure's y=1.0 edge, which
# is what created a large dead-space gap between the legend/title block and
# the subplots below it. `fig.tight_layout(rect=[...])` must be called
# BEFORE `fig.legend`/`fig.suptitle` are added -- tight_layout also reserves
# extra room to avoid overlapping any figure-level artists already present,
# so adding the legend/suptitle first (as a naive reading of "legend then
# tight_layout" suggests) silently re-introduces a big gap above GRID_TOP.
SUPTITLE_Y = 1.0
LEGEND_Y = 0.93
GRID_TOP = 0.9

COV_LABELS = {
    "count": "Count",
    "count_batch": "Count + batch",
    "count_plate": "Count + plate",
    "count_batch_plate": "Count + batch + plate",
    "control_centered": "Control-centered (batch x condition)",
}
COV_COLORS = {
    "count": "#c9ccd1",
    "count_batch": "#5a5f66",
    "count_plate": "#7ec8f2",
    "count_batch_plate": "#1f77b4",
    "control_centered": "#ff7f0e",
}

CALLS = [
    ("activity", "Activity calls", "Same compound vs plate-matched DMSO controls", "Compounds"),
    (
        "distinctiveness",
        "Distinctiveness calls",
        "Same compound vs other {condition} compounds",
        "Compounds",
    ),
    ("consistency", "Consistency calls", "Same target vs different targets", "Target groups"),
]
# `make_proteomics_copairs_summary_figure` puts multiple conditions in ONE
# figure (unlike `make_copairs_summary_figure`, one figure per condition), so
# CALLS' distinctiveness subtitle -- the only one with a `{condition}` slot --
# can't be filled with a single condition name there; each x-axis group is
# already labeled with its own condition, so this static phrasing covers it
# without an awkward "FFA/IL6"-style join.
PROTEOMICS_CALL_SUBTITLE_OVERRIDES = {
    "distinctiveness": "Same compound vs other compounds in the same condition",
}
# Overrides for CALLS' consistency entry (subtitle, unit) when grouping by
# something other than the default Metadata_target.
CONSISTENCY_GROUPBY_LABELS = {
    "Metadata_moa": ("Same MoA vs different MoAs", "MoA groups"),
}


def _load_results(
    parquet_dir: Path,
    feature_spaces: list,
    covariate_sets: list,
    condition_tag: str,
    call_tags: dict,
) -> dict:
    results = {}
    for space in feature_spaces:
        for cov_key in covariate_sets:
            for call_name, *_ in CALLS:
                tag = call_tags.get(call_name, "")
                path = parquet_dir / f"{space}{condition_tag}_{cov_key}_{call_name}{tag}.parquet"
                results[(space, cov_key, call_name)] = pd.read_parquet(path)
    return results


def make_copairs_summary_figure(
    out_dir: Path,
    feature_spaces: list,
    covariate_sets: list,
    condition: str = "FFA",  # mirrors imaging.load.DEFAULT_CONDITION
    consistency_groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> Path:
    """Build a call-count/nMAP summary figure from this run's result
    parquets in `out_dir/parquet/`, saved to `out_dir/figures/` as
    `reproduced_figure.png` (or `reproduced_figure_<condition>.png` for a
    non-default condition, with an extra `_moa` suffix if consistency was
    grouped by Metadata_moa). Returns the saved path."""
    condition_tag = "" if condition == "FFA" else f"_{condition}"
    consistency_tag = "" if consistency_groupby == DEFAULT_CONSISTENCY_GROUPBY else "_moa"
    call_tags = {"consistency": consistency_tag}
    results = _load_results(
        out_dir / "parquet", feature_spaces, covariate_sets, condition_tag, call_tags
    )

    calls = [
        (call_name, title, *CONSISTENCY_GROUPBY_LABELS.get(consistency_groupby, (subtitle, unit)))
        if call_name == "consistency"
        else (call_name, title, subtitle, unit)
        for call_name, title, subtitle, unit in CALLS
    ]

    fig, axes = plt.subplots(2, len(calls), figsize=(20, 13))

    for col, (call_name, title, subtitle, unit) in enumerate(calls):
        ax_bar, ax_violin = axes[0, col], axes[1, col]
        n_total = len(results[(feature_spaces[0], covariate_sets[0], call_name)])

        group_width = 0.8
        n_cov = len(covariate_sets)
        bar_width = group_width / n_cov
        violin_positions = []
        violin_data = []
        violin_colors = []

        for space_ix, space in enumerate(feature_spaces):
            for cov_ix, cov_key in enumerate(covariate_sets):
                df = results[(space, cov_key, call_name)]
                n_calls = int(df["below_corrected_p"].sum())
                x = space_ix + (cov_ix - (n_cov - 1) / 2) * bar_width
                ax_bar.bar(
                    x, n_calls, width=bar_width * 0.95, color=COV_COLORS[cov_key]
                )
                ax_bar.text(
                    x, n_calls, str(n_calls), ha="center", va="bottom", fontsize=8
                )

                nmap = df["normalized_average_precision"].dropna().to_numpy()
                violin_positions.append(x)
                violin_data.append(nmap if len(nmap) > 0 else np.array([0.0]))
                violin_colors.append(COV_COLORS[cov_key])

        ax_bar.set_xticks(range(len(feature_spaces)))
        ax_bar.set_xticklabels(feature_spaces)
        ax_bar.set_title(f"{title}\n{subtitle.format(condition=condition)}", fontsize=PANEL_TITLE_FONTSIZE)
        ax_bar.set_ylabel(f"Significant {unit.lower()} (of {n_total:,})", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bar.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        parts = ax_violin.violinplot(
            violin_data, positions=violin_positions, widths=bar_width * 0.9,
            showmedians=True, showextrema=False,
        )
        for body, color in zip(parts["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.9)
        parts["cmedians"].set_color("black")

        ax_violin.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_violin.set_xticks(range(len(feature_spaces)))
        ax_violin.set_xticklabels(feature_spaces)
        ax_violin.set_ylabel(
            f"{unit[:-1]} nMAP" if unit.endswith("s") else "nMAP", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax_violin.set_ylim(-0.25, 1.05)
        ax_violin.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 1, GRID_TOP])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COV_COLORS[k]) for k in covariate_sets
    ]
    fig.legend(
        handles,
        [COV_LABELS[k] for k in covariate_sets],
        title="Variables in the Ridge model",
        loc="upper center",
        ncol=len(covariate_sets),
        bbox_to_anchor=(0.5, LEGEND_Y),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    fig.suptitle(
        "Plate residualization changes the number of statistically significant hits,\n"
        f"and mean normalized AP (reproduction from cpg0014 {condition} data)",
        fontsize=SUPTITLE_FONTSIZE,
        y=SUPTITLE_Y,
    )

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"reproduced_figure{condition_tag}{consistency_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_copairs_calls_grid(
    axes: np.ndarray,
    calls: list,
    results: dict,
    groups: list,
    hues: list,
    hue_colors: dict,
    group_labels: Optional[list] = None,
) -> None:
    """Shared call-count-bar/nMAP-violin drawing for
    `make_copairs_cross_condition_figure` and
    `make_copairs_feature_space_figure`: `results` maps
    `(group, hue, call_name) -> result dataframe`. `groups` are the x-axis
    tick groups and `hues` are the grouped/colored bars within each group --
    the same 3-column-of-(count-bar, nMAP-violin) layout as
    `make_copairs_summary_figure`, just parameterized over which axis plays
    the group/hue role (that function keeps its own inlined copy rather than
    calling this, since feature_space/covariate_set are always its
    group/hue -- this helper only exists for the two figures that need that
    swapped or fixed)."""
    group_labels = group_labels if group_labels is not None else [str(g) for g in groups]
    for col, (call_name, title, subtitle, unit) in enumerate(calls):
        ax_bar, ax_violin = axes[0, col], axes[1, col]
        n_total = len(results[(groups[0], hues[0], call_name)])

        group_width = 0.8
        n_hues = len(hues)
        bar_width = group_width / n_hues
        violin_positions, violin_data, violin_colors = [], [], []

        for group_ix, group in enumerate(groups):
            for hue_ix, hue in enumerate(hues):
                df = results[(group, hue, call_name)]
                n_calls = int(df["below_corrected_p"].sum())
                x = group_ix + (hue_ix - (n_hues - 1) / 2) * bar_width
                ax_bar.bar(x, n_calls, width=bar_width * 0.95, color=hue_colors[hue])
                ax_bar.text(x, n_calls, str(n_calls), ha="center", va="bottom", fontsize=9)

                nmap = df["normalized_average_precision"].dropna().to_numpy()
                violin_positions.append(x)
                violin_data.append(nmap if len(nmap) > 0 else np.array([0.0]))
                violin_colors.append(hue_colors[hue])

        ax_bar.set_xticks(range(len(groups)))
        ax_bar.set_xticklabels(group_labels)
        ax_bar.set_title(f"{title}\n{subtitle}", fontsize=PANEL_TITLE_FONTSIZE)
        ax_bar.set_ylabel(f"Significant {unit.lower()} (of {n_total:,})", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bar.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        parts = ax_violin.violinplot(
            violin_data, positions=violin_positions, widths=bar_width * 0.9,
            showmedians=True, showextrema=False,
        )
        for body, color in zip(parts["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.9)
        parts["cmedians"].set_color("black")

        ax_violin.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_violin.set_xticks(range(len(groups)))
        ax_violin.set_xticklabels(group_labels)
        ax_violin.set_ylabel(
            f"{unit[:-1]} nMAP" if unit.endswith("s") else "nMAP", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax_violin.set_ylim(-0.25, 1.05)
        ax_violin.tick_params(labelsize=TICK_LABEL_FONTSIZE)


def _copairs_calls(consistency_groupby: str) -> list:
    """`CALLS`, with the consistency title/unit swapped per
    `CONSISTENCY_GROUPBY_LABELS` and the distinctiveness subtitle's
    `{condition}` slot filled with `PROTEOMICS_CALL_SUBTITLE_OVERRIDES`'s
    condition-agnostic phrasing -- shared by the two multi-condition
    figures below, neither of which has a single `condition` to fill that
    slot with (unlike `make_copairs_summary_figure`)."""
    return [
        (call_name, title, *CONSISTENCY_GROUPBY_LABELS.get(consistency_groupby, (subtitle, unit)))
        if call_name == "consistency"
        else (call_name, title, PROTEOMICS_CALL_SUBTITLE_OVERRIDES.get(call_name, subtitle), unit)
        for call_name, title, subtitle, unit in CALLS
    ]


def make_copairs_cross_condition_figure(
    out_dirs: dict,
    feature_spaces: list,
    covariate_set: str,
    save_dir: Path,
    consistency_groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> Path:
    """Cross-condition companion to `make_copairs_summary_figure`: ONE
    figure covering every condition in `out_dirs` (`{condition: out_dir}` --
    each condition's copairs run lives in its own `out_dir`, e.g.
    results/imaging/copairs_v2/processed/<condition>/, since
    `commands.imaging.copairs_main` runs one condition at a time), fixed to
    a single `covariate_set` (pass the "best" residualization method, e.g.
    "count_batch_plate") so the comparison isolates the condition-to-condition
    axis instead of also varying by covariate set. Feature spaces are the
    x-axis groups (as in `make_copairs_summary_figure`) and conditions are
    the grouped/colored bars within each group.

    Reads each condition's already-computed parquets under
    `out_dirs[condition] / "parquet"` -- does not re-run copairs. Saved to
    `save_dir/reproduced_figure_all_conditions_<covariate_set>.png`."""
    conditions = list(out_dirs)
    consistency_tag = "" if consistency_groupby == DEFAULT_CONSISTENCY_GROUPBY else "_moa"
    call_tags = {"consistency": consistency_tag}
    calls = _copairs_calls(consistency_groupby)

    results = {}
    for condition in conditions:
        condition_tag = "" if condition == "FFA" else f"_{condition}"
        for space in feature_spaces:
            for call_name, *_ in CALLS:
                tag = call_tags.get(call_name, "")
                path = (
                    out_dirs[condition] / "parquet"
                    / f"{space}{condition_tag}_{covariate_set}_{call_name}{tag}.parquet"
                )
                results[(space, condition, call_name)] = pd.read_parquet(path)

    palette = plt.get_cmap("tab10").colors
    condition_colors = {c: palette[i % len(palette)] for i, c in enumerate(conditions)}

    fig, axes = plt.subplots(2, len(calls), figsize=(20, 13))
    _draw_copairs_calls_grid(
        axes, calls, results, feature_spaces, conditions, condition_colors,
        group_labels=feature_spaces,
    )

    fig.tight_layout(rect=[0, 0, 1, GRID_TOP])

    handles = [plt.Rectangle((0, 0), 1, 1, color=condition_colors[c]) for c in conditions]
    fig.legend(
        handles, conditions, title="Condition", loc="upper center",
        ncol=len(conditions), bbox_to_anchor=(0.5, LEGEND_Y), frameon=False,
        fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE,
    )
    fig.suptitle(
        "Statistically significant hits and mean normalized AP across conditions\n"
        f"({COV_LABELS[covariate_set]} residualization, all feature spaces)",
        fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"reproduced_figure_all_conditions_{covariate_set}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_copairs_feature_space_figure(
    out_dirs: dict,
    feature_space: str,
    covariate_sets: list,
    save_dir: Path,
    consistency_groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> Path:
    """Single-feature-space companion to `make_copairs_summary_figure`: ONE
    figure covering every condition in `out_dirs` (`{condition: out_dir}`)
    for one `feature_space` (e.g. "CellProfiler"), with conditions as the
    x-axis groups and Ridge covariate sets as the grouped/colored bars
    within each group -- the mirror image of
    `make_copairs_cross_condition_figure` (which fixes the covariate set
    and varies feature space instead).

    Reads each condition's already-computed parquets under
    `out_dirs[condition] / "parquet"` -- does not re-run copairs. Saved to
    `save_dir/reproduced_figure_<feature_space>_all_conditions.png`."""
    conditions = list(out_dirs)
    consistency_tag = "" if consistency_groupby == DEFAULT_CONSISTENCY_GROUPBY else "_moa"
    call_tags = {"consistency": consistency_tag}
    calls = _copairs_calls(consistency_groupby)

    results = {}
    for condition in conditions:
        condition_tag = "" if condition == "FFA" else f"_{condition}"
        for cov_key in covariate_sets:
            for call_name, *_ in CALLS:
                tag = call_tags.get(call_name, "")
                path = (
                    out_dirs[condition] / "parquet"
                    / f"{feature_space}{condition_tag}_{cov_key}_{call_name}{tag}.parquet"
                )
                results[(condition, cov_key, call_name)] = pd.read_parquet(path)

    fig, axes = plt.subplots(2, len(calls), figsize=(20, 13))
    _draw_copairs_calls_grid(
        axes, calls, results, conditions, covariate_sets, COV_COLORS, group_labels=conditions,
    )

    fig.tight_layout(rect=[0, 0, 1, GRID_TOP])

    handles = [plt.Rectangle((0, 0), 1, 1, color=COV_COLORS[k]) for k in covariate_sets]
    fig.legend(
        handles, [COV_LABELS[k] for k in covariate_sets], title="Variables in the Ridge model",
        loc="upper center", ncol=len(covariate_sets), bbox_to_anchor=(0.5, LEGEND_Y), frameon=False,
        fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE,
    )
    fig.suptitle(
        f"{feature_space}: statistically significant hits and mean normalized AP\n"
        "across conditions and Ridge covariate sets",
        fontsize=SUPTITLE_FONTSIZE, y=SUPTITLE_Y,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"reproduced_figure_{feature_space}_all_conditions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_proteomics_copairs_summary_figure(
    out_dir: Path,
    conditions: list,
    processed_tags: list,
    consistency_groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> Path:
    """`run_proteomics_copairs.py`'s analogue of
    `make_copairs_summary_figure`: proteomics has only one "feature space",
    so `processed_tags` (e.g. `"raw"` vs a `"<method>_<covariates>"`
    corrected tag) plays the covariate_sets role instead, grouped/colored
    per bar, with `conditions` (e.g. FFA/IL6) on the x-axis in place of
    feature_spaces. Reads
    `out_dir/parquet/proteomics_<condition>_<tag>_<call>[_moa].parquet` --
    written by `run_proteomics_copairs.py`, one file per (condition,
    processed_tag, call). A single `processed_tags` entry (the common case:
    one Hail Batch job only computes one processed state) still renders
    fine, just without a raw-vs-corrected comparison. Saved to
    `out_dir/figures/proteomics_reproduced_figure_<tags>.png`."""
    consistency_tag = "" if consistency_groupby == DEFAULT_CONSISTENCY_GROUPBY else "_moa"
    call_tags = {"consistency": consistency_tag}
    calls = [
        (call_name, title, *CONSISTENCY_GROUPBY_LABELS.get(consistency_groupby, (subtitle, unit)))
        if call_name == "consistency"
        else (call_name, title, PROTEOMICS_CALL_SUBTITLE_OVERRIDES.get(call_name, subtitle), unit)
        for call_name, title, subtitle, unit in CALLS
    ]

    results = {}
    for condition in conditions:
        for tag in processed_tags:
            for call_name, *_ in CALLS:
                ctag = call_tags.get(call_name, "")
                path = out_dir / "parquet" / f"proteomics_{condition}_{tag}_{call_name}{ctag}.parquet"
                results[(condition, tag, call_name)] = pd.read_parquet(path)

    palette = plt.get_cmap("tab10").colors
    tag_colors = {tag: palette[i % len(palette)] for i, tag in enumerate(processed_tags)}

    fig, axes = plt.subplots(2, len(calls), figsize=(20, 13))

    for col, (call_name, title, subtitle, unit) in enumerate(calls):
        ax_bar, ax_violin = axes[0, col], axes[1, col]
        # Each condition has its own compound/target universe (unlike
        # make_copairs_summary_figure's single shared `condition`), but that
        # universe doesn't depend on `tag` (same panel, just corrected
        # differently) -- so one "(n=...)" per condition, on the x-tick,
        # mirrors the reference figure's plain-count bar labels + single
        # shared "called out of N" denominator instead of cluttering every
        # bar with its own "x/y" fraction.
        n_per_condition = [len(results[(c, processed_tags[0], call_name)]) for c in conditions]

        group_width = 0.8
        n_tags = len(processed_tags)
        bar_width = group_width / n_tags
        violin_positions = []
        violin_data = []
        violin_colors = []

        for cond_ix, condition in enumerate(conditions):
            for tag_ix, tag in enumerate(processed_tags):
                df = results[(condition, tag, call_name)]
                n_calls = int(df["below_corrected_p"].sum())
                x = cond_ix + (tag_ix - (n_tags - 1) / 2) * bar_width
                ax_bar.bar(x, n_calls, width=bar_width * 0.95, color=tag_colors[tag])
                ax_bar.text(
                    x, n_calls, str(n_calls), ha="center", va="bottom", fontsize=8,
                )

                nmap = df["normalized_average_precision"].dropna().to_numpy()
                violin_positions.append(x)
                violin_data.append(nmap if len(nmap) > 0 else np.array([0.0]))
                violin_colors.append(tag_colors[tag])

        ax_bar.set_xticks(range(len(conditions)))
        ax_bar.set_xticklabels(
            [f"{c}\n(out of {n:,})" for c, n in zip(conditions, n_per_condition)]
        )
        ax_bar.set_title(f"{title}\n{subtitle}", fontsize=PANEL_TITLE_FONTSIZE)
        ax_bar.set_ylabel(f"Significant {unit.lower()}", fontsize=AXIS_LABEL_FONTSIZE)
        ax_bar.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        parts = ax_violin.violinplot(
            violin_data, positions=violin_positions, widths=bar_width * 0.9,
            showmedians=True, showextrema=False,
        )
        for body, color in zip(parts["bodies"], violin_colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.9)
        parts["cmedians"].set_color("black")

        ax_violin.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_violin.set_xticks(range(len(conditions)))
        ax_violin.set_xticklabels(conditions)
        ax_violin.set_ylabel(
            f"{unit[:-1]} nMAP" if unit.endswith("s") else "nMAP", fontsize=AXIS_LABEL_FONTSIZE
        )
        ax_violin.set_ylim(-0.25, 1.05)
        ax_violin.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    fig.tight_layout(rect=[0, 0, 1, GRID_TOP])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=tag_colors[t]) for t in processed_tags
    ]
    fig.legend(
        handles,
        processed_tags,
        title="Processed state",
        loc="upper center",
        ncol=len(processed_tags),
        bbox_to_anchor=(0.5, LEGEND_Y),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_FONTSIZE,
    )
    fig.suptitle(
        "Batch/plate correction changes the number of statistically significant hits,\n"
        "and mean normalized AP (proteomics copairs)",
        fontsize=SUPTITLE_FONTSIZE,
        y=SUPTITLE_Y,
    )

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"proteomics_reproduced_figure_{'_'.join(processed_tags)}{consistency_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path




def _categorical_panel(
    ax, coords: np.ndarray, labels: pd.Series, title: str, xlabel: str, ylabel: str
) -> None:
    """Scatter `coords` colored by a categorical `labels` series (tab20 up
    to 20 categories, else viridis), with a legend when there are few
    enough categories to fit one legibly."""
    categories = labels.astype("category")
    codes = categories.cat.codes.to_numpy()
    n_cat = max(len(categories.cat.categories), 1)
    cmap = plt.get_cmap("tab20" if n_cat <= 20 else "viridis")
    ax.scatter(
        coords[:, 0], coords[:, 1], c=codes, cmap=cmap, s=6, alpha=0.7,
        vmin=0, vmax=max(n_cat - 1, 1),
    )
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    if n_cat <= 12:
        handles = [
            plt.Line2D(
                [], [], marker="o", linestyle="",
                color=cmap(i / max(n_cat - 1, 1)), label=str(cat),
            )
            for i, cat in enumerate(categories.cat.categories)
        ]
        ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="best")


def _continuous_panel(
    fig: plt.Figure, ax, coords: np.ndarray, values: np.ndarray,
    title: str, xlabel: str, ylabel: str, cmap: str = "viridis",
) -> None:
    """Scatter `coords` colored by a continuous `values` array, with a
    colorbar -- the continuous-covariate counterpart to `_categorical_panel`
    (e.g. cell count, where a discrete legend doesn't make sense)."""
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=6, alpha=0.7)
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)


def _scatter_grid(
    coords_before: np.ndarray,
    coords_after: np.ndarray,
    meta: pd.DataFrame,
    batch_col: str,
    condition_col: str,
    title: str,
    axis_labels: Optional[list] = None,
) -> plt.Figure:
    """2x2 grid: rows = before/after residualization, columns = colored by
    batch / colored by condition. `axis_labels`, if given, is
    `[(xlabel_before, ylabel_before), (xlabel_after, ylabel_after)]` -- one
    (x, y) label pair per row, e.g. PCA's own "PC1 (12.3% var)" (each row is
    its own PCA fit, so before/after have different explained-variance
    ratios). Defaults to "Dim 1"/"Dim 2" for embeddings with no comparable
    per-axis quantity (e.g. UMAP)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rows = [("Before residualization", coords_before), ("After residualization", coords_after)]
    cols = [batch_col, condition_col]
    for r, (row_label, coords) in enumerate(rows):
        xlabel, ylabel = axis_labels[r] if axis_labels is not None else ("Dim 1", "Dim 2")
        for c, col in enumerate(cols):
            _categorical_panel(
                axes[r, c], coords, meta[col], f"{row_label}\ncolored by {col}", xlabel, ylabel
            )
    fig.suptitle(title, fontsize=SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def make_batch_report_figures(
    feats_before: np.ndarray,
    feats_after: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    file_stub: str,
    title_prefix: str,
    batch_col: str = "Metadata_batch",
    condition_col: str = "Metadata_condition",
    seed: int = 0,
) -> None:
    """Save a PCA `<file_stub>_pca.png` (always) and UMAP `_umap.png` (if
    umap-learn is installed) 2x2 grid -- rows before/after residualization,
    columns colored by `batch_col`/`condition_col` -- under `out_dir`, from
    `feats_before`/`feats_after` (already row-aligned with `meta`; the
    caller decides how large a sample to pass in)."""
    pca_model_before = PCA(n_components=2, random_state=seed).fit(feats_before)
    pca_model_after = PCA(n_components=2, random_state=seed).fit(feats_after)
    pca_before = pca_model_before.transform(feats_before)
    pca_after = pca_model_after.transform(feats_after)

    def _pc_labels(model: PCA) -> tuple:
        var = model.explained_variance_ratio_
        return f"PC1 ({var[0]:.1%} var)", f"PC2 ({var[1]:.1%} var)"

    fig = _scatter_grid(
        pca_before, pca_after, meta, batch_col, condition_col, title=f"{title_prefix} -- PCA",
        axis_labels=[_pc_labels(pca_model_before), _pc_labels(pca_model_after)],
    )
    fig.savefig(out_dir / f"{file_stub}_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if umap is not None:
        umap_before = umap.UMAP(n_components=2, random_state=seed).fit_transform(feats_before)
        umap_after = umap.UMAP(n_components=2, random_state=seed).fit_transform(feats_after)
        fig = _scatter_grid(
            umap_before, umap_after, meta, batch_col, condition_col, title=f"{title_prefix} -- UMAP",
            axis_labels=[("UMAP 1", "UMAP 2"), ("UMAP 1", "UMAP 2")],
        )
        fig.savefig(out_dir / f"{file_stub}_umap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("umap-learn not installed; skipping UMAP plot", flush=True)


def make_batch_effect_pca_figure(
    feats: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    file_stub: str,
    title_prefix: str,
    count_col: str = "Metadata_cell_count",
    plate_col: str = "Metadata_Plate",
    batch_col: str = "Metadata_batch",
    seed: int = 0,
) -> Path:
    """Single-PCA-fit, 3-panel snapshot of the batch effects already
    present in `feats` (raw, pre-residualization) -- colored by cell count
    (continuous), plate, and batch. Unlike `make_batch_report_figures`'
    before/after grid, this doesn't depend on a residualization method, so
    callers compute it once per feature space rather than once per
    (feature_space, method) pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_model = PCA(n_components=2, random_state=seed).fit(feats)
    coords = pca_model.transform(feats)
    var = pca_model.explained_variance_ratio_
    xlabel, ylabel = f"PC1 ({var[0]:.1%} var)", f"PC2 ({var[1]:.1%} var)"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _continuous_panel(
        fig, axes[0], coords, meta[count_col].to_numpy(dtype=float),
        f"Colored by {count_col}", xlabel, ylabel,
    )
    _categorical_panel(axes[1], coords, meta[plate_col], f"Colored by {plate_col}", xlabel, ylabel)
    _categorical_panel(axes[2], coords, meta[batch_col], f"Colored by {batch_col}", xlabel, ylabel)
    fig.suptitle(
        f"{title_prefix} -- PCA batch-effect snapshot (raw features)",
        fontsize=SUPTITLE_FONTSIZE,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = out_dir / f"{file_stub}_pca.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


REVERSION_FIGURE_SAMPLE_SIZE = 5000


def _subsample_index(n: int, sample_size: Optional[int], seed: int) -> np.ndarray:
    """Row indices for a plotting subsample -- all rows if `n <= sample_size`
    (or `sample_size` is None), else a fixed-seed draw without replacement."""
    if sample_size is None or n <= sample_size:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample_size, replace=False)


def make_reversion_diagnostic_figures(
    feats_before: np.ndarray,
    feats_after: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    file_stub: str,
    title_prefix: str,
    batch_col: str = "Metadata_batch",
    condition_col: str = "Metadata_condition",
    sample_size: Optional[int] = REVERSION_FIGURE_SAMPLE_SIZE,
    seed: int = 0,
) -> None:
    """PCA/UMAP before-vs-after-residualization diagnostic for one jointly
    loaded Baseline+stress reversion run
    (`imaging.reversion.load_joint_residualized`'s `feats_before`/
    `feats_after`, row-aligned with `meta`): does the residualization step
    remove batch separation while leaving the Baseline-vs-stress condition
    separation (the reversion axis itself) intact? Subsamples to
    `sample_size` rows first (fixed seed, same convention as
    `imaging.batch_report.compute_report`) since this is a quick visual
    sanity check, not a precise embedding, and delegates to
    `make_batch_report_figures` for the actual 2x2 PCA/UMAP grids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _subsample_index(len(feats_before), sample_size, seed)
    meta_sample = meta.iloc[idx].reset_index(drop=True)
    make_batch_report_figures(
        feats_before[idx],
        feats_after[idx],
        meta_sample,
        out_dir,
        file_stub,
        title_prefix,
        batch_col=batch_col,
        condition_col=condition_col,
        seed=seed,
    )


def make_copairs_pc_figure(
    feats_before: np.ndarray,
    feats_after: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    file_stub: str,
    title_prefix: str,
    batch_col: str = "Metadata_batch",
    plate_col: str = "Metadata_Plate",
    sample_size: Optional[int] = REVERSION_FIGURE_SAMPLE_SIZE,
    seed: int = 0,
) -> None:
    """PCA before-vs-after-residualization diagnostic for one
    `imaging.commands.copairs_main` (feature_space, covariate_set) run:
    does residualization remove batch structure? `commands.imaging.copairs_main`
    runs a single Metadata_condition at a time, so unlike
    `make_reversion_diagnostic_figures` (jointly loaded Baseline+stress,
    colored by batch/condition) there's no condition contrast to show here
    -- the second column is colored by `plate_col` instead, since that's
    the unit every Ridge covariate set actually corrects (see
    `imaging.batch_report`'s silhouette_plate). Subsamples to
    `sample_size` rows first, same convention as
    `make_reversion_diagnostic_figures`, and delegates to
    `make_batch_report_figures` for the actual 2x2 PCA/UMAP grids."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _subsample_index(len(feats_before), sample_size, seed)
    meta_sample = meta.iloc[idx].reset_index(drop=True)
    make_batch_report_figures(
        feats_before[idx],
        feats_after[idx],
        meta_sample,
        out_dir,
        file_stub,
        title_prefix,
        batch_col=batch_col,
        condition_col=plate_col,
        seed=seed,
    )


def make_covariate_comparison_figure(
    metrics_by_space: dict,
    out_dir: Path,
    covariate_sets: list,
    extra_methods: list = (),
) -> Path:
    """One panel per feature space: post-residualization silhouette_batch,
    silhouette_condition and silhouette_plate across `covariate_sets`, with
    covariate sets whose name contains "plate" shaded. A covariate set
    over-corrects when its silhouette_condition line dips below zero
    alongside silhouette_batch -- real condition signal getting washed out
    along with batch/plate noise -- rather than batch dropping while
    condition stays flat or improves. silhouette_plate is inflated by
    condition itself (every plate is single-condition), so read it relative
    to silhouette_condition rather than against zero: converging toward the
    condition line means within-condition plate drift was absorbed, staying
    well above it means plate structure survives within a condition.
    """
    methods = list(covariate_sets) + list(extra_methods)
    feature_spaces = list(metrics_by_space)
    fig, axes = plt.subplots(
        1, len(feature_spaces), figsize=(5 * len(feature_spaces), 5), sharey=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(methods))

    for ax, space in zip(axes, feature_spaces):
        metrics = metrics_by_space[space]
        batch_after = [metrics[m]["after"]["silhouette_batch"] for m in methods]
        cond_after = [metrics[m]["after"]["silhouette_condition"] for m in methods]
        plate_after = [metrics[m]["after"]["silhouette_plate"] for m in methods]

        for i, cov_key in enumerate(covariate_sets):
            if "plate" in cov_key:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4b6b6", alpha=0.4, zorder=0)
        if extra_methods:
            ax.axvline(len(covariate_sets) - 0.5, color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        ax.plot(x, batch_after, marker="o", color="#1f77b4", label="silhouette_batch (after)")
        ax.plot(x, cond_after, marker="o", color="#d62728", label="silhouette_condition (after)")
        ax.plot(
            x, plate_after, marker="^", color="#2ca02c", linestyle="--",
            label="silhouette_plate (after)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_xlim(-0.5, len(methods) - 0.5)
        ax.set_title(space)

    axes[0].set_ylabel("Silhouette score (after residualization)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#f4b6b6", alpha=0.4))
    labels.append("covariate set includes plate")
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False
    )
    fig.suptitle(
        "Batch correction by method -- shaded sets over-correct when\n"
        "silhouette_condition drops below 0 along with silhouette_batch",
        fontsize=12, y=1.18,
    )
    fig.tight_layout()

    out_path = out_dir / "covariate_set_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_local_mixing_comparison_figure(
    metrics_by_space: dict,
    out_dir: Path,
    covariate_sets: list,
    extra_methods: list = (),
) -> Path:
    """Companion to `make_covariate_comparison_figure`: post-residualization
    local-mixing metrics (imaging.batch_report) across `covariate_sets`, one
    column per feature space, covariate sets whose name contains "plate"
    shaded (same convention as the sibling figure).

    Two rows, kept on separate axes because the metrics don't share a
    scale:
    - top: `silhouette_batch_stratified` and `silhouette_condition`
      (silhouette scale, [-1, 1]) -- the condition-stratified counterpart
      to `make_covariate_comparison_figure`'s pooled `silhouette_batch`,
      immune to a large between-condition gap masking local batch
      sub-clustering.
    - bottom: kBET acceptance rate (`1 - kbet_rejection_rate`) and `ilisi`
      (both [0, 1], higher = better mixed) -- single-cell metrics computed
      from each point's nearest neighbors only.

    A method that pushes stratified batch silhouette down *and* condition
    silhouette down (top row), or kBET/iLISI up while `clisi` (not plotted
    here, see the saved metrics JSON) also rises, washed out real
    condition signal along with batch noise rather than cleanly removing
    batch.
    """
    methods = list(covariate_sets) + list(extra_methods)
    feature_spaces = list(metrics_by_space)
    fig, axes = plt.subplots(
        2, len(feature_spaces), figsize=(5 * len(feature_spaces), 9),
        sharex=True, sharey="row",
    )
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(2, -1)
    x = np.arange(len(methods))

    for col, space in enumerate(feature_spaces):
        metrics = metrics_by_space[space]
        strat_batch = [metrics[m]["after"]["silhouette_batch_stratified"] for m in methods]
        cond_after = [metrics[m]["after"]["silhouette_condition"] for m in methods]
        kbet_accept = [1.0 - metrics[m]["after"]["kbet_rejection_rate"] for m in methods]
        ilisi_after = [metrics[m]["after"]["ilisi"] for m in methods]

        top, bottom = axes[0, col], axes[1, col]
        for ax in (top, bottom):
            for i, cov_key in enumerate(covariate_sets):
                if "plate" in cov_key:
                    ax.axvspan(i - 0.5, i + 0.5, color="#f4b6b6", alpha=0.4, zorder=0)
            if extra_methods:
                ax.axvline(len(covariate_sets) - 0.5, color="black", linestyle=":", linewidth=1)

        top.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        top.plot(
            x, strat_batch, marker="o", color="#1f77b4",
            label="silhouette_batch_stratified (after)",
        )
        top.plot(
            x, cond_after, marker="o", color="#d62728",
            label="silhouette_condition (after)",
        )
        top.set_title(space)

        bottom.set_ylim(-0.05, 1.05)
        bottom.plot(
            x, kbet_accept, marker="s", color="#9467bd",
            label="kBET acceptance rate (after)",
        )
        bottom.plot(
            x, ilisi_after, marker="^", color="#ff7f0e", linestyle="--",
            label="iLISI (after)",
        )
        bottom.set_xticks(x)
        bottom.set_xticklabels(methods, rotation=30, ha="right")
        bottom.set_xlim(-0.5, len(methods) - 0.5)

    axes[0, 0].set_ylabel("Silhouette score (after residualization)")
    axes[1, 0].set_ylabel("Local mixing score (after residualization)")

    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[1, 0].get_legend_handles_labels()
    handles = top_handles + bottom_handles
    labels = top_labels + bottom_labels
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#f4b6b6", alpha=0.4))
    labels.append("covariate set includes plate")
    fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), frameon=False
    )
    fig.suptitle(
        "Local-neighborhood mixing by method -- immune to a large\n"
        "between-condition gap masking local batch sub-clustering",
        fontsize=12, y=1.1,
    )
    fig.tight_layout()

    out_path = out_dir / "local_mixing_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_feature_space_comparison_figures(
    metrics_by_space: dict,
    out_dir: Path,
    covariate_sets: list,
    extra_methods: list = (),
) -> list:
    """One figure per feature space, combining that space's panel from
    `make_covariate_comparison_figure` (left) with its two panels from
    `make_local_mixing_comparison_figure` (right, stacked) -- lets a single
    feature space's full metric set be viewed without the other spaces'
    panels crowding the figure. Unlike the two sibling comparison figures,
    this drops the "covariate set includes plate" shading/legend entry and
    the descriptive suptitle, since only one covariate-set axis is shown per
    figure and that context isn't needed here.
    """
    methods = list(covariate_sets) + list(extra_methods)
    x = np.arange(len(methods))
    out_paths = []

    for space, metrics in metrics_by_space.items():
        batch_after = [metrics[m]["after"]["silhouette_batch"] for m in methods]
        cond_after = [metrics[m]["after"]["silhouette_condition"] for m in methods]
        plate_after = [metrics[m]["after"]["silhouette_plate"] for m in methods]
        strat_batch = [metrics[m]["after"]["silhouette_batch_stratified"] for m in methods]
        kbet_accept = [1.0 - metrics[m]["after"]["kbet_rejection_rate"] for m in methods]
        ilisi_after = [metrics[m]["after"]["ilisi"] for m in methods]

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
        left = fig.add_subplot(gs[:, 0])
        top_right = fig.add_subplot(gs[0, 1])
        bottom_right = fig.add_subplot(gs[1, 1], sharex=top_right)

        for ax in (left, top_right):
            if extra_methods:
                ax.axvline(len(covariate_sets) - 0.5, color="black", linestyle=":", linewidth=1)
        if extra_methods:
            bottom_right.axvline(
                len(covariate_sets) - 0.5, color="black", linestyle=":", linewidth=1
            )

        left.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        left.plot(x, batch_after, marker="o", color="#1f77b4", label="silhouette_batch")
        left.plot(x, cond_after, marker="o", color="#d62728", label="silhouette_condition")
        left.plot(
            x, plate_after, marker="^", color="#2ca02c", linestyle="--",
            label="silhouette_plate",
        )
        left.set_xticks(x)
        left.set_xticklabels(methods, rotation=30, ha="right")
        left.set_xlim(-0.5, len(methods) - 0.5)
        left.set_ylabel("Silhouette score (after residualization)")
        left.set_title("Covariate-set comparison")
        left.legend(frameon=False)

        top_right.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        top_right.plot(
            x, strat_batch, marker="o", color="#1f77b4",
            label="silhouette_batch_stratified",
        )
        top_right.plot(
            x, cond_after, marker="o", color="#d62728", label="silhouette_condition",
        )
        top_right.set_ylabel("Silhouette score (after residualization)")
        top_right.set_title("Local mixing comparison")
        top_right.legend(frameon=False)
        plt.setp(top_right.get_xticklabels(), visible=False)

        bottom_right.set_ylim(-0.05, 1.05)
        bottom_right.plot(
            x, kbet_accept, marker="s", color="#9467bd", label="kBET acceptance rate",
        )
        bottom_right.plot(
            x, ilisi_after, marker="^", color="#ff7f0e", linestyle="--", label="iLISI",
        )
        bottom_right.set_xticks(x)
        bottom_right.set_xticklabels(methods, rotation=30, ha="right")
        bottom_right.set_xlim(-0.5, len(methods) - 0.5)
        bottom_right.set_ylabel("Local mixing score (after residualization)")
        bottom_right.legend(frameon=False)

        fig.suptitle(space, fontsize=SUPTITLE_FONTSIZE)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = out_dir / f"{space}_comparison.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def _grouped_bars(
    ax, categories: list, representations: list, values: dict,
    value_fmt: str = "{:.2f}", ylim: Optional[tuple] = None,
    colors: Optional[dict] = None,
) -> None:
    """One bar per (category, representation), colored by `colors` (default
    `REPRESENTATION_COLORS`; pass a different {key: color} mapping for an
    axis whose groups aren't CellProfiler/CPCNN/UniDino -- e.g. proteomics'
    processed_tag axis). `values[rep][category]` is the bar height;
    missing/NaN is drawn as a zero-height bar labeled "n/a" rather than
    silently omitted -- e.g. too few reversion nominees to compute a
    stability/effect-size estimate is itself informative, not a zero.
    `ylim`, if given, is applied BEFORE placing value labels (not after --
    labels offset from an autoscaled range that a caller then zooms
    end up floating outside the visible axes)."""
    colors = REPRESENTATION_COLORS if colors is None else colors
    n_reps = len(representations)
    width = 0.8 / n_reps
    x = np.arange(len(categories))
    bar_groups = []
    for i, rep in enumerate(representations):
        raw = [values[rep].get(cat, float("nan")) for cat in categories]
        heights = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in raw]
        offsets = x + (i - (n_reps - 1) / 2) * width
        bars = ax.bar(
            offsets, heights, width=width * 0.9,
            color=colors.get(rep, "#888888"), label=rep, zorder=3,
        )
        bar_groups.append((bars, raw))

    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylim is not None:
        ax.set_ylim(*ylim)

    y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-6)
    for bars, raw in bar_groups:
        for bar, v in zip(bars, raw):
            is_na = v is None or (isinstance(v, float) and np.isnan(v))
            label = "n/a" if is_na else value_fmt.format(v)
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            offset = (0.015 if va == "bottom" else -0.015) * y_span
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + offset, label,
                ha="center", va=va, fontsize=7, color="#0b0b0b",
            )


def make_tier_a_figure(a1_metrics_by_space: dict, out_dir: Path) -> Path:
    """Tier A1 -- technical quality: silhouette_batch/plate/condition before
    vs. after residualization, native dimension, per representation
    (imaging.batch_report, run via run_cellrep_benchmark.py's `_run_tier_a1`).
    Batch/plate should drop toward/below zero after correction; condition
    should survive or improve -- see docs/batch_effect_conclusions.md."""
    representations = list(a1_metrics_by_space)
    categories = ["Batch", "Plate", "Condition"]
    metric_keys = ["silhouette_batch", "silhouette_plate", "silhouette_condition"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, stage, title in zip(axes, ("before", "after"), ("Before correction", "After correction")):
        values = {
            rep: {cat: a1_metrics_by_space[rep][stage][key] for cat, key in zip(categories, metric_keys)}
            for rep in representations
        }
        _grouped_bars(ax, categories, representations, values)
        ax.set_title(title)
    axes[0].set_ylabel("Silhouette score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(representations),
        bbox_to_anchor=(0.5, 1.05), frameon=False,
    )
    fig.suptitle(
        "Tier A -- technical quality: batch/plate silhouette before vs. after\n"
        "residualization (native dimension)",
        fontsize=12, y=1.18,
    )
    fig.tight_layout()

    out_path = out_dir / "tier_a_technical_quality.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_tier_b_figure(summary_rows: pd.DataFrame, out_dir: Path) -> Path:
    """Tier B1 -- condition separability: Baseline-vs-stress linear-probe
    AUROC per representation per condition, on the matched-dimension
    features (imaging.benchmark.condition_separability). A space that fails
    this can't produce a meaningful reversion signal regardless of
    downstream (Tier C) results -- y-axis is zoomed since all three
    routinely sit near ceiling."""
    representations = sorted(summary_rows["representation"].unique())
    conditions = list(dict.fromkeys(summary_rows["condition"]))
    values = {
        rep: dict(
            zip(
                summary_rows.loc[summary_rows["representation"] == rep, "condition"],
                summary_rows.loc[summary_rows["representation"] == rep, "condition_separability_auroc"],
            )
        )
        for rep in representations
    }

    all_vals = [v for rep in values.values() for v in rep.values() if v is not None and not np.isnan(v)]
    lo = min(all_vals) if all_vals else 0.9
    ylim = (max(0, lo - 0.02), 1.005)

    fig, ax = plt.subplots(figsize=(2.2 * len(conditions) + 2, 4.5))
    _grouped_bars(ax, conditions, representations, values, value_fmt="{:.3f}", ylim=ylim)
    ax.set_ylabel("Baseline-vs-stress AUROC")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(representations),
        bbox_to_anchor=(0.5, 1.05), frameon=False,
    )
    fig.suptitle("Tier B1 -- condition separability (matched-dimension features)", fontsize=12, y=1.16)
    fig.tight_layout()

    out_path = out_dir / "tier_b_condition_separability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_tier_c_figure(summary_rows: pd.DataFrame, out_dir: Path) -> Path:
    """Tier C -- downstream, decision-relevant performance: C1 (reversion
    hit count), C2 (leave-one-out replicate-split stability, imaging.
    benchmark.replicate_split_stability -- both the binary hit-list Jaccard,
    a high-confidence-floor stress test, and the continuous Spearman
    correlation of RI_spec between folds, which isn't sensitive to
    threshold-cliff near-misses; see that function's docstring) and C3
    (cross-representation effect size, imaging.benchmark.
    cross_representation_effect_size -- does a representation's hits also
    score highly on a DIFFERENT representation's independently-computed
    reversion score; the same-representation imaging.benchmark.
    effect_size_separation is circular here, see its docstring, and is not
    used for this panel), per representation per condition. "n/a" bars are
    conditions with too few nominated/overlapping compounds to compute a
    stability/effect-size estimate, not zeros."""
    representations = sorted(summary_rows["representation"].unique())
    conditions = list(dict.fromkeys(summary_rows["condition"]))

    def _values(col):
        return {
            rep: dict(
                zip(
                    summary_rows.loc[summary_rows["representation"] == rep, "condition"],
                    summary_rows.loc[summary_rows["representation"] == rep, col],
                )
            )
            for rep in representations
        }

    panels = [
        ("n_nominated_robust_ci", "C1 -- reversion hit count", "n hits", "{:.0f}"),
        ("c2_reversion_jaccard_mean", "C2 -- stability (hit-list floor)", "Jaccard (reversion hits)", "{:.2f}"),
        ("c2_reversion_score_spearman_mean", "C2 -- stability (score rank)", "Spearman rho (RI_spec)", "{:.2f}"),
        ("cross_rep_effect_size_auroc", "C3 -- cross-representation validation", "AUROC (vs. other reps)", "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        _grouped_bars(ax, conditions, representations, _values(col), value_fmt=fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(representations),
        bbox_to_anchor=(0.5, 1.1), frameon=False,
    )
    fig.suptitle("Tier C -- downstream task performance", fontsize=12, y=1.2)
    fig.tight_layout()

    out_path = out_dir / "tier_c_downstream_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_tier_d_figure(concordance_rows: pd.DataFrame, out_dir: Path) -> Path:
    """Tier D2 -- proteomics cross-modality concordance: Jaccard between
    each imaging representation's hit lists and the independent proteomic
    call (proteomics.concordance, imaging.benchmark.proteomic_concordance),
    per condition. Only conditions with a profiled proteomics plate appear
    (Baseline/FFA/IL6 -- no Low Gluc, see proteomics.imaging_metadata)."""
    representations = sorted(concordance_rows["representation"].unique())
    conditions = list(dict.fromkeys(concordance_rows["condition"]))

    def _values(col):
        return {
            rep: dict(
                zip(
                    concordance_rows.loc[concordance_rows["representation"] == rep, "condition"],
                    concordance_rows.loc[concordance_rows["representation"] == rep, col],
                )
            )
            for rep in representations
        }

    panels = [
        ("active_jaccard", "Active compounds"),
        ("allowlist_jaccard", "Active ∩ distinct"),
        ("reversion_jaccard", "Reversion hits"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.5), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        _grouped_bars(ax, conditions, representations, _values(col), value_fmt="{:.2f}")
        ax.set_title(title)
    axes[0].set_ylabel("Jaccard vs. independent proteomic call")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(representations),
        bbox_to_anchor=(0.5, 1.1), frameon=False,
    )
    fig.suptitle("Tier D -- proteomics cross-modality concordance", fontsize=12, y=1.2)
    fig.tight_layout()

    out_path = out_dir / "tier_d_proteomics_concordance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_tier_e_figure(
    summary_rows: pd.DataFrame,
    cross_condition_jaccard: dict,
    out_dir: Path,
) -> Path:
    """Tier E -- copairs-level cross-representation/cross-condition
    agreement and biological plausibility, one step upstream of reversion
    (imaging.benchmark's hit_overlap/mean_pairwise_jaccard and
    copairs_call_enrichment, wired in by run_cellrep_benchmark.py).

    Left to right: E1 cross-representation agreement, per condition (how
    much do CellProfiler/CPCNN/UniDino agree on which compounds/terms they
    call) for the activity ∩ distinctiveness allowlist and for consistency;
    E3 MoA/target permutation enrichment of the allowlist, per condition
    (does a representation's copairs-called compounds concentrate on a
    biologically coherent mechanism -- e.g. anti-inflammatory MoAs among
    IL6 hits -- not just clear a statistical threshold); and E2
    cross-condition agreement, per representation across call types (does
    a representation call the same compounds active/distinct/consistent in
    MORE THAN ONE stress condition -- a generic/promiscuous signal rather
    than a condition-specific one -- lower is more condition-specific).

    `cross_condition_jaccard` is `{representation: {call: mean_jaccard}}`
    (e.g. loaded from `{space}_cross_condition_jaccard.json`); the other
    four panels come straight from `summary_rows` (the same
    {space}_{condition}_summary.json rows Tiers B/C use)."""
    representations = sorted(summary_rows["representation"].unique())
    conditions = list(dict.fromkeys(summary_rows["condition"]))

    def _values(col):
        return {
            rep: dict(
                zip(
                    summary_rows.loc[summary_rows["representation"] == rep, "condition"],
                    summary_rows.loc[summary_rows["representation"] == rep, col],
                )
            )
            for rep in representations
        }

    panels = [
        (
            "copairs_cross_rep_allowlist_jaccard",
            "E1 -- cross-rep agreement\n(active ∩ distinct)",
            "Jaccard vs. other reps",
            "{:.2f}",
        ),
        (
            "copairs_cross_rep_consistency_jaccard",
            "E1 -- cross-rep agreement\n(consistency)",
            "Jaccard vs. other reps",
            "{:.2f}",
        ),
        (
            "copairs_allowlist_moa_n_significant_q10",
            "E3 -- MoA enrichment\n(allowlist)",
            "n MoA terms (q<=0.10)",
            "{:.0f}",
        ),
        (
            "copairs_allowlist_target_n_significant_q10",
            "E3 -- target enrichment\n(allowlist)",
            "n target terms (q<=0.10)",
            "{:.0f}",
        ),
    ]
    n_panels = len(panels) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        _grouped_bars(ax, conditions, representations, _values(col), value_fmt=fmt)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)

    call_types = ["activity", "distinctiveness", "allowlist", "consistency"]
    cross_cond_values = {
        rep: {call: cross_condition_jaccard.get(rep, {}).get(call, float("nan")) for call in call_types}
        for rep in representations
    }
    ax = axes[-1]
    _grouped_bars(ax, call_types, representations, cross_cond_values, value_fmt="{:.2f}")
    ax.set_title("E2 -- cross-condition agreement\n(same rep, across conditions)", fontsize=10)
    ax.set_ylabel("Jaccard across conditions")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(representations),
        bbox_to_anchor=(0.5, 1.12), frameon=False,
    )
    fig.suptitle(
        "Tier E -- copairs-level agreement and biological plausibility\n"
        "(cross-representation, cross-condition, MoA/target enrichment)",
        fontsize=12, y=1.24,
    )
    fig.tight_layout()

    out_path = out_dir / "tier_e_copairs_agreement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_proteomics_tier_e_figure(
    summary_rows: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Proteomics Tier E, E3 only: MoA/target preranked-GSEA enrichment of
    the copairs allowlist (active ∩ distinctive), per (condition,
    processed_tag) -- computed by `commands.proteomics.tier_e_main`
    (`proteomics.benchmark` -- a separate implementation from
    `imaging.benchmark`, not a reuse of it, per this repo's
    module-independence rule; see that module's docstring). Proteomics has
    no "representation" axis -- a single feature space -- so `processed_tag`
    (raw vs. a batch/plate-corrected variant) plays that role instead, the
    same swap `make_proteomics_copairs_summary_figure` already makes for the
    copairs summary figure -- including its `tab10` palette convention
    rather than imaging Tier E's fixed `REPRESENTATION_COLORS`.

    `summary_rows` is the `{tag}_{condition}_tier_e_summary.json` rows
    `tier_e_main` writes (one per (processed_tag, condition))."""
    processed_tags = sorted(summary_rows["processed_tag"].unique())
    conditions = list(dict.fromkeys(summary_rows["condition"]))
    palette = plt.get_cmap("tab10").colors
    tag_colors = {tag: palette[i % len(palette)] for i, tag in enumerate(processed_tags)}

    def _values(col):
        return {
            tag: dict(
                zip(
                    summary_rows.loc[summary_rows["processed_tag"] == tag, "condition"],
                    summary_rows.loc[summary_rows["processed_tag"] == tag, col],
                )
            )
            for tag in processed_tags
        }

    panels = [
        (
            "copairs_allowlist_moa_n_significant_q10",
            "E3 -- MoA enrichment\n(allowlist)",
            "n MoA terms (q<=0.10)",
            "{:.0f}",
        ),
        (
            "copairs_allowlist_target_n_significant_q10",
            "E3 -- target enrichment\n(allowlist)",
            "n target terms (q<=0.10)",
            "{:.0f}",
        ),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        _grouped_bars(ax, conditions, processed_tags, _values(col), value_fmt=fmt, colors=tag_colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(processed_tags),
        bbox_to_anchor=(0.5, 1.15), frameon=False,
    )
    fig.suptitle(
        "Tier E3 -- MoA/target enrichment of the copairs allowlist (proteomics)",
        fontsize=12, y=1.28,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proteomics_tier_e_copairs_agreement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
