"""All matplotlib figure-building for the imaging pipeline, so plotting code
lives in one place rather than being scattered across driver modules.

- `make_copairs_summary_figure`: recreates the imaging_results.png layout from the copairs
  pipeline outputs saved by run_pipeline.py -- a 3x2 grid of (activity,
  distinctiveness, consistency) x (call-count bars, nMAP violins), one group
  of covariate-set bars/violins per feature space.
- `make_batch_report_figures`: PCA/UMAP 2x2 grids (before/after
  residualization x colored-by-batch/colored-by-condition) for
  imaging.batch_report.
- `make_covariate_comparison_figure`: one panel per feature space plotting
  post-residualization silhouette_batch/silhouette_condition across
  covariate sets, shading covariate sets that include "plate" -- makes it
  visually obvious when adding plate over-corrects (condition silhouette
  drops below zero alongside batch silhouette, instead of batch dropping
  while condition is preserved).

All are meant for direct use (e.g. called from run_pipeline.py right after
loading/residualizing) rather than as `__main__`-guarded scripts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from .copairs_pipeline import DEFAULT_CONSISTENCY_GROUPBY
from .load import DEFAULT_CONDITION

try:
    import umap
except ImportError:
    umap = None

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
    condition: str = DEFAULT_CONDITION,
    consistency_groupby: str = DEFAULT_CONSISTENCY_GROUPBY,
) -> Path:
    """Build a call-count/nMAP summary figure from this run's result
    parquets in `out_dir/parquet/`, saved to `out_dir/figures/` as
    `reproduced_figure.png` (or `reproduced_figure_<condition>.png` for a
    non-default condition, with an extra `_moa` suffix if consistency was
    grouped by Metadata_moa). Returns the saved path."""
    condition_tag = "" if condition == DEFAULT_CONDITION else f"_{condition}"
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
        ax_bar.set_title(f"{title}\n{subtitle.format(condition=condition)}", fontsize=12)
        ax_bar.set_ylabel(f"{unit} called out of {n_total:,}")

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
        ax_violin.set_ylabel(f"{unit[:-1]} nMAP" if unit.endswith("s") else "nMAP")
        ax_violin.set_ylim(-0.25, 1.05)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COV_COLORS[k]) for k in covariate_sets
    ]
    fig.legend(
        handles,
        [COV_LABELS[k] for k in covariate_sets],
        title="Variables in the Ridge model",
        loc="upper center",
        ncol=len(covariate_sets),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.suptitle(
        "Plate residualization changes call counts and mean normalized AP\n"
        f"(reproduction from cpg0014 {condition} data)",
        fontsize=16,
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"reproduced_figure{condition_tag}{consistency_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _scatter_grid(
    coords_before: np.ndarray,
    coords_after: np.ndarray,
    meta: pd.DataFrame,
    batch_col: str,
    condition_col: str,
    title: str,
) -> plt.Figure:
    """2x2 grid: rows = before/after residualization, columns = colored by
    batch / colored by condition."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    rows = [("Before residualization", coords_before), ("After residualization", coords_after)]
    cols = [batch_col, condition_col]
    for r, (row_label, coords) in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r, c]
            categories = meta[col].astype("category")
            codes = categories.cat.codes.to_numpy()
            n_cat = max(len(categories.cat.categories), 1)
            cmap = plt.get_cmap("tab20" if n_cat <= 20 else "viridis")
            ax.scatter(
                coords[:, 0], coords[:, 1], c=codes, cmap=cmap, s=6, alpha=0.7,
                vmin=0, vmax=max(n_cat - 1, 1),
            )
            ax.set_title(f"{row_label}\ncolored by {col}", fontsize=10)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            if n_cat <= 12:
                handles = [
                    plt.Line2D(
                        [], [], marker="o", linestyle="",
                        color=cmap(i / max(n_cat - 1, 1)), label=str(cat),
                    )
                    for i, cat in enumerate(categories.cat.categories)
                ]
                ax.legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=13)
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
    pca_before = PCA(n_components=2, random_state=seed).fit_transform(feats_before)
    pca_after = PCA(n_components=2, random_state=seed).fit_transform(feats_after)
    fig = _scatter_grid(
        pca_before, pca_after, meta, batch_col, condition_col, title=f"{title_prefix} -- PCA"
    )
    fig.savefig(out_dir / f"{file_stub}_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if umap is not None:
        umap_before = umap.UMAP(n_components=2, random_state=seed).fit_transform(feats_before)
        umap_after = umap.UMAP(n_components=2, random_state=seed).fit_transform(feats_after)
        fig = _scatter_grid(
            umap_before, umap_after, meta, batch_col, condition_col, title=f"{title_prefix} -- UMAP"
        )
        fig.savefig(out_dir / f"{file_stub}_umap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("umap-learn not installed; skipping UMAP plot", flush=True)


def make_covariate_comparison_figure(
    metrics_by_space: dict,
    out_dir: Path,
    covariate_sets: list,
    extra_methods: list = (),
) -> Path:
    """One panel per feature space: post-residualization silhouette_batch
    and silhouette_condition across `covariate_sets`, with covariate sets
    whose name contains "plate" shaded. A covariate set over-corrects when
    its silhouette_condition line dips below zero alongside
    silhouette_batch -- real condition signal getting washed out along
    with batch/plate noise -- rather than batch dropping while condition
    stays flat or improves.
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

        for i, cov_key in enumerate(covariate_sets):
            if "plate" in cov_key:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4b6b6", alpha=0.4, zorder=0)
        if extra_methods:
            ax.axvline(len(covariate_sets) - 0.5, color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
        ax.plot(x, batch_after, marker="o", color="#1f77b4", label="silhouette_batch (after)")
        ax.plot(x, cond_after, marker="o", color="#d62728", label="silhouette_condition (after)")

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
