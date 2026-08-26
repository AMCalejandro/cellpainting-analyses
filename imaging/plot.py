"""Recreate the imaging_results.png layout from the copairs pipeline outputs
saved by run_pipeline.py: a 3x2 grid of (activity, distinctiveness,
consistency) x (call-count bars, nMAP violins), one group of covariate-set
bars/violins per feature space.

Exposes `make_figure` for direct use (e.g. called from run_pipeline.py right
after it writes results) rather than a `__main__`-guarded script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load import DEFAULT_CONDITION

COV_LABELS = {
    "count": "Count",
    "count_batch": "Count + batch",
    "count_plate": "Count + plate",
    "count_batch_plate": "Count + batch + plate",
}
COV_COLORS = {
    "count": "#c9ccd1",
    "count_batch": "#5a5f66",
    "count_plate": "#7ec8f2",
    "count_batch_plate": "#1f77b4",
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


def _load_results(
    out_dir: Path, feature_spaces: list, covariate_sets: list, condition_tag: str
) -> dict:
    results = {}
    for space in feature_spaces:
        for cov_key in covariate_sets:
            for call_name, *_ in CALLS:
                path = out_dir / f"{space}{condition_tag}_{cov_key}_{call_name}.parquet"
                results[(space, cov_key, call_name)] = pd.read_parquet(path)
    return results


def make_figure(
    out_dir: Path,
    feature_spaces: list,
    covariate_sets: list,
    condition: str = DEFAULT_CONDITION,
) -> Path:
    """Build a call-count/nMAP summary figure from this run's result
    parquets in `out_dir`, saved alongside them as `reproduced_figure.png`
    (or `reproduced_figure_<condition>.png` for a non-default condition).
    Returns the saved path."""
    condition_tag = "" if condition == DEFAULT_CONDITION else f"_{condition}"
    results = _load_results(out_dir, feature_spaces, covariate_sets, condition_tag)

    fig, axes = plt.subplots(2, len(CALLS), figsize=(20, 13))

    for col, (call_name, title, subtitle, unit) in enumerate(CALLS):
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

    out_path = out_dir / f"reproduced_figure{condition_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
