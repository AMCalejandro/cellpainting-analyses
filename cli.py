#!/usr/bin/env python
"""Unified CLI entrypoint for cellpainting-analyses.

Consolidates the former root run_*.py / make_benchmark_figures.py driver
scripts into nested subcommands. This module only builds the argparse tree
and dispatches -- all actual logic lives in commands/imaging.py,
commands/proteomics.py, and commands/benchmark.py (which in turn call into
the imaging/ and proteomics/ packages, unchanged).

Usage:
    python cli.py imaging copairs ...
    python cli.py imaging batch_report ...
    python cli.py imaging reversion ...
    python cli.py proteomics processing ...
    python cli.py proteomics copairs ...
    python cli.py proteomics tier_e ...
    python cli.py proteomics concordance ...
    python cli.py proteomics batch_report ...
    python cli.py benchmark run ...
    python cli.py benchmark figures ...

Run `python cli.py <group> <command> --help` for a command's full flag list.
"""

import argparse

from commands import benchmark as benchmark_cmds
from commands import imaging as imaging_cmds
from commands import proteomics as proteomics_cmds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description=__doc__)
    groups = parser.add_subparsers(dest="group", required=True)

    imaging_parser = groups.add_parser("imaging", help="Imaging (cpg0014) copairs/reversion pipeline.")
    imaging_sub = imaging_parser.add_subparsers(dest="command", required=True)
    imaging_cmds.add_copairs_parser(
        imaging_sub.add_parser("copairs", help="Copairs activity/distinctiveness/consistency pipeline.")
    )
    imaging_cmds.add_copairs_compare_parser(
        imaging_sub.add_parser(
            "copairs-compare",
            help="Cross-condition companion figures built from already computed copairs parquets.",
        )
    )
    imaging_cmds.add_batch_report_parser(
        imaging_sub.add_parser("batch_report", help="Batch/condition silhouette report.")
    )
    imaging_cmds.add_reversion_parser(
        imaging_sub.add_parser("reversion", help="Compound reversion scoring.")
    )

    proteomics_parser = groups.add_parser("proteomics", help="Proteomics normalize/correct/copairs pipeline.")
    proteomics_sub = proteomics_parser.add_subparsers(dest="command", required=True)
    proteomics_cmds.add_processing_parser(
        proteomics_sub.add_parser("processing", help="Batch/plate correction + silhouette report.")
    )
    proteomics_cmds.add_copairs_parser(
        proteomics_sub.add_parser("copairs", help="Activity/distinctiveness/consistency copairs calls.")
    )
    proteomics_cmds.add_tier_e_parser(
        proteomics_sub.add_parser("tier_e", help="Copairs-level Tier E agreement + MoA/target enrichment.")
    )
    proteomics_cmds.add_concordance_parser(
        proteomics_sub.add_parser("concordance", help="Proteomic hit calling + imaging concordance (Tier D).")
    )
    proteomics_cmds.add_batch_report_parser(
        proteomics_sub.add_parser("batch_report", help="Preprocessing/batch-correction bake-off.")
    )

    benchmark_parser = groups.add_parser("benchmark", help="Feature-representation benchmark (Tiers A-E).")
    benchmark_sub = benchmark_parser.add_subparsers(dest="command", required=True)
    benchmark_cmds.add_run_parser(
        benchmark_sub.add_parser("run", help="Run the matched-dimension representation benchmark.")
    )
    benchmark_cmds.add_figures_parser(
        benchmark_sub.add_parser("figures", help="Assemble Tier A-E figures from saved artifacts.")
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
