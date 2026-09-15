import argparse
import json
from pathlib import Path
from shlex import quote

import hailtop.batch as hb


def _setup_commands(j: "hb.Job", repo_cfg: dict) -> None:
    j.command("apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates")
    j.command(
        f"git clone --branch {quote(repo_cfg['branch'])} --single-branch "
        f"{quote(repo_cfg['url'])} repo"
    )
    j.command("cd repo")
    j.command("curl -fsSL https://pixi.sh/install.sh | sh")
    j.command('export PATH="/root/.pixi/bin:$PATH"')
    j.command("for attempt in {1..10}; do pixi install && break; done")


def build_batch(config: dict, pipeline: str) -> hb.Batch:
    hb_cfg = config["hail-batch"]
    repo_cfg = config["repo"]

    backend = hb.ServiceBackend(
        billing_project=hb_cfg["billing-project"],
        remote_tmpdir=hb_cfg["remote-tmpdir"],
        regions=hb_cfg["regions"],
    )
    b = hb.Batch(backend=backend, name="cellpainting-copairs")

    if pipeline == "imaging":
        _add_imaging_jobs(b, config, hb_cfg, repo_cfg)
    elif pipeline == "proteomics":
        _add_proteomics_jobs(b, config, hb_cfg, repo_cfg)
    elif pipeline == "reversion":
        _add_reversion_jobs(b, config, hb_cfg, repo_cfg)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}")

    return b


def _add_imaging_jobs(b: "hb.Batch", config: dict, hb_cfg: dict, repo_cfg: dict) -> None:
    pipeline_cfg = config["pipeline"]

    feature_spaces = ",".join(pipeline_cfg["feature-spaces"])
    covariate_sets = ",".join(pipeline_cfg["covariate-sets"])
    null_size = pipeline_cfg["null-size"]
    preprocess_flag = "--preprocess" if pipeline_cfg.get("preprocess") else ""

    for condition, condition_tag in pipeline_cfg["conditions"].items():
        j = b.new_job(name=f"copairs {condition_tag}")
        j._machine_type = hb_cfg["machine-type"]
        j.storage(hb_cfg["storage"])

        data_inputs = {
            filename: b.read_input(gcs_path)
            for filename, gcs_path in config["data-files"].items()
        }

        _setup_commands(j, repo_cfg)

        j.command("mkdir -p data/imaging")
        for filename, local_input in data_inputs.items():
            j.command(f"cp {local_input} data/imaging/{quote(filename)}")

        j.command(
            "pixi run python cli.py imaging copairs "
            f"--condition {quote(condition)} "
            "--out-dir results "
            f"--null-size {null_size} "
            f"--feature-spaces {quote(feature_spaces)} "
            f"--covariate-sets {quote(covariate_sets)} "
            f"{preprocess_flag}"
        )

        j.command("tar -czf results.tar.gz -C results .")
        j.command(f"mv results.tar.gz {j.ofile}")
        output_dir = config["output-dir"].rstrip("/")
        b.write_output(j.ofile, f"{output_dir}/{condition_tag}/results.tar.gz")


def _add_proteomics_jobs(b: "hb.Batch", config: dict, hb_cfg: dict, repo_cfg: dict) -> None:
    """One job per (`processed`, `method`) combination -- same one-axis-
    per-job split `_add_imaging_jobs` uses for imaging conditions, just with
    `--processed`/`--method` as the new axes instead of `--condition`.
    `--conditions` (FFA/IL6) and `--covariates` (proteomics.correction.
    COVARIATE_SETS keys, e.g. plate/batch/batch_plate) are both swept
    in-process by `cli.py proteomics copairs` within each job -- one full
    copairs run per covariates entry -- matching how each imaging job
    already sweeps --feature-spaces/--covariate-sets in-process.

    `method` (and `covariates`) only affect the batch/plate correction step,
    which `--processed false` skips entirely (see `proteomics.pipeline.
    load_raw_proteomics`), so the method axis is only swept for
    `processed=true` -- `processed=false` gets a single "raw" job."""
    prote_cfg = config["proteomics-pipeline"]

    conditions = ",".join(prote_cfg["conditions"])
    covariates = ",".join(prote_cfg["covariates"])
    output_dir = config["proteomics-output-dir"].rstrip("/")
    methods = prote_cfg["method"]
    if isinstance(methods, str):
        methods = [methods]

    for processed in prote_cfg["processed"]:
        processed_str = "true" if processed else "false"
        methods_to_run = methods if processed else methods[:1]

        for method in methods_to_run:
            j = b.new_job(name=f"proteomics-copairs processed={processed_str} method={method}")
            j._machine_type = hb_cfg["machine-type"]
            j.storage(hb_cfg["storage"])

            metadata_input = b.read_input(config["data-files"]["metadata_cpg0014.tsv"])
            proteomics_inputs = {
                filename: b.read_input(gcs_path)
                for filename, gcs_path in config["proteomics-data-files"].items()
            }

            _setup_commands(j, repo_cfg)

            j.command("mkdir -p data/imaging data/proteomics")
            j.command(f"cp {metadata_input} data/imaging/metadata_cpg0014.tsv")
            for filename, local_input in proteomics_inputs.items():
                j.command(f"cp {local_input} data/proteomics/{quote(filename)}")

            j.command(
                "pixi run python cli.py proteomics copairs "
                f"--processed {processed_str} "
                f"--conditions {quote(conditions)} "
                f"--method {quote(method)} "
                f"--covariates {quote(covariates)} "
                f"--null-size {prote_cfg['null-size']} "
                f"--consistency-groupby {quote(prote_cfg['consistency-groupby'])} "
                "--out-dir results"
            )

            j.command("tar -czf results.tar.gz -C results .")
            j.command(f"mv results.tar.gz {j.ofile}")
            tag = f"processed_{processed_str}_{method}" if processed else f"processed_{processed_str}"
            b.write_output(j.ofile, f"{output_dir}/{tag}/results.tar.gz")


def _add_reversion_jobs(b: "hb.Batch", config: dict, hb_cfg: dict, repo_cfg: dict) -> None:
    """One job per (stress_condition, feature_space, covariate_set) triple --
    `cli.py imaging copairs-reversion` scores one feature space / covariate
    set / stress condition per invocation (same one-shot convention as the
    axis-based `reversion` command, unlike `copairs`'s in-process sweep over
    --feature-spaces/--covariate-sets), so all three axes are swept here
    instead, one job each.

    `copairs-reversion` now requires a PRENOMINATED compound allowlist --
    a per-condition `compute_activity` call for the stress condition and one
    for baseline, each computed on that condition's OWN (not jointly
    residualized) feature space by `--pipeline imaging`, not by this
    reversion pipeline. Each job here therefore reads the already uploaded
    `results.tar.gz` for its stress condition and for baseline from
    `output-dir` (the same GCS layout `_add_imaging_jobs` writes) and
    extracts the one activity parquet it needs, tagged with
    `reversion-pipeline.activity-covariate-set` (independent of this job's
    own `--covariate-set`, which is for the JOINT Baseline+stress
    residualization instead). This is a hard prerequisite: run
    `--pipeline imaging` (with every condition in `reversion-pipeline`'s
    `stress-conditions` plus `baseline-condition` present in
    `pipeline.conditions`) to completion before `--pipeline reversion`.
    """
    rev_cfg = config["reversion-pipeline"]
    baseline_condition = rev_cfg["baseline-condition"]
    null_size = rev_cfg["null-size"]
    output_dir = config["reversion-output-dir"].rstrip("/")
    activity_covariate_set = rev_cfg["activity-covariate-set"]
    imaging_output_dir = config["output-dir"].rstrip("/")
    imaging_condition_tags = config["pipeline"]["conditions"]
    # Matches imaging.load.DEFAULT_CONDITION: copairs_main only tags a
    # result filename with its condition when it isn't this one.
    default_imaging_condition = "FFA"

    def _activity_tarball(condition: str) -> str:
        tag = imaging_condition_tags[condition]
        return f"{imaging_output_dir}/{tag}/results.tar.gz"

    def _activity_parquet_rel_path(condition: str, feature_space: str) -> str:
        condition_tag = "" if condition == default_imaging_condition else f"_{condition}"
        return f"parquet/{feature_space}{condition_tag}_{activity_covariate_set}_activity.parquet"

    for stress_condition, condition_tag in rev_cfg["stress-conditions"].items():
        for feature_space in rev_cfg["feature-spaces"]:
            for covariate_set in rev_cfg["covariate-sets"]:
                j = b.new_job(
                    name=f"copairs-reversion {condition_tag} {feature_space} {covariate_set}"
                )
                j._machine_type = hb_cfg["machine-type"]
                j.storage(hb_cfg["storage"])

                data_inputs = {
                    filename: b.read_input(gcs_path)
                    for filename, gcs_path in config["data-files"].items()
                }
                stress_activity_tarball = b.read_input(_activity_tarball(stress_condition))
                baseline_activity_tarball = b.read_input(_activity_tarball(baseline_condition))

                _setup_commands(j, repo_cfg)

                j.command("mkdir -p data/imaging")
                for filename, local_input in data_inputs.items():
                    j.command(f"cp {local_input} data/imaging/{quote(filename)}")

                j.command("mkdir -p stress_activity baseline_activity")
                j.command(f"tar -xzf {stress_activity_tarball} -C stress_activity")
                j.command(f"tar -xzf {baseline_activity_tarball} -C baseline_activity")
                activity_parquet = (
                    f"stress_activity/{_activity_parquet_rel_path(stress_condition, feature_space)}"
                )
                baseline_activity_parquet = (
                    f"baseline_activity/{_activity_parquet_rel_path(baseline_condition, feature_space)}"
                )

                j.command(
                    "pixi run python cli.py imaging copairs-reversion "
                    f"--feature-space {quote(feature_space)} "
                    f"--covariate-set {quote(covariate_set)} "
                    f"--stress-condition {quote(stress_condition)} "
                    f"--baseline-condition {quote(baseline_condition)} "
                    f"--activity-parquet {quote(activity_parquet)} "
                    f"--baseline-activity-parquet {quote(baseline_activity_parquet)} "
                    f"--null-size {null_size} "
                    "--out-dir results"
                )

                j.command("tar -czf results.tar.gz -C results .")
                j.command(f"mv results.tar.gz {j.ofile}")
                tag = f"{condition_tag}/{feature_space}_{covariate_set}"
                b.write_output(j.ofile, f"{output_dir}/{tag}/results.tar.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.json"),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["imaging", "proteomics", "reversion"],
        help="Which pipeline to submit to Hail Batch.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    batch = build_batch(config, args.pipeline)
    batch.run()
