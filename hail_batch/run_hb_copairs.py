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
    `--conditions` (FFA/IL6) is swept in-process by `cli.py proteomics
    copairs` within each job, matching how each imaging job already sweeps
    --feature-spaces/--covariate-sets in-process.

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
        choices=["imaging", "proteomics"],
        help="Which pipeline to submit to Hail Batch.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    batch = build_batch(config, args.pipeline)
    batch.run()
