import argparse
import json
from pathlib import Path
from shlex import quote

import hailtop.batch as hb


def build_batch(config: dict) -> hb.Batch:
    hb_cfg = config["hail-batch"]
    repo_cfg = config["repo"]
    pipeline_cfg = config["pipeline"]

    backend = hb.ServiceBackend(
        billing_project=hb_cfg["billing-project"],
        remote_tmpdir=hb_cfg["remote-tmpdir"],
        regions=hb_cfg["regions"],
    )
    b = hb.Batch(backend=backend, name="cellpainting-copairs")

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

        j.command("apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates")
        j.command(
            f"git clone --branch {quote(repo_cfg['branch'])} --single-branch "
            f"{quote(repo_cfg['url'])} repo"
        )
        j.command("cd repo")
        j.command("curl -fsSL https://pixi.sh/install.sh | sh")
        j.command('export PATH="/root/.pixi/bin:$PATH"')
        j.command("for attempt in {1..10}; do pixi install && break; done")

        j.command("mkdir -p data/imaging")
        for filename, local_input in data_inputs.items():
            j.command(f"cp {local_input} data/imaging/{quote(filename)}")

        j.command(
            "pixi run python run_pipeline.py "
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

    return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config.json"),
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        config = json.load(fh)

    batch = build_batch(config)
    batch.run()
