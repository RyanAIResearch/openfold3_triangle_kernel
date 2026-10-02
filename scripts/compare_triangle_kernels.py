"""Train the setup-pdb-subset mini set a few steps with each triangle-kernel backend.

Every run uses the same runner YAML, seed and TF32 setting (cuEquivariance's attention
backward needs TF32), skips validation and checkpoints, and reports its wall time.

Usage, after `pixi run -e openfold3 setup-pdb-subset`:
    pixi run -e openfold3 compare-triangle-kernels
    pixi run -e openfold3 compare-triangle-kernels --steps 50 --backends pytorch cueq
"""

import argparse
import subprocess
import time
from pathlib import Path

import yaml

BACKENDS = {
    "pytorch": None,
    "triton": "use_triton_triangle_kernels",
    "cueq": "use_cueq_triangle_kernels",
    "turbopairformer": "use_turbopairformer_triangle_kernels",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", type=int, default=20, help="training steps per run")
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    args = parser.parse_args()

    output = Path("runs")
    output.mkdir(exist_ok=True)
    seconds = {}
    for backend in args.backends:
        config = yaml.safe_load(Path("datasets/train_pdb_subset.yaml").read_text())
        config["experiment_settings"].update(
            output_dir=str(output / backend), restart_checkpoint_path=None
        )
        config["data_module_args"]["epoch_len"] = args.steps
        config["checkpoint_config"] = {"save_top_k": 0, "save_last": False}
        config["pl_trainer_args"].update(
            max_epochs=1,
            max_steps=args.steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        memory = config["model_update"]["custom"]["settings"]["memory"]["train"]
        memory.update(
            {flag: flag == BACKENDS[backend] for flag in BACKENDS.values() if flag}
        )
        memory["use_deepspeed_evo_attention"] = False
        runner_yaml = output / f"{backend}.yaml"
        runner_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

        start = time.time()
        cmd = ["run_openfold", "train", "--runner-yaml", str(runner_yaml)]
        if subprocess.run([*cmd, "--seed", "42", "--use_tf32", "true"]).returncode == 0:
            seconds[backend] = time.time() - start
        else:
            print(f"{backend}: training failed")

    print(f"\nWall time for {args.steps} training steps:")
    for backend, s in seconds.items():
        speedup = (
            f"{seconds['pytorch'] / s:.2f}x pytorch" if "pytorch" in seconds else ""
        )
        print(f"  {backend:16s} {s:6.0f} s  {speedup}")


if __name__ == "__main__":
    main()
