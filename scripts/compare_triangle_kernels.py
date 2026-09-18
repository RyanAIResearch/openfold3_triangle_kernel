"""Train a few steps with each triangle-kernel backend and chart the wall time.

Prepares a small PDB subset with the dataset scripts next to this file, then
trains the same runner YAML and seed with PyTorch, Triton, cuEquivariance and
TurboPairFormer, writing <output-dir>/backend_comparison.png.

Usage (from the OpenFold3 checkout):
    python scripts/compare_triangle_kernels.py
    python scripts/compare_triangle_kernels.py --steps 50 --backends pytorch cueq
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

BACKEND_FLAGS = {
    "pytorch": {},
    "triton": {"use_triton_triangle_kernels": True},
    "cueq": {"use_cueq_triangle_kernels": True},
    "turbopairformer": {
        "use_turbopairformer_triangle_kernels": True,
    },
}
KERNEL_FLAGS = {"use_deepspeed_evo_attention": False} | {
    flag: False for flags in BACKEND_FLAGS.values() for flag in flags
}


def run(*cmd):
    subprocess.run([str(arg) for arg in cmd], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-dir", default="datasets", help="PDB subset location")
    parser.add_argument(
        "--output-dir", default="runs", help="runner YAMLs, logs, chart"
    )
    parser.add_argument("--steps", type=int, default=20, help="training steps per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", nargs="+", default=list(BACKEND_FLAGS))
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass --use_tf32 to every run; cuEquivariance attention backward needs it",
    )
    args = parser.parse_args(argv)

    scripts = Path(__file__).resolve().parent / "datasets"
    run(
        sys.executable,
        scripts / "generate_subset_cache.py",
        "--target-dir",
        args.target_dir,
    )
    run(sys.executable, scripts / "download_subset.py", "--target-dir", args.target_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    for backend in args.backends:
        subset_yaml = Path(args.target_dir) / "train_pdb_subset.yaml"
        cfg = yaml.safe_load(subset_yaml.read_text())
        cfg["experiment_settings"].update(
            output_dir=str(output / backend), restart_checkpoint_path=None
        )
        cfg["data_module_args"]["epoch_len"] = args.steps
        cfg["checkpoint_config"] = {"save_top_k": 0, "save_last": False}
        cfg["pl_trainer_args"].update(
            max_epochs=1,
            max_steps=args.steps,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        )
        memory = cfg["model_update"]["custom"]["settings"]["memory"]["train"]
        memory.update(KERNEL_FLAGS | BACKEND_FLAGS[backend])
        runner_yaml = output / f"{backend}.yaml"
        runner_yaml.write_text(yaml.safe_dump(cfg))

        start = time.time()
        try:
            run(
                "run_openfold",
                "train",
                "--runner-yaml",
                runner_yaml,
                "--seed",
                args.seed,
                "--use_tf32",
                str(args.tf32).lower(),
            )
        except subprocess.CalledProcessError as error:
            print(f"{backend}: training failed (exit {error.returncode}), skipping")
            continue
        results[backend] = time.time() - start

    for backend, seconds in results.items():
        print(f"{backend:16s} {seconds:7.0f} s for {args.steps} steps")
    if not results:
        sys.exit("no backend completed")
    save_chart(results, args.steps, output / "backend_comparison.png")


def save_chart(results, steps, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("chart skipped: pip install matplotlib to write", path)
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(list(results), list(results.values()), color="#4C72B0", width=0.6)
    ax.bar_label(bars, fmt="%.0f s", padding=3)
    ax.set_ylabel(f"seconds for {steps} training steps")
    ax.set_title("OpenFold3 training on one GPU, 384-token crops")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"chart: {path}")


if __name__ == "__main__":
    main()
