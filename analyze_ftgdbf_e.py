"""Collect FTGDBF energy distributions for correct and incorrect decisions."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    PROJECT_DIR / "experiments" / "experiment_ftgdbf_random.json"
)
DEFAULT_OUTPUT = PROJECT_DIR / "error_distribution_by_e"
ALGORITHM = "fixed threshold gradient descent bit-flipping"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect per-iteration FTGDBF decision-energy distributions."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=100_00)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_experiment(config_path):
    config = load_json(str(config_path))
    experiment_config = config["experiment"]
    codec_config = experiment_config["codec"]
    if codec_config.get("algorithm") != ALGORITHM:
        raise ValueError(f"config must use {ALGORITHM!r}")
    return LdpcExperimentInstance(
        LdpcExperimentSettings(**experiment_config)
    )


def append_values(file_handle, values):
    np.ascontiguousarray(values, dtype=np.float32).tofile(file_handle)


def file_values(path):
    if path.stat().st_size == 0:
        return None
    return np.memmap(path, dtype=np.float32, mode="r")


def chunked_histogram(values, bin_edges, chunk_size=1_000_000):
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    for start in range(0, values.size, chunk_size):
        counts += np.histogram(
            values[start:start + chunk_size],
            bins=bin_edges,
        )[0]
    widths = np.diff(bin_edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths)
    return counts / (total * widths)


def plot_histogram(iteration_dir, bins):
    correct = file_values(iteration_dir / "correct_decisions.float32.bin")
    incorrect = file_values(iteration_dir / "incorrect_decisions.float32.bin")
    if correct is None and incorrect is None:
        return

    bin_edges = np.linspace(-5.0, 5.0, bins + 1)

    figure, axis = plt.subplots(figsize=(9, 6))
    if correct is not None:
        axis.stairs(
            chunked_histogram(correct, bin_edges),
            bin_edges,
            fill=True,
            alpha=0.55,
            label="Correct decisions",
        )
    if incorrect is not None:
        axis.stairs(
            chunked_histogram(incorrect, bin_edges),
            bin_edges,
            fill=True,
            alpha=0.55,
            label="Incorrect decisions",
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xlim(-5.0, 5.0)
    axis.set_xlabel("E(x)")
    axis.set_ylabel("Probability density")
    axis.set_title(f"FTGDBF decision energy, iteration {int(iteration_dir.name):03d}")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(iteration_dir / "histogram.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    experiment = load_experiment(config_path)
    decoder = experiment.codec.decoder_impl
    n_iterations = decoder.n_iterations
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_dirs = []
    correct_counts = np.zeros(n_iterations, dtype=np.int64)
    incorrect_counts = np.zeros(n_iterations, dtype=np.int64)
    decoded_words = 0

    with ExitStack() as stack:
        correct_files = []
        incorrect_files = []
        for iteration in range(n_iterations):
            iteration_dir = output_dir / f"{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            iteration_dirs.append(iteration_dir)
            correct_files.append(stack.enter_context(
                (iteration_dir / "correct_decisions.float32.bin").open("wb")
            ))
            incorrect_files.append(stack.enter_context(
                (iteration_dir / "incorrect_decisions.float32.bin").open("wb")
            ))

        rng = np.random.default_rng(args.seed)
        for trial in range(args.trials):
            if not experiment.codec.is_azcw():
                experiment.codec.generate(
                    rng,
                    experiment.iwd,
                    experiment.tx_bits,
                )
            experiment.run_channel(args.snr, rng)
            decoded, trajectory = decoder.trace(experiment.llr_in)
            if decoded:
                decoded_words += 1
            transmitted_symbols = (
                1 - 2 * experiment.tx_bits.astype(np.int8)
            )
            for iteration, (state, energy) in enumerate(trajectory):
                should_flip = state != transmitted_symbols
                did_flip = energy <= 0
                correct_decision = did_flip == should_flip
                correct_values = energy[correct_decision]
                incorrect_values = energy[~correct_decision]
                append_values(correct_files[iteration], correct_values)
                append_values(incorrect_files[iteration], incorrect_values)
                correct_counts[iteration] += correct_values.size
                incorrect_counts[iteration] += incorrect_values.size

            if (trial + 1) % 1000 == 0:
                print(
                    f"{trial + 1}/{args.trials} words, "
                    f"decoded={decoded_words}",
                    flush=True,
                )

    for iteration_dir in iteration_dirs:
        plot_histogram(iteration_dir, args.bins)

    metadata = {
        "algorithm": ALGORITHM,
        "config": str(config_path),
        "snr_db": args.snr,
        "trials": args.trials,
        "decoded_words": decoded_words,
        "failed_words": args.trials - decoded_words,
        "iterations": n_iterations,
        "threshold": 0.0,
        "seed": args.seed,
        "value_dtype": "float32",
        "correct_decision_counts": correct_counts.tolist(),
        "incorrect_decision_counts": incorrect_counts.tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved distributions to {output_dir}")


if __name__ == "__main__":
    main()
