"""Collect correct and incorrect zero-crossing distributions for soft GDBF."""

import argparse
import json
import shutil
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
    PROJECT_DIR / "experiments" / "experiment_soft_gdbf_random.json"
)
DEFAULT_OUTPUT = PROJECT_DIR / "error_distribution_by_e"
ALGORITHM = "soft gradient descent bit-flipping"


def parse_args(description="Collect Soft GDBF steps that cross zero."):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_experiment(config_path):
    config = load_json(str(config_path))
    experiment_config = config["experiment"]
    algorithm = experiment_config["codec"].get("algorithm")
    if algorithm != ALGORITHM:
        raise ValueError(f"config must use {ALGORITHM!r}")
    return LdpcExperimentInstance(
        LdpcExperimentSettings(**experiment_config)
    )


def append_values(file_handle, values):
    np.ascontiguousarray(values, dtype=np.float32).tofile(file_handle)


def chunked_histogram(values, bin_edges, chunk_size=1_000_000):
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    for start in range(0, values.size, chunk_size):
        counts += np.histogram(
            values[start:start + chunk_size],
            bins=bin_edges,
        )[0]
    total = counts.sum()
    if total == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float64)
    return counts / (total * np.diff(bin_edges))


def file_values(path):
    if path.stat().st_size == 0:
        return None
    return np.memmap(path, dtype=np.float32, mode="r")


def plot_histogram(iteration_dir, bins, correct_label, incorrect_label, title):
    correct = file_values(iteration_dir / "correct_steps.float32.bin")
    incorrect = file_values(iteration_dir / "incorrect_steps.float32.bin")
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
            label=correct_label,
        )
    if incorrect is not None:
        axis.stairs(
            chunked_histogram(incorrect, bin_edges),
            bin_edges,
            fill=True,
            alpha=0.55,
            label=incorrect_label,
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=1.2)
    axis.set_xlim(-5.0, 5.0)
    axis.set_xlabel(r"step$_n$ = $\eta_t v_n$")
    axis.set_ylabel("Probability density")
    axis.set_title(f"{title}, iteration {int(iteration_dir.name):03d}")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(iteration_dir / "histogram.png", dpi=180)
    plt.close(figure)


def main(mode="zero_crossings"):
    if mode == "zero_crossings":
        description = "Collect Soft GDBF steps that cross zero."
        correct_label = "Correct crossings"
        incorrect_label = "Incorrect crossings"
        title = "Soft GDBF zero crossings"
        tracked_event = (
            "step crossed zero into a correct or incorrect hard decision"
        )
    elif mode == "confidence_changes":
        description = "Collect correct and incorrect Soft GDBF confidence changes."
        correct_label = "Correct confidence changes"
        incorrect_label = "Incorrect confidence changes"
        title = "Soft GDBF confidence changes"
        tracked_event = (
            "nonzero step increased or decreased confidence in the true symbol"
        )
    else:
        raise ValueError(f"unknown analysis mode: {mode!r}")

    args = parse_args(description)
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    config_path = args.config.resolve()
    output_dir = args.output.resolve()
    experiment = load_experiment(config_path)
    decoder = experiment.codec.decoder_impl
    n_iterations = min(args.iterations, decoder.n_iterations)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    iteration_dirs = []
    correct_counts = np.zeros(n_iterations, dtype=np.int64)
    incorrect_counts = np.zeros(n_iterations, dtype=np.int64)
    active_word_counts = np.zeros(n_iterations, dtype=np.int64)
    decoded_words = 0

    with ExitStack() as stack:
        correct_files = []
        incorrect_files = []
        for iteration in range(n_iterations):
            iteration_dir = output_dir / f"{iteration:03d}"
            iteration_dir.mkdir()
            iteration_dirs.append(iteration_dir)
            correct_files.append(stack.enter_context(
                (iteration_dir / "correct_steps.float32.bin").open("wb")
            ))
            incorrect_files.append(stack.enter_context(
                (iteration_dir / "incorrect_steps.float32.bin").open("wb")
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

            received = experiment.llr_in.copy()
            x = received.copy()
            velocity = np.zeros_like(x)
            transmitted_symbols = (
                1 - 2 * experiment.tx_bits.astype(np.int8)
            )

            for iteration in range(n_iterations):
                hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
                if np.all(decoder.bpsk_syndrome(hard_x) == 1):
                    decoded_words += 1
                    break

                active_word_counts[iteration] += 1
                gradient = decoder.objective_gradient(x, received)
                velocity = (
                    decoder.momentum * velocity
                    + (1 - decoder.momentum) * gradient
                )
                learning_rate = decoder.learning_rate / np.sqrt(
                    1 + decoder.learning_rate_decay * iteration
                )
                optimizer_step = learning_rate * velocity
                next_x = decoder.gamma * (x + optimizer_step)
                step = next_x - x
                next_hard_x = np.where(next_x >= 0, 1, -1).astype(np.int8)

                if mode == "zero_crossings":
                    crossed_zero = next_hard_x != hard_x
                    correct_step = (
                        crossed_zero & (next_hard_x == transmitted_symbols)
                    )
                    incorrect_step = (
                        crossed_zero & (next_hard_x != transmitted_symbols)
                    )
                else:
                    true_confidence_change = transmitted_symbols * step
                    correct_step = true_confidence_change > 0
                    incorrect_step = true_confidence_change < 0

                correct_values = step[correct_step]
                incorrect_values = step[incorrect_step]
                append_values(correct_files[iteration], correct_values)
                append_values(incorrect_files[iteration], incorrect_values)
                correct_counts[iteration] += correct_values.size
                incorrect_counts[iteration] += incorrect_values.size
                x = next_x
            else:
                hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
                if np.all(decoder.bpsk_syndrome(hard_x) == 1):
                    decoded_words += 1

            if (trial + 1) % 1000 == 0:
                print(
                    f"{trial + 1}/{args.trials} words, "
                    f"decoded={decoded_words}",
                    flush=True,
                )

    for iteration_dir in iteration_dirs:
        plot_histogram(
            iteration_dir,
            args.bins,
            correct_label,
            incorrect_label,
            title,
        )

    metadata = {
        "algorithm": ALGORITHM,
        "config": str(config_path),
        "snr_db": args.snr,
        "trials": args.trials,
        "decoded_words": decoded_words,
        "not_decoded_words": args.trials - decoded_words,
        "iterations": n_iterations,
        "seed": args.seed,
        "value": "gamma * (current_x + learning_rate * momentum_velocity) - current_x",
        "analysis_mode": mode,
        "tracked_event": tracked_event,
        "value_dtype": "float32",
        "active_word_counts": active_word_counts.tolist(),
        "correct_step_counts": correct_counts.tolist(),
        "incorrect_step_counts": incorrect_counts.tolist(),
        "decoder_params": {
            "learning_rate": decoder.learning_rate,
            "learning_rate_decay": decoder.learning_rate_decay,
            "momentum": decoder.momentum,
            "regularization": decoder.regularization,
            "alpha": decoder.alpha,
            "gamma": decoder.gamma,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved distributions to {output_dir}")


if __name__ == "__main__":
    main()
