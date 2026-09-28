"""Grid-search C++ E-GDBF V3 parameters (alpha, eta, p) at one SNR point."""

import argparse
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_py.cpp_bin_ldpc_egdbf import lib_compile as egdbf_compile
from simulator_awgn_python.tools import load_json
from tune_egdbf import (
    comma_separated_floats,
    default_workers,
    evaluate_candidate,
    init_worker,
    result_score,
    save_best,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_egdbf_v3.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_egdbf_v3.txt"
V3_ALGORITHM = "cpp edge-wise gradient descent bit-flipping v3"

DEFAULT_ALPHAS = tuple(np.round(np.arange(1.0, 3.001, 0.1), 2))
DEFAULT_ETAS = tuple(np.round(np.arange(0.2, 5.001, 0.1), 2))
DEFAULT_PROBABILITIES = (0.95, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search C++ E-GDBF V3 hyperparameters using FER at a fixed SNR. "
            "Every new best result is saved immediately."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=2_000_000)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--alphas",
        type=comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--etas",
        type=comma_separated_floats,
        default=DEFAULT_ETAS,
    )
    parser.add_argument(
        "--probabilities",
        type=comma_separated_floats,
        default=DEFAULT_PROBABILITIES,
    )
    return parser.parse_args()


def validate_args(args):
    if not np.isfinite(args.snr):
        raise ValueError("--snr must be finite")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.alphas):
        raise ValueError("all alphas must be finite and positive")
    if any(not np.isfinite(value) or value <= 0 for value in args.etas):
        raise ValueError("all etas must be finite and positive")
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in args.probabilities):
        raise ValueError("all probabilities must be finite and in [0, 1]")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != V3_ALGORITHM:
        raise ValueError("the selected config must use the C++ E-GDBF V3 decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args):
    candidates = []
    seen = set()
    for alpha, eta, probability in itertools.product(
        args.alphas, args.etas, args.probabilities,
    ):
        candidate = {
            "alpha": float(alpha),
            "eta": float(eta),
            "p": float(probability),
        }
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    egdbf_compile()
    candidates = parameter_grid(args)
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".jsonl").write_text("", encoding="utf-8")
    print(
        f"E-GDBF V3 search: SNR={args.snr:g} dB, "
        f"max_trials={args.trials}, target_errors={args.max_errors}, "
        f"parameter_sets={len(candidates)}, workers={workers}",
        flush=True,
    )

    best_result = None
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            base_experiment,
            args.snr,
            args.trials,
            args.max_errors,
            args.seed,
        ),
    ) as pool:
        results = pool.imap_unordered(
            evaluate_candidate,
            enumerate(candidates),
            chunksize=1,
        )
        for completed, result in enumerate(results, start=1):
            with output_path.with_suffix(".jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(result) + "\n")
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, completed, len(candidates))
                params = json.dumps(result["decoder_params"], separators=(",", ":"))
                print(
                    f"NEW BEST [{completed}/{len(candidates)}] "
                    f"FER={result['fer']:.6g} "
                    f"({result['frame_errors']} errors / "
                    f"{result['trials']} trials), "
                    f"BER={result['ber']:.6g}, "
                    f"avg_iter={result['average_iterations']:.3f}\n"
                    f"params={params}\n"
                    f"saved to {output_path}",
                    flush=True,
                )

    print(f"Search completed. Best parameters: {output_path}", flush=True)


if __name__ == "__main__":
    main()
