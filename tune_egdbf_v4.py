"""Grid-search C++ E-GDBF V4 (sign-only GDMS) parameters at one SNR point."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_py.cpp_bin_ldpc_gdms import lib_compile as gdms_compile
from simulator_awgn_python.tools import load_json
from tune_gdms import (
    comma_separated_floats,
    default_workers,
    evaluate_candidate,
    init_worker,
    parameter_grid,
    result_score,
    save_best,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_egdbf_v4.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_egdbf_v4.txt"
V4_ALGORITHM = "cpp edge-wise gradient descent bit-flipping v4"

# 15 * 11 * 26 * 16 = 68 640 combinations (plus the config baseline).
DEFAULT_LEARNING_RATES = tuple(np.round(np.arange(0.02, 0.301, 0.02), 2))
# Zero plus a log-spaced grid: over 50 iterations the rate falls by
# sqrt(1 + 50 * decay), i.e. from ~1.02x at 0.001 to ~7x at 1.
DEFAULT_LEARNING_RATE_DECAYS = (0.0,) + tuple(
    float(f"{value:.2g}") for value in np.geomspace(0.001, 1.0, 10)
)
DEFAULT_ALPHAS = tuple(np.round(np.arange(0.5, 3.001, 0.1), 2))
DEFAULT_L2 = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
              2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search C++ E-GDBF V4 hyperparameters using FER at a fixed SNR. "
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
        "--learning-rates",
        type=comma_separated_floats,
        default=DEFAULT_LEARNING_RATES,
    )
    parser.add_argument(
        "--learning-rate-decays",
        type=comma_separated_floats,
        default=DEFAULT_LEARNING_RATE_DECAYS,
    )
    parser.add_argument(
        "--alphas",
        type=comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument("--l2-values", type=comma_separated_floats, default=DEFAULT_L2)
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
    if any(not np.isfinite(v) or v <= 0 for v in args.learning_rates):
        raise ValueError("all learning rates must be finite and positive")
    if any(not np.isfinite(v) or v < 0 for v in args.learning_rate_decays):
        raise ValueError("all learning-rate decays must be finite and non-negative")
    if any(not np.isfinite(v) or v <= 0 for v in args.alphas):
        raise ValueError("all alphas must be finite and positive")
    if any(not np.isfinite(v) or v < 0 for v in args.l2_values):
        raise ValueError("l2 values must be finite and non-negative")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != V4_ALGORITHM:
        raise ValueError("the selected config must use the C++ E-GDBF V4 decoder")
    return experiment, config.get("simulation", {})


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    gdms_compile()
    candidates = parameter_grid(
        args,
        base_experiment["codec"]["decoder_params"],
    )
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".jsonl").write_text("", encoding="utf-8")
    print(
        f"E-GDBF V4 search: SNR={args.snr:g} dB, "
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
            params = json.dumps(result["decoder_params"], separators=(",", ":"))
            if result["invalid"]:
                print(
                    f"INVALID [{completed}/{len(candidates)}] params={params}",
                    flush=True,
                )
                continue
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, completed, len(candidates))
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
