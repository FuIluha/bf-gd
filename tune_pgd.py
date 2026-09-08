"""Grid-search C++ probabilistic gradient decoder parameters at one SNR."""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_pgd import lib_compile as pgd_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_pgd.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_pgd.txt"

DEFAULT_LEARNING_RATES = (0.01, 0.03, 0.05, 0.1, 0.2, 0.4)
DEFAULT_LEARNING_RATE_DECAYS = (0.0, 0.01, 0.05, 0.1, 0.25)
DEFAULT_MOMENTA = (0.0, 0.3, 0.5, 0.7, 0.9)
DEFAULT_ALPHAS = (0.25, 0.5, 1.0, 1.5, 2.0)

_BASE_EXPERIMENT = None
_SNR_DB = None
_MAX_TRIALS = None
_MAX_ERRORS = None
_SEED = None


def comma_separated_floats(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid number list: {value!r}"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search C++ PGD hyperparameters using FER at a fixed SNR. "
            "Every new best result is saved immediately."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.4)
    parser.add_argument("--trials", type=int, default=100_000_000)
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
        "--momenta",
        type=comma_separated_floats,
        default=DEFAULT_MOMENTA,
    )
    parser.add_argument(
        "--alphas",
        type=comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    return parser.parse_args()


def validate_args(args):
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(value <= 0 for value in args.learning_rates):
        raise ValueError("all learning rates must be positive")
    if any(value < 0 for value in args.learning_rate_decays):
        raise ValueError("all learning-rate decays must be non-negative")
    if any(not 0 <= value < 1 for value in args.momenta):
        raise ValueError("all momenta must be in [0, 1)")
    if any(value < 0 for value in args.alphas):
        raise ValueError("all alpha values must be non-negative")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != (
        "cpp probabilistic gradient descent decoder"
    ):
        raise ValueError(
            "the selected config must use the C++ probabilistic gradient decoder"
        )
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    baseline = {
        "learning_rate": float(base_params["learning_rate"]),
        "learning_rate_decay": float(base_params["learning_rate_decay"]),
        "momentum": float(base_params["momentum"]),
        "alpha": float(base_params["alpha"]),
    }
    candidates = [baseline]
    for values in itertools.product(
        args.learning_rates,
        args.learning_rate_decays,
        args.momenta,
        args.alphas,
    ):
        candidates.append({
            "learning_rate": values[0],
            "learning_rate_decay": values[1],
            "momentum": values[2],
            "alpha": values[3],
        })

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate.items())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def init_worker(base_experiment, snr_db, max_trials, max_errors, seed):
    global _BASE_EXPERIMENT, _SNR_DB, _MAX_TRIALS, _MAX_ERRORS, _SEED
    _BASE_EXPERIMENT = base_experiment
    _SNR_DB = snr_db
    _MAX_TRIALS = max_trials
    _MAX_ERRORS = max_errors
    _SEED = seed


def evaluate_candidate(index_and_params):
    index, decoder_params = index_and_params
    experiment_config = copy.deepcopy(_BASE_EXPERIMENT)
    experiment_config["codec"]["decoder_params"] = decoder_params
    experiment = LdpcExperimentInstance(
        LdpcExperimentSettings(**experiment_config)
    )

    frame_errors = 0
    bit_errors = 0.0
    iterations = 0
    trials_completed = 0
    for trial_index in range(_MAX_TRIALS):
        rng = np.random.default_rng([_SEED, trial_index])
        try:
            result = experiment.run(_SNR_DB, rng)
        except FloatingPointError:
            return {
                "index": index,
                "decoder_params": decoder_params,
                "invalid": True,
            }
        trials_completed += 1
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        iterations += int(result.n_iter)
        if frame_errors >= _MAX_ERRORS:
            break

    return {
        "index": index,
        "decoder_params": decoder_params,
        "invalid": False,
        "trials": trials_completed,
        "frame_errors": frame_errors,
        "fer": frame_errors / trials_completed,
        "ber": bit_errors / trials_completed,
        "average_iterations": iterations / trials_completed,
    }


def result_score(result):
    return result["fer"], result["ber"], result["average_iterations"]


def save_best(path, result, args, completed, total):
    payload = {
        "snr_db": args.snr,
        "max_trials": args.trials,
        "target_frame_errors": args.max_errors,
        "trials": result["trials"],
        "seed": args.seed,
        "completed_parameter_sets": completed,
        "total_parameter_sets": total,
        "frame_errors": result["frame_errors"],
        "fer": result["fer"],
        "ber": result["ber"],
        "average_iterations": result["average_iterations"],
        "decoder_params": result["decoder_params"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def default_workers(simulation_config):
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus:
        return int(allocated_cpus)
    local_cpus = os.cpu_count() or 1
    return min(
        int(simulation_config.get("n_workers", local_cpus)),
        local_cpus,
    )


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    pgd_compile()
    candidates = parameter_grid(
        args,
        base_experiment["codec"]["decoder_params"],
    )
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(
        args.workers or default_workers(simulation_config),
        len(candidates),
    )
    output_path = args.output.resolve()
    print(
        f"PGD search: SNR={args.snr:g} dB, "
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
            if result["invalid"]:
                params = json.dumps(
                    result["decoder_params"],
                    separators=(",", ":"),
                )
                print(
                    f"INVALID [{completed}/{len(candidates)}] "
                    f"params={params}",
                    flush=True,
                )
                continue
            if (
                best_result is None
                or result_score(result) < result_score(best_result)
            ):
                best_result = result
                save_best(
                    output_path,
                    result,
                    args,
                    completed,
                    len(candidates),
                )
                params = json.dumps(
                    result["decoder_params"],
                    separators=(",", ":"),
                )
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

    print(
        f"Search completed. Best parameters: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
