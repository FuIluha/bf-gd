"""Grid-search soft GDBF decoder parameters at one SNR point."""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_soft_gdbf import lib_compile as cpp_soft_gdbf_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_soft_gdbf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_soft_gdbf.txt"

DEFAULT_LEARNING_RATES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_UPDATE_PROBABILITIES = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_BETA1_VALUES = (0.0, 0.5, 0.8, 0.9, 0.99)
DEFAULT_BETA2_VALUES = (0.8, 0.9, 0.99, 0.999)
DEFAULT_ADAM_EPSILONS = (1e-8, 1e-6, 1e-4, 1e-2)

_BASE_EXPERIMENT = None
_SNR_DB = None
_MAX_TRIALS = None
_MAX_ERRORS = None
_SEED = None


def comma_separated_floats(value):
    """Parse a comma-separated command-line list of floats."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search soft GDBF hyperparameters using FER at a fixed SNR. "
            "The best result is saved after every improvement."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=10_000_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-configs",
        type=int,
        help="evaluate only the first N sets; useful for a quick test",
    )
    parser.add_argument(
        "--learning-rates",
        type=comma_separated_floats,
        default=DEFAULT_LEARNING_RATES,
    )
    parser.add_argument(
        "--update-probabilities",
        type=comma_separated_floats,
        default=DEFAULT_UPDATE_PROBABILITIES,
    )
    parser.add_argument(
        "--beta1-values",
        type=comma_separated_floats,
        default=DEFAULT_BETA1_VALUES,
    )
    parser.add_argument(
        "--beta2-values",
        type=comma_separated_floats,
        default=DEFAULT_BETA2_VALUES,
    )
    parser.add_argument(
        "--adam-epsilons",
        type=comma_separated_floats,
        default=DEFAULT_ADAM_EPSILONS,
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
    if any(not 0 < value <= 1 for value in args.update_probabilities):
        raise ValueError("all update probabilities must be in (0, 1]")
    if any(not 0 <= value < 1 for value in args.beta1_values):
        raise ValueError("all beta1 values must be in [0, 1)")
    if any(not 0 <= value < 1 for value in args.beta2_values):
        raise ValueError("all beta2 values must be in [0, 1)")
    if any(value <= 0 for value in args.adam_epsilons):
        raise ValueError("all Adam epsilons must be positive")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") not in (
        "soft gradient descent bit-flipping",
        "cpp soft gradient descent bit-flipping",
    ):
        raise ValueError("the selected config does not use the soft GDBF decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    baseline = {
        "learning_rate": float(base_params["learning_rate"]),
        "update_probability": float(base_params["update_probability"]),
        "beta1": float(base_params["beta1"]),
        "beta2": float(base_params["beta2"]),
        "adam_epsilon": float(base_params["adam_epsilon"]),
    }
    candidates = [baseline]

    for values in itertools.product(
        args.learning_rates,
        args.update_probabilities,
        args.beta1_values,
        args.beta2_values,
        args.adam_epsilons,
    ):
        candidates.append({
            "learning_rate": values[0],
            "update_probability": values[1],
            "beta1": values[2],
            "beta2": values[3],
            "adam_epsilon": values[4],
        })

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
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
        result = experiment.run(_SNR_DB, rng)
        trials_completed += 1
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        iterations += int(result.n_iter)
        if frame_errors >= _MAX_ERRORS:
            break

    return {
        "index": index,
        "decoder_params": decoder_params,
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
    return min(int(simulation_config.get("n_workers", local_cpus)), local_cpus)


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    if base_experiment["codec"]["algorithm"].startswith("cpp "):
        cpp_soft_gdbf_compile()
    candidates = parameter_grid(
        args,
        base_experiment["codec"]["decoder_params"],
    )
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    print(
        f"Soft GDBF search: SNR={args.snr:g} dB, "
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
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, completed, len(candidates))
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

    print(f"Search completed. Best parameters: {output_path}", flush=True)


if __name__ == "__main__":
    main()
