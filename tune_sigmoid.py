"""Grid-search sigmoid gradient decoder parameters at one SNR point."""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experement_sigmoid.json"
DEFAULT_OUTPUT = PROJECT_DIR / "sigmoid_params.json"
DEFAULT_THETAS = np.round(np.arange(0.01, 1.001, 0.01), 2)
DEFAULT_BETAS = np.round(np.arange(0.25, 4.01, 0.25), 2)

_BASE_EXPERIMENT = None
_SNR_DB = None
_MAX_TRIALS = None
_MAX_ERRORS = None
_SEED = None


def comma_separated_floats(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search sigmoid decoder theta and beta at a fixed SNR."
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--thetas", type=comma_separated_floats, default=DEFAULT_THETAS)
    parser.add_argument("--betas", type=comma_separated_floats, default=DEFAULT_BETAS)
    return parser.parse_args()


def validate_args(args):
    if args.trials <= 0 or args.max_errors <= 0:
        raise ValueError("--trials and --max-errors must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(theta <= 0 for theta in args.thetas):
        raise ValueError("all theta values must be positive")
    if any(beta <= 0 for beta in args.betas):
        raise ValueError("all beta values must be positive")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != (
        "sigmoid multi gradient descent bit-flipping"
    ):
        raise ValueError("the selected config does not use the sigmoid decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    baseline = {
        "theta": float(base_params["theta"]),
        "beta": float(base_params.get("beta", 1.0)),
    }
    candidates = [baseline]
    candidates.extend(
        {"theta": theta, "beta": beta}
        for theta, beta in itertools.product(args.thetas, args.betas)
    )
    unique = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


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
    settings = LdpcExperimentSettings(**experiment_config)
    experiment = LdpcExperimentInstance(settings)

    frame_errors = 0
    bit_errors = 0.0
    iterations = 0
    for trial_index in range(_MAX_TRIALS):
        rng = np.random.default_rng([_SEED, trial_index])
        result = experiment.run(_SNR_DB, rng)
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        iterations += int(result.n_iter)
        if frame_errors >= _MAX_ERRORS:
            break

    trials = trial_index + 1
    return {
        "index": index,
        "decoder_params": decoder_params,
        "trials": trials,
        "frame_errors": frame_errors,
        "fer": frame_errors / trials,
        "ber": bit_errors / trials,
        "average_iterations": iterations / trials,
    }


def result_score(result):
    return result["fer"], result["ber"], result["average_iterations"]


def save_best(path, result, args, completed, total):
    payload = {
        "snr_db": args.snr,
        "max_trials": args.trials,
        "target_frame_errors": args.max_errors,
        "completed_parameter_sets": completed,
        "total_parameter_sets": total,
        "seed": args.seed,
        **result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def default_workers(simulation_config):
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated:
        return int(allocated)
    return min(int(simulation_config.get("n_workers", os.cpu_count() or 1)), os.cpu_count() or 1)


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    candidates = parameter_grid(args, base_experiment["codec"]["decoder_params"])
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    print(
        f"Sigmoid search: SNR={args.snr:g} dB, parameter_sets={len(candidates)}, "
        f"workers={workers}",
        flush=True,
    )
    best_result = None
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(base_experiment, args.snr, args.trials, args.max_errors, args.seed),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(evaluate_candidate, enumerate(candidates), chunksize=1),
            start=1,
        ):
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(args.output.resolve(), result, args, completed, len(candidates))
                print(
                    f"NEW BEST [{completed}/{len(candidates)}] "
                    f"FER={result['fer']:.6g}, BER={result['ber']:.6g}, "
                    f"params={result['decoder_params']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()