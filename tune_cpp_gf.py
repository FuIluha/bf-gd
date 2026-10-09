"""Grid-search the C++ gradient flow (GF) parameters at one SNR point.

The GF update is  x <- x - eta * (x - y + gamma * grad h_{alpha,beta}(x)).
grad h is linear in (alpha, beta), so only the products gamma*alpha and gamma*beta
matter: gamma is kept in the interface but its default grid is the single value 1.
"""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_gf import lib_compile as gf_compile
from lbc_encoder.lbc_encoder import lib_compile as lbc_compile
from simulator_awgn_python.channel import lib_compile as chan_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_gf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_gf.txt"

# 14 * 14 * 1 * 16 = 3136 combinations, denser around the region found
# useful in earlier searches (alpha ~ 0.1-0.5, beta ~ 1-3, eta ~ 0.007-0.014).
DEFAULT_ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
DEFAULT_BETAS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)
DEFAULT_GAMMAS = (1.0,)
DEFAULT_ETAS = (
    0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
    0.011, 0.012, 0.014, 0.016, 0.020, 0.025, 0.030, 0.040,
)

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
        description=(
            "Search C++ GF hyperparameters using FER at a fixed SNR. "
            "Every new best result is saved immediately."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=-1.0)
    parser.add_argument("--trials", type=int, default=100_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--iterations", type=int,
                        help="override n_iterations from the config")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--alphas", type=comma_separated_floats, default=DEFAULT_ALPHAS)
    parser.add_argument("--betas", type=comma_separated_floats, default=DEFAULT_BETAS)
    parser.add_argument("--gammas", type=comma_separated_floats, default=DEFAULT_GAMMAS)
    parser.add_argument("--etas", type=comma_separated_floats, default=DEFAULT_ETAS)
    return parser.parse_args()


def validate_args(args):
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(not np.isfinite(v) or v < 0 for v in args.alphas + args.betas + args.gammas):
        raise ValueError("alpha, beta and gamma values must be finite and non-negative")
    if any(not np.isfinite(v) or v <= 0 for v in args.etas):
        raise ValueError("all eta values must be positive")


def load_base_experiment(config_path, iterations):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != "cpp gf":
        raise ValueError('the selected config must use the "cpp gf" algorithm')
    if iterations is not None:
        experiment["codec"]["n_iterations"] = iterations
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    baseline = {
        "alpha": float(base_params["alpha"]),
        "beta": float(base_params["beta"]),
        "gamma": float(base_params["gamma"]),
        "eta": float(base_params["eta"]),
    }
    candidates = [baseline]
    for alpha, beta, gamma, eta in itertools.product(
        args.alphas, args.betas, args.gammas, args.etas,
    ):
        candidates.append({"alpha": alpha, "beta": beta, "gamma": gamma, "eta": eta})

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


def format_params(decoder_params):
    return "  ".join(f"{name}={value:g}" for name, value in decoder_params.items())


def save_best(path, result, args, iterations, completed, total):
    payload = {
        "snr_db": args.snr,
        "n_iterations": iterations,
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
    base_experiment, simulation_config = load_base_experiment(args.config, args.iterations)
    iterations = base_experiment["codec"]["n_iterations"]

    chan_compile()
    lbc_compile()
    gf_compile()
    candidates = parameter_grid(args, base_experiment["codec"]["decoder_params"])
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    print(
        f"GF search: SNR={args.snr:g} dB, iterations={iterations}, "
        f"max_trials={args.trials}, target_errors={args.max_errors}, "
        f"parameter_sets={len(candidates)}, workers={workers}",
        flush=True,
    )

    output_path.with_suffix(".jsonl").write_text("", encoding="utf-8")
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
            if result["invalid"]:
                print(
                    f"INVALID [{completed}/{len(candidates)}] "
                    f"{format_params(result['decoder_params'])}",
                    flush=True,
                )
                continue
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, iterations, completed, len(candidates))
                print(
                    f"NEW BEST [{completed}/{len(candidates)}] "
                    f"FER={result['fer']:.6g} "
                    f"({result['frame_errors']} errors / "
                    f"{result['trials']} trials), "
                    f"BER={result['ber']:.6g}, "
                    f"avg_iter={result['average_iterations']:.3f}\n"
                    f"params: {format_params(result['decoder_params'])}\n"
                    f"saved to {output_path}",
                    flush=True,
                )

    print(f"Search completed. Best parameters: {output_path}", flush=True)


if __name__ == "__main__":
    main()
