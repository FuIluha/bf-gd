"""Grid-search Sigmoid1 probabilistic momentum decoder parameters at one SNR point."""

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
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_sigmoid1.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params.txt"

DEFAULT_DELTAS = np.round(np.arange(0.8, 1.201, 0.05), 2)
DEFAULT_ALPHAS = np.round(np.arange(1.5, 2.001, 0.05), 2)
DEFAULT_BETAS = np.round(np.arange(0.25, 4.01, 0.25), 2)
DEFAULT_PROBABILITIES = np.round(np.arange(0.8, 1.001, 0.05), 2)
DEFAULT_REGULARIZATIONS = np.round(np.arange(0.0, 0.51, 0.05), 2)

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


def rho_profile(value):
    """Parse one comma-separated momentum profile."""
    values = comma_separated_floats(value)
    if not values:
        raise argparse.ArgumentTypeError("rho must not be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search Sigmoid1 decoder hyperparameters using FER at a fixed SNR. "
            "The best result is rewritten to params.txt after every improvement."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument(
        "--trials",
        type=int,
        default=10_000_000,
        help="maximum frames for every parameter set (default: 10000000)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="stop a parameter set after this many frame errors (default: 10)",
    )
    parser.add_argument(
        "--screen-trials",
        type=int,
        default=300,
        help="frames per candidate in the screening stage (default: 300)",
    )
    parser.add_argument(
        "--screen-max-errors",
        type=int,
        default=5,
        help="errors per candidate in screening (default: 5)",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=10,
        help="number of screening winners for the full run (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="parallel parameter sets (default: allocated Slurm CPUs or local CPUs)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-configs",
        type=int,
        help="evaluate only the first N sets; useful for a quick test",
    )
    parser.add_argument(
        "--deltas",
        type=comma_separated_floats,
        default=DEFAULT_DELTAS,
    )
    parser.add_argument(
        "--alphas",
        type=comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--betas",
        type=comma_separated_floats,
        default=DEFAULT_BETAS,
    )
    parser.add_argument(
        "--probabilities",
        type=comma_separated_floats,
        default=DEFAULT_PROBABILITIES,
    )
    parser.add_argument(
        "--regularizations",
        type=comma_separated_floats,
        default=DEFAULT_REGULARIZATIONS,
    )
    parser.add_argument(
        "--rho",
        type=rho_profile,
        action="append",
        dest="rho_profiles",
        help=(
            "momentum profile, for example --rho 2,2,2,2,2,1,1; "
            "repeat the option to search several profiles"
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if args.screen_trials <= 0:
        raise ValueError("--screen-trials must be positive")
    if args.screen_max_errors <= 0:
        raise ValueError("--screen-max-errors must be positive")
    if args.finalists <= 0:
        raise ValueError("--finalists must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(beta <= 0 for beta in args.betas):
        raise ValueError("all beta values must be positive")
    if any(probability <= 0 or probability > 1 for probability in args.probabilities):
        raise ValueError("all probabilities must be in (0, 1]")
    if any(reg < 0 for reg in args.regularizations):
        raise ValueError("all regularization values must be non-negative")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    codec = experiment["codec"]
    if codec.get("algorithm") != (
        "sigmoid1 probabilistic momentum gradient descent bit-flipping"
    ):
        raise ValueError("the selected config does not use the Sigmoid1 decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    rho_profiles = args.rho_profiles or [tuple(base_params["rho"])]
    baseline = {
        "delta": float(base_params["delta"]),
        "alpha": float(base_params["alpha"]),
        "beta": float(base_params.get("beta", 1.0)),
        "p": float(base_params["p"]),
        "regularization": float(base_params.get("regularization", 0.0)),
        "rho": list(base_params["rho"]),
        "L": int(base_params["L"]),
    }

    candidates = [baseline]
    for delta, alpha, beta, probability, regularization, rho in itertools.product(
        args.deltas,
        args.alphas,
        args.betas,
        args.probabilities,
        args.regularizations,
        rho_profiles,
    ):
        candidates.append(
            {
                "delta": delta,
                "alpha": alpha,
                "beta": beta,
                "p": probability,
                "regularization": regularization,
                "rho": list(rho),
                "L": len(rho),
            }
        )

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


def evaluate_candidate(candidate_and_budget):
    index, decoder_params, max_trials, max_errors = candidate_and_budget
    experiment_config = copy.deepcopy(_BASE_EXPERIMENT)
    experiment_config["codec"]["decoder_params"] = decoder_params
    settings = LdpcExperimentSettings(**experiment_config)
    experiment = LdpcExperimentInstance(settings)

    frame_errors = 0
    bit_errors = 0.0
    iterations = 0
    trials_completed = 0
    for trial_index in range(max_trials):
        # Identical seeds make every candidate see the same channel realizations.
        rng = np.random.default_rng([_SEED, trial_index])
        result = experiment.run(_SNR_DB, rng)
        trials_completed += 1
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        iterations += int(result.n_iter)
        if frame_errors >= max_errors:
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
    """FER is primary; BER and decoding work break statistically equal ties."""
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


def print_best(result, completed, total, output_path):
    params = json.dumps(result["decoder_params"], separators=(",", ":"))
    print(
        f"NEW BEST [{completed}/{total}] "
        f"FER={result['fer']:.6g} "
        f"({result['frame_errors']} errors / {result['trials']} trials), "
        f"BER={result['ber']:.6g}, "
        f"avg_iter={result['average_iterations']:.3f}\n"
        f"params={params}\n"
        f"saved to {output_path}",
        flush=True,
    )


def default_workers(simulation_config):
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus:
        return int(allocated_cpus)
    local_cpus = os.cpu_count() or 1
    return min(int(simulation_config.get("n_workers", local_cpus)), local_cpus)


def evaluate_stage(pool, candidates, max_trials, max_errors):
    jobs = [
        (index, decoder_params, max_trials, max_errors)
        for index, decoder_params in enumerate(candidates)
    ]
    return list(pool.imap_unordered(evaluate_candidate, jobs, chunksize=1))


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    base_params = base_experiment["codec"]["decoder_params"]
    candidates = parameter_grid(args, base_params)
    if args.max_configs is not None:
        candidates = candidates[: args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    print(
        f"Sigmoid1 search: SNR={args.snr:g} dB, "
        f"screen={args.screen_trials} frames/{args.screen_max_errors} errors, "
        f"final={args.trials} frames/{args.max_errors} errors, "
        f"parameter_sets={len(candidates)}, finalists={args.finalists}, "
        f"workers={workers}",
        flush=True,
    )

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
        screen_results = evaluate_stage(
            pool,
            candidates,
            args.screen_trials,
            args.screen_max_errors,
        )
        screen_results.sort(key=result_score)
        finalists = screen_results[: min(args.finalists, len(screen_results))]
        print(
            f"Screening completed: {len(screen_results)} candidates; "
            f"running {len(finalists)} finalists",
            flush=True,
        )

        final_candidates = [candidates[result["index"]] for result in finalists]
        final_results = evaluate_stage(
            pool,
            final_candidates,
            args.trials,
            args.max_errors,
        )

        best_result = None
        for completed, result in enumerate(final_results, start=1):
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, completed, len(final_results))
                print_best(result, completed, len(final_results), output_path)

    print(f"Search completed. Best parameters: {output_path}", flush=True)


if __name__ == "__main__":
    main()
