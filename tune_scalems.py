"""Grid-search llr_scale for the layered min-sum LDPC decoder at one SNR point."""

import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiment.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params.txt"

DEFAULT_LLR_SCALES = np.round(np.arange(0.05, 1.001, 0.05), 3)

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
            "Search the h_layered_min_sum llr_scale hyperparameter using FER at a "
            "fixed SNR. Only the overall best result is reported at the end."
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
        default=200,
        help="stop a parameter set after this many frame errors (default: 200)",
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
        help="evaluate only the first N values; useful for a quick test",
    )
    parser.add_argument(
        "--llr-scales",
        type=comma_separated_floats,
        default=DEFAULT_LLR_SCALES,
        help="comma-separated list of llr_scale values to try",
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
    if any(scale <= 0 for scale in args.llr_scales):
        raise ValueError("all llr_scale values must be positive")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    codec = experiment["codec"]
    if codec.get("algorithm") != "h_layered_min_sum":
        raise ValueError("the selected config does not use the h_layered_min_sum decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_llr_scale):
    candidates = [float(base_llr_scale)]
    for scale in args.llr_scales:
        candidates.append(float(scale))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = round(candidate, 6)
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
    index, llr_scale, max_trials, max_errors = candidate_and_budget
    experiment_config = copy.deepcopy(_BASE_EXPERIMENT)
    experiment_config["codec"]["llr_scale"] = llr_scale
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
        "llr_scale": llr_scale,
        "trials": trials_completed,
        "frame_errors": frame_errors,
        "fer": frame_errors / trials_completed,
        "ber": bit_errors / trials_completed,
        "average_iterations": iterations / trials_completed,
    }


def result_score(result):
    """FER is primary; BER and decoding work break statistically equal ties."""
    return result["fer"], result["ber"], result["average_iterations"]


def save_best(path, result, args):
    payload = {
        "snr_db": args.snr,
        "max_trials": args.trials,
        "target_frame_errors": args.max_errors,
        "trials": result["trials"],
        "seed": args.seed,
        "frame_errors": result["frame_errors"],
        "fer": result["fer"],
        "ber": result["ber"],
        "average_iterations": result["average_iterations"],
        "llr_scale": result["llr_scale"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def print_best(result, total, output_path):
    print(
        f"BEST of {total} llr_scale values: "
        f"llr_scale={result['llr_scale']:g} "
        f"FER={result['fer']:.6g} "
        f"({result['frame_errors']} errors / {result['trials']} trials), "
        f"BER={result['ber']:.6g}, "
        f"avg_iter={result['average_iterations']:.3f}\n"
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
        (index, llr_scale, max_trials, max_errors)
        for index, llr_scale in enumerate(candidates)
    ]
    return list(pool.imap_unordered(evaluate_candidate, jobs, chunksize=1))


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    base_llr_scale = base_experiment["codec"].get("llr_scale", 1.0)
    candidates = parameter_grid(args, base_llr_scale)
    if args.max_configs is not None:
        candidates = candidates[: args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    print(
        f"llr_scale search: SNR={args.snr:g} dB, "
        f"trials={args.trials}/errors={args.max_errors}, "
        f"parameter_sets={len(candidates)}, workers={workers}",
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
        results = evaluate_stage(pool, candidates, args.trials, args.max_errors)

    best_result = min(results, key=result_score)
    save_best(output_path, best_result, args)
    print_best(best_result, len(results), output_path)


if __name__ == "__main__":
    main()