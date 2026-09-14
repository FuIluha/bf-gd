"""Grid-search the C++ E-GDBF parameters at one SNR point."""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_egdbf import lib_compile as egdbf_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_egdbf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_egdbf.txt"

# 13 * 8 = 104 deterministic hard-message configurations.
DEFAULT_ALPHAS = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
    1.8, 2.0, 2.25, 2.5, 3.0, 4.0,
)
DEFAULT_RHO_PROFILES = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.25, 0.25, 0.25, 0.25, 0.25, 0.125, 0.125),
    (0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25),
    (1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5),
    (1.5, 1.5, 1.5, 1.5, 1.5, 0.75, 0.75),
    (2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0),
    (2.5, 2.5, 2.5, 2.5, 2.5, 1.25, 1.25),
    (3.0, 3.0, 3.0, 3.0, 3.0, 1.5, 1.5),
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
            "Search C++ E-GDBF hyperparameters using FER at a fixed SNR. "
            "Every new best result is saved immediately."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=-0.2)
    parser.add_argument("--trials", type=int, default=10_000_000)
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
        "--rho",
        type=comma_separated_floats,
        action="append",
        dest="rho_profiles",
        help=(
            "momentum profile, for example --rho 0.5,0.5,0.5,0.5,0.5,0.25,0.25; "
            "repeat the option to search several profiles"
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not np.isfinite(args.snr):
        raise ValueError("--snr must be finite")
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
    profiles = args.rho_profiles or DEFAULT_RHO_PROFILES
    for profile in profiles:
        if not profile or any(not np.isfinite(value) for value in profile):
            raise ValueError("rho profiles must be non-empty and finite")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != (
        "cpp edge-wise gradient descent bit-flipping"
    ):
        raise ValueError("the selected config must use the C++ E-GDBF decoder")
    return experiment, config.get("simulation", {})


def parameter_grid(args, base_params):
    rho_profiles = args.rho_profiles or DEFAULT_RHO_PROFILES
    baseline = {
        "alpha": float(base_params["alpha"]),
        "rho": [float(value) for value in base_params["rho"]],
        "L": int(base_params["L"]),
    }
    candidates = [baseline]
    for rho, alpha in itertools.product(
        rho_profiles,
        args.alphas,
    ):
        candidates.append({
            "alpha": float(alpha),
            "rho": [float(value) for value in rho],
            "L": len(rho),
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
        # Every candidate receives the same independent channel-frame seeds.
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
    egdbf_compile()
    candidates = parameter_grid(
        args,
        base_experiment["codec"]["decoder_params"],
    )
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(args.workers or default_workers(simulation_config), len(candidates))
    output_path = args.output.resolve()
    print(
        f"E-GDBF search: SNR={args.snr:g} dB, "
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
