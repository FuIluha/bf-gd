"""Grid-search the C++ E-GDBF parameters at one SNR point."""

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import os
import math
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_egdbf import lib_compile as egdbf_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_egdbf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_egdbf.txt"

# Dense search: 50 channel weights and all unique two-level, length-7
# momentum profiles generated from the ranges below (63,250 sets total).
DEFAULT_ALPHAS = tuple(np.round(np.arange(0.1, 5.001, 0.1), 2))
DEFAULT_RHO_EARLY_VALUES = tuple(np.round(np.arange(0.0, 4.001, 0.25), 2))
DEFAULT_RHO_LATE_VALUES = tuple(np.round(np.arange(0.0, 3.001, 0.25), 2))
DEFAULT_RHO_SPLITS = (1, 2, 3, 4, 5, 6, 7)

_BASE_EXPERIMENT = None
_SNR_DB = None
_MAX_TRIALS = None
_MAX_ERRORS = None
_SEED = None
_STAGE = None
_FIXED_TRIALS = False


def comma_separated_floats(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid number list: {value!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError("the list must not be empty")
    return values


def comma_separated_ints(value):
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
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
    parser.add_argument("--mode", choices=("staged", "grid"), default="staged")
    parser.add_argument("--candidates", type=int, default=2048,
                        help="number of broad candidates in staged mode")
    parser.add_argument("--screen-trials", type=int, default=2000)
    parser.add_argument("--refine-top", type=int, default=24)
    parser.add_argument("--refine-trials", type=int, default=5000)
    parser.add_argument("--final-top", type=int, default=8)
    parser.add_argument("--final-errors", type=int, default=500,
                        help="target errors in the best finalist; capped by --trials")
    parser.add_argument("--rho-max", type=float, default=4.0)
    parser.add_argument("--rho-step", type=float, default=0.25)
    parser.add_argument("--rho-unconstrained-fraction", type=float, default=0.25,
                        help="share of full rho profiles without monotonicity")
    parser.add_argument(
        "--alphas",
        type=comma_separated_floats,
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument(
        "--deltas",
        type=comma_separated_floats,
        help="threshold offsets; staged default is 0..2 by 0.1, grid uses config delta",
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
    parser.add_argument(
        "--rho-early-values",
        type=comma_separated_floats,
        default=DEFAULT_RHO_EARLY_VALUES,
        help="values before the momentum-profile split",
    )
    parser.add_argument(
        "--rho-late-values",
        type=comma_separated_floats,
        default=DEFAULT_RHO_LATE_VALUES,
        help="values from the momentum-profile split onward",
    )
    parser.add_argument(
        "--rho-splits",
        type=comma_separated_ints,
        default=DEFAULT_RHO_SPLITS,
        help="counts of leading entries using the early rho value",
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
    if any(value <= 0 for value in (
        args.candidates, args.screen_trials, args.refine_top,
        args.refine_trials, args.final_top, args.final_errors,
    )):
        raise ValueError("staged search counts must be positive")
    if not np.isfinite(args.rho_max) or args.rho_max < 0:
        raise ValueError("--rho-max must be finite and non-negative")
    if not np.isfinite(args.rho_step) or args.rho_step <= 0:
        raise ValueError("--rho-step must be finite and positive")
    if not np.isfinite(args.rho_unconstrained_fraction) or not (
        0 <= args.rho_unconstrained_fraction <= 1
    ):
        raise ValueError("--rho-unconstrained-fraction must be in [0, 1]")
    if args.mode == "staged" and args.screen_trials > args.trials:
        raise ValueError("--screen-trials must not exceed --trials")
    if args.mode == "staged" and args.refine_trials > args.trials:
        raise ValueError("--refine-trials must not exceed --trials")
    if any(not np.isfinite(value) or value <= 0 for value in args.alphas):
        raise ValueError("all alphas must be finite and positive")
    if args.deltas is not None and any(
        not np.isfinite(value) or value < 0 for value in args.deltas
    ):
        raise ValueError("all deltas must be finite and non-negative")
    if any(
        not np.isfinite(value) or value < 0
        for value in args.rho_early_values + args.rho_late_values
    ):
        raise ValueError("generated rho values must be finite and non-negative")
    if any(split <= 0 for split in args.rho_splits):
        raise ValueError("rho splits must be positive")
    for profile in args.rho_profiles or ():
        if not profile or any(not np.isfinite(value) for value in profile):
            raise ValueError("rho profiles must be non-empty and finite")


def load_base_experiment(config_path):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") not in (
        "cpp edge-wise gradient descent bit-flipping",
        "cpp edge-wise gradient descent bit-flipping v2",
    ):
        raise ValueError("the selected config must use a C++ E-GDBF decoder")
    return experiment, config.get("simulation", {})


def generated_rho_profiles(args, momentum_length):
    if args.rho_profiles:
        return args.rho_profiles
    if momentum_length == 0:
        return [()]
    if any(split > momentum_length for split in args.rho_splits):
        raise ValueError("rho splits must not exceed the configured L")

    profiles = []
    seen = set()
    for early, late, split in itertools.product(
        args.rho_early_values,
        args.rho_late_values,
        args.rho_splits,
    ):
        profile = (
            (float(early),) * split
            + (float(late),) * (momentum_length - split)
        )
        if profile not in seen:
            seen.add(profile)
            profiles.append(profile)
    return profiles


def parameter_grid(args, base_params):
    rho_profiles = generated_rho_profiles(args, int(base_params["L"]))
    base_delta = float(base_params.get("delta", 0.0))
    delta_values = args.deltas if args.deltas is not None else (base_delta,)
    baseline = {
        "alpha": float(base_params["alpha"]),
        "delta": base_delta,
        "rho": [float(value) for value in base_params["rho"]],
        "L": int(base_params["L"]),
    }
    candidates = [baseline]
    for rho, alpha, delta in itertools.product(
        rho_profiles,
        args.alphas,
        delta_values,
    ):
        candidates.append({
            "alpha": float(alpha),
            "delta": float(delta),
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


def unique_candidates(candidates):
    result = []
    seen = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def broad_candidates(args, base_params):
    """Sample the full rho vector, plus alpha and delta, reproducibly."""
    rng = np.random.default_rng(args.seed)
    momentum_length = int(base_params["L"])
    delta_values = args.deltas or tuple(np.round(np.arange(0, 2.001, 0.1), 2))
    rho_values = np.round(
        np.arange(0, args.rho_max + args.rho_step / 2, args.rho_step), 6,
    )
    baseline = {
        "alpha": float(base_params["alpha"]),
        "delta": float(base_params.get("delta", 0.0)),
        "rho": [float(value) for value in base_params["rho"]],
        "L": momentum_length,
    }
    target = min(args.candidates, args.max_configs or args.candidates)
    candidates = [baseline]
    if momentum_length > 0 and target > 1:
        candidates.append({**baseline, "rho": [], "L": 0})
    seen = {json.dumps(candidate, sort_keys=True) for candidate in candidates}
    for _ in range(target * 20):
        if len(candidates) >= target:
            break
        if args.rho_profiles:
            rho = list(args.rho_profiles[rng.integers(len(args.rho_profiles))])
        elif momentum_length == 0 or rng.random() < 0.02:
            rho = []
        elif rng.random() < 0.25:
            early, late = sorted(
                rng.choice(rho_values, size=2), reverse=True,
            )
            split = int(rng.integers(1, momentum_length + 1))
            rho = [float(early)] * split + [float(late)] * (momentum_length - split)
        else:
            rho = [float(value) for value in rng.choice(rho_values, size=momentum_length)]
            if rng.random() >= args.rho_unconstrained_fraction:
                rho.sort(reverse=True)
        candidate = {
            "alpha": float(args.alphas[rng.integers(len(args.alphas))]),
            "delta": float(delta_values[rng.integers(len(delta_values))]),
            "rho": rho,
            "L": len(rho),
        }
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return candidates


def local_candidates(best_results):
    """Refine each selected coordinate at half the broad-grid spacing."""
    candidates = []
    for result in best_results:
        base = result["decoder_params"]
        candidates.append(base)
        for offset in (-0.05, 0.05):
            changed = copy.deepcopy(base)
            changed["alpha"] = round(base["alpha"] + offset, 6)
            if changed["alpha"] > 0:
                candidates.append(changed)
            changed = copy.deepcopy(base)
            changed["delta"] = round(base["delta"] + offset, 6)
            if changed["delta"] >= 0:
                candidates.append(changed)
        for index in range(base["L"]):
            for offset in (-0.125, 0.125):
                changed = copy.deepcopy(base)
                value = round(changed["rho"][index] + offset, 6)
                if value < 0:
                    continue
                changed["rho"][index] = value
                candidates.append(changed)
    return unique_candidates(candidates)


def init_worker(base_experiment, snr_db, max_trials, max_errors, seed,
                stage=0, fixed_trials=False):
    global _BASE_EXPERIMENT, _SNR_DB, _MAX_TRIALS, _MAX_ERRORS, _SEED
    global _STAGE, _FIXED_TRIALS
    _BASE_EXPERIMENT = base_experiment
    _SNR_DB = snr_db
    _MAX_TRIALS = max_trials
    _MAX_ERRORS = max_errors
    _SEED = seed
    _STAGE = stage
    _FIXED_TRIALS = fixed_trials


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
        rng = np.random.default_rng([_SEED, _STAGE, trial_index])
        result = experiment.run(_SNR_DB, rng)
        trials_completed += 1
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        iterations += int(result.n_iter)
        if not _FIXED_TRIALS and frame_errors >= _MAX_ERRORS:
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


def save_best(path, result, args, completed, total, stage=None):
    payload = {
        "snr_db": args.snr,
        "max_trials": args.trials,
        "target_frame_errors": (
            args.max_errors if stage is None else
            args.final_errors if stage in ("final", "verify") else None
        ),
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
    if stage is not None:
        payload["stage"] = stage
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


def run_staged_phase(args, base_experiment, candidates, trials, stage,
                     output_path, workers):
    workers = min(workers, len(candidates))
    print(
        f"{stage}: {len(candidates)} parameter sets, "
        f"{trials} common trials per set, {workers} workers",
        flush=True,
    )
    results_seen = []
    best_result = None
    with mp.get_context("spawn").Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(base_experiment, args.snr, trials, args.max_errors,
                  args.seed,
                  {"screen": 1, "refine": 2, "final": 3, "verify": 4}[stage],
                  True),
    ) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(evaluate_candidate, enumerate(candidates), chunksize=1),
            start=1,
        ):
            result["stage"] = stage
            results_seen.append(result)
            with output_path.with_suffix(".jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(result) + "\n")
            if best_result is None or result_score(result) < result_score(best_result):
                best_result = result
                save_best(output_path, result, args, completed, len(candidates), stage)
                print(
                    f"{stage} best [{completed}/{len(candidates)}]: "
                    f"FER={result['fer']:.6g}, "
                    f"errors={result['frame_errors']}/{trials}, "
                    f"params={json.dumps(result['decoder_params'], separators=(',', ':'))}",
                    flush=True,
                )
    return sorted(results_seen, key=result_score)


def run_staged_search(args, base_experiment, output_path, workers):
    base_params = base_experiment["codec"]["decoder_params"]
    broad = broad_candidates(args, base_params)
    screened = run_staged_phase(
        args, base_experiment, broad, args.screen_trials, "screen",
        output_path, workers,
    )
    refined_candidates = local_candidates(screened[:args.refine_top])
    refined = run_staged_phase(
        args, base_experiment, refined_candidates, args.refine_trials,
        "refine", output_path, workers,
    )
    finalists = [item["decoder_params"] for item in refined[:args.final_top]]
    estimated_fer = max(refined[0]["fer"], 1 / args.refine_trials)
    final_trials = min(
        args.trials,
        max(args.refine_trials, math.ceil(args.final_errors / estimated_fer)),
    )
    final = run_staged_phase(
        args, base_experiment, finalists, final_trials, "final",
        output_path, workers,
    )
    while final[0]["frame_errors"] < args.final_errors and final_trials < args.trials:
        factor = max(
            2,
            math.ceil(args.final_errors / max(final[0]["frame_errors"], 1)),
        )
        final_trials = min(args.trials, final_trials * factor)
        final = run_staged_phase(
            args, base_experiment, finalists, final_trials, "final",
            output_path, workers,
        )
    selected = final[0]["decoder_params"]
    verified = run_staged_phase(
        args, base_experiment, [selected], final_trials, "verify",
        output_path, workers,
    )
    verify_trials = final_trials
    while verified[0]["frame_errors"] < args.final_errors and verify_trials < args.trials:
        factor = max(
            2,
            math.ceil(args.final_errors / max(verified[0]["frame_errors"], 1)),
        )
        verify_trials = min(args.trials, verify_trials * factor)
        verified = run_staged_phase(
            args, base_experiment, [selected], verify_trials, "verify",
            output_path, workers,
        )
    print(
        f"Selected parameters: {json.dumps(selected, separators=(',', ':'))}; "
        f"independent FER={verified[0]['fer']:.6g} "
        f"({verified[0]['frame_errors']}/{verify_trials}); "
        f"saved to {output_path}",
        flush=True,
    )
    if verified[0]["frame_errors"] < args.final_errors:
        print(
            "Verification precision target was not reached before the trial cap; "
            "increase --trials for a tighter FER estimate.",
            flush=True,
        )


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    if args.mode == "staged" and base_experiment["codec"]["algorithm"] != (
        "cpp edge-wise gradient descent bit-flipping"
    ):
        raise ValueError("staged search currently targets E-GDBF V1")
    egdbf_compile()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".jsonl").write_text("", encoding="utf-8")
    workers = args.workers or default_workers(simulation_config)
    if args.mode == "staged":
        run_staged_search(args, base_experiment, output_path, workers)
        return
    candidates = parameter_grid(
        args,
        base_experiment["codec"]["decoder_params"],
    )
    if args.max_configs is not None:
        candidates = candidates[:args.max_configs]

    workers = min(workers, len(candidates))
    print(
        f"E-GDBF search: SNR={args.snr:g} dB, "
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
            0,
            False,
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
