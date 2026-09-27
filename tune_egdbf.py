"""Tune C++ E-GDBF or diagnose its failed decoding frames."""

import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import math
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.bin_ldpc_egdbf import BinLdpcEgdbfDecoder
from ldpc_py.cpp_bin_ldpc_egdbf import CppBinLdpcEgdbfDecoder
from ldpc_py.cpp_bin_ldpc_egdbf import SOURCE_HASH, lib_compile as egdbf_compile
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
_DIAGNOSTIC_EXPERIMENT = None
_DIAGNOSTIC_SEED = None


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
    parser.add_argument("--mode", choices=("staged", "grid", "diagnose"), default="staged")
    parser.add_argument("--no-momentum", action="store_true",
                        help="search with L=0 and rho=[] regardless of the config")
    parser.add_argument("--snrs", type=comma_separated_floats,
                        default=(0.3, 0.5, 0.8),
                        help="SNR points for diagnostic mode")
    parser.add_argument("--failures-per-snr", type=int, default=100,
                        help="failed frames to capture at each diagnostic SNR")
    parser.add_argument("--replay-only", action="store_true",
                        help="re-analyze saved diagnostic frames without new trials")
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
    if any(not np.isfinite(snr) for snr in args.snrs):
        raise ValueError("--snrs must contain only finite values")
    if len(set(args.snrs)) != len(args.snrs):
        raise ValueError("--snrs must not contain duplicates")
    if args.failures_per_snr <= 0:
        raise ValueError("--failures-per-snr must be positive")
    if args.replay_only and args.mode != "diagnose":
        raise ValueError("--replay-only requires --mode diagnose")
    if args.no_momentum and args.mode == "diagnose":
        raise ValueError("--no-momentum is only available for parameter search")
    if args.no_momentum and args.rho_profiles:
        raise ValueError("--rho cannot be combined with --no-momentum")
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
    if args.no_momentum:
        grid_args = copy.copy(args)
        grid_args.deltas = delta_values
        candidates = parameter_grid(grid_args, base_params)
        return candidates[:args.max_configs] if args.max_configs else candidates
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


def init_diagnostic_worker(base_experiment, seed):
    global _DIAGNOSTIC_EXPERIMENT, _DIAGNOSTIC_SEED
    _DIAGNOSTIC_EXPERIMENT = LdpcExperimentInstance(
        LdpcExperimentSettings(**base_experiment)
    )
    _DIAGNOSTIC_SEED = seed


def collect_diagnostic_batch(task):
    snr_index, snr_db, first_trial, batch_size = task
    experiment = _DIAGNOSTIC_EXPERIMENT
    failures = []
    disagreements = []
    for trial_index in range(first_trial, first_trial + batch_size):
        rng = np.random.default_rng([_DIAGNOSTIC_SEED, snr_index, trial_index])
        result = experiment.run(snr_db, rng)
        actual_error = bool(np.any((experiment.llr_out < 0) != experiment.tx_bits))
        if bool(result.fe_cum) != actual_error:
            disagreements.append(trial_index)
        if actual_error:
            failures.append({
                "snr_index": snr_index,
                "snr_db": snr_db,
                "trial_index": trial_index,
                "received": experiment.llr_in.copy(),
                "decoded": experiment.llr_out.copy(),
                "tx_bits": experiment.tx_bits.copy(),
                "iterations": int(result.n_iter),
                "reported_error": bool(result.fe_cum),
            })
    return failures, disagreements


def collect_diagnostic_failures(args, base_experiment, workers):
    all_failures = []
    trial_counts = {}
    discrepancy_counts = {}
    batch_size = 256
    for snr_index, snr_db in enumerate(args.snrs):
        failures = []
        disagreements = 0
        trials_completed = 0
        tasks = (
            (snr_index, snr_db, first, min(batch_size, args.trials - first))
            for first in range(0, args.trials, batch_size)
        )
        print(f"Collecting {args.failures_per_snr} failures at {snr_db:g} dB", flush=True)
        with mp.get_context("spawn").Pool(
            processes=workers,
            initializer=init_diagnostic_worker,
            initargs=(base_experiment, args.seed),
        ) as pool:
            for found, mismatches in pool.imap(collect_diagnostic_batch, tasks):
                for record in found:
                    failures.append(record)
                    if len(failures) == args.failures_per_snr:
                        break
                if len(failures) == args.failures_per_snr:
                    trials_completed = failures[-1]["trial_index"] + 1
                    disagreements += sum(index < trials_completed for index in mismatches)
                    break
                disagreements += len(mismatches)
                trials_completed += batch_size
                if len(failures) and len(failures) % 25 < len(found):
                    print(f"  {len(failures)} failures in {trials_completed} trials", flush=True)
        trials_completed = min(trials_completed, args.trials)
        trial_counts[str(snr_db)] = trials_completed
        discrepancy_counts[str(snr_db)] = disagreements
        all_failures.extend(failures)
        print(f"  {len(failures)} failures in {trials_completed} trials", flush=True)
        if len(failures) < args.failures_per_snr:
            raise RuntimeError(
                f"Only {len(failures)} failures at {snr_db:g} dB before "
                f"the {args.trials} trial cap"
            )
    return all_failures, trial_counts, discrepancy_counts


def trace_egdbf_frame(decoder, received, tx_bits):
    """Replay the Python V1 update, retaining compact per-iteration diagnostics."""
    limit = decoder.n_iterations
    channel_signs = np.where(received >= 0, 1, -1).astype(np.int8)
    q = channel_signs[decoder.edge_vn].copy()
    ages = np.full(decoder.edges_count, decoder.L + 1, dtype=np.int32) if decoder.L else None
    target = (1 - 2 * tx_bits.astype(np.int16)).astype(np.int8)
    syndrome_weights = np.full(limit + 1, -1, dtype=np.int32)
    bit_errors = np.full(limit + 1, -1, dtype=np.int32)
    flip_counts = np.full(limit, -1, dtype=np.int32)
    thresholds = np.full(limit, np.nan, dtype=np.float64)
    seen_states = {}
    cycle_start = None
    cycle_period = None
    previous_word = None
    unchanged_tail = 0
    longest_unchanged = 0
    for iteration in range(limit + 1):
        state = q.tobytes() + (ages.tobytes() if ages is not None else b"")
        if cycle_start is None:
            previous_iteration = seen_states.get(state)
            if previous_iteration is not None:
                cycle_start = previous_iteration
                cycle_period = iteration - previous_iteration
            else:
                seen_states[state] = iteration

        check_messages = decoder.check_to_variable_messages(q)
        incoming = decoder.incoming_sums(check_messages)
        x = decoder.hard_word(decoder.posterior_score(received, incoming), channel_signs)
        syndrome = decoder.bpsk_syndrome(x)
        syndrome_weights[iteration] = np.count_nonzero(syndrome != 1)
        bit_errors[iteration] = np.count_nonzero(x != target)
        if previous_word is not None and np.array_equal(x, previous_word):
            unchanged_tail += 1
        else:
            unchanged_tail = 0
        longest_unchanged = max(longest_unchanged, unchanged_tail)
        previous_word = x.copy()

        if syndrome_weights[iteration] == 0 or iteration == limit:
            return {
                "iterations": iteration,
                "decoded": x,
                "syndrome_weights": syndrome_weights,
                "bit_errors": bit_errors,
                "flip_counts": flip_counts,
                "thresholds": thresholds,
                "cycle_start": cycle_start,
                "cycle_period": cycle_period,
                "unchanged_tail": unchanged_tail,
                "longest_unchanged": longest_unchanged,
                "error_support": np.flatnonzero(x != target),
                "unsatisfied_checks": np.flatnonzero(syndrome != 1),
            }

        if ages is not None:
            np.minimum(ages, decoder.L, out=ages)
            ages += 1
        energies = decoder.edge_energies(received, q, check_messages, incoming, ages)
        threshold = decoder.energy_threshold(energies)
        flipped = energies <= threshold
        flip_counts[iteration] = np.count_nonzero(flipped)
        thresholds[iteration] = threshold
        q[flipped] *= -1
        if ages is not None:
            ages[flipped] = 0


def qc_shift_is_automorphism(pcm, lift=54):
    checks, bits = pcm.shape
    if checks % lift or bits % lift:
        return False
    blocks = pcm.reshape(checks // lift, lift, bits // lift, lift)
    return np.array_equal(blocks, np.roll(blocks, 1, axis=(1, 3)))


def canonical_error_support(support, lift=54):
    return min(
        tuple(sorted((index // lift) * lift + (index + shift) % lift
                     for index in support))
        for shift in range(lift)
    )


def empirical_curve_summary(txt_path):
    if not txt_path.exists():
        return None
    data = np.genfromtxt(txt_path, names=True)
    points = {}
    for snr in (0.3, 0.5, 0.8):
        matches = data[np.isclose(data["snr"], snr)]
        if len(matches) != 1:
            continue
        row = matches[0]
        fer = float(row["fe_cum"] / row["tests"])
        points[str(snr)] = {
            "trials": int(row["tests"]),
            "frame_errors": int(row["fe_cum"]),
            "raw_fer": fer,
            "confidence_95": [
                fer - float(row["fer_e_minus"]),
                fer + float(row["fer_e_plus"]),
            ],
            "plotted_fit": float(row["fer_fit"]),
        }
    slopes = {}
    for lo, hi in ((0.2, 0.5), (0.5, 0.8)):
        segment = data[(data["snr"] >= lo - 1e-9) &
                       (data["snr"] <= hi + 1e-9)]
        if len(segment) >= 2 and np.all(segment["fe_cum"] > 0):
            slopes[f"{lo:g}:{hi:g}"] = float(np.polyfit(
                segment["snr"],
                np.log10(segment["fe_cum"] / segment["tests"]),
                1,
            )[0])
    return {"source": str(txt_path), "points": points,
            "log10_fer_slope_per_db": slopes}


def diagnose_failures(args, base_experiment, failures, trial_counts,
                      discrepancy_counts, output_path):
    settings = LdpcExperimentSettings(**base_experiment)
    codec = LdpcExperimentInstance(settings).codec
    cpp = codec.decoder_impl
    if not codec.is_azcw():
        raise ValueError("diagnostic mode currently requires the all-zero codeword")
    parameters = base_experiment["codec"]["decoder_params"]
    common = dict(
        pcm=cpp.pcm, block_length=cpp.block_length, n_checks=cpp.n_checks,
        is_systematic=cpp.is_systematic, **parameters,
    )
    python = BinLdpcEgdbfDecoder(None, n_iterations=300, **common)
    longer = {
        cap: CppBinLdpcEgdbfDecoder(None, n_iterations=cap, **common)
        for cap in (600, 1200)
    }
    qc_equivalent = qc_shift_is_automorphism(cpp.pcm)
    traces = []
    extended_outputs = {cap: [] for cap in longer}
    extended_iterations = {cap: [] for cap in longer}
    classifications = []
    forced_bit_indices = []
    signatures = Counter()
    canonical_supports = Counter()
    for index, record in enumerate(failures):
        received = record["received"]
        trace = trace_egdbf_frame(python, received, record["tx_bits"])
        if trace["iterations"] != record["iterations"] or not np.array_equal(
            trace["decoded"], record["decoded"]
        ):
            raise AssertionError(
                f"Python/C++ mismatch at {record['snr_db']} dB, "
                f"trial {record['trial_index']}"
            )
        traces.append(trace)
        for cap, decoder in longer.items():
            output = np.empty_like(received)
            iteration = decoder.decode(received, output)
            extended_outputs[cap].append(output)
            extended_iterations[cap].append(iteration)

        support = trace["error_support"]
        checks = trace["unsatisfied_checks"]
        desired_signs = 1 - 2 * record["tx_bits"].astype(np.int16)
        forced = np.flatnonzero(
            desired_signs * cpp.alpha * received + cpp.variable_degrees < 0
        )
        forced_bit_indices.append(forced)
        signatures[(record["snr_db"], len(support), len(checks))] += 1
        if qc_equivalent and len(support) <= 20:
            canonical_supports[(record["snr_db"], canonical_error_support(support))] += 1
        recovered_600 = not np.any((extended_outputs[600][-1] < 0) != record["tx_bits"])
        recovered_1200 = not np.any((extended_outputs[1200][-1] < 0) != record["tx_bits"])
        if len(forced):
            category = "channel_dominates_checks"
        elif trace["iterations"] < 300:
            category = "wrong_valid_word"
        elif recovered_600:
            category = "recovered_by_600"
        elif recovered_1200:
            category = "recovered_by_1200"
        elif trace["cycle_period"] is not None:
            category = "exact_cycle"
        elif trace["unchanged_tail"] >= 50 and np.any(trace["flip_counts"][-50:] > 0):
            category = "hard_word_stagnation"
        else:
            category = "other_nonconvergence"
        classifications.append(category)
        if (index + 1) % 25 == 0:
            print(f"Replayed {index + 1}/{len(failures)} failures", flush=True)

    summary = {
        "mode": "diagnose",
        "seed": args.seed,
        "source_config": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "cpp_source_hash": SOURCE_HASH,
        "decoder_params": parameters,
        "n_iterations": 300,
        "python_cpp_checked_frames": len(failures),
        "variable_degrees": sorted(set(int(value) for value in cpp.variable_degrees)),
        "requested_failures_per_snr": args.failures_per_snr,
        "qc_shift_54_verified": qc_equivalent,
        "existing_curve": empirical_curve_summary(Path(settings.filename).with_suffix(".txt")),
        "by_snr": {},
    }
    for snr_db in args.snrs:
        indices = [i for i, item in enumerate(failures) if item["snr_db"] == snr_db]
        labels = Counter(classifications[i] for i in indices)
        cycles = Counter(traces[i]["cycle_period"] for i in indices
                         if traces[i]["cycle_period"] is not None)
        patterns = [
            {"bit_errors": bits, "unsatisfied_checks": checks, "count": count}
            for (point, bits, checks), count in signatures.most_common()
            if point == snr_db
        ][:10]
        repeated = [
            {"support": [int(bit) for bit in support], "count": count}
            for (point, support), count in canonical_supports.most_common()
            if point == snr_db and count > 1
        ][:10]
        examples = {}
        for i in indices:
            label = classifications[i]
            if label not in examples:
                examples[label] = {
                    "trial_index": failures[i]["trial_index"],
                    "bit_errors_at_300": len(traces[i]["error_support"]),
                    "unsatisfied_checks_at_300": len(traces[i]["unsatisfied_checks"]),
                    "error_support": traces[i]["error_support"].tolist(),
                    "unsatisfied_check_indices": traces[i]["unsatisfied_checks"].tolist(),
                    "forced_bit_indices": forced_bit_indices[i].tolist(),
                    "forced_received": failures[i]["received"][forced_bit_indices[i]].tolist(),
                    "cycle_start": traces[i]["cycle_start"],
                    "cycle_period": traces[i]["cycle_period"],
                    "unchanged_tail": traces[i]["unchanged_tail"],
                }
        summary["by_snr"][str(snr_db)] = {
            "trials": trial_counts[str(snr_db)],
            "captured_failures": len(indices),
            "reported_vs_actual_disagreements": discrepancy_counts[str(snr_db)],
            "channel_dominates_checks": sum(bool(len(forced_bit_indices[i]))
                                             for i in indices),
            "channel_dominance_rate_lower_bound": sum(
                bool(len(forced_bit_indices[i])) for i in indices
            ) / trial_counts[str(snr_db)],
            "categories": dict(labels),
            "exact_cycles": sum(traces[i]["cycle_period"] is not None for i in indices),
            "hard_word_unchanged_last_50": sum(
                traces[i]["unchanged_tail"] >= 50 for i in indices
            ),
            "cycles_by_period": {str(k): v for k, v in cycles.items()},
            "mean_bit_errors_at_300": float(np.mean([
                len(traces[i]["error_support"]) for i in indices
            ])),
            "median_bit_errors_at_300": float(np.median([
                len(traces[i]["error_support"]) for i in indices
            ])),
            "mean_syndrome_weight_at_300": float(np.mean([
                len(traces[i]["unsatisfied_checks"]) for i in indices
            ])),
            "median_flips_per_iteration_last_50": float(np.median([
                np.mean(traces[i]["flip_counts"][
                    max(0, traces[i]["iterations"] - 50):traces[i]["iterations"]
                ]) if traces[i]["iterations"] else 0.0
                for i in indices
            ])),
            "recovered_by_600": sum(
                not np.any((extended_outputs[600][i] < 0) != failures[i]["tx_bits"])
                for i in indices
            ),
            "recovered_by_1200": sum(
                not np.any((extended_outputs[1200][i] < 0) != failures[i]["tx_bits"])
                for i in indices
            ),
            "residual_patterns": patterns,
            "repeated_qc_equivalent_supports": repeated,
            "examples": examples,
        }

    serialized_summary = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        snr_db=np.asarray([item["snr_db"] for item in failures]),
        trial_index=np.asarray([item["trial_index"] for item in failures]),
        received=np.stack([item["received"] for item in failures]),
        decoded_300=np.stack([item["decoded"] for item in failures]),
        tx_bits=np.stack([item["tx_bits"] for item in failures]),
        iterations_300=np.asarray([item["iterations"] for item in failures]),
        reported_error=np.asarray([item["reported_error"] for item in failures]),
        syndrome_weight=np.stack([item["syndrome_weights"] for item in traces]),
        bit_errors=np.stack([item["bit_errors"] for item in traces]),
        flips=np.stack([item["flip_counts"] for item in traces]),
        threshold=np.stack([item["thresholds"] for item in traces]),
        cycle_start=np.asarray([item["cycle_start"] if item["cycle_start"] is not None
                                else -1 for item in traces]),
        cycle_period=np.asarray([item["cycle_period"] if item["cycle_period"] is not None
                                 else -1 for item in traces]),
        forced_bit_mask=np.stack([
            np.isin(np.arange(cpp.block_length), indices)
            for indices in forced_bit_indices
        ]),
        classification=np.asarray(classifications),
        decoded_600=np.stack(extended_outputs[600]),
        iterations_600=np.asarray(extended_iterations[600]),
        decoded_1200=np.stack(extended_outputs[1200]),
        iterations_1200=np.asarray(extended_iterations[1200]),
    )
    output_path.with_suffix(".json").write_text(serialized_summary, encoding="utf-8")
    return summary


def run_diagnosis(args, base_experiment, simulation_config):
    if base_experiment["codec"]["algorithm"] != (
        "cpp edge-wise gradient descent bit-flipping"
    ):
        raise ValueError("diagnostic mode requires C++ E-GDBF V1")
    if base_experiment["codec"]["n_iterations"] != 300:
        raise ValueError("diagnostic mode requires the 300-iteration V1 config")
    output_path = args.output.resolve()
    if args.output == DEFAULT_OUTPUT:
        output_path = PROJECT_DIR / "data" / "egdbf_v1_diagnosis"
    if args.replay_only:
        previous = json.loads(output_path.with_suffix(".json").read_text())
        fingerprint = hashlib.sha256(args.config.read_bytes()).hexdigest()
        if previous["config_sha256"] != fingerprint:
            raise ValueError("saved frames belong to a different experiment config")
        if previous.get("cpp_source_hash") not in (None, SOURCE_HASH):
            raise ValueError("saved frames belong to a different C++ decoder build")
        if set(previous["by_snr"]) != {str(snr) for snr in args.snrs}:
            raise ValueError("--snrs must match the saved diagnostic sample")
        args.seed = previous["seed"]
        args.failures_per_snr = previous["requested_failures_per_snr"]
        with np.load(output_path.with_suffix(".npz")) as saved:
            failures = [
                {
                    "snr_index": args.snrs.index(float(saved["snr_db"][i])),
                    "snr_db": float(saved["snr_db"][i]),
                    "trial_index": int(saved["trial_index"][i]),
                    "received": saved["received"][i].copy(),
                    "decoded": saved["decoded_300"][i].copy(),
                    "tx_bits": saved["tx_bits"][i].copy(),
                    "iterations": int(saved["iterations_300"][i]),
                    "reported_error": bool(saved["reported_error"][i]),
                }
                for i in range(len(saved["snr_db"]))
            ]
        trials = {snr: item["trials"] for snr, item in previous["by_snr"].items()}
        disagreements = {
            snr: item["reported_vs_actual_disagreements"]
            for snr, item in previous["by_snr"].items()
        }
    else:
        workers = args.workers or default_workers(simulation_config)
        failures, trials, disagreements = collect_diagnostic_failures(
            args, base_experiment, workers,
        )
    summary = diagnose_failures(
        args, base_experiment, failures, trials, disagreements, output_path,
    )
    for snr_db, item in summary["by_snr"].items():
        print(
            f"{snr_db} dB: {item['captured_failures']} failures / {item['trials']} "
            f"trials; cycles={item['exact_cycles']}, "
            f"stagnant={item['hard_word_unchanged_last_50']}, "
            f"recovered by 600/1200={item['recovered_by_600']}/"
            f"{item['recovered_by_1200']}",
            flush=True,
        )
    print(f"Saved {output_path.with_suffix('.npz')} and {output_path.with_suffix('.json')}",
          flush=True)


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config)
    if args.no_momentum:
        base_experiment = copy.deepcopy(base_experiment)
        base_experiment["codec"]["decoder_params"].update(L=0, rho=[])
    if args.mode == "staged" and base_experiment["codec"]["algorithm"] != (
        "cpp edge-wise gradient descent bit-flipping"
    ):
        raise ValueError("staged search currently targets E-GDBF V1")
    egdbf_compile()
    if args.mode == "diagnose":
        run_diagnosis(args, base_experiment, simulation_config)
        return
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
