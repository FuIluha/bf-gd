"""Full MPGD grid at one SNR, collecting a fixed number of frame errors."""

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.bin_ldpc_mpgd import BinLdpcMpgdDecoder
from ldpc_py.cpp_bin_ldpc_mpgd import lib_compile
from simulator_awgn_python.tools import load_json
from .tune_gdms import comma_separated_floats, default_workers, result_score


PROJECT_DIR = Path(__file__).resolve().parent.parent
# Reuse the existing code/channel experiment, replacing only its decoder.
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_gdms.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_mpgd_snr0.txt"
# Refined grid: 11 * 8 * 8 * 8 * 8 = 45,056 combinations.
DEFAULT_ALPHAS = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)
DEFAULT_ETA_RS = (0.7, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
DEFAULT_ETA_VS = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0)
DEFAULT_TAU_RS = (0.0, 0.003, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
DEFAULT_TAU_VS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45)

_BASE_EXPERIMENT = _SNR_DB = _MAX_TRIALS = _MAX_ERRORS = _SEED = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                        help="existing BinaryLdpcCodec experiment; decoder is replaced with cpp mpgd")
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=50, help="fixed limit of complete CN/VN cycles (default: 50)")
    parser.add_argument("--trials", type=int, default=100_000_000)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--dry-run", action="store_true", help="print grid size/settings without compiling or simulating")
    for flag, defaults in (("alphas", DEFAULT_ALPHAS), ("eta-rs", DEFAULT_ETA_RS),
                           ("eta-vs", DEFAULT_ETA_VS), ("tau-rs", DEFAULT_TAU_RS),
                           ("tau-vs", DEFAULT_TAU_VS)):
        parser.add_argument(f"--{flag}", type=comma_separated_floats, default=defaults)
    return parser.parse_args(argv)


def parameter_axes(args):
    return {
        "alpha": tuple(dict.fromkeys(args.alphas)),
        "eta_r": tuple(dict.fromkeys(args.eta_rs)),
        "eta_v": tuple(dict.fromkeys(args.eta_vs)),
        "tau_r": tuple(dict.fromkeys(args.tau_rs)),
        "tau_v": tuple(dict.fromkeys(args.tau_vs)),
    }


def validate_args(args):
    for key in ("iterations", "trials", "max_errors", "workers", "max_configs"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.iterations >= np.iinfo(np.uint32).max:
        raise ValueError("--iterations must fit uint32")
    if args.seed < 0 or not np.isfinite(args.snr):
        raise ValueError("seed must be non-negative and SNR must be finite")
    if args.output.suffix == ".jsonl":
        raise ValueError("--output is the best-parameter file; .jsonl is reserved for the sweep log")
    axes = parameter_axes(args)
    for spec in BinLdpcMpgdDecoder.describe().parameters:
        if not axes[spec.key]:
            raise ValueError(f"empty grid for {spec.key}")
        for value in axes[spec.key]:
            spec.parse(value)


def parameter_grid(args):
    """Only explicitly requested grid values; no hidden baseline candidate."""
    axes = parameter_axes(args)
    candidates = (dict(zip(axes, values)) for values in itertools.product(*axes.values()))
    return itertools.islice(candidates, args.max_configs)


def load_base_experiment(config_path, iterations):
    config = load_json(str(config_path))
    experiment = copy.deepcopy(config["experiment"])
    if experiment["codec"].get("type") != "BinaryLdpcCodec":
        raise ValueError("MPGD tuner requires a BinaryLdpcCodec experiment")
    experiment["codec"]["algorithm"] = "cpp mpgd"
    experiment["codec"]["n_iterations"] = iterations
    experiment["codec"]["decoder_params"] = {}
    return experiment, config.get("simulation", {})


def init_worker(base_experiment, snr_db, max_trials, max_errors, seed):
    global _BASE_EXPERIMENT, _SNR_DB, _MAX_TRIALS, _MAX_ERRORS, _SEED
    _BASE_EXPERIMENT, _SNR_DB = base_experiment, snr_db
    _MAX_TRIALS, _MAX_ERRORS, _SEED = max_trials, max_errors, seed


def evaluate_candidate(index_and_params):
    index, params = index_and_params
    config = copy.deepcopy(_BASE_EXPERIMENT)
    config["codec"]["decoder_params"] = params
    experiment = LdpcExperimentInstance(LdpcExperimentSettings(**config))
    frame_errors = trials = iterations = 0
    bit_errors = 0.0
    for trial in range(_MAX_TRIALS):
        rng = np.random.default_rng([_SEED, trial])
        try:
            result = experiment.run(_SNR_DB, rng)
        except FloatingPointError as exc:
            return {"index": index, "decoder_params": params, "invalid": True, "reason": str(exc)}
        # The generic AZCW runner assumes early termination means correctness.
        # Here count actual output errors, including convergence to a wrong codeword.
        ber = experiment.codec.decoder_impl.output_ber(
            experiment.llr_out, experiment.tx_bits, result.n_iter,
        )
        trials += 1
        frame_errors += int(ber > 0)
        bit_errors += float(ber)
        iterations += int(result.n_iter)
        if frame_errors >= _MAX_ERRORS:
            break
    return {
        "index": index, "decoder_params": params, "invalid": False,
        "trials": trials, "frame_errors": frame_errors,
        "target_reached": frame_errors >= _MAX_ERRORS,
        "fer": frame_errors / trials, "ber": bit_errors / trials,
        "average_iterations": iterations / trials,
    }


def save_best(path, result, args, base_experiment, completed, total):
    codec = copy.deepcopy(base_experiment["codec"])
    codec["decoder_params"] = result["decoder_params"]
    payload = {
        **result, "snr_db": args.snr, "seed": args.seed,
        "max_trials": args.trials, "target_frame_errors": args.max_errors,
        "completed_parameter_sets": completed, "total_parameter_sets": total,
        "epsilon": BinLdpcMpgdDecoder.EPSILON,
        "codec": codec, "channel_output": base_experiment["channel_output"],
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation = load_base_experiment(args.config, args.iterations)
    total = math.prod(len(axis) for axis in parameter_axes(args).values())
    if args.max_configs is not None:
        total = min(total, args.max_configs)
    workers = min(args.workers or default_workers(simulation), total)
    print(
        f"MPGD search: SNR={args.snr:g} dB, iterations={args.iterations}, "
        f"max_trials={args.trials}, target_errors={args.max_errors}, "
        f"parameter_sets={total}, workers={workers}, seed={args.seed}", flush=True,
    )
    print(f"grid={json.dumps(parameter_axes(args))}", flush=True)
    if args.dry_run:
        return
    lib_compile()  # Compile once before starting workers.
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    best = None
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as log:
        with mp.get_context("spawn").Pool(
            processes=workers, initializer=init_worker,
            initargs=(base_experiment, args.snr, args.trials, args.max_errors, args.seed),
        ) as pool:
            for completed, result in enumerate(pool.imap_unordered(
                evaluate_candidate, enumerate(parameter_grid(args)), chunksize=1,
            ), start=1):
                log.write(json.dumps(result) + "\n")
                log.flush()
                params = json.dumps(result["decoder_params"], separators=(",", ":"))
                if result["invalid"]:
                    print(f"INVALID [{completed}/{total}] params={params}", flush=True)
                    continue
                # Keep final ties deterministic even with unordered workers.
                is_best = best is None or (
                    result_score(result), result["index"]
                ) < (result_score(best), best["index"])
                status = "NEW BEST" if is_best else "PROGRESS"
                capped = " [trial cap reached]" if not result["target_reached"] else ""
                print(
                    f"{status} [{completed}/{total}] FER={result['fer']:.6g} "
                    f"({result['frame_errors']} errors / {result['trials']} trials), "
                    f"BER={result['ber']:.6g}, avg_iter={result['average_iterations']:.3f}{capped}\n"
                    f"params={params}", flush=True,
                )
                if is_best:
                    best = result
                    save_best(output, result, args, base_experiment, completed, total)
                    print(f"saved to {output}", flush=True)
    if best is None:
        raise RuntimeError("No valid MPGD candidate completed")
    print(f"Search completed. Best parameters: {output}", flush=True)


if __name__ == "__main__":
    main()
