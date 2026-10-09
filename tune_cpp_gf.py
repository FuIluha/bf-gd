"""Search the C++ gradient flow (GF) parameters at one SNR point.

The GF update is  x <- x - eta * (x - y + gamma * grad h_{alpha,beta}(x))  and grad h is
linear in (alpha, beta), so only the products gamma*alpha and gamma*beta matter.
The search therefore runs over three parameters (alpha, beta, eta) with gamma fixed to 1,
all of them on a logarithmic scale.

Search strategy
  1. Latin-hypercube sample of the whole box (plus the paper defaults as a reference).
  2. Successive halving: all candidates are decoded on the same noise realizations
     (common random numbers), the worse part is dropped after each round and the
     survivors get several times more frames.
  3. Local refinement: shrinking boxes around the current best, evaluated on the
     same frames as the incumbent.
  4. Confirmation: the finalists are re-evaluated on fresh noise, so the reported
     result is not inflated by selecting the luckiest candidate.
Only the best parameters are printed and saved.
"""

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.cpp_bin_ldpc_gf import lib_compile as gf_compile
from lbc_encoder.lbc_encoder import lib_compile as lbc_compile
from simulator_awgn_python.channel import lib_compile as chan_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_gf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_gf.txt"

# Reference point from the paper (Table 2 with gamma = 1).
REFERENCE_PARAMS = (1.0, 2.0, 0.01)  # alpha, beta, eta

SEARCH_STREAM = 0
CONFIRM_STREAM = 1
INSTANCE_CACHE_SIZE = 3

_BASE_EXPERIMENT = None
_SNR_DB = None
_SEED = None
_INSTANCES = {}


@dataclass
class Candidate:
    """Parameters (alpha, beta, eta) together with the statistics collected so far."""
    params: tuple
    frames: int = 0
    frame_errors: int = 0
    bit_errors: float = 0.0
    invalid: bool = False

    @property
    def ber(self):
        return self.bit_errors / self.frames if self.frames else math.inf

    @property
    def fer(self):
        return self.frame_errors / self.frames if self.frames else math.inf

    def score(self, metric):
        if self.invalid or not self.frames:
            return (math.inf, math.inf)
        if metric == "fer":
            return (self.fer, self.ber)
        return (self.ber, self.fer)

    def decoder_params(self):
        alpha, beta, eta = self.params
        return {"alpha": alpha, "beta": beta, "gamma": 1.0, "eta": eta}


def float_pair(value):
    try:
        low, high = (float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected 'low,high', got {value!r}") from exc
    if not 0 < low < high:
        raise argparse.ArgumentTypeError("need 0 < low < high")
    return low, high


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search C++ GF parameters (alpha, beta, eta; gamma is fixed to 1) at a "
            "fixed SNR. Only the best result is printed and saved."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=-2.0,
                        help="simulator SNR in dB (Es/N0)")
    parser.add_argument("--metric", choices=("ber", "fer"), default="ber",
                        help="quantity to minimize; the other one breaks ties")
    parser.add_argument("--iterations", type=int,
                        help="override n_iterations from the config")
    parser.add_argument("--alpha-range", type=float_pair, default=(0.1, 10.0))
    parser.add_argument("--beta-range", type=float_pair, default=(0.1, 20.0))
    parser.add_argument("--eta-range", type=float_pair, default=(0.002, 0.05))
    parser.add_argument("--candidates", type=int, default=512,
                        help="size of the initial global sample")
    parser.add_argument("--min-frames", type=int, default=300,
                        help="frames per candidate in the first halving round")
    parser.add_argument("--halving-factor", type=int, default=3,
                        help="survivors are 1/factor, frames grow by factor")
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--refine-rounds", type=int, default=4)
    parser.add_argument("--refine-samples", type=int, default=64)
    parser.add_argument("--refine-shrink", type=float, default=0.6,
                        help="box half-width (in log units) is multiplied by this per round")
    parser.add_argument("--confirm-frames", type=int, default=5000,
                        help="initial frames per finalist on fresh noise")
    parser.add_argument("--confirm-errors", "--max-errors", dest="confirm_errors",
                        type=int, default=200,
                        help="stop doubling when the best finalist has this many frame errors")
    parser.add_argument("--max-confirm-frames", "--trials", dest="max_confirm_frames",
                        type=int, default=400_000,
                        help="upper limit of frames per finalist in the confirmation stage")
    parser.add_argument("--chunk", type=int, default=25,
                        help="frames per parallel task")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true",
                        help="print progress to stderr")
    return parser.parse_args()


def validate_args(args):
    positive = (
        "candidates", "min_frames", "halving_factor", "finalists", "refine_samples",
        "confirm_frames", "confirm_errors", "max_confirm_frames", "chunk",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.halving_factor < 2:
        raise ValueError("--halving-factor must be at least 2")
    if args.refine_rounds < 0:
        raise ValueError("--refine-rounds must be non-negative")
    if not 0 < args.refine_shrink <= 1:
        raise ValueError("--refine-shrink must be in (0, 1]")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")


def load_base_experiment(config_path, iterations):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != "cpp gf":
        raise ValueError('the selected config must use the "cpp gf" algorithm')
    if iterations is not None:
        experiment["codec"]["n_iterations"] = iterations
    return experiment, config.get("simulation", {})


def default_workers(simulation_config):
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus:
        return int(allocated_cpus)
    local_cpus = os.cpu_count() or 1
    return min(int(simulation_config.get("n_workers", local_cpus)), local_cpus)


def log(args, message):
    if args.verbose:
        print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- workers

def init_worker(base_experiment, snr_db, seed):
    global _BASE_EXPERIMENT, _SNR_DB, _SEED
    _BASE_EXPERIMENT = base_experiment
    _SNR_DB = snr_db
    _SEED = seed


def get_instance(params):
    instance = _INSTANCES.get(params)
    if instance is None:
        if len(_INSTANCES) >= INSTANCE_CACHE_SIZE:
            _INSTANCES.clear()
        experiment_config = copy.deepcopy(_BASE_EXPERIMENT)
        alpha, beta, eta = params
        experiment_config["codec"]["decoder_params"] = {
            "alpha": alpha, "beta": beta, "gamma": 1.0, "eta": eta,
        }
        instance = LdpcExperimentInstance(LdpcExperimentSettings(**experiment_config))
        _INSTANCES[params] = instance
    return instance


def run_chunk(task):
    """Decode frames [start, end) of one noise stream with one parameter set."""
    key, params, stream, start, end = task
    instance = get_instance(params)
    frame_errors = 0
    bit_errors = 0.0
    for frame in range(start, end):
        rng = np.random.default_rng([_SEED, stream, frame])
        try:
            result = instance.run(_SNR_DB, rng)
        except FloatingPointError:
            return key, True, 0, 0.0, 0
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
    return key, False, frame_errors, bit_errors, end - start


# ------------------------------------------------------------ search logic

def evaluate(pool, candidates, target_frames, stream, chunk):
    """Bring every candidate to exactly target_frames frames of the given noise stream."""
    tasks = []
    for key, candidate in enumerate(candidates):
        if candidate.invalid:
            continue
        for start in range(candidate.frames, target_frames, chunk):
            tasks.append((
                key, candidate.params, stream, start, min(start + chunk, target_frames),
            ))
    for key, invalid, frame_errors, bit_errors, frames in pool.imap_unordered(
        run_chunk, tasks, chunksize=1,
    ):
        candidate = candidates[key]
        if invalid:
            candidate.invalid = True
            continue
        candidate.frame_errors += frame_errors
        candidate.bit_errors += bit_errors
        candidate.frames += frames


def sample_log_box(count, low, high, seed):
    """Latin-hypercube sample of a box, uniform in the logarithm of each coordinate."""
    low = np.log(np.asarray(low, dtype=np.float64))
    high = np.log(np.asarray(high, dtype=np.float64))
    unit = qmc.LatinHypercube(d=len(low), seed=seed).random(count)
    return [tuple(float(v) for v in np.exp(low + point * (high - low))) for point in unit]


def successive_halving(pool, args, candidates):
    frames = args.min_frames
    while True:
        evaluate(pool, candidates, frames, SEARCH_STREAM, args.chunk)
        candidates = [c for c in candidates if not c.invalid]
        if not candidates:
            raise RuntimeError("all candidates diverged; narrow the parameter ranges")
        candidates.sort(key=lambda c: c.score(args.metric))
        log(args, f"halving: {len(candidates)} candidates x {frames} frames, "
                  f"best {args.metric}={candidates[0].score(args.metric)[0]:.4g} "
                  f"params={candidates[0].params}")
        if len(candidates) <= args.finalists:
            return candidates, frames
        keep = max(args.finalists, math.ceil(len(candidates) / args.halving_factor))
        candidates = candidates[:keep]
        frames *= args.halving_factor


def refine(pool, args, survivors, frames, global_low, global_high):
    """Shrinking log-boxes around the incumbent, all compared on the same frames."""
    evaluated = list(survivors)
    incumbent = evaluated[0]
    half_width = np.log(4.0)  # first box: a factor of 4 in each direction
    for round_index in range(args.refine_rounds):
        center = np.log(np.asarray(incumbent.params))
        low = np.maximum(center - half_width, np.log(global_low))
        high = np.minimum(center + half_width, np.log(global_high))
        points = sample_log_box(
            args.refine_samples, np.exp(low), np.exp(high), args.seed + 1 + round_index,
        )
        fresh = [Candidate(point) for point in points]
        evaluate(pool, fresh, frames, SEARCH_STREAM, args.chunk)
        evaluated.extend(fresh)
        evaluated.sort(key=lambda c: c.score(args.metric))
        incumbent = evaluated[0]
        half_width *= args.refine_shrink
        log(args, f"refine {round_index + 1}/{args.refine_rounds}: "
                  f"best {args.metric}={incumbent.score(args.metric)[0]:.4g} "
                  f"params={incumbent.params}")
    return [c for c in evaluated if not c.invalid][:args.finalists]


def confirm(pool, args, finalists):
    """Re-evaluate the finalists and the reference on fresh noise until the best is reliable."""
    entries = [Candidate(c.params) for c in finalists]
    reference = next((c for c in entries if c.params == REFERENCE_PARAMS), None)
    if reference is None:
        reference = Candidate(REFERENCE_PARAMS)
        entries.append(reference)

    frames = args.confirm_frames
    while True:
        evaluate(pool, entries, frames, CONFIRM_STREAM, args.chunk)
        best = min(entries, key=lambda c: c.score(args.metric))
        log(args, f"confirm: {frames} frames, best {args.metric}="
                  f"{best.score(args.metric)[0]:.4g} ({best.frame_errors} frame errors)")
        if best.frame_errors >= args.confirm_errors or frames >= args.max_confirm_frames:
            return best, reference, frames
        frames = min(frames * 2, args.max_confirm_frames)


def result_payload(candidate, args, frames, iterations):
    return {
        "snr_db": args.snr,
        "metric": args.metric,
        "n_iterations": iterations,
        "confirmation_frames": frames,
        "frame_errors": candidate.frame_errors,
        "fer": candidate.fer,
        "ber": candidate.ber,
        "decoder_params": candidate.decoder_params(),
    }


def save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    temporary_path.replace(path)


def main():
    args = parse_args()
    validate_args(args)
    os.chdir(PROJECT_DIR)
    base_experiment, simulation_config = load_base_experiment(args.config, args.iterations)
    iterations = base_experiment["codec"]["n_iterations"]

    chan_compile()
    lbc_compile()
    gf_compile()

    low = (args.alpha_range[0], args.beta_range[0], args.eta_range[0])
    high = (args.alpha_range[1], args.beta_range[1], args.eta_range[1])
    candidates = [Candidate(point) for point in
                  sample_log_box(args.candidates, low, high, args.seed)]
    if REFERENCE_PARAMS not in {c.params for c in candidates}:
        candidates.append(Candidate(REFERENCE_PARAMS))

    workers = args.workers or default_workers(simulation_config)
    log(args, f"GF search: SNR={args.snr:g} dB, metric={args.metric}, "
              f"iterations={iterations}, candidates={len(candidates)}, workers={workers}")

    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(base_experiment, args.snr, args.seed),
    ) as pool:
        survivors, frames = successive_halving(pool, args, candidates)
        finalists = refine(pool, args, survivors, frames, low, high)
        best, reference, confirm_frames = confirm(pool, args, finalists)

    payload = result_payload(best, args, confirm_frames, iterations)
    payload["reference"] = {
        "description": "paper defaults (alpha=1, beta=2, gamma=1, eta=0.01)",
        "fer": reference.fer,
        "ber": reference.ber,
        "frame_errors": reference.frame_errors,
    }
    save_result(args.output.resolve(), payload)

    params = best.decoder_params()
    print(
        f"Best GF parameters at SNR={args.snr:g} dB ({iterations} iterations):\n"
        f"  alpha={params['alpha']:.6g}  beta={params['beta']:.6g}  "
        f"gamma={params['gamma']:g}  eta={params['eta']:.6g}\n"
        f"  BER={best.ber:.4g}  FER={best.fer:.4g}  "
        f"({best.frame_errors} frame errors / {best.frames} frames)\n"
        f"  saved to {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
