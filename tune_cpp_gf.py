"""Search the C++ gradient flow (GF) parameters at one SNR point.

The GF update is
    x <- clip(x - eta * (x - y + gamma * grad h_{alpha,beta}(x)), -theta, theta)
starting from x(0) = 0, y or sign(y) (init). grad h is linear in (alpha, beta), so
only gamma*alpha and gamma*beta matter: gamma is kept in the interface but its
default grid is the single value 1.

Search strategy
  1. Grid: every combination of alpha, beta, gamma, eta, theta and init
     (plus the parameters from the config as a reference).
  2. Successive halving: all candidates are decoded on the same noise
     realizations (common random numbers), the worse 2/3 is dropped after each
     rung and the survivors get 3 times more frames. A candidate stops getting
     frames once it has --max-errors frame errors or --trials frames, and a chunk
     of frames stops early after --abort-errors errors, so hopeless candidates
     are cheap.
  3. Zoom: a finer grid (midpoints to the neighbouring grid values) around the
     best survivors, raced on the same frames and merged with the survivors.
  4. Confirmation: the finalists and the reference are re-evaluated on fresh
     noise (again up to --max-errors errors or --trials frames each), so the
     reported FER is not inflated by picking the luckiest one.
Every evaluated candidate is logged to the .jsonl file next to --output, and
the current best is saved to --output after every stage.
"""

import argparse
import copy
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ldpc_experiment import LdpcExperimentInstance, LdpcExperimentSettings
from ldpc_py.bin_ldpc_gf import INIT_MODES
from ldpc_py.cpp_bin_ldpc_gf import lib_compile as gf_compile
from lbc_encoder.lbc_encoder import lib_compile as lbc_compile
from simulator_awgn_python.channel import lib_compile as chan_compile
from simulator_awgn_python.tools import load_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "experiments" / "experiment_cpp_gf.json"
DEFAULT_OUTPUT = PROJECT_DIR / "params_cpp_gf.txt"

DEFAULT_ITERATIONS = 100

# 11 * 10 * 1 * 10 * 5 * 3 = 16500 combinations.
# alpha, beta and eta are roughly log-spaced; alpha and beta set the strength of
# the code terms relative to the channel term x - y (coefficient 1). alpha = 0 is
# kept because theta = 1 already pulls the state to the box; alpha = 5 freezes the
# signs almost at once and beta = 12 lets the check term (variable degree up to
# 18) swamp the channel, so both are left out.
# The etas are for 100 iterations (integration time T = 100 * eta): below
# eta = 0.01 (T < 1) the state hardly leaves x(0); unclipped Euler is stable only
# for eta < ~2 / (1 + 8 alpha + 2 beta d_v), but clipping to theta ~ 1 keeps
# eta ~ 0.1-0.15 useful, and at 0.2 and above almost every frame fails. With
# another --iterations scale --etas inversely.
# theta is dense near 1, where clipping changes the dynamics: for init zero and
# hard the state stays below ~1.3-1.9 anyway, so theta = 1.6 is close to "no
# clipping" and larger values would only repeat it; for init y a large theta
# lets the state oscillate between -theta and theta.
DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
DEFAULT_BETAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_GAMMAS = (1.0,)
DEFAULT_ETAS = (0.01, 0.015, 0.02, 0.03, 0.045, 0.065, 0.08, 0.1, 0.13, 0.16)
DEFAULT_THETAS = (1.0, 1.1, 1.2, 1.35, 1.6)
DEFAULT_INITS = INIT_MODES

SEARCH_STREAM = 0
CONFIRM_STREAM = 1
INSTANCE_CACHE_SIZE = 4

_BASE_EXPERIMENT = None
_SNR_DB = None
_SEED = None
_INSTANCES = {}


@dataclass
class Candidate:
    """One parameter set together with the statistics collected so far."""
    params: dict
    source: str = "grid"
    frames: int = 0
    next_frame: int = 0
    frame_errors: int = 0
    bit_errors: float = 0.0
    invalid: bool = False
    rung: int = 0

    @property
    def key(self):
        return tuple(self.params.items())

    @property
    def fer(self):
        return self.frame_errors / self.frames if self.frames else math.inf

    @property
    def ber(self):
        return self.bit_errors / self.frames if self.frames else math.inf

    def score(self):
        if self.invalid or not self.frames:
            return math.inf, math.inf
        return self.fer, self.ber

    def record(self, stage):
        return {
            "stage": stage,
            "source": self.source,
            "decoder_params": self.params,
            "invalid": self.invalid,
            "rung": self.rung,
            "frames": self.frames,
            "frame_errors": self.frame_errors,
            "fer": self.fer if self.frames else None,
            "ber": self.ber if self.frames else None,
        }


def comma_separated_inits(value):
    values = tuple(item.strip() for item in value.split(","))
    unknown = [item for item in values if item not in INIT_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown init modes {unknown}, expected some of {INIT_MODES}"
        )
    return values


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
            "Search C++ GF hyperparameters using FER at a fixed SNR: grid, successive "
            "halving on common noise, zoom around the best and confirmation on fresh noise."
        )
    )
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--snr", type=float, default=-1.0)
    parser.add_argument("--trials", type=int, default=100_000_000,
                        help="maximum frames per candidate")
    parser.add_argument("--max-errors", type=int, default=100,
                        help="a candidate stops getting frames after this many frame errors")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="decoder iterations, overrides n_iterations from the config")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-configs", type=int,
                        help="evaluate only the first grid points (for quick tests)")

    grid = parser.add_argument_group("grid")
    grid.add_argument("--alphas", type=comma_separated_floats, default=DEFAULT_ALPHAS)
    grid.add_argument("--betas", type=comma_separated_floats, default=DEFAULT_BETAS)
    grid.add_argument("--gammas", type=comma_separated_floats, default=DEFAULT_GAMMAS)
    grid.add_argument("--etas", type=comma_separated_floats, default=DEFAULT_ETAS)
    grid.add_argument("--thetas", type=comma_separated_floats, default=DEFAULT_THETAS,
                      help="clipping levels theta >= 1 (inf disables clipping)")
    grid.add_argument("--inits", type=comma_separated_inits, default=DEFAULT_INITS,
                      help=f"initial points x(0) to search, some of {','.join(INIT_MODES)}")

    halving = parser.add_argument_group("successive halving")
    halving.add_argument("--first-frames", type=int, default=500,
                         help="frames per candidate in the first rung")
    halving.add_argument("--factor", type=int, default=3,
                         help="survivors are 1/factor, frames grow by factor")
    halving.add_argument("--finalists", type=int, default=8)
    halving.add_argument("--abort-errors", type=int, default=50,
                         help="a chunk of frames stops after this many frame errors")
    halving.add_argument("--zoom-centers", type=int, default=3,
                         help="best survivors to zoom around (0 disables the zoom)")
    halving.add_argument("--chunk", type=int, default=500,
                         help="frames per parallel task")

    confirm = parser.add_argument_group("confirmation on fresh noise")
    confirm.add_argument("--confirm-frames", type=int, default=20_000,
                         help="initial frames per finalist")
    parser.add_argument("--show-invalid", action="store_true",
                        help="print every diverged parameter set (they are always logged to .jsonl)")
    return parser.parse_args()


def validate_args(args):
    positive = (
        "trials", "max_errors", "iterations", "first_frames", "finalists",
        "abort_errors", "chunk", "confirm_frames",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.factor < 2:
        raise ValueError("--factor must be at least 2")
    if args.zoom_centers < 0:
        raise ValueError("--zoom-centers must be non-negative")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.max_configs is not None and args.max_configs <= 0:
        raise ValueError("--max-configs must be positive")
    if any(not np.isfinite(v) or v < 0 for v in args.alphas + args.betas + args.gammas):
        raise ValueError("alpha, beta and gamma values must be finite and non-negative")
    if any(not np.isfinite(v) or v <= 0 for v in args.etas):
        raise ValueError("all eta values must be positive")
    if any(not v >= 1 for v in args.thetas):
        raise ValueError("all theta values must be at least 1")


def load_base_experiment(config_path, iterations):
    config = load_json(str(config_path))
    experiment = config["experiment"]
    if experiment["codec"].get("algorithm") != "cpp gf":
        raise ValueError('the selected config must use the "cpp gf" algorithm')
    experiment["codec"]["n_iterations"] = iterations
    return experiment, config.get("simulation", {})


def make_params(alpha, beta, gamma, eta, theta, init):
    return {
        "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
        "eta": float(eta), "theta": float(theta), "init": init,
    }


def reference_params(base_params):
    return make_params(
        base_params["alpha"], base_params["beta"], base_params["gamma"],
        base_params["eta"], base_params.get("theta", math.inf),
        base_params.get("init", "zero"),
    )


def parameter_grid(args, base_params):
    candidates = [Candidate(reference_params(base_params), source="reference")]
    # theta and init vary fastest, so every (alpha, beta, gamma, eta) is tried
    # with all of them next to each other, even when --max-configs cuts the grid.
    for alpha, beta, gamma, eta, theta, init in itertools.product(
        args.alphas, args.betas, args.gammas, args.etas, args.thetas, args.inits,
    ):
        candidates.append(Candidate(make_params(alpha, beta, gamma, eta, theta, init)))
    return unique(candidates)


def unique(candidates, known=()):
    seen = set(known)
    result = []
    for candidate in candidates:
        if candidate.key not in seen:
            seen.add(candidate.key)
            result.append(candidate)
    return result


def midpoint(low, high, linear):
    if linear or low <= 0:
        return float(f"{(low + high) / 2:.6g}")
    return float(f"{math.sqrt(low * high):.6g}")


def neighbour_values(value, grid, linear=False):
    """The value and the midpoints to its neighbours in the grid."""
    if not math.isfinite(value):
        return [value]
    points = sorted(set(v for v in grid if math.isfinite(v)) | {value})
    index = points.index(value)
    values = [value]
    if index > 0:
        values.append(midpoint(points[index - 1], value, linear))
    if index + 1 < len(points):
        values.append(midpoint(value, points[index + 1], linear))
    return values


def zoom_candidates(args, centers):
    candidates = []
    for center in centers:
        p = center.params
        for alpha, beta, eta, theta in itertools.product(
            neighbour_values(p["alpha"], args.alphas),
            neighbour_values(p["beta"], args.betas),
            neighbour_values(p["eta"], args.etas),
            neighbour_values(p["theta"], args.thetas, linear=True),
        ):
            params = make_params(alpha, beta, p["gamma"], eta, theta, p["init"])
            candidates.append(Candidate(params, source="zoom"))
    return candidates


# ---------------------------------------------------------------- workers

def init_worker(base_experiment, snr_db, seed):
    global _BASE_EXPERIMENT, _SNR_DB, _SEED
    _BASE_EXPERIMENT = base_experiment
    _SNR_DB = snr_db
    _SEED = seed


def get_instance(params):
    key = tuple(params.items())
    instance = _INSTANCES.get(key)
    if instance is None:
        if len(_INSTANCES) >= INSTANCE_CACHE_SIZE:
            _INSTANCES.clear()
        experiment_config = copy.deepcopy(_BASE_EXPERIMENT)
        experiment_config["codec"]["decoder_params"] = params
        instance = LdpcExperimentInstance(LdpcExperimentSettings(**experiment_config))
        _INSTANCES[key] = instance
    return instance


def run_chunk(task):
    """Decode frames [start, end) of one noise stream with one parameter set."""
    index, params, stream, start, end, abort_errors = task
    instance = get_instance(params)
    frame_errors = 0
    bit_errors = 0.0
    frames = 0
    for frame in range(start, end):
        rng = np.random.default_rng([_SEED, stream, frame])
        try:
            result = instance.run(_SNR_DB, rng)
        except FloatingPointError:
            return index, True, 0, 0.0, 0
        frames += 1
        frame_errors += int(result.fe_cum)
        bit_errors += float(result.be_cum)
        if abort_errors is not None and frame_errors >= abort_errors:
            break
    return index, False, frame_errors, bit_errors, frames


# ------------------------------------------------------------ search logic

def evaluate(pool, candidates, target_frames, stream, chunk, abort_errors, max_errors):
    """Run every valid candidate up to target_frames frames of the noise stream.

    Candidates that already have max_errors frame errors get no more frames.
    """
    tasks = []
    for index, candidate in enumerate(candidates):
        if candidate.invalid or candidate.frame_errors >= max_errors:
            continue
        for start in range(candidate.next_frame, target_frames, chunk):
            end = min(start + chunk, target_frames)
            tasks.append((index, candidate.params, stream, start, end, abort_errors))
        candidate.next_frame = max(candidate.next_frame, target_frames)
    for index, invalid, frame_errors, bit_errors, frames in pool.imap_unordered(
        run_chunk, tasks, chunksize=1,
    ):
        candidate = candidates[index]
        if invalid:
            candidate.invalid = True
            continue
        candidate.frame_errors += frame_errors
        candidate.bit_errors += bit_errors
        candidate.frames += frames


class Search:
    """Shared state of one search run: pool, logging and progress output."""

    def __init__(self, pool, args, iterations, log_path):
        self.pool = pool
        self.args = args
        self.iterations = iterations
        self.log_path = log_path
        self.started = time.perf_counter()
        self.invalid_count = 0

    def say(self, message):
        elapsed = time.perf_counter() - self.started
        print(f"[{elapsed / 60:7.1f} min] {message}", flush=True)

    def log(self, candidates, stage):
        with self.log_path.open("a", encoding="utf-8") as log:
            for candidate in candidates:
                log.write(json.dumps(candidate.record(stage)) + "\n")

    def drop_invalid(self, candidates, stage):
        invalid = [c for c in candidates if c.invalid]
        self.log(invalid, stage)
        self.invalid_count += len(invalid)
        if self.args.show_invalid:
            for candidate in invalid:
                self.say(f"INVALID {format_params(candidate.params)}")
        return [c for c in candidates if not c.invalid]

    def halving(self, candidates, stage, final_frames=None):
        """Successive halving on the search stream.

        Without final_frames it runs until at most --finalists candidates are
        left; with final_frames it runs until the candidates have that many
        frames and returns all survivors of the last rung.
        """
        args = self.args
        frames = min(args.first_frames, args.trials)
        rung = 0
        while True:
            if final_frames is not None:
                frames = min(frames, final_frames)
            evaluate(self.pool, candidates, frames, SEARCH_STREAM, args.chunk,
                     args.abort_errors, args.max_errors)
            candidates = self.drop_invalid(candidates, stage)
            if not candidates:
                raise RuntimeError(f"{stage}: all candidates diverged")
            candidates.sort(key=Candidate.score)
            for candidate in candidates:
                candidate.rung = rung
            best = candidates[0]
            self.say(
                f"{stage} rung {rung}: {len(candidates)} candidates x {frames} frames, "
                f"best FER={best.fer:.4g} ({best.frame_errors} errors)  "
                f"{format_params(best.params)}"
            )
            if final_frames is None:
                done = len(candidates) <= args.finalists
            else:
                done = frames >= final_frames
            if done:
                return candidates, frames
            keep = max(args.finalists, math.ceil(len(candidates) / args.factor))
            self.log(candidates[keep:], stage)
            candidates = candidates[:keep]
            frames = min(frames * args.factor, args.trials)
            rung += 1

    def confirm(self, finalists, reference):
        """Re-evaluate the finalists and the reference on fresh noise."""
        args = self.args
        entries = [Candidate(c.params, source=c.source) for c in finalists]
        if reference.key not in {c.key for c in entries}:
            entries.append(Candidate(reference.params, source="reference"))
        frames = min(args.confirm_frames, args.trials)
        while True:
            evaluate(self.pool, entries, frames, CONFIRM_STREAM, args.chunk, None,
                     args.max_errors)
            valid = [c for c in entries if not c.invalid]
            if not valid:
                raise RuntimeError("all finalists diverged on fresh noise")
            best = min(valid, key=Candidate.score)
            self.say(
                f"confirm: {frames} frames, best FER={best.fer:.4g} "
                f"({best.frame_errors} errors)  {format_params(best.params)}"
            )
            if best.frame_errors >= args.max_errors or frames >= args.trials:
                entries.sort(key=Candidate.score)
                self.log(entries, "confirm")
                return entries
            frames = min(frames * 2, args.trials)


def format_params(decoder_params):
    return "  ".join(
        f"{name}={value:g}" if isinstance(value, float) else f"{name}={value}"
        for name, value in decoder_params.items()
    )


def wilson_interval(errors, frames, z=1.96):
    """95% Wilson score interval of a frame error rate."""
    if not frames:
        return None
    p = errors / frames
    denominator = 1 + z * z / frames
    center = (p + z * z / (2 * frames)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / frames + z * z / (4 * frames ** 2)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def stats(candidate):
    return {
        "source": candidate.source,
        "decoder_params": candidate.params,
        "frames": candidate.frames,
        "frame_errors": candidate.frame_errors,
        "fer": candidate.fer,
        "fer_95ci": wilson_interval(candidate.frame_errors, candidate.frames),
        "ber": candidate.ber,
    }


def save_result(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
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
    reference = Candidate(reference_params(base_experiment["codec"]["decoder_params"]))

    workers = args.workers or default_workers(simulation_config)
    output_path = args.output.resolve()
    log_path = output_path.with_suffix(".jsonl")
    log_path.write_text("", encoding="utf-8")
    common = {
        "snr_db": args.snr,
        "n_iterations": iterations,
        "seed": args.seed,
        "max_trials": args.trials,
        "target_frame_errors": args.max_errors,
        "grid_size": len(candidates),
    }
    print(
        f"GF search: SNR={args.snr:g} dB, iterations={iterations}, "
        f"trials={args.trials}, max_errors={args.max_errors}, "
        f"parameter_sets={len(candidates)}, workers={workers}",
        flush=True,
    )

    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(base_experiment, args.snr, args.seed),
    ) as pool:
        search = Search(pool, args, iterations, log_path)

        survivors, frames = search.halving(candidates, "grid")
        save_result(output_path, {
            **common, "stage": "grid halving (search noise, not confirmed)",
            "best": stats(survivors[0]),
        })

        if args.zoom_centers:
            known = {c.key for c in candidates}
            zoomed = unique(zoom_candidates(args, survivors[:args.zoom_centers]), known)
            search.say(f"zoom: {len(zoomed)} new candidates around the best "
                       f"{min(args.zoom_centers, len(survivors))}")
            if zoomed:
                zoomed, _ = search.halving(zoomed, "zoom", final_frames=frames)
                survivors = sorted(survivors + zoomed, key=Candidate.score)
            best_so_far = survivors[0]
            save_result(output_path, {
                **common, "stage": "zoom (search noise, not confirmed)",
                "best": stats(best_so_far),
            })

        finalists = survivors[:args.finalists]
        search.log(survivors[args.finalists:], "zoom" if args.zoom_centers else "grid")
        search.log(finalists, "finalist")
        confirmed = search.confirm(finalists, reference)

    best = confirmed[0]
    reference_entry = next(c for c in confirmed if c.key == reference.key)
    save_result(output_path, {
        **common,
        "stage": "confirmed on fresh noise",
        "diverged_parameter_sets": search.invalid_count,
        "best": stats(best),
        "reference": stats(reference_entry),
        "finalists": [stats(c) for c in confirmed],
        "decoder_params": best.params,
    })
    print(
        f"Search completed in {(time.perf_counter() - search.started) / 60:.1f} min. "
        f"{search.invalid_count} parameter sets diverged.\n"
        f"Best (fresh noise): FER={best.fer:.4g} "
        f"({best.frame_errors} errors / {best.frames} frames), BER={best.ber:.4g}\n"
        f"  {format_params(best.params)}\n"
        f"Reference: FER={reference_entry.fer:.4g}  {format_params(reference_entry.params)}\n"
        f"Saved to {output_path}, all candidates in {log_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
