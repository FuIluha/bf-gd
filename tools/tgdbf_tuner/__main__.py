"""Run TGDBF tuning and serve its live dashboard."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
from threading import Thread

from .core import RunState, TuningConfig, run
from .dashboard import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE = PROJECT_ROOT / "codes" / "ldpc_savin_4_8_12_24_Zc54.json"


def _rho(value):
    try:
        result = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rho must be comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("rho must not be empty")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--snr", type=float, required=True, help="SNR in dB")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ratio", type=float, help="minimum correct/incorrect count N for threshold tuning")
    mode.add_argument("--fixed-delta", type=float, help="fixed upper threshold; observe actual flip ratio N")
    parser.add_argument("--bin-width", type=float, help="fixed width in E-Emin units for threshold tuning")
    parser.add_argument("--train-frames", type=int, default=10_000)
    parser.add_argument("--eval-frames", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.8)
    parser.add_argument("--rho", type=_rho, default=(0.0,) * 7)
    parser.add_argument("--L", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PROJECT_ROOT / "logs" / f"tgdbf_tune_{stamp}.jsonl"
    config = TuningConfig(
        code=args.code, snr_db=args.snr, ratio=args.ratio,
        bin_width=args.bin_width, fixed_delta=args.fixed_delta,
        train_frames=args.train_frames,
        eval_frames=args.eval_frames, seed=args.seed, workers=args.workers,
        iterations=args.iterations, alpha=args.alpha, rho=args.rho,
        L=args.L, output=output,
    )
    config.validate()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    state = RunState(config.metadata())

    def work():
        try:
            run(config, state)
        except Exception as exc:
            state.update(status="failed", message=f"{type(exc).__name__}: {exc}")
            print(f"TGDBF tuning failed: {type(exc).__name__}: {exc}", flush=True)

    worker = Thread(target=work, name="tgdbf-tuning", daemon=True)
    worker.start()
    print(f"TGDBF log: {Path(output).expanduser().resolve()}", flush=True)
    print(f"Host for SSH tunnel: {socket.gethostname()}", flush=True)
    print(f"Dashboard on {args.host}:{args.port}", flush=True)
    create_app(state).run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
