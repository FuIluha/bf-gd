"""Command-line entry point for the interactive energy explorer."""

import argparse
from pathlib import Path

from .app import create_app
from .data import DatasetConfig, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CODE = PROJECT_ROOT / "codes" / "ldpc_savin_4_8_12_24_Zc54.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive LDPC decoder diagnostic explorer.",
    )
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--frames", type=int, default=1_000)
    parser.add_argument("--snr", type=float, default=1.0, help="SNR in dB")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker threads for encoding and decoder stepping",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset = load_dataset(DatasetConfig(
        code_path=args.code,
        frames=args.frames,
        snr_db=args.snr,
        seed=args.seed,
        workers=args.workers,
    ))
    print(
        f"Loaded {dataset.batch.frames} frames of length "
        f"{dataset.batch.block_length} at SNR={args.snr:g} dB"
    )
    app = create_app(dataset, seed=args.seed, workers=args.workers)
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True,
    )


if __name__ == "__main__":
    main()
