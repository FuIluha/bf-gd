"""Plot FER curves for BP, min-sum, GD, soft GDBF, and PMGDBF decoders."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT = PROJECT_DIR / "fer_decoder_comparison.png"


def latest_result(pattern, required=True):
    files = list(DATA_DIR.glob(pattern))
    if not files:
        if not required:
            return None
        raise FileNotFoundError(
            f"No result files matching {pattern!r} in {DATA_DIR}"
        )
    return max(files, key=lambda path: path.stat().st_mtime)


def load_fer(path):
    data = np.genfromtxt(path, names=True)
    data = np.atleast_1d(data)
    valid = (
        np.isfinite(data["snr"])
        & np.isfinite(data["fer"])
        & (data["tests"] > 0)
        & (data["fer"] > 0)
    )
    return (
        data["snr"][valid],
        data["fer"][valid],
        data["fer_e_minus"][valid],
        data["fer_e_plus"][valid],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot simulated FER for BP, min-sum, GD, soft GDBF, and PMGDBF."
        )
    )
    parser.add_argument(
        "--bp",
        type=Path,
        help="sum-product BP text result; the latest matching file is used by default",
    )
    parser.add_argument(
        "--soft-bf",
        type=Path,
        help="soft BF text result; the latest matching file is used by default",
    )
    parser.add_argument(
        "--min-sum",
        type=Path,
        help="min-sum scale 1.0 text result; the latest matching file is used by default",
    )
    parser.add_argument(
        "--min-sum-075",
        type=Path,
        help="min-sum scale 0.75 text result; added automatically when available",
    )
    parser.add_argument(
        "--pmgdbf",
        type=Path,
        help="PMGDBF text result; the latest matching file is used by default",
    )
    parser.add_argument(
        "--gd",
        type=Path,
        help="GD text result; added automatically when available",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def add_fer_curve(axis, path, label, color, marker):
    snr, fer, error_minus, error_plus = load_fer(path)
    axis.errorbar(
        snr,
        fer,
        yerr=np.vstack((error_minus, error_plus)),
        label=label,
        color=color,
        marker=marker,
        markersize=4,
        linewidth=1.5,
        capsize=2,
    )


def main():
    args = parse_args()
    bp_path = args.bp or latest_result("*sum_product*.txt")
    min_sum_path = args.min_sum or latest_result("*min_sum*scale_1.000*.txt")
    min_sum_075_path = args.min_sum_075 or latest_result(
        "*min_sum*scale_0.750*.txt", required=False
    )
    soft_bf_path = args.soft_bf or latest_result(
        "*cpp soft gradient descent bit-flipping*.txt"
    )
    pmgdbf_path = args.pmgdbf or latest_result(
        "*probabilistic momentum gradient descent bit-flipping*.txt"
    )
    gd_path = args.gd or latest_result(
        "*gradient descent decoder*.txt", required=False
    )

    figure, axis = plt.subplots(figsize=(9, 6))
    add_fer_curve(axis, bp_path, "BP (Sum-Product)", "tab:blue", "o")
    add_fer_curve(axis, min_sum_path, "Min-Sum (scale=1.0)", "tab:orange", "s")
    if min_sum_075_path is not None:
        add_fer_curve(
            axis,
            min_sum_075_path,
            "Min-Sum (scale=0.75)",
            "tab:purple",
            "v",
        )
    add_fer_curve(axis, soft_bf_path, "Soft GDBF", "tab:green", "^")
    add_fer_curve(axis, pmgdbf_path, "PMGDBF", "tab:red", "D")
    if gd_path is not None:
        add_fer_curve(axis, gd_path, "GD", "tab:brown", "P")

    axis.set_yscale("log")
    axis.set_xlabel("SNR, dB")
    axis.set_ylabel("FER")
    axis.set_title("LDPC decoder FER comparison")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    print(f"BP data: {bp_path}")
    print(f"Min-sum data: {min_sum_path}")
    if min_sum_075_path is not None:
        print(f"Min-sum scale 0.75 data: {min_sum_075_path}")
    print(f"Soft GDBF data: {soft_bf_path}")
    print(f"PMGDBF data: {pmgdbf_path}")
    if gd_path is not None:
        print(f"GD data: {gd_path}")
    print(f"Plot saved to: {output_path}")

    plt.close(figure)


if __name__ == "__main__":
    main()
