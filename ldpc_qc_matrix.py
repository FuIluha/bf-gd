"""Utilities for expanding QC-LDPC protographs."""

import argparse
import json
from pathlib import Path

import numpy as np

from ldpc_common.alist import Alist

PROTOGRAPH_DIRECTORY = Path("ldpc_qc_data")
CODES_DIRECTORY = Path("codes")


def expand_qc_matrix(protograph, zc):
    """Expand a QC-LDPC protograph into a binary parity-check matrix.

    A ``-1`` entry represents an all-zero ``zc`` by ``zc`` block. Each
    non-negative entry represents an identity matrix circularly shifted
    to the right by that number of positions.
    """
    protograph = np.asarray(protograph)

    if protograph.ndim != 2:
        raise ValueError("protograph must be a two-dimensional array")
    if not isinstance(zc, (int, np.integer)) or zc <= 0:
        raise ValueError("Zc must be a positive integer")
    if not np.issubdtype(protograph.dtype, np.integer):
        raise ValueError("protograph entries must be integers")
    if np.any(protograph < -1):
        raise ValueError("protograph entries must be -1 or non-negative")

    n_block_rows, n_block_columns = protograph.shape
    matrix = np.zeros(
        (n_block_rows * zc, n_block_columns * zc),
        dtype=np.uint8,
    )

    identity_rows = np.arange(zc)
    block_rows, block_columns = np.nonzero(protograph >= 0)

    for block_row, block_column in zip(block_rows, block_columns):
        shift = int(protograph[block_row, block_column])
        rows = block_row * zc + identity_rows
        columns = block_column * zc + (identity_rows + shift) % zc
        matrix[rows, columns] = 1

    return matrix


def generate_qc_code(protograph_filename, zc):
    """Generate the ALIST and code configuration used by the simulator."""
    protograph_path = PROTOGRAPH_DIRECTORY / protograph_filename
    protograph = np.loadtxt(protograph_path, dtype=np.int64)
    matrix = expand_qc_matrix(protograph, zc)

    CODES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    code_name = f"{protograph_path.stem}_Zc{zc}"
    pcm_path = CODES_DIRECTORY / f"{code_name}_pcm.alist"
    config_path = CODES_DIRECTORY / f"{code_name}.json"

    Alist.write(matrix, pcm_path)

    config = {
        "name": code_name,
        "pcm": pcm_path.name,
        "punctured": 0,
        "is_systematic": False,
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a QC-LDPC parity-check matrix"
    )
    parser.add_argument(
        "-c",
        "--protograph",
        required=True,
        help="protograph filename in the ldpc_qc_data directory",
    )
    parser.add_argument(
        "--Zc",
        dest="zc",
        required=True,
        type=int,
        help="circulant size",
    )
    arguments = parser.parse_args()
    generated_path = generate_qc_code(arguments.protograph, arguments.zc)
    print(f"Successfully generated {generated_path}")
