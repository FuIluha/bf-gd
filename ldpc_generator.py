"""Generate a binary generator matrix from an ALIST parity-check matrix."""

import argparse
from pathlib import Path

import numpy as np

from ldpc_common.alist import Alist


def generator_from_parity_check(parity_check, dimension=None):
    """Return independent rows from the GF(2) null space of parity_check."""
    reduced = np.asarray(parity_check, dtype=np.uint8).copy()
    if reduced.ndim != 2:
        raise ValueError("parity-check matrix must be two-dimensional")

    n_rows, n_columns = reduced.shape
    pivot_columns = []
    pivot_row = 0
    for column in range(n_columns):
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if candidates.size == 0:
            continue
        selected_row = pivot_row + int(candidates[0])
        if selected_row != pivot_row:
            reduced[[pivot_row, selected_row]] = reduced[
                [selected_row, pivot_row]
            ]

        rows_to_eliminate = np.flatnonzero(reduced[:, column])
        rows_to_eliminate = rows_to_eliminate[rows_to_eliminate != pivot_row]
        reduced[rows_to_eliminate] ^= reduced[pivot_row]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == n_rows:
            break

    pivot_columns = np.asarray(pivot_columns, dtype=np.int64)
    free_columns = np.setdiff1d(
        np.arange(n_columns, dtype=np.int64),
        pivot_columns,
        assume_unique=True,
    )
    generator = np.zeros((free_columns.size, n_columns), dtype=np.uint8)
    generator[:, free_columns] = np.eye(free_columns.size, dtype=np.uint8)
    generator[:, pivot_columns] = reduced[
        :pivot_columns.size,
        free_columns,
    ].T

    if dimension is not None:
        if dimension <= 0 or dimension > generator.shape[0]:
            raise ValueError(
                f"dimension must be in [1, {generator.shape[0]}]"
            )
        generator = generator[:dimension]
    return generator, pivot_columns.size


def main():
    parser = argparse.ArgumentParser(
        description="Generate a binary generator matrix from an ALIST file."
    )
    parser.add_argument("pcm", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dimension", type=int)
    args = parser.parse_args()

    parity_check = Alist.read(args.pcm).astype(np.uint8)
    generator, rank = generator_from_parity_check(
        parity_check,
        args.dimension,
    )
    if np.any((generator @ parity_check.T) % 2):
        raise RuntimeError("generated matrix does not satisfy G H^T = 0")
    np.savetxt(args.output, generator, fmt="%d")
    print(
        f"H shape={parity_check.shape}, rank={rank}, "
        f"G shape={generator.shape}, saved to {args.output}"
    )


if __name__ == "__main__":
    main()
