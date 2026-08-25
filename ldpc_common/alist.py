"""Reader and writer for binary matrices in ALIST format."""

import numpy as np


class Alist:
    """Parse and write binary matrices in ALIST format."""

    @staticmethod
    def write(matr, filename):
        """Write a two-dimensional NumPy array to an ALIST file."""
        column_weights = np.sum(matr > 0, axis=0)
        row_weights = np.sum(matr > 0, axis=1)
        row_max = np.max(row_weights)
        col_max = np.max(column_weights)

        with open(filename, 'w', encoding='utf-8') as file:
            print(f'{matr.shape[1]} {matr.shape[0]}', file=file)
            print(f'{col_max} {row_max}', file=file)
            print(Alist.to_string(column_weights), file=file)
            print(Alist.to_string(row_weights), file=file)

            for i in range(matr.shape[1]):
                print(
                    Alist.to_string(
                        np.nonzero(matr[:, i])[0] + 1,
                        col_max,
                    ),
                    file=file,
                )
            for i in range(matr.shape[0]):
                print(
                    Alist.to_string(
                        np.nonzero(matr[i, :])[0] + 1,
                        row_max,
                    ),
                    file=file,
                )

    @staticmethod
    def read(filename):
        """Read an ALIST file into a two-dimensional binary NumPy array."""
        with open(filename, 'r', encoding='utf-8') as file:
            matrix_size = np.fromstring(
                file.readline(), sep=' ', dtype=np.uint
            )
            matr = np.zeros((matrix_size[1], matrix_size[0]), dtype=np.uint)
            max_counts = np.fromstring(
                file.readline(), sep=' ', dtype=np.uint
            )
            row_weights = np.fromstring(
                file.readline(), sep=' ', dtype=np.uint
            )
            col_weights = np.fromstring(
                file.readline(), sep=' ', dtype=np.uint
            )

            for i in range(matr.shape[1]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[0]
                idx = idx[idx > 0] - 1
                assert len(idx) == row_weights[i]
                matr[idx, i] = 1

            for i in range(matr.shape[0]):
                idx = np.fromstring(file.readline(), sep=' ', dtype=np.uint)
                assert len(idx) == max_counts[1]
                idx = idx[idx > 0] - 1
                assert len(idx) == col_weights[i]
                assert np.sum(matr[i, :]) == len(idx)
                assert np.sum(matr[i, idx]) == len(idx)

        return matr

    @staticmethod
    def read_shape(filename):
        """Read only the matrix shape from an ALIST file."""
        with open(filename, 'r', encoding='utf-8') as file:
            matrix_size = np.fromstring(
                file.readline(), sep=' ', dtype=np.uint
            )
        return matrix_size[1], matrix_size[0]

    @staticmethod
    def to_string(np_arr, length=None):
        """Convert an array to an ALIST-compatible space-separated string."""
        if length:
            np_arr = np.hstack(
                [np_arr, np.array([0] * (length - len(np_arr)))]
            )
        return ' '.join(map(str, np_arr.astype(np.uint).tolist()))
