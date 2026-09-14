"""Edge-wise Gradient Descent Bit-Flipping decoder."""

import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcEgdbfDecoder(BinLdpcDecoderBase):
    """GDBF with binary extrinsic messages and edge-wise energies."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = float(kwargs["delta"])
        self.alpha = float(kwargs["alpha"])
        self.p = float(kwargs["p"])
        self.L = int(kwargs["L"])
        rho = np.asarray(kwargs["rho"], dtype=np.float64)

        if not np.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.p) or not 0 < self.p <= 1:
            raise ValueError("p must be finite and in (0, 1]")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if len(rho) != self.L:
            raise ValueError("Momentum length must be equal to L")
        if not np.all(np.isfinite(rho)):
            raise ValueError("Momentum values must be finite")

        self.rho = np.concatenate((rho, np.array([0.0])))

        edge_cn, edge_vn = np.nonzero(self.pcm)
        self.edge_cn = edge_cn.astype(np.int32)
        self.edge_vn = edge_vn.astype(np.int32)
        self.edges_count = len(self.edge_cn)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks,
        )
        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))
        self.variable_degrees = np.bincount(
            self.edge_vn,
            minlength=self.block_length,
        )
        if np.any(self.variable_degrees == 0):
            raise ValueError("E-GDBF does not support degree-zero variable nodes")

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def check_to_variable_messages(self, x, check_syndromes=None):
        """Return r[a->i], the product of all other bits in check a."""
        if check_syndromes is None:
            check_syndromes = self.bpsk_syndrome(x)
        edge_values = x[self.edge_vn]
        return check_syndromes[self.edge_cn] * edge_values

    def edge_energies(self, x, y, check_messages, ages):
        """Return one extrinsic check opinion/energy for every graph edge."""
        intrinsic_and_momentum = (
            self.alpha * x * y + self.rho[ages - 1]
        )
        check_alignment = x[self.edge_vn] * check_messages
        return check_alignment + intrinsic_and_momentum[self.edge_vn]

    def posterior_energies(self, edge_energies):
        """Aggregate all edge energies for the variable-node flip decision."""
        return np.bincount(
            self.edge_vn,
            weights=edge_energies,
            minlength=self.block_length,
        )

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        y = llr_in.copy()
        x = np.where(y >= 0, 1, -1).astype(np.int8)
        ages = np.full(self.block_length, self.L + 1, dtype=np.int32)

        for iteration in range(self.n_iterations):
            check_syndromes = self.bpsk_syndrome(x)
            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration

            np.minimum(ages, self.L, out=ages)
            ages += 1

            check_messages = self.check_to_variable_messages(
                x,
                check_syndromes,
            )
            edge_energy = self.edge_energies(
                x,
                y,
                check_messages,
                ages,
            )
            posterior_energy = self.posterior_energies(edge_energy)

            threshold = np.min(posterior_energy) + self.delta
            selected = rng.random(self.block_length) < self.p
            flip_mask = (posterior_energy <= threshold) & selected
            x[flip_mask] *= -1
            ages[flip_mask] = 0

        llr_out[:] = x
        return self.n_iterations
