"""Thresholded edge-wise hard message-passing decoder."""

import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcEgdbfDecoder(BinLdpcDecoderBase):
    """Hard extrinsic messages with a word-wide edge-energy threshold."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
        self.delta = float(kwargs.get("delta", 0.0))
        self.L = int(kwargs["L"])
        rho = np.asarray(kwargs["rho"], dtype=np.float64)

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.delta) or self.delta < 0:
            raise ValueError("delta must be finite and non-negative")
        if self.L < 0:
            raise ValueError("L must be non-negative")
        if len(rho) != self.L:
            raise ValueError("Momentum length must be equal to L")
        if not np.all(np.isfinite(rho)):
            raise ValueError("Momentum values must be finite")

        # rho[L] is used before an edge message has changed for the first time.
        self.rho = np.concatenate((rho, np.array([0.0]))) if self.L else rho

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
        variable_degrees = np.bincount(
            self.edge_vn,
            minlength=self.block_length,
        )
        if np.any(variable_degrees == 0):
            raise ValueError("E-GDBF does not support degree-zero variable nodes")
        self.variable_degrees = variable_degrees

    def bpsk_syndrome(self, x):
        """Return parity-check products for one hard word."""
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def check_to_variable_messages(self, variable_messages):
        """Compute r[a->i] from the other q[j->a] signs."""
        check_products = np.multiply.reduceat(
            variable_messages,
            self.check_offsets[:-1],
        )
        return check_products[self.edge_cn] * variable_messages

    def incoming_sums(self, check_messages):
        """Sum all check opinions incident to every variable node."""
        return np.bincount(
            self.edge_vn,
            weights=check_messages,
            minlength=self.block_length,
        )

    def posterior_score(self, y, incoming_sums):
        """Return the full bit score used for the hard-word decision."""
        return self.alpha * y + incoming_sums

    def edge_energies(
        self,
        y,
        variable_messages,
        check_messages,
        incoming_sums,
        ages,
    ):
        """Compute q[i->a] times the extrinsic score, plus edge momentum."""
        variables = self.edge_vn
        score = (
            self.alpha * y[variables]
            + incoming_sums[variables]
            - check_messages
        )
        energy = variable_messages * score
        if self.L:
            energy += self.rho[ages - 1]
        return energy

    def energy_threshold(self, energies):
        """V1: threshold relative to the least energetic edge."""
        return np.min(energies) + self.delta

    @staticmethod
    def hard_sign(values, previous):
        """Quantize to +/-1, retaining the previous sign on an exact tie."""
        return np.where(
            values > 0,
            1,
            np.where(values < 0, -1, previous),
        ).astype(np.int8)

    def hard_word(self, posterior_score, channel_signs):
        """Make the a-posteriori hard decision; channel breaks exact ties."""
        return self.hard_sign(posterior_score, channel_signs)

    def decode(self, llr_in, llr_out, rng=None):
        del rng  # E-GDBF message updates are deterministic.
        y = np.asarray(llr_in, dtype=np.float64)
        channel_signs = np.where(y >= 0, 1, -1).astype(np.int8)
        variable_messages = channel_signs[self.edge_vn].copy()
        ages = np.full(self.edges_count, self.L + 1, dtype=np.int32) if self.L else None

        for iteration in range(self.n_iterations):
            check_messages = self.check_to_variable_messages(
                variable_messages,
            )
            incoming = self.incoming_sums(check_messages)
            posterior = self.posterior_score(y, incoming)
            x = self.hard_word(posterior, channel_signs)
            if np.all(self.bpsk_syndrome(x) == 1):
                llr_out[:] = x
                return iteration

            if self.L:
                np.minimum(ages, self.L, out=ages)
                ages += 1
            energies = self.edge_energies(
                y,
                variable_messages,
                check_messages,
                incoming,
                ages,
            )
            flip = energies <= self.energy_threshold(energies)
            variable_messages[flip] *= -1
            if self.L:
                ages[flip] = 0

        check_messages = self.check_to_variable_messages(variable_messages)
        incoming = self.incoming_sums(check_messages)
        posterior = self.posterior_score(y, incoming)
        llr_out[:] = self.hard_word(posterior, channel_signs)
        return self.n_iterations
