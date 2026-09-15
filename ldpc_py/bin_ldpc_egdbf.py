"""Hard-decision Edge-wise Gradient Descent Bit-Flipping decoder."""

import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcEgdbfDecoder(BinLdpcDecoderBase):
    """Hard message passing with persistent variable-to-check signs."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
        self.L = int(kwargs["L"])
        rho = np.asarray(kwargs["rho"], dtype=np.float64)

        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if self.L <= 0:
            raise ValueError("L must be positive")
        if len(rho) != self.L:
            raise ValueError("Momentum length must be equal to L")
        if not np.all(np.isfinite(rho)):
            raise ValueError("Momentum values must be finite")

        # rho[L] is used before an edge message has changed for the first time.
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
        variable_degrees = np.bincount(
            self.edge_vn,
            minlength=self.block_length,
        )
        if np.any(variable_degrees == 0):
            raise ValueError("E-GDBF does not support degree-zero variable nodes")

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

    def posterior_gradient(self, y, incoming_sums):
        """Return the full gradient used only for the hard-word decision."""
        return self.alpha * y + incoming_sums

    def extrinsic_gradients(
        self,
        y,
        variable_messages,
        check_messages,
        incoming_sums,
        ages,
    ):
        """Return one recipient-excluding gradient for every outgoing edge."""
        variables = self.edge_vn
        return (
            self.alpha * y[variables]
            + incoming_sums[variables]
            - check_messages
            + self.rho[ages - 1] * variable_messages
        )

    @staticmethod
    def hard_sign(values, previous):
        """Quantize to +/-1, retaining the previous sign on an exact tie."""
        return np.where(
            values > 0,
            1,
            np.where(values < 0, -1, previous),
        ).astype(np.int8)

    def hard_word(self, posterior_gradient, channel_signs):
        """Make the a-posteriori hard decision; channel breaks exact ties."""
        return self.hard_sign(posterior_gradient, channel_signs)

    def decode(self, llr_in, llr_out, rng=None):
        del rng  # E-GDBF message updates are deterministic.
        y = np.asarray(llr_in, dtype=np.float64)
        channel_signs = np.where(y >= 0, 1, -1).astype(np.int8)
        variable_messages = channel_signs[self.edge_vn].copy()
        ages = np.full(self.edges_count, self.L + 1, dtype=np.int32)

        for iteration in range(self.n_iterations):
            check_messages = self.check_to_variable_messages(
                variable_messages,
            )
            incoming = self.incoming_sums(check_messages)
            posterior = self.posterior_gradient(y, incoming)
            x = self.hard_word(posterior, channel_signs)
            if np.all(self.bpsk_syndrome(x) == 1):
                llr_out[:] = x
                return iteration

            np.minimum(ages, self.L, out=ages)
            ages += 1
            gradients = self.extrinsic_gradients(
                y,
                variable_messages,
                check_messages,
                incoming,
                ages,
            )
            new_messages = self.hard_sign(
                gradients,
                variable_messages,
            )
            changed = new_messages != variable_messages
            variable_messages = new_messages
            ages[changed] = 0

        check_messages = self.check_to_variable_messages(variable_messages)
        incoming = self.incoming_sums(check_messages)
        posterior = self.posterior_gradient(y, incoming)
        llr_out[:] = self.hard_word(posterior, channel_signs)
        return self.n_iterations
