import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSoftGdbfDecoder(BinLdpcDecoderBase):
    """Extrinsic edge-state decoder with L2 decay of messages and decisions."""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.alpha = kwargs["alpha"]

        self.l2 = float(kwargs.get("l2", 1.0))
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and non-negative")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0:
            raise ValueError("Learning rate decay must be non-negative")

        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)

        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        self.edges_count = len(self.edge_cn)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks
        )

        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def check_to_variable_messages(self, edge_values):
        """Calculate min-sum check-to-variable messages."""
        edge_signs = np.where(edge_values < 0, -1, 1)
        edge_magnitudes = np.abs(edge_values)

        check_signs = np.multiply.reduceat(
            edge_signs,
            self.check_offsets[:-1],
        )
        first_minima = np.minimum.reduceat(
            edge_magnitudes,
            self.check_offsets[:-1],
        )
        is_first_minimum = (
            edge_magnitudes == first_minima[self.edge_cn]
        )
        first_minimum_counts = np.add.reduceat(
            is_first_minimum,
            self.check_offsets[:-1],
        )
        second_minima = np.minimum.reduceat(
            np.where(is_first_minimum, np.inf, edge_magnitudes),
            self.check_offsets[:-1],
        )

        use_second_minimum = (
            is_first_minimum
            & (first_minimum_counts[self.edge_cn] == 1)
        )
        extrinsic_magnitudes = np.where(
            use_second_minimum,
            second_minima[self.edge_cn],
            first_minima[self.edge_cn],
        )
        extrinsic_signs = check_signs[self.edge_cn] * edge_signs
        return extrinsic_signs * extrinsic_magnitudes

    def variable_to_check_messages(self, y, check_messages):
        """Exclude the recipient check from each outgoing edge message."""
        total = self.objective_gradient(y, check_messages)
        return total[self.edge_vn] - check_messages

    def objective_gradient(self, y, check_messages):
        """Channel plus all current check messages, used as the bit-update direction."""
        return self.alpha * y + np.bincount(
            self.edge_vn, weights=check_messages, minlength=self.block_length,
        )

    def update_state(self, y, x, outgoing, iteration):
        incoming = self.check_to_variable_messages(outgoing)
        total = self.objective_gradient(y, incoming)
        eta = self.learning_rate / np.sqrt(1 + self.learning_rate_decay * iteration)
        next_q = outgoing + eta * (total[self.edge_vn] - incoming - self.l2 * outgoing)
        next_x = x + eta * (total - self.l2 * x)
        if not np.all(np.isfinite(next_q)) or not np.all(np.isfinite(next_x)):
            raise FloatingPointError("Non-finite soft GDBF state")
        return next_x, next_q

    def decode(self, llr_in, llr_out, rng=None):
        y = llr_in.astype(np.float64, copy=True)
        x = y.copy()
        outgoing = y[self.edge_vn].copy()

        for iteration in range(self.n_iterations): # iteration loop
            hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
            check_syndromes = self.bpsk_syndrome(hard_x) # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration # exit the iteration loop;

            x, outgoing = self.update_state(y, x, outgoing, iteration)

        llr_out[:] = x
        return self.n_iterations
