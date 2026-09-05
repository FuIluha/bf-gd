import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSoftGdbfDecoder(BinLdpcDecoderBase):
    """Implementation of projected gradient ascent bit-flipping decoder."""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.momentum = kwargs["momentum"]
        self.regularization = kwargs["regularization"]
        self.alpha = kwargs["alpha"]

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0:
            raise ValueError("Learning rate decay must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")
        if self.regularization < 0:
            raise ValueError("Regularization must be non-negative")

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

    def check_to_variable_messages(self, x):
        """Calculate min-sum check-to-variable messages."""
        edge_values = x[self.edge_vn]
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

    def objective_gradient(self, x, y):
        """Calculate a received-scale soft bit-update direction."""
        edge_messages = self.check_to_variable_messages(x)
        check_message_sum = np.bincount(
            self.edge_vn,
            weights=edge_messages,
            minlength=self.block_length,
        )
        return (
            self.alpha * y
            + check_message_sum
            - self.regularization * x
        )

    def decode(self, llr_in, llr_out, rng=None):
        y = llr_in.copy()
        x = y.copy()
        velocity = np.zeros_like(x)

        for iteration in range(self.n_iterations): # iteration loop
            hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
            check_syndromes = self.bpsk_syndrome(hard_x) # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration # exit the iteration loop;

            grad = self.objective_gradient(x, y)
            velocity = (
                self.momentum * velocity
                + (1 - self.momentum) * grad
            )
            current_learning_rate = self.learning_rate / np.sqrt(
                1 + self.learning_rate_decay * iteration
            )
            x = x + current_learning_rate * velocity

        llr_out[:] = x
        return self.n_iterations
