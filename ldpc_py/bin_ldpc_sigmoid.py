import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSigmoidDecoder(BinLdpcDecoderBase):
    """Implementation of sigmoid multi gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.theta = kwargs["theta"]
        self.mu = kwargs["mu"]
        self.beta = kwargs.get("beta", 1.0)
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

    def objective_gradient(self, x, y):
        """Calculate the score for moving each bit along the objective gradient."""
        edge_values = x[self.edge_vn]
        check_products = np.multiply.reduceat(
            edge_values,
            self.check_offsets[:-1],
        )
        extrinsic_messages = check_products[self.edge_cn] * edge_values
        sigmoid = 1.0 / (
            1.0 + np.exp(-self.beta * extrinsic_messages)
        )
        soft_messages = 2.0 * sigmoid - 1.0
        check_message_sum = np.bincount(
            self.edge_vn,
            weights=soft_messages,
            minlength=self.block_length,
        )
        gradient = y + check_message_sum
        return x * gradient

    def decode(self, llr_in, llr_out, rng=None):
        mu = self.mu
        y = llr_in.copy()

        # step 1
        x = (2 * (y >= 0) - 1).astype(np.int8)  # sign, zero is positive
        for iteration in range(self.n_iterations):
            check_syndromes = self.bpsk_syndrome(x)

            # step 2
            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration

            # step 3
            f1 = (np.dot(x, y) + np.sum(check_syndromes))
            gradient = self.objective_gradient(x, y)

            if mu == 0:
                # step 3.1 (multi-bit mode)
                x[gradient < self.theta] *= -1
                updated_check_syndromes = self.bpsk_syndrome(x)
                f2 = (np.dot(x, y) + np.sum(updated_check_syndromes))
                if f1 > f2:
                    mu = 1
            else:
                # step 3.2 (single-bit mode)
                x[np.argmin(gradient)] *= -1

        # step 4
        llr_out[:] = x
        return self.n_iterations
        