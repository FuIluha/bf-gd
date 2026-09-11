import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSigmoidDecoder(BinLdpcDecoderBase):
    """Implementation of sigmoid multi gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.theta = kwargs["theta"]
        self.beta = kwargs.get("beta", 1.0)
        self.alpha = kwargs.get("alpha", 1.0)
        self.regularization = kwargs.get("regularization", 0.1)

        if self.theta <= 0:
            raise ValueError("Theta (learning rate) must be positive")
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

    def objective_gradient(self, x, y):
        """Calculate gradient """
        s = np.tanh(self.beta * x / 2.0)
        ds = (self.beta / 2.0) * (1.0 - s ** 2)

        edge_s = s[self.edge_vn]
        edge_ds = ds[self.edge_vn]

        check_products = np.multiply.reduceat(
            edge_s,
            self.check_offsets[:-1],
        )

        zero_mask = edge_s == 0.0
        zero_counts = np.add.reduceat(
            zero_mask,
            self.check_offsets[:-1],
        )
        extrinsic_products = np.zeros(self.edges_count, dtype=np.float64)

        no_zero_checks = zero_counts == 0
        no_zero_edges = no_zero_checks[self.edge_cn]
        extrinsic_products[no_zero_edges] = (
            check_products[self.edge_cn[no_zero_edges]]
            / edge_s[no_zero_edges]
        )

        for check_index in np.flatnonzero(zero_counts == 1):
            start = self.check_offsets[check_index]
            stop = self.check_offsets[check_index + 1]
            check_edge_values = edge_s[start:stop]
            extrinsic_products[start:stop] = np.prod(
                check_edge_values[check_edge_values != 0.0]
            )

        edge_contrib = edge_ds * extrinsic_products
        check_message_sum = np.bincount(
            self.edge_vn,
            weights=edge_contrib,
            minlength=self.block_length,
        )

        return self.alpha * y + check_message_sum - self.regularization * x

    def decode(self, llr_in, llr_out, rng=None):
        y = llr_in.copy()
        x = y.copy()

        for iteration in range(self.n_iterations):  # iteration loop
            hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
            check_syndromes = self.bpsk_syndrome(hard_x)  # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration  # exit the iteration loop;

            grad = self.objective_gradient(x, y)
            x = x + self.theta * grad

        llr_out[:] = x
        return self.n_iterations