import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcGdDecoder(BinLdpcDecoderBase):
    """Gradient decoder using the exact sum-product objective gradient."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.regularization = kwargs["regularization"]
        self.alpha = kwargs["alpha"]

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.regularization < 0:
            raise ValueError("Regularization must be non-negative")
        if self.alpha < 0:
            raise ValueError("Alpha must be non-negative")

        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)
        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks,
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
        """Calculate exact sum-product check-to-variable LLRs."""
        edge_values = x[self.edge_vn]
        edge_tanh = np.tanh(edge_values * 0.5)

        is_zero = edge_tanh == 0
        zero_counts = np.add.reduceat(
            is_zero,
            self.check_offsets[:-1],
        )
        nonzero_products = np.multiply.reduceat(
            np.where(is_zero, 1, edge_tanh),
            self.check_offsets[:-1],
        )

        check_zero_counts = zero_counts[self.edge_cn]
        check_nonzero_products = nonzero_products[self.edge_cn]
        extrinsic_products = np.zeros_like(edge_tanh)

        no_zeros = check_zero_counts == 0
        np.divide(
            check_nonzero_products,
            edge_tanh,
            out=extrinsic_products,
            where=no_zeros,
        )

        only_zero_edge = (check_zero_counts == 1) & is_zero
        extrinsic_products[only_zero_edge] = check_nonzero_products[
            only_zero_edge
        ]

        one = np.array(1, dtype=x.dtype)
        zero = np.array(0, dtype=x.dtype)
        atanh_limit = np.nextafter(one, zero)
        np.clip(
            extrinsic_products,
            -atanh_limit,
            atanh_limit,
            out=extrinsic_products,
        )
        return 2 * np.arctanh(extrinsic_products)

    def objective_gradient(self, x, channel_llr):
        check_messages = self.check_to_variable_messages(x)
        edge_values = x[self.edge_vn]
        edge_gradients = 0.5 * (
            np.tanh(0.5 * (edge_values + check_messages))
            - np.tanh(0.5 * (edge_values - check_messages))
        )
        check_gradient = np.bincount(
            self.edge_vn,
            weights=edge_gradients,
            minlength=self.block_length,
        )
        return (
            self.alpha * channel_llr
            + check_gradient
            - self.regularization * x
        )

    def decode(self, llr_in, llr_out, rng=None):
        channel_llr = llr_in.copy()
        x = channel_llr.copy()

        for iteration in range(self.n_iterations):
            hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
            if np.all(self.bpsk_syndrome(hard_x) == 1):
                llr_out[:] = x
                return iteration

            gradient = self.objective_gradient(x, channel_llr)
            x += self.learning_rate * gradient

        llr_out[:] = x
        return self.n_iterations
