import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcPgdDecoder(BinLdpcDecoderBase):
    """Probabilistic gradient decoder operating in the received domain."""

    requires_sigma_noise = True

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.learning_rate_decay = kwargs["learning_rate_decay"]
        self.momentum = kwargs["momentum"]
        self.alpha = kwargs["alpha"]
        self.beta = kwargs["beta"]

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0:
            raise ValueError("Learning rate decay must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")
        if self.alpha < 0:
            raise ValueError("Alpha must be non-negative")
        if self.beta <= 0:
            raise ValueError("Beta must be positive")

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

    @staticmethod
    def probabilities(x, sigma_squared, beta):
        """Convert received-domain values to P(X=+1 | x)."""
        scaled = 2 * beta * x / sigma_squared
        return np.exp(-np.logaddexp(0, -scaled))

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def extrinsic_bipolar_products(self, bipolar_probabilities):
        """Calculate products of all other bipolar means on every edge."""
        edge_values = bipolar_probabilities[self.edge_vn]
        is_zero = edge_values == 0
        zero_counts = np.add.reduceat(
            is_zero,
            self.check_offsets[:-1],
        )
        nonzero_products = np.multiply.reduceat(
            np.where(is_zero, 1, edge_values),
            self.check_offsets[:-1],
        )

        check_zero_counts = zero_counts[self.edge_cn]
        check_nonzero_products = nonzero_products[self.edge_cn]
        extrinsic_products = np.zeros_like(edge_values)

        no_zeros = check_zero_counts == 0
        np.divide(
            check_nonzero_products,
            edge_values,
            out=extrinsic_products,
            where=no_zeros,
        )

        only_zero_edge = (check_zero_counts == 1) & is_zero
        extrinsic_products[only_zero_edge] = check_nonzero_products[
            only_zero_edge
        ]
        return extrinsic_products

    def objective_gradient(self, x, received, sigma_squared):
        probabilities = self.probabilities(x, sigma_squared, self.beta)
        bipolar_probabilities = 2 * probabilities - 1
        edge_products = self.extrinsic_bipolar_products(
            bipolar_probabilities
        )
        check_product_sums = np.bincount(
            self.edge_vn,
            weights=edge_products,
            minlength=self.block_length,
        )
        return (
            self.alpha * received
            + (4 * self.beta / sigma_squared)
            * probabilities
            * (1 - probabilities)
            * check_product_sums
        )

    def decode(self, llr_in, llr_out, rng=None, sigma_noise=None):
        if sigma_noise is None or sigma_noise <= 0:
            raise ValueError("A positive channel noise sigma is required")

        received = llr_in.copy()
        x = received.copy()
        velocity = np.zeros_like(x)
        sigma_squared = sigma_noise * sigma_noise

        for iteration in range(self.n_iterations):
            hard_x = np.where(x >= 0, 1, -1).astype(np.int8)
            if np.all(self.bpsk_syndrome(hard_x) == 1):
                llr_out[:] = x
                return iteration

            gradient = self.objective_gradient(
                x,
                received,
                sigma_squared,
            )
            velocity = (
                self.momentum * velocity
                + (1 - self.momentum) * gradient
            )
            current_learning_rate = self.learning_rate / np.sqrt(
                1 + self.learning_rate_decay * iteration
            )
            x = x + current_learning_rate * velocity

        llr_out[:] = x
        return self.n_iterations
