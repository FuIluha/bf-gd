"""Gradient Flow (GF) decoder, Wadayama & Wei, IEEE Access 2025.

The GF-ODE  dx/dt = -(x - y + gamma * grad h(x))  is solved by the Euler
method with step eta; every decoder iteration is one Euler step, so
n_iterations = N and the sampling time is T = N * eta.

Code potential energy:
    h(x) = alpha * sum_j (x_j^2 - 1)^2
         + beta  * sum_i (prod_{j in A(i)} x_j - 1)^2
"""

import numpy as np
from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcGfDecoder(BinLdpcDecoderBase):
    """Classic gradient flow decoder for the AWGN channel (Euler method)."""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
        self.beta = float(kwargs["beta"])
        self.gamma = float(kwargs["gamma"])
        self.eta = float(kwargs["eta"])

        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("alpha, beta and gamma must be non-negative")

        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)

        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks
        )

        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

        if np.any(check_degrees < 1):
            raise ValueError("GF requires every check to have at least one edge")

    def check_products(self, x):
        """Return prod_{j in A(i)} x_j and prod_{j in A(i) minus k} x_j per edge.

        The exclusive product is computed without division, so zero
        components (e.g. the initial point x = 0) are handled exactly.
        """
        edge_values = x[self.edge_vn]
        is_zero = edge_values == 0
        nonzero_values = np.where(is_zero, 1.0, edge_values)

        zero_counts = np.add.reduceat(is_zero.astype(np.int32), self.check_offsets[:-1])
        nonzero_products = np.multiply.reduceat(nonzero_values, self.check_offsets[:-1])
        full_products = np.where(zero_counts == 0, nonzero_products, 0.0)

        edge_zero_counts = zero_counts[self.edge_cn]
        edge_nonzero_products = nonzero_products[self.edge_cn]
        exclusive_products = np.where(
            edge_zero_counts == 0,
            edge_nonzero_products / nonzero_values,
            np.where(is_zero & (edge_zero_counts == 1), edge_nonzero_products, 0.0),
        )
        return full_products, exclusive_products

    def potential_energy(self, x):
        """Code potential energy h_{alpha,beta}(x), eq. (6)."""
        full_products, _ = self.check_products(x)
        return (
            self.alpha * np.sum((x * x - 1) ** 2)
            + self.beta * np.sum((full_products - 1) ** 2)
        )

    def potential_gradient(self, x):
        """Gradient of the code potential energy, eq. (12)."""
        full_products, exclusive_products = self.check_products(x)
        check_terms = (full_products[self.edge_cn] - 1) * exclusive_products
        return (
            4 * self.alpha * (x * x - 1) * x
            + 2 * self.beta * np.bincount(
                self.edge_vn, weights=check_terms, minlength=self.block_length,
            )
        )

    def objective(self, x, y):
        """Potential energy f(x) = 0.5 * ||x - y||^2 + gamma * h(x), eq. (5)."""
        return 0.5 * np.sum((x - y) ** 2) + self.gamma * self.potential_energy(x)

    def step(self, x, y):
        """One Euler step of the GF-ODE, eq. (10)."""
        next_x = x - self.eta * (x - y + self.gamma * self.potential_gradient(x))
        if not np.all(np.isfinite(next_x)):
            raise FloatingPointError("Non-finite GF state")
        return next_x

    def decode(self, llr_in, llr_out, rng=None):
        y = llr_in.astype(np.float64, copy=True)
        x = np.zeros_like(y)

        for _ in range(self.n_iterations):
            x = self.step(x, y)

        llr_out[:] = x
        return self.n_iterations
