import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSoftGdbfDecoder(BinLdpcDecoderBase):
    """Implementation of soft gradient descent bit-flipping decoder."""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = float(kwargs.get("learning_rate", 1.0))
        self.update_probability = float(kwargs.get("update_probability", 1.0))
        self.beta1 = float(kwargs.get("beta1", 0.9))
        self.beta2 = float(kwargs.get("beta2", 0.999))
        self.adam_epsilon = float(kwargs.get("adam_epsilon", 1e-8))

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not 0 < self.update_probability <= 1:
            raise ValueError("Update probability must be in (0, 1]")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if self.adam_epsilon <= 0:
            raise ValueError("Adam epsilon must be positive")

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
        """Calculate the gradient of the soft GDBF objective function."""
        edge_values = x[self.edge_vn]
        edge_is_zero = edge_values == 0

        nonzero_edge_values = np.where(edge_is_zero, 1, edge_values)

        # Keep the sign and magnitude separate because log is undefined for
        # negative values.
        check_sign_products = np.multiply.reduceat(
            np.sign(nonzero_edge_values),
            self.check_offsets[:-1],
        )
        edge_log_magnitudes = np.log(np.abs(nonzero_edge_values))
        check_log_magnitude_sums = np.add.reduceat(
            edge_log_magnitudes,
            self.check_offsets[:-1],
        )
        check_zero_counts = np.add.reduceat(
            edge_is_zero,
            self.check_offsets[:-1],
        )

        edge_check_signs = check_sign_products[self.edge_cn]
        edge_check_log_magnitudes = check_log_magnitude_sums[self.edge_cn]
        edge_zero_counts = check_zero_counts[self.edge_cn]
        extrinsic_products = np.zeros_like(edge_values)

        # No zeros in a check: remove the current edge in the log domain.
        no_zero_mask = edge_zero_counts == 0
        extrinsic_products[no_zero_mask] = (
            edge_check_signs[no_zero_mask]
            * np.sign(edge_values[no_zero_mask])
            * np.exp(
                edge_check_log_magnitudes[no_zero_mask]
                - edge_log_magnitudes[no_zero_mask]
            )
        )

        # Exactly one zero: only that zero receives a nonzero product.
        single_zero_edge_mask = (edge_zero_counts == 1) & edge_is_zero
        extrinsic_products[single_zero_edge_mask] = (
            edge_check_signs[single_zero_edge_mask]
            * np.exp(edge_check_log_magnitudes[single_zero_edge_mask])
        )

        return y + np.bincount(
            self.edge_vn,
            weights=extrinsic_products,
            minlength=self.block_length,
        )

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        y = llr_in.copy()
        x = y.copy()
        gradient_history = np.zeros_like(x)
        second_moment = np.zeros_like(x)

        for iteration in range(self.n_iterations): # iteration loop
            hard_x = (2 * (x >= 0) - 1).astype(np.int8)
            check_syndromes = self.bpsk_syndrome(hard_x) # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = hard_x
                return iteration # exit the iteration loop;

            grad = self.objective_gradient(x, y)
            penalized_grad = grad - self.beta1 * gradient_history

            second_moment *= self.beta2
            second_moment += (
                (1 - self.beta2) * penalized_grad * penalized_grad
            )

            step = iteration + 1
            corrected_second_moment = second_moment / (1 - self.beta2 ** step)
            adam_update = self.learning_rate * penalized_grad / (
                np.sqrt(corrected_second_moment) + self.adam_epsilon
            )
            update_mask = (
                rng.random(self.block_length) < self.update_probability
            )
            x[update_mask] += adam_update[update_mask]

            # Remember only directions in which an update was actually made.
            gradient_history *= self.beta1
            gradient_history[update_mask] += (
                (1 - self.beta1) * grad[update_mask]
            )

        llr_out[:] = 2 * (x >= 0) - 1
        return self.n_iterations
