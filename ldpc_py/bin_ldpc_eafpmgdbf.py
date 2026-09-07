import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcEafpmgdbfDecoder(BinLdpcDecoderBase):
    """Implementation of erasure add probabilistic momentum gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.delta_e = kwargs["delta_e"]
        self.alpha = kwargs["alpha"]
        self.p = kwargs["p"]
        self.zeros_in_init = kwargs["zeros_in_init"]
        rho = np.asarray(kwargs["rho"], dtype=np.float32)
        self.L = kwargs["L"]

        if len(rho) != self.L:
            raise ValueError("Momentum length must be equal to L")

        self.rho = np.concatenate((
            rho,
            np.array([0.0], dtype=np.float32),
        ))
        self.edge_cn, self.edge_vn = np.nonzero(self.pcm)

        self.edge_cn = self.edge_cn.astype(np.int32)
        self.edge_vn = self.edge_vn.astype(np.int32)

        self.edges_count = len(self.edge_cn)

        check_degrees = np.bincount(
            self.edge_cn,
            minlength=self.n_checks
        )


        self.check_degrees = check_degrees

        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def check_state(self, x):
        edge_values = x[self.edge_vn]
        edge_erasure_mask = edge_values == 0

        check_nonzero_products = np.multiply.reduceat(
            np.where(edge_erasure_mask, 1, edge_values),
            self.check_offsets[:-1],
        )
        check_erasure_counts = np.add.reduceat(
            edge_erasure_mask,
            self.check_offsets[:-1],
            dtype=np.int16,
        )

        check_syndromes = check_nonzero_products.copy()
        check_syndromes[check_erasure_counts != 0] = 0

        return (
            edge_values,
            check_nonzero_products,
            check_erasure_counts,
            check_syndromes,
        )

    def erasure_recovery_sums(
        self,
        edge_values,
        check_nonzero_products,
        check_erasure_counts,
        recovery_mask,
    ):
        single_erasure_mask = (
            (edge_values == 0)
            & (check_erasure_counts[self.edge_cn] == 1)
            & recovery_mask[self.edge_vn]
        )

        return np.bincount(
            self.edge_vn[single_erasure_mask],
            weights=check_nonzero_products[
                self.edge_cn[single_erasure_mask]
            ],
            minlength=self.block_length,
        )

    def extrinsic_syndrome_sums(self, x):
        (
            edge_values,
            check_nonzero_products,
            check_erasure_counts,
            _,
        ) = self.check_state(x)

        edge_erasure_counts = check_erasure_counts[self.edge_cn]
        edge_nonzero_products = check_nonzero_products[self.edge_cn]

        edge_messages = np.zeros(self.edges_count, dtype=np.int8)

        no_erasure_mask = edge_erasure_counts == 0
        edge_messages[no_erasure_mask] = (
            edge_nonzero_products[no_erasure_mask]
            * edge_values[no_erasure_mask]
        )

        single_erasure_mask = (
            (edge_erasure_counts == 1)
            & (edge_values == 0)
        )
        edge_messages[single_erasure_mask] = edge_nonzero_products[
            single_erasure_mask
        ]

        return np.bincount(
            self.edge_vn,
            weights=edge_messages,
            minlength=self.block_length,
        )

    def correct_erasures_immediately(self, x, target_mask, original_values):

        if not np.any(target_mask):
            return x

        edge_values, check_nonzero_products, check_erasure_counts, _ = (
            self.check_state(x)
        )
        edge_erasure_mask = edge_values == 0

        degree1_edge_mask = (
            edge_erasure_mask
            & target_mask[self.edge_vn]
            & (self.check_degrees[self.edge_cn] == 1)
        )
        resolved_mask = np.zeros(self.block_length, dtype=bool)
        if np.any(degree1_edge_mask):
            vars1 = self.edge_vn[degree1_edge_mask]
            vals1 = check_nonzero_products[self.edge_cn[degree1_edge_mask]]
            x[vars1] = vals1
            resolved_mask[vars1] = True

        edge_values, check_nonzero_products, check_erasure_counts, _ = (
            self.check_state(x)
        )
        edge_erasure_mask = edge_values == 0
        remaining_mask = target_mask & ~resolved_mask & (x == 0)
        single_erasure_edge_mask = (
            edge_erasure_mask
            & remaining_mask[self.edge_vn]
            & (check_erasure_counts[self.edge_cn] == 1)
        )
        if np.any(single_erasure_edge_mask):
            vars2 = self.edge_vn[single_erasure_edge_mask]
            vals2 = check_nonzero_products[self.edge_cn[single_erasure_edge_mask]]

            vote_sum = np.bincount(
                vars2, weights=vals2, minlength=self.block_length
            )
            majority_val = np.sign(vote_sum)  

            apply_mask = remaining_mask & (majority_val != 0)
            x[apply_mask] = majority_val[apply_mask]

        still_unresolved_mask = target_mask & (x == 0)
        x[still_unresolved_mask] = original_values[still_unresolved_mask]

        return x

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        y = llr_in
        if self.zeros_in_init:
            x = np.zeros(self.block_length, dtype=np.int8)
            x[y >= 0.5] = 1
            x[y <= -0.5] = -1
        else:
            x = (2 * (y >= 0) - 1).astype(np.int8)  # sign, zero is positive

        l = np.full(
            self.block_length,
            self.L + 1,
            dtype=np.int16,
        )
        for iteration in range(self.n_iterations): # iteration loop
            original_values = x.copy()
            x_erasure_mask = x == 0
            has_erasures = np.any(x_erasure_mask)
            if has_erasures:
                x = self.correct_erasures_immediately(
                    x, x_erasure_mask, original_values
                )
                x_erasure_mask = x == 0
                has_erasures = np.any(x_erasure_mask)

            if has_erasures:
                (
                    edge_values,
                    check_nonzero_products,
                    check_erasure_counts,
                    check_syndromes,
                ) = self.check_state(x)
            else:
                check_syndromes = self.bpsk_syndrome(x)

            if np.all(check_syndromes == 1):
                unresolved = x == 0
                x[unresolved] = np.where(y[unresolved] >= 0, 1, -1)
                llr_out[:] = x
                return iteration # exit the iteration loop;

            incident_syndrome_sums = np.bincount(
                self.edge_vn,
                weights=check_syndromes[self.edge_cn],
                minlength=self.block_length,
            )
            np.minimum(l, self.L, out=l)
            l += 1
            E = self.alpha * x * y + incident_syndrome_sums + self.rho[l - 1] # local energy computation

            minimum_energy = np.min(E)
            E_th = minimum_energy + self.delta
            E_th_e = minimum_energy + self.delta_e
            selected_mask = rng.random(self.block_length) < self.p
            update_mask = (E <= E_th) & selected_mask
            bit_mask = update_mask & ~x_erasure_mask

            x[bit_mask] *= -1 # bit-flipping
            l[bit_mask] = 0

            new_erasure_mask = (
                (E > E_th)
                & (E <= E_th_e)
                & selected_mask
                & ~x_erasure_mask
            )
            if np.any(new_erasure_mask):
                x[new_erasure_mask] = 0
                l[new_erasure_mask] = 0

        unresolved = x == 0
        x[unresolved] = np.where(y[unresolved] >= 0, 1, -1)
        llr_out[:] = x
        return self.n_iterations