import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcEafmgdbfDecoder(BinLdpcDecoderBase):
    """Implementation of erasura add  probabilistic momentum gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.alpha = kwargs["alpha"]
        self.p = kwargs["p"]
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

        self.check_offsets = np.concatenate((
            np.array([0]),
            np.cumsum(check_degrees),
        ))

    def bpsk_syndrome(self, x):
        return np.multiply.reduceat(
            x[self.edge_vn],
            self.check_offsets[:-1],
        )

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        y = llr_in.copy()
        x = (2 * (y >= 0) - 1).astype(np.int8)  # sign, zero is positive
        l = np.repeat(self.L + 1, self.block_length)

        for iteration in range(self.n_iterations): # iteration loop
            check_syndromes = self.bpsk_syndrome(x) # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration # exit the iteration loop;

            incident_syndrome_sums = np.bincount(
                self.edge_vn,
                weights=check_syndromes[self.edge_cn],
                minlength=self.block_length,
            )
            l = np.minimum(l, self.L) + 1
            E = self.alpha * x * y + incident_syndrome_sums + self.rho[l - 1] # local energy computation

            E_th = np.min(E) + self.delta
            rand = rng.random(self.block_length)
            mask = (E <= E_th) & (rand < self.p)
            x_copy = x.copy()
            x[mask] = 0
            l[mask] = 0

            erasured = (x == 0)
            if np.any(erasured):
                erasures_per_check = np.bincount(
                    self.edge_cn,
                    weights=erasured[self.edge_vn],
                    minlength=self.n_checks,
                )
                x_wo_erased = np.where(erasured, 1, x).astype(np.int8)
                vote_value = self.bpsk_syndrome(x_wo_erased)

                useful_edge = erasured[self.edge_vn] & (erasures_per_check[self.edge_cn] == 1)

                votes_sum = np.bincount(
                    self.edge_vn,
                    weights=np.where(useful_edge, vote_value[self.edge_cn], 0),
                    minlength=self.block_length,
                )
                votes_count = np.bincount(
                    self.edge_vn,
                    weights=useful_edge,
                    minlength=self.block_length,
                )

                majority = np.sign(votes_sum).astype(np.int8)

                fixed = erasured & (votes_count > 0) & (majority != 0)
                unresolved = ~fixed & erasured

                x[fixed] = majority[fixed]
                x[unresolved] = x_copy[unresolved]


        llr_out[:] = x
        return self.n_iterations
