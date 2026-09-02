import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcEpmgdbfDecoder(BinLdpcDecoderBase):
    """Implementation of erasure probabilistic momentum gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.delta_e = kwargs["delta_e"]
        self.p = kwargs["p"]
        self.lambd = kwargs["lambd"]
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

    def extrinsic_syndrome_sums(self, x):
        edge_values = x[self.edge_vn]
        nonzero_edge_values = np.where(edge_values == 0, 1, edge_values)

        check_nonzero_products = np.multiply.reduceat(
            nonzero_edge_values,
            self.check_offsets[:-1],
        )
        check_erasure_counts = np.add.reduceat(
            (edge_values == 0).astype(np.int32),
            self.check_offsets[:-1],
        )

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


    
    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        y = llr_in.copy()
        x = np.zeros(self.block_length)
        x[y >= 0.5] = 1
        x[y <= -0.5] = -1

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
            E_th_e = np.min(E) + self.delta_e
            rand = rng.random(self.block_length)
            mask = (E <= E_th) & (rand < self.p)
            bit_mask = mask & (x != 0)
            erasure_mask = mask & (x == 0)

            has_erasure_updates = np.any(erasure_mask)
            if has_erasure_updates:
                extrinsic_syndrome_sums = self.extrinsic_syndrome_sums(x)

            x[bit_mask] *= -1 # bit-flipping
            if has_erasure_updates:
                x[erasure_mask] = np.sign(
                    extrinsic_syndrome_sums[erasure_mask]
                )
            l[mask] = 0

            rand = rng.random(self.block_length)
            mask = (E > E_th) & (E <= E_th_e) & (rand < self.p)
            bit_mask = mask & (x != 0)
            erasure_mask = mask & (x == 0)
            x[bit_mask] = 0

        llr_out[:] = x
        return self.n_iterations







