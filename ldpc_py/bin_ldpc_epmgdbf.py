import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcEpmgdbfDecoder(BinLdpcDecoderBase):
    """Implementation of erasure probabilistic momentum gradient descent bit-flipping decoder"""
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
            bit_mask = mask & (x != 0)
            erasure_mask = mask & (x == 0)

            x[bit_mask] = 0
            x[erasure_mask] = np.sign(incident_syndrome_sums[erasure_mask])

            l[mask] = 0

        llr_out[:] = x
        return self.n_iterations









