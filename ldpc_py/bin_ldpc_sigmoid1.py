import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcSigmoid1Decoder(BinLdpcDecoderBase):
    """Implementation of sigmoid1 probabilistic momentum gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.alpha = kwargs["alpha"]
        self.p = kwargs["p"]
        self.beta = kwargs["beta"]
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

    def sigmoid_syndrome(self, z):
        s = 2.0 / (1.0 + np.exp(-self.beta * z)) - 1.0
        s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)

        s_edges = s[self.edge_vn]
        signs = np.sign(s_edges)
        log_abs = np.log(np.abs(s_edges))

        prod_signs = np.multiply.reduceat(signs, self.check_offsets[:-1])
        sum_log_abs = np.add.reduceat(log_abs, self.check_offsets[:-1])

        return prod_signs * np.exp(sum_log_abs)

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        y = llr_in.copy()
        x = (2 * (y >= 0) - 1).astype(np.int8)  # sign, zero is positive
        z = x.astype(np.float32) * np.abs(y).astype(np.float32)  # soft bipolar state
        l = np.repeat(self.L + 1, self.block_length)
        for iteration in range(self.n_iterations): # iteration loop
            check_syndromes = self.bpsk_syndrome(x) # syndrome

            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration # exit the iteration loop;

            sigmoid_check_syndromes = self.sigmoid_syndrome(z)

            incident_syndrome_sums = np.bincount(
                self.edge_vn,
                weights=sigmoid_check_syndromes[self.edge_cn],
                minlength=self.block_length,
            )
            l = np.minimum(l, self.L) + 1
            E = self.alpha * x * y + incident_syndrome_sums + self.rho[l - 1] # local energy computation

            E_th = np.min(E) + self.delta
            rand = rng.random(self.block_length)
            mask = (E <= E_th) & (rand < self.p)
            x[mask] *= -1 # bit-flipping
            z[mask] *= -1 # keep soft state in sync with hard decision
            l[mask] = 0

        llr_out[:] = x
        return self.n_iterations