import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcMgdbfDecoder(BinLdpcDecoderBase):
    """Implementation of multi gradient descent bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.theta = kwargs["theta"]
        self.mu = kwargs["mu"]
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

    def decode(self, llr_in, llr_out):
        mu = self.mu
        y = llr_in.copy()

        # step 1
        x = (2 * (y >= 0) - 1).astype(np.int8)  # sign, zero is positive
        for iteration in range(self.n_iterations):
            check_syndromes = self.bpsk_syndrome(x)

            # step 2
            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration

            # step 3
            f1 = (np.dot(x, y) + np.sum(check_syndromes))
            incident_syndrome_sums = np.bincount(
                self.edge_vn,
                weights=check_syndromes[self.edge_cn],
                minlength=self.block_length,
            )
            delta = x * y + incident_syndrome_sums

            if mu == 0:
                # step 3.1 (multi-bit mode)
                x[delta < self.theta] *= -1
                updated_check_syndromes = self.bpsk_syndrome(x)
                f2 = (np.dot(x, y) + np.sum(updated_check_syndromes))
                if f1 > f2:
                    mu = 1
            else:
                # step 3.2 (single-bit mode)
                x[np.argmin(delta)] *= -1

        # step 4
        llr_out[:] = x
        return self.n_iterations
        