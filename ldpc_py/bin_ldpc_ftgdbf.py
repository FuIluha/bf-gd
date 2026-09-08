import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


class BinLdpcFtgdbfDecoder(BinLdpcDecoderBase):
    """Deterministic GDBF decoder with a fixed zero flipping threshold."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
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

    def local_energy(self, x, received, check_syndromes):
        incident_syndrome_sums = np.bincount(
            self.edge_vn,
            weights=check_syndromes[self.edge_cn],
            minlength=self.block_length,
        )
        return self.alpha * x * received + incident_syndrome_sums

    def decode(self, llr_in, llr_out, rng=None):
        received = llr_in.copy()
        x = np.where(received >= 0, 1, -1).astype(np.int8)

        for iteration in range(self.n_iterations):
            check_syndromes = self.bpsk_syndrome(x)
            if np.all(check_syndromes == 1):
                llr_out[:] = x
                return iteration

            energy = self.local_energy(x, received, check_syndromes)
            x[energy <= 0] *= -1

        llr_out[:] = x
        return self.n_iterations

    def trace(self, llr_in):
        """Return pre-flip states and energies for one decoding trajectory."""
        received = llr_in.copy()
        x = np.where(received >= 0, 1, -1).astype(np.int8)
        trajectory = []

        for _ in range(self.n_iterations):
            check_syndromes = self.bpsk_syndrome(x)
            if np.all(check_syndromes == 1):
                return True, trajectory

            energy = self.local_energy(x, received, check_syndromes)
            trajectory.append((x.copy(), energy.astype(np.float32)))
            x[energy <= 0] *= -1

        decoded = np.all(self.bpsk_syndrome(x) == 1)
        return decoded, trajectory
