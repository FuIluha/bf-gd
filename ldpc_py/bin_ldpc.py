import numpy as np

from ldpc_common.alist import Alist


class BinLdpcDecoderBase:
    """Common interface and utilities for LDPC decoders."""

    def __init__(self, alist_filename, **kwargs):
        self.pcm = Alist.read(alist_filename).astype(np.uint8)

        self.block_length = kwargs["block_length"]
        self.n_checks = kwargs["n_checks"]
        self.n_iterations = kwargs["n_iterations"]
        self.is_systematic = kwargs["is_systematic"]

        if self.pcm.shape != (self.n_checks, self.block_length):
            raise ValueError(
                "Parity-check matrix shape does not match decoder settings"
            )

        if self.n_iterations <= 0:
            raise ValueError(
                "n_iterations must be positive"
            )

    def decode(self, llr_in, llr_out, rng=None):
        """Decode llr_in into llr_out and return iteration count"""
        raise NotImplementedError

    def syndrome(self, bits):
        """Return syndrome"""
        return np.asarray(bits @ self.pcm.T % 2, dtype=np.uint8)

    def output_ber(self, llr_out, tx_bits, _n_iter):
        """Calculate ber"""
        if self.is_systematic:
            k = self.block_length - self.n_checks
            ber = np.mean((llr_out[:k] < 0) != tx_bits[:k])
            return ber
        return np.mean((llr_out < 0) != tx_bits)
