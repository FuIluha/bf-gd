import numpy as np
from .bin_ldpc import BinLdpcDecoderBase

class BinLdpcBfDecoder(BinLdpcDecoderBase):
    """Implementation of bit-flipping decoder"""
    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.variable_node_degrees = np.sum(self.pcm, axis=0)

    def decode(self, llr_in, llr_out):
        decoded_bits = (llr_in < 0).astype(np.uint8)
        
        for iteration in range(self.n_iterations):
            syndrome = self.syndrome(decoded_bits)
            if not np.any(syndrome):
                llr_out[:] = 1.0 - 2.0 * decoded_bits
                return iteration

            unsatisfied_check_counts = syndrome @ self.pcm
            unsatisfied_check_ratios = (
                unsatisfied_check_counts / self.variable_node_degrees
            )
            flip_mask = unsatisfied_check_ratios > 0.5

            decoded_bits = decoded_bits ^ flip_mask

        llr_out[:] = 1.0 - 2.0 * decoded_bits
        return self.n_iterations
