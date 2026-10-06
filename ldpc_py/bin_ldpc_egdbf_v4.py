"""E-GDBF V4: GDMS whose checks send only the product of signs."""

import numpy as np

from .bin_ldpc_gdms import BinLdpcGdmsDecoder


class BinLdpcEgdbfV4Decoder(BinLdpcGdmsDecoder):
    """GDMS without the minimum magnitude in check-to-variable messages."""

    def check_to_variable_messages(self, edge_values):
        edge_signs = np.where(edge_values < 0, -1, 1)
        check_signs = np.multiply.reduceat(
            edge_signs,
            self.check_offsets[:-1],
        )
        return (check_signs[self.edge_cn] * edge_signs).astype(np.float64)
