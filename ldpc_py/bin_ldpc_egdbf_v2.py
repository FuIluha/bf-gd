"""E-GDBF V2: use the minimum mean edge energy of a bit as threshold."""

import numpy as np

from .bin_ldpc_egdbf import BinLdpcEgdbfDecoder


class BitMeanEnergyThreshold:
    """Threshold policy shared by the Python and C++ V2 wrappers."""

    def __init__(self, alist_filename, **kwargs):
        probability = float(kwargs.get("p", 1.0))
        if probability != 1.0:
            raise ValueError("E-GDBF V2 supports only deterministic flips (p=1)")
        super().__init__(alist_filename, **kwargs)

    def energy_threshold(self, energies):
        bit_means = np.bincount(
            self.edge_vn,
            weights=energies,
            minlength=self.block_length,
        ) / self.variable_degrees
        return np.min(bit_means) + self.delta


class BinLdpcEgdbfV2Decoder(BitMeanEnergyThreshold, BinLdpcEgdbfDecoder):
    """Keep V1 edge updates but anchor the threshold to bit mean energies."""
