"""C++ implementation of E-GDBF V2."""

from .cpp_bin_ldpc_egdbf import CppBinLdpcEgdbfDecoder
from .bin_ldpc_egdbf_v2 import BitMeanEnergyThreshold


class CppBinLdpcEgdbfV2Decoder(BitMeanEnergyThreshold, CppBinLdpcEgdbfDecoder):
    """Use the bit-mean threshold in the shared C++ edge decoder."""

    _create_function = "cpp_egdbf_v2_create"
