"""C++ implementation of E-GDBF V3."""

from .cpp_bin_ldpc_egdbf import CppBinLdpcEgdbfDecoder
from .bin_ldpc_egdbf_v3 import GradientStepRounding


class CppBinLdpcEgdbfV3Decoder(GradientStepRounding, CppBinLdpcEgdbfDecoder):
    """Use the rounded gradient step in the shared C++ edge decoder."""

    _create_function = "cpp_egdbf_v3_create"
