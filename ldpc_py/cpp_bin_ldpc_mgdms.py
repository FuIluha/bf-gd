"""Momentum GDMS using the shared C++ min-sum implementation."""

from .cpp_bin_ldpc_gdms import CppBinLdpcGdmsDecoder


class CppBinLdpcMgdmsDecoder(CppBinLdpcGdmsDecoder):
    """C++ momentum gradient-descent min-sum decoder."""

    MOMENTUM_ENABLED = True
