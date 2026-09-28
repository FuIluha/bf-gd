"""C++ implementation of E-GDBF V4 (sign-only GDMS)."""

from .cpp_bin_ldpc_gdms import CppBinLdpcGdmsDecoder


class CppBinLdpcEgdbfV4Decoder(CppBinLdpcGdmsDecoder):
    """Use the sign-only check update in the shared C++ GDMS decoder."""

    _create_function = "cpp_egdbf_v4_create"
