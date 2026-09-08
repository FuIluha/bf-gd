from .bin_ldpc_bf import BinLdpcBfDecoder
from .bin_ldpc_ms import BinLdpcMsDecoder
from .bin_ldpc_mgdbf import BinLdpcMgdbfDecoder
from .bin_ldpc_pmgdbf import BinLdpcPmgdbfDecoder
from .bin_ldpc_epmgdbf import BinLdpcEpmgdbfDecoder
from .bin_ldpc_soft_gdbf import BinLdpcSoftGdbfDecoder
from .bin_ldpc_gd import BinLdpcGdDecoder
from .bin_ldpc_pgd import BinLdpcPgdDecoder
from .bin_ldpc_ftgdbf import BinLdpcFtgdbfDecoder
from .cpp_bin_ldpc_gd import CppBinLdpcGdDecoder
from .cpp_bin_ldpc_pgd import CppBinLdpcPgdDecoder
from .cpp_bin_ldpc_soft_gdbf import CppBinLdpcSoftGdbfDecoder
from .cpp_bin_ldpc_sp_gdbf import CppBinLdpcSpGdbfDecoder

_DECODER_TYPES = {
    "bit-flipping": BinLdpcBfDecoder,
    "min-sum": BinLdpcMsDecoder,
    "multi gradient descent bit-flipping": BinLdpcMgdbfDecoder,
    "probabilistic momentum gradient descent bit-flipping": BinLdpcPmgdbfDecoder,
    "erasure probabilistic momentum gradient descent bit-flipping": BinLdpcEpmgdbfDecoder,
    "soft gradient descent bit-flipping": BinLdpcSoftGdbfDecoder,
    "gradient descent decoder": BinLdpcGdDecoder,
    "probabilistic gradient descent decoder": BinLdpcPgdDecoder,
    "fixed threshold gradient descent bit-flipping": BinLdpcFtgdbfDecoder,
    "cpp gradient descent decoder": CppBinLdpcGdDecoder,
    "cpp probabilistic gradient descent decoder": CppBinLdpcPgdDecoder,
    "cpp soft gradient descent bit-flipping": CppBinLdpcSoftGdbfDecoder,
    "cpp sum-product gradient descent bit-flipping": CppBinLdpcSpGdbfDecoder,
}

def create_decoder(algorithm, alist_filename, **kwargs):
    try:
        decoder_type = _DECODER_TYPES[algorithm]
    except KeyError as exc:
        supported = ", ".join(_DECODER_TYPES.keys())
        raise ValueError(
            f"Unknown algorithm {algorithm!r}. "
            f"Supported algorithms: {supported}"
        ) from exc

    return decoder_type(alist_filename, **kwargs)
