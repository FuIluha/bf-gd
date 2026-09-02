from .bin_ldpc_bf import BinLdpcBfDecoder
from .bin_ldpc_ms import BinLdpcMsDecoder
from .bin_ldpc_mgdbf import BinLdpcMgdbfDecoder
from .bin_ldpc_pmgdbf import BinLdpcPmgdbfDecoder
from .bin_ldpc_epmgdbf import BinLdpcEpmgdbfDecoder
from .cpp_bin_ldpc_epmgdbf import CppBinLdpcEpmgdbfDecoder

_DECODER_TYPES = {
    "bit-flipping": BinLdpcBfDecoder,
    "min-sum": BinLdpcMsDecoder,
    "multi gradient descent bit-flipping": BinLdpcMgdbfDecoder,
    "probabilistic momentum gradient descent bit-flipping": BinLdpcPmgdbfDecoder,
    "erasure probabilistic momentum gradient descent bit-flipping": BinLdpcEpmgdbfDecoder,
    "cpp erasure probabilistic momentum gradient descent bit-flipping": CppBinLdpcEpmgdbfDecoder,
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
