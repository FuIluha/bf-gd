from .bin_ldpc_bf import BinLdpcBfDecoder
from .bin_ldpc_ms import BinLdpcMsDecoder
from .bin_ldpc_mgdbf import BinLdpcMgdbfDecoder
from .bin_ldpc_sigmoid1 import BinLdpcSigmoid1Decoder
from .bin_ldpc_pmgdbf import BinLdpcPmgdbfDecoder
from .bin_ldpc_epmgdbf import BinLdpcEpmgdbfDecoder
from .bin_ldpc_eafpmgdbf import BinLdpcEafpmgdbfDecoder
from .bin_ldpc_egdbf import BinLdpcEgdbfDecoder
from .cpp_bin_ldpc_egdbf import CppBinLdpcEgdbfDecoder
from .bin_ldpc_gdms import BinLdpcGdmsDecoder
from .bin_ldpc_mgdms import BinLdpcMgdmsDecoder
from .cpp_bin_ldpc_gdms import CppBinLdpcGdmsDecoder
from .cpp_bin_ldpc_mgdms import CppBinLdpcMgdmsDecoder

_DECODER_TYPES = {
    "bit-flipping": BinLdpcBfDecoder,
    "min-sum": BinLdpcMsDecoder,
    "multi gradient descent bit-flipping": BinLdpcMgdbfDecoder,
    "sigmoid1 probabilistic momentum gradient descent bit-flipping": BinLdpcSigmoid1Decoder,
    "probabilistic momentum gradient descent bit-flipping": BinLdpcPmgdbfDecoder,
    "erasure probabilistic momentum gradient descent bit-flipping": BinLdpcEpmgdbfDecoder,
    "erasure add probabilistic momentum gradient descent bit-flipping": BinLdpcEafpmgdbfDecoder,
    "edge-wise gradient descent bit-flipping": BinLdpcEgdbfDecoder,
    "cpp edge-wise gradient descent bit-flipping": CppBinLdpcEgdbfDecoder,
    "gradient descent min-sum": BinLdpcGdmsDecoder,
    "momentum gradient descent min-sum": BinLdpcMgdmsDecoder,
    "cpp momentum gradient descent min-sum": CppBinLdpcMgdmsDecoder,
    "soft gradient descent bit-flipping": BinLdpcGdmsDecoder,
    "cpp gradient descent min-sum": CppBinLdpcGdmsDecoder,
    "cpp soft gradient descent bit-flipping": CppBinLdpcGdmsDecoder,
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
