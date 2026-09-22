"""The only registration point for visualisable decoders.

Add a ``module:ClassName`` entry to SUPPORTED_DECODERS after implementing the
VisualizableDecoderBase interface in the production decoder itself.
"""

from importlib import import_module

from .base import VisualizableDecoderBase


SUPPORTED_DECODERS = (
    "ldpc_py.bin_ldpc_ftgdbf:BinLdpcFtgdbfDecoder",
    "ldpc_py.bin_ldpc_pmgdbf:BinLdpcPmgdbfDecoder",
    "ldpc_py.bin_ldpc_gdms:BinLdpcGdmsDecoder",
    "ldpc_py.bin_ldpc_tgdbf:BinLdpcTgdbfDecoder",
)


def decoder_catalog():
    result = {}
    for entry in SUPPORTED_DECODERS:
        module_name, class_name = entry.split(":", 1)
        decoder_class = getattr(import_module(module_name), class_name)
        if not issubclass(decoder_class, VisualizableDecoderBase):
            raise TypeError(f"{entry} is not a visualisable decoder")
        spec = decoder_class.describe()
        spec.validate()
        if spec.key in result:
            raise ValueError(f"Duplicate decoder key: {spec.key}")
        result[spec.key] = decoder_class
    return result


def decoder_class(key):
    try:
        return decoder_catalog()[key]
    except KeyError as exc:
        raise ValueError(f"Unknown visualisable decoder: {key}") from exc
