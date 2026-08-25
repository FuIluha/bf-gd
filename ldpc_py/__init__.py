"""Public interface for LDPC decoders."""

from .decoder_factory import create_decoder

__all__ = [
    "create_decoder"
]