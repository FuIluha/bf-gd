"""ctypes wrapper for the shared C++ gradient flow (GF) decoder core."""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .bin_ldpc import BinLdpcDecoderBase


SOURCE_PATH = Path(__file__).with_suffix(".cpp")
SOURCE_HASH = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16]
LIBRARY_PATH = Path(tempfile.gettempdir()) / (
    f"bf_gd_cpp_gf_{SOURCE_HASH}.so"
)
INVALID_RESULT = np.iinfo(np.uint32).max


def lib_compile():
    """Compile the C++ GF shared library."""
    if LIBRARY_PATH.exists():
        return
    temporary_library = Path(f"{LIBRARY_PATH}.{os.getpid()}.tmp")
    try:
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-O3",
                "-fPIC",
                "-shared",
                str(SOURCE_PATH),
                "-o",
                str(temporary_library),
            ],
            check=True,
        )
        os.replace(temporary_library, LIBRARY_PATH)
    finally:
        temporary_library.unlink(missing_ok=True)


def load_library():
    """Load the C++ library and configure its ctypes interface."""
    lib_compile()
    library = ctypes.CDLL(str(LIBRARY_PATH))

    uint32_array = np.ctypeslib.ndpointer(
        dtype=np.uint32,
        ndim=1,
        flags="C_CONTIGUOUS",
    )
    float32_array = np.ctypeslib.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="C_CONTIGUOUS",
    )
    float64_array = np.ctypeslib.ndpointer(
        dtype=np.float64,
        ndim=1,
        flags="C_CONTIGUOUS",
    )

    library.cpp_gf_create.restype = ctypes.c_void_p
    library.cpp_gf_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,  # alpha
        ctypes.c_double,  # beta
        ctypes.c_double,  # gamma
        ctypes.c_double,  # eta
        uint32_array,
        uint32_array,
    ]
    library.cpp_gf_decode_float32.restype = ctypes.c_uint32
    library.cpp_gf_decode_float32.argtypes = [
        ctypes.c_void_p,
        float32_array,
        float32_array,
    ]
    library.cpp_gf_decode_float64.restype = ctypes.c_uint32
    library.cpp_gf_decode_float64.argtypes = [
        ctypes.c_void_p,
        float64_array,
        float64_array,
    ]
    library.cpp_gf_free.restype = None
    library.cpp_gf_free.argtypes = [ctypes.c_void_p]
    return library


class CppBinLdpcGfDecoder(BinLdpcDecoderBase):
    """C++ implementation of the classic gradient flow decoder."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.alpha = float(kwargs["alpha"])
        self.beta = float(kwargs["beta"])
        self.gamma = float(kwargs["gamma"])
        self.eta = float(kwargs["eta"])

        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.alpha < 0 or self.beta < 0 or self.gamma < 0:
            raise ValueError("alpha, beta and gamma must be non-negative")

        edge_cn, edge_vn = np.nonzero(self.pcm)
        self.edge_vn = np.ascontiguousarray(edge_vn, dtype=np.uint32)
        check_degrees = np.bincount(edge_cn, minlength=self.n_checks)
        self.check_offsets = np.ascontiguousarray(
            np.concatenate((np.array([0]), np.cumsum(check_degrees))),
            dtype=np.uint32,
        )

        self._library = load_library()
        self._decoder = self._library.cpp_gf_create(
            self.block_length,
            self.n_checks,
            self.n_iterations,
            self.alpha,
            self.beta,
            self.gamma,
            self.eta,
            self.edge_vn,
            self.check_offsets,
        )
        if not self._decoder:
            raise RuntimeError("Failed to create C++ GF decoder")

    def decode(self, llr_in, llr_out, rng=None):
        if llr_in.dtype != llr_out.dtype:
            raise TypeError("llr_in and llr_out must have the same dtype")

        if llr_in.dtype == np.float32:
            result = self._library.cpp_gf_decode_float32(
                self._decoder,
                llr_in,
                llr_out,
            )
        elif llr_in.dtype == np.float64:
            result = self._library.cpp_gf_decode_float64(
                self._decoder,
                llr_in,
                llr_out,
            )
        else:
            raise TypeError("C++ GF supports only float32 and float64 LLRs")

        if result == INVALID_RESULT:
            raise FloatingPointError(
                "GF state exceeded the output floating-point range"
            )
        return result

    def __del__(self):
        decoder = getattr(self, "_decoder", None)
        library = getattr(self, "_library", None)
        if decoder and library:
            library.cpp_gf_free(decoder)
            self._decoder = None
