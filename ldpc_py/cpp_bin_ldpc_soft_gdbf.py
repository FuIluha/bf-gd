"""ctypes wrapper for the C++ soft GDBF decoder."""

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
    f"bf_gd_cpp_soft_gdbf_{SOURCE_HASH}.so"
)
INVALID_RESULT = np.iinfo(np.uint32).max


def lib_compile():
    """Compile the C++ soft GDBF shared library."""
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

    library.cpp_soft_gdbf_create.restype = ctypes.c_void_p
    library.cpp_soft_gdbf_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        uint32_array,
        uint32_array,
    ]
    library.cpp_soft_gdbf_decode_float32.restype = ctypes.c_uint32
    library.cpp_soft_gdbf_decode_float32.argtypes = [
        ctypes.c_void_p,
        float32_array,
        float32_array,
    ]
    library.cpp_soft_gdbf_decode_float64.restype = ctypes.c_uint32
    library.cpp_soft_gdbf_decode_float64.argtypes = [
        ctypes.c_void_p,
        float64_array,
        float64_array,
    ]
    library.cpp_soft_gdbf_free.restype = None
    library.cpp_soft_gdbf_free.argtypes = [ctypes.c_void_p]
    return library


class CppBinLdpcSoftGdbfDecoder(BinLdpcDecoderBase):
    """C++ implementation of the soft GDBF decoder."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.learning_rate = float(kwargs["learning_rate"])
        self.learning_rate_decay = float(kwargs["learning_rate_decay"])
        self.momentum = float(kwargs["momentum"])
        self.regularization = float(kwargs["regularization"])
        self.alpha = float(kwargs["alpha"])

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate_decay < 0:
            raise ValueError("Learning rate decay must be non-negative")
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be in [0, 1)")
        if self.regularization < 0:
            raise ValueError("Regularization must be non-negative")

        edge_cn, edge_vn = np.nonzero(self.pcm)
        edge_cn = edge_cn.astype(np.uint32)
        self.edge_vn = np.ascontiguousarray(edge_vn, dtype=np.uint32)
        check_degrees = np.bincount(edge_cn, minlength=self.n_checks)
        self.check_offsets = np.ascontiguousarray(
            np.concatenate((np.array([0]), np.cumsum(check_degrees))),
            dtype=np.uint32,
        )

        self._library = load_library()
        self._decoder = self._library.cpp_soft_gdbf_create(
            self.block_length,
            self.n_checks,
            self.n_iterations,
            self.learning_rate,
            self.learning_rate_decay,
            self.momentum,
            self.regularization,
            self.alpha,
            self.edge_vn,
            self.check_offsets,
        )
        if not self._decoder:
            raise RuntimeError("Failed to create C++ soft GDBF decoder")

    def decode(self, llr_in, llr_out, rng=None):
        if llr_in.dtype != llr_out.dtype:
            raise TypeError("llr_in and llr_out must have the same dtype")

        if llr_in.dtype == np.float32:
            result = self._library.cpp_soft_gdbf_decode_float32(
                self._decoder,
                llr_in,
                llr_out,
            )
        elif llr_in.dtype == np.float64:
            result = self._library.cpp_soft_gdbf_decode_float64(
                self._decoder,
                llr_in,
                llr_out,
            )
        else:
            raise TypeError("C++ soft GDBF supports only float32 and float64 LLRs")

        if result == INVALID_RESULT:
            raise FloatingPointError(
                "soft GDBF state exceeded the output floating-point range"
            )
        return result

    def __del__(self):
        decoder = getattr(self, "_decoder", None)
        library = getattr(self, "_library", None)
        if decoder and library:
            library.cpp_soft_gdbf_free(decoder)
            self._decoder = None
