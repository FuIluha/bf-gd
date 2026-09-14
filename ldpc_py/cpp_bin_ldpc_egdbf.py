"""ctypes wrapper for the C++ E-GDBF decoder."""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .bin_ldpc_egdbf import BinLdpcEgdbfDecoder


SOURCE_PATH = Path(__file__).with_suffix(".cpp")
SOURCE_HASH = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16]
LIBRARY_PATH = Path(tempfile.gettempdir()) / (
    f"bf_gd_cpp_egdbf_{SOURCE_HASH}.so"
)


def lib_compile():
    """Compile the C++ E-GDBF shared library."""
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
    float64_array = np.ctypeslib.ndpointer(
        dtype=np.float64,
        ndim=1,
        flags="C_CONTIGUOUS",
    )
    float32_array = np.ctypeslib.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="C_CONTIGUOUS",
    )

    library.cpp_egdbf_create.restype = ctypes.c_void_p
    library.cpp_egdbf_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        float64_array,
        ctypes.c_uint32,
        uint32_array,
        uint32_array,
    ]
    library.cpp_egdbf_decode_float32.restype = ctypes.c_uint32
    library.cpp_egdbf_decode_float32.argtypes = [
        ctypes.c_void_p,
        float32_array,
        float32_array,
        ctypes.c_uint64,
    ]
    library.cpp_egdbf_decode_float64.restype = ctypes.c_uint32
    library.cpp_egdbf_decode_float64.argtypes = [
        ctypes.c_void_p,
        float64_array,
        float64_array,
        ctypes.c_uint64,
    ]
    library.cpp_egdbf_free.restype = None
    library.cpp_egdbf_free.argtypes = [ctypes.c_void_p]
    return library


class CppBinLdpcEgdbfDecoder(BinLdpcEgdbfDecoder):
    """C++ implementation of the E-GDBF decoder."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.edge_vn = np.ascontiguousarray(self.edge_vn, dtype=np.uint32)
        self.check_offsets = np.ascontiguousarray(
            self.check_offsets,
            dtype=np.uint32,
        )
        self.rho_values = np.ascontiguousarray(
            self.rho[:self.L],
            dtype=np.float64,
        )

        self._library = load_library()
        self._decoder = self._library.cpp_egdbf_create(
            self.block_length,
            self.n_checks,
            self.n_iterations,
            self.delta,
            self.alpha,
            self.p,
            self.rho_values,
            self.L,
            self.edge_vn,
            self.check_offsets,
        )
        if not self._decoder:
            raise RuntimeError("Failed to create C++ E-GDBF decoder")

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if llr_in.dtype != llr_out.dtype:
            raise TypeError("llr_in and llr_out must have the same dtype")
        if not llr_in.flags.c_contiguous or not llr_out.flags.c_contiguous:
            raise TypeError("llr_in and llr_out must be C-contiguous")

        seed = int(rng.bit_generator.random_raw())
        if llr_in.dtype == np.float32:
            return self._library.cpp_egdbf_decode_float32(
                self._decoder,
                llr_in,
                llr_out,
                seed,
            )
        if llr_in.dtype == np.float64:
            return self._library.cpp_egdbf_decode_float64(
                self._decoder,
                llr_in,
                llr_out,
                seed,
            )
        raise TypeError("C++ E-GDBF supports only float32 and float64 LLRs")

    def __del__(self):
        decoder = getattr(self, "_decoder", None)
        library = getattr(self, "_library", None)
        if decoder and library:
            library.cpp_egdbf_free(decoder)
            self._decoder = None
