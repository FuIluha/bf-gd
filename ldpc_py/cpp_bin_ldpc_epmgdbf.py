"""ctypes wrapper for the C++ EPMGDBF decoder."""

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
LIBRARY_PATH = (
    Path(tempfile.gettempdir()) / f"bf_gd_cpp_epmgdbf_{SOURCE_HASH}.so"
)


def lib_compile():
    """Compile the C++ EPMGDBF shared library."""
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

    library.cpp_epmgdbf_create.restype = ctypes.c_void_p
    library.cpp_epmgdbf_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        uint32_array,
        uint32_array,
    ]

    library.cpp_epmgdbf_decode_float32.restype = ctypes.c_uint32
    library.cpp_epmgdbf_decode_float32.argtypes = [
        ctypes.c_void_p,
        float32_array,
        float32_array,
        ctypes.c_uint64,
    ]
    library.cpp_epmgdbf_decode_float64.restype = ctypes.c_uint32
    library.cpp_epmgdbf_decode_float64.argtypes = [
        ctypes.c_void_p,
        float64_array,
        float64_array,
        ctypes.c_uint64,
    ]
    library.cpp_epmgdbf_free.restype = None
    library.cpp_epmgdbf_free.argtypes = [ctypes.c_void_p]
    return library


class CppBinLdpcEpmgdbfDecoder(BinLdpcDecoderBase):
    """C++ implementation of the EPMGDBF decoder."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self.delta = kwargs["delta"]
        self.delta_e = kwargs["delta_e"]
        self.alpha = kwargs["alpha"]
        self.p = kwargs["p"]
        self.zeros_in_init = kwargs["zeros_in_init"]
        self.L = kwargs["L"]
        self.rho = np.asarray(kwargs["rho"], dtype=np.float32)

        if len(self.rho) != self.L:
            raise ValueError("Momentum length must be equal to L")

        edge_cn, edge_vn = np.nonzero(self.pcm)
        edge_cn = edge_cn.astype(np.uint32)
        self.edge_vn = np.ascontiguousarray(edge_vn, dtype=np.uint32)
        check_degrees = np.bincount(edge_cn, minlength=self.n_checks)
        self.check_offsets = np.ascontiguousarray(
            np.concatenate((np.array([0]), np.cumsum(check_degrees))),
            dtype=np.uint32,
        )

        self._library = load_library()
        self._decoder = self._library.cpp_epmgdbf_create(
            self.block_length,
            self.n_checks,
            self.n_iterations,
            self.delta,
            self.delta_e,
            self.alpha,
            self.p,
            self.zeros_in_init,
            self.rho.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.L,
            self.edge_vn,
            self.check_offsets,
        )
        if not self._decoder:
            raise RuntimeError("Failed to create C++ EPMGDBF decoder")

    def decode(self, llr_in, llr_out, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if llr_in.dtype != llr_out.dtype:
            raise TypeError("llr_in and llr_out must have the same dtype")

        seed = int(rng.bit_generator.random_raw())
        if llr_in.dtype == np.float32:
            return self._library.cpp_epmgdbf_decode_float32(
                self._decoder,
                llr_in,
                llr_out,
                seed,
            )
        if llr_in.dtype == np.float64:
            return self._library.cpp_epmgdbf_decode_float64(
                self._decoder,
                llr_in,
                llr_out,
                seed,
            )
        raise TypeError("C++ EPMGDBF supports only float32 and float64 LLRs")

    def __del__(self):
        decoder = getattr(self, "_decoder", None)
        library = getattr(self, "_library", None)
        if decoder and library:
            library.cpp_epmgdbf_free(decoder)
            self._decoder = None
