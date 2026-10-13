"""ctypes frontend for the MPGD C++ decoder; parameters share Python validation."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .bin_ldpc_mpgd import BinLdpcMpgdDecoder


SOURCE_PATH = Path(__file__).with_suffix(".cpp")
SOURCE_HASH = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()[:16]
LIBRARY_PATH = Path(tempfile.gettempdir()) / f"bf_gd_cpp_mpgd_{SOURCE_HASH}.so"
INVALID_RESULT = np.iinfo(np.uint32).max


def lib_compile():
    if LIBRARY_PATH.exists():
        return
    temporary = Path(f"{LIBRARY_PATH}.{os.getpid()}.tmp")
    try:
        subprocess.run([
            "g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O3",
            "-ffp-contract=off", "-fPIC", "-shared", str(SOURCE_PATH), "-o", str(temporary),
        ], check=True)
        os.replace(temporary, LIBRARY_PATH)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def load_library():
    lib_compile()
    lib = ctypes.CDLL(str(LIBRARY_PATH))
    indices = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1, flags="C_CONTIGUOUS")
    lib.cpp_mpgd_create.argtypes = [ctypes.c_uint32] * 3 + [ctypes.c_double] * 6 + [indices] * 2
    lib.cpp_mpgd_create.restype = ctypes.c_void_p
    for dtype, name in ((np.float32, "float32"), (np.float64, "float64")):
        array = np.ctypeslib.ndpointer(dtype=dtype, ndim=1, flags="C_CONTIGUOUS")
        function = getattr(lib, f"cpp_mpgd_decode_{name}")
        function.argtypes = [ctypes.c_void_p, array, array]
        function.restype = ctypes.c_uint32
    lib.cpp_mpgd_free.argtypes = [ctypes.c_void_p]
    lib.cpp_mpgd_free.restype = None
    return lib


class CppBinLdpcMpgdDecoder(BinLdpcMpgdDecoder):
    """Native numerical loop, with the same graph, metadata and parameters."""

    def __init__(self, alist_filename, **kwargs):
        super().__init__(alist_filename, **kwargs)
        if not self.n_iterations < INVALID_RESULT:
            raise ValueError("MPGD iteration limit must fit uint32")
        p = self.parameters
        self._library = load_library()
        self._decoder = self._library.cpp_mpgd_create(
            self.block_length, self.n_checks, self.n_iterations,
            p["alpha"], p["eta_r"], p["eta_v"], p["l2_r"], p["l2_v"], p["max_value"],
            np.ascontiguousarray(self.edge_vn, dtype=np.uint32),
            np.ascontiguousarray(self.check_offsets, dtype=np.uint32),
        )
        if not self._decoder:
            raise RuntimeError("Failed to create C++ MPGD decoder")

    def decode(self, llr_in, llr_out, rng=None):
        if llr_in.dtype != llr_out.dtype or llr_in.dtype not in (np.float32, np.float64):
            raise TypeError("C++ MPGD requires matching float32 or float64 input/output")
        if llr_in.shape != (self.block_length,) or llr_out.shape != llr_in.shape:
            raise ValueError("MPGD input/output must have block_length values")
        if not np.all(np.isfinite(llr_in)):
            raise ValueError("MPGD input must be finite")
        source = np.ascontiguousarray(llr_in)
        output = llr_out if llr_out.flags.c_contiguous else np.empty_like(source)
        function = getattr(self._library, f"cpp_mpgd_decode_{llr_in.dtype.name}")
        result = function(self._decoder, source, output)
        if result == INVALID_RESULT:
            raise FloatingPointError("Non-finite MPGD state")
        if output is not llr_out:
            llr_out[:] = output
        return int(result)

    def __del__(self):
        decoder = getattr(self, "_decoder", None)
        library = getattr(self, "_library", None)
        if decoder and library:
            library.cpp_mpgd_free(decoder)
            self._decoder = None
