"""
This module implements the soft decoder of binary LDPC codes.
C++ implementation is stored in impl directory.
cTypes are used to execute decoder from python
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/ldpc_soft_py/.
# Copyright (c) 2023 Kirill Andreev, Alexey Frolov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import ctypes
import os
from enum import IntEnum

import numpy as np

from ldpc_common.alist import Alist


# C++ implementation: compilation, linking, and execution routines
LIB_PATH = 'bin_ldpc_soft_decoder.so'


def lib_compile():
    """
    Compile the C++ SCL decoder implementation
    NOTE: In normalized offset min-sum, the offset option is implemented under OFFSETS_ENABLED ifdef
    NOTE: To enable a full-spport of offsets, add -DOFFSETS_ENABLED flag.
    NOTE: Mind performance issues if enabling offsets
    NOTE: To check inf/NaN values in the output, use -DOUTPUT_SANITY_CHECK flag.
    """
    wdir = os.path.dirname(__file__)
    # if os.path.isfile(LIB_PATH):
    #     return
    src = ['bin_alist_loader', 'bin_ldpc_soft_ctypes']
    src_abs = [os.path.join(wdir, s) for s in src]

    if not os.popen('which g++').read():
        raise RuntimeError('g++ not found.')

    for src_file in src_abs:
        os.system(f'g++ -Wall -Werror -O3 -fPIC -c -o {src_file}.o {src_file}.cpp')
    os.system(
        'g++ -shared -o ' +
        os.path.join(wdir, LIB_PATH) + ' ' +
        ''.join([s + '.o ' for s in src_abs])
    )
    obj_files = os.path.join(wdir, '*.o')
    os.system(f'rm {obj_files}')


def decoder_settings_fields(ctypes_data):
    """
    Decoder settings. Replicates the C++ strucutre
    See bin_ldpc_soft_settings.h for more detail.
    """
    return [
        # Total number of variable nodes, including punctured
        ("block_length", ctypes.c_uint32),
        # The number of parity checks
        ("n_checks", ctypes.c_uint32),
        # Whether a code is systematic. Required to evaluate FER
        # If systematic, the information positions are the first k = block_length - n_checks.
        ("is_systematic", ctypes.c_bool),
        # Syndrome-convergence termination
        ("early_termination", ctypes.c_bool),
        # For all-zeros codeword (AZCW), the early termination criterion simpler:
        # all output LLRs are strictly positive
        ("is_azcw", ctypes.c_bool),
        # Maximum number of decoding iterations
        ("n_iterations", ctypes.c_uint32),
        # LLR scales (per-column)
        ("scale_array", ctypes.POINTER(ctypes_data)),
        # LLR offsets (per-column)
        ("offset_array", ctypes.POINTER(ctypes_data))
    ]


class DecoderSettings32(ctypes.Structure):
    """
    float32
    """
    _fields_ = decoder_settings_fields(ctypes.c_float)


class DecoderSettings64(ctypes.Structure):
    """
    float64
    """
    _fields_ = decoder_settings_fields(ctypes.c_double)


def load_lib():
    """
    Load shared library and all supported functions
    """
    def __decode_args(dtype):
        """
        Return decoder function arguments depending on data type
        """
        return [
            ctypes.c_uint,                        # Command
            ctypes.c_void_p,                      # LDPC pointer
            np.ctypeslib.ndpointer(dtype=dtype),  # Channel LLR
            np.ctypeslib.ndpointer(dtype=dtype),  # Output LLRs
        ]

    def __init_args(stype):
        """
        Return decoder init arguments depending on settings type
        """
        return [
            ctypes.c_char_p,       # Alist path
            ctypes.POINTER(stype)  # LDPC decoder settings
        ]

    def __output_ber_args(stype, dtype):
        return [
            ctypes.POINTER(stype),                  # LDPC decoder settings
            np.ctypeslib.ndpointer(dtype=dtype),    # Channel LLR
            np.ctypeslib.ndpointer(dtype=np.uint8), # Transmitted bits
            ctypes.c_uint                           # Resulting iterations
        ]

    wdir = os.path.abspath(os.path.dirname(__file__))
    lib = ctypes.CDLL(os.path.join(wdir, LIB_PATH))

    # Tanner graph initializer
    lib.init_ldpc_float32.restype = ctypes.c_void_p
    lib.init_ldpc_float32.argtypes = __init_args(DecoderSettings32)

    lib.init_ldpc_float64.restype = ctypes.c_void_p
    lib.init_ldpc_float64.argtypes = __init_args(DecoderSettings64)

    # Decode functions
    lib.decode_siso_float32.restype = ctypes.c_uint
    lib.decode_siso_float32.argtypes = __decode_args(np.float32)

    lib.decode_siso_float64.restype = ctypes.c_uint
    lib.decode_siso_float64.argtypes = __decode_args(np.float64)

    # Output BER calculation functions:
    lib.output_ber_float64.restype = ctypes.c_double
    lib.output_ber_float64.argtypes = __output_ber_args(
        DecoderSettings64,
        np.float64
    )

    lib.output_ber_float32.restype = ctypes.c_double
    lib.output_ber_float32.argtypes = __output_ber_args(
        DecoderSettings32,
        np.float32
    )

    # Destroy LDPC object
    lib.free_ldpc_float32.restype = None
    lib.free_ldpc_float32.argtypes = [ctypes.c_void_p]

    lib.free_ldpc_float64.restype = None
    lib.free_ldpc_float64.argtypes = [ctypes.c_void_p]
    return lib

class DecoderType(IntEnum):
    """
    Decoding algorithm codes, see bin_ldpc_ctypes.cpp
    """
    SUM_PRODUCT = 0
    MIN_SUM = 2
    H_LAYERED_MIN_SUM = 3
    V_LAYERED_MIN_SUM = 4
    H_LAYERED_MIN_SUM_GLDPC = 5
    SUM_PRODUCT_GLDPC = 6

    @classmethod
    def get_decoder_methods(cls, decoder_class) -> dict[str, int]:
        """
        For each decoder type, get the list of supported decoding algorithms
        """
        if decoder_class.__name__ == "BinLdpcSoftDecoder":
            return {
                'sum_product': cls.SUM_PRODUCT,
                'min_sum': cls.MIN_SUM,
                'h_layered_min_sum': cls.H_LAYERED_MIN_SUM,
                'v_layered_min_sum': cls.V_LAYERED_MIN_SUM
            }
        if decoder_class.__name__ == "BinGldpcSoftDecoder":
            return {
                'h_layered_min_sum': cls.H_LAYERED_MIN_SUM_GLDPC,
                'sum_product': cls.SUM_PRODUCT_GLDPC
            }
        return {}


class BinLdpcSoftDecoderBase:
    """
    This class creates basic LDPC codec instance given alist file
    and provides all LDPC decoding routines
    NOTE: the class does not provide input parameters sanity checks
    """

    def __init__(self, alist_filename, **kwargs):
        self.ldpc_ptr = None
        self.lib = load_lib()

        # Data and strucutres types
        llr_type = kwargs['llr_type']

        # Decoding functions
        self.lib_decode_soft = self.lib_decode_soft_float32
        init_fcn = getattr(self.lib, 'init_ldpc_' + llr_type)
        self.free_fcn = getattr(self.lib, 'free_ldpc_' + llr_type)
        self.lib_decode_soft = getattr(self, 'lib_decode_soft_' + llr_type)
        self.output_ber = getattr(self, 'output_ber_' + llr_type)

        # Settings structure
        if llr_type == 'float32':
            SettingsType = DecoderSettings32
            ctypes_data=ctypes.c_float
        else:  # float64
            SettingsType = DecoderSettings64
            ctypes_data=ctypes.c_double

        np_data = getattr(np, llr_type)
        llr_scale = kwargs['llr_scale']
        block_length=kwargs['block_length']
        n_checks=kwargs['n_checks']
        if n_checks >= block_length:
            raise ValueError('The number of parity checks should not exceed block length')
        # Scale and offset array should be  kept to avoid garbage collection
        self.scale_array = (np.ones((block_length,)) * llr_scale).astype(np_data)
        self.offset_array = np.zeros((block_length,), dtype=np_data)
        self.settings= SettingsType(
            block_length=block_length,
            n_checks=n_checks,
            is_systematic=['is_systematic'],
            early_termination=True,  # By default, use early termination
            is_azcw=kwargs['is_azcw'],
            n_iterations=kwargs['n_iterations'],
            scale_array=self.scale_array.ctypes.data_as(ctypes.POINTER(ctypes_data)),
            offset_array=self.offset_array.ctypes.data_as(ctypes.POINTER(ctypes_data))
        )

        # Initialize LDPC decoder
        self.ldpc_ptr = init_fcn(alist_filename.encode(), self.settings)
        if not self.ldpc_ptr:
            raise ValueError('Failed to initialize decoder.')

    def lib_decode_soft_float64(self, llr_in, llr_out, decoder_type):
        """
        Float-64 decoder. All input and output arrays have float64 types
        """
        n_iter = self.lib.decode_siso_float64(
            decoder_type,
            self.ldpc_ptr,
            llr_in,
            llr_out
        )
        return n_iter

    def lib_decode_soft_float32(self, llr_in, llr_out, decoder_type):
        """
        Float-32 decoder.
        """
        n_iter = self.lib.decode_siso_float32(
            decoder_type,
            self.ldpc_ptr,
            llr_in,
            llr_out
        )
        return n_iter

    def output_ber_float32(self, llr_out, tx_bits, n_iter_got):
        """
        Output BER estimation
        """
        ber = self.lib.output_ber_float32(
            self.settings,
            llr_out,
            tx_bits,
            n_iter_got
        )
        return ber

    def output_ber_float64(self, llr_out, tx_bits, n_iter_got):
        """
        Output BER estimation
        """
        ber = self.lib.output_ber_float64(
            self.settings,
            llr_out,
            tx_bits,
            n_iter_got
        )
        return ber

    def __del__(self):
        if self.ldpc_ptr is not None:
            self.free_fcn(self.ldpc_ptr)

    def _decode_soft(self, llr_in, llr_out, decoder_type):
        """
        Run C++ implementation
        """
        n_iter = self.lib_decode_soft(llr_in, llr_out, decoder_type)
        return n_iter

    def _create_decoder_methods(self):
        """
        Dynamic method creation (for inherited classes)
        """
        decoder_methods = DecoderType.get_decoder_methods(self.__class__)

        for method_name, decoder_type in decoder_methods.items():
            def create_decode_method(dec_type):
                def decode_method(llr_in, llr_out):
                    return self._decode_soft(llr_in, llr_out, dec_type)
                return decode_method

            setattr(self, method_name, create_decode_method(decoder_type))


class BinLdpcSoftDecoder(BinLdpcSoftDecoderBase):
    """
    Binary soft LDPC decoder instance
    """
    def __init__(self, alist_filename: str, **kwargs):
        super().__init__(alist_filename, **kwargs)
        self._create_decoder_methods()


class BinGldpcSoftDecoder(BinLdpcSoftDecoderBase):
    """
    Binary soft Generalized LDPC decoder instance.
    Provide a parity check matrix shat should be extended by Cordaro-Wagner code.
    The actual number of parity checks in the equivalent binary PCM will be twice larger
    than in the provided PCM.
    """
    def __init__(self, alist_filename: str, **kwargs):
        super().__init__(alist_filename, **kwargs)
        if not self.settings.is_azcw:
            raise ValueError('GLDPC decoder does not support random codeword')
        self._create_decoder_methods()


if __name__ == '__main__':
    lib_compile()
