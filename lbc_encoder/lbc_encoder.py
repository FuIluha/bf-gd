"""
Linear block code encoder
Generator matrix is stored in the form of compressed bitfield,
resulting in faster generator matrix multiplication
"""
import os
import ctypes
import numpy as np

SRC_FILE = 'lbc_encoder_impl'
LIB_PATH = SRC_FILE + '.so'


def lib_compile():
    """
    Compilation routines
    """
    wdir = os.path.dirname(__file__)
    src_file = os.path.join(wdir, SRC_FILE)
    os.system(f'g++ -O3 -fPIC -c -o {src_file}.o {src_file}.cpp')
    os.system('g++ -shared -o ' + os.path.join(wdir, LIB_PATH) + f' {src_file}.o')


class LBCEncoder:
    """
    Linear block code encoder implementation
    Wrapper class for C++ implementation
    """
    def __init__(self, generator_filename):
        """
        :param generator_filename: path to the generator matrix (read by np.loadtxt)
        The size of matrix should be k (information bit count) rows and n (block length) columns
        """
        self.generator_path = generator_filename
        self.lib = LBCEncoder.__load_lib()

        gen_mtx = np.loadtxt(generator_filename, dtype=np.uint8)
        self.inf_bits, self.cwd_length = gen_mtx.shape

        # Calculate the size of compressed generator matrix
        c_size = int(np.ceil(self.inf_bits / 32)) * self.cwd_length
        self.generator = np.zeros((c_size,), dtype=np.uint32)  # Compressed generator matrix
        self.lib.compress_generator(
            gen_mtx.T.reshape(-1),  # Properly unrolled matrix
            self.generator,
            self.inf_bits,
            self.cwd_length
        )

    @staticmethod
    def __load_lib():
        """
        Load shared object
        """
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), LIB_PATH))

        lib.generator_multiply.restype = None
        lib.generator_multiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint32),  # Generator matrix
            np.ctypeslib.ndpointer(dtype=np.uint8),  # Information word
            np.ctypeslib.ndpointer(dtype=np.uint8),  # Codeword
            ctypes.c_uint,  # information word length
            ctypes.c_uint,  # code word length
        ]

        lib.compress_generator.restype = None
        lib.compress_generator.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint8),  # Generator matrix
            np.ctypeslib.ndpointer(dtype=np.uint32),  # Compressed generator matrix
            ctypes.c_uint,  # information word length
            ctypes.c_uint,  # code word length
        ]
        return lib

    def encode(self, iwd):
        """

        :param iwd: information word (np array of type np.uint8)
        :return: codeword (np.array of type np.uint8)
        """
        cwd = np.zeros((self.cwd_length,), dtype=np.uint8)
        self.lib.generator_multiply(
            self.generator,
            iwd.astype(np.uint8),  # Information word, converted to np.uint8
            cwd,  # Buffer for a codeword
            self.inf_bits,
            self.cwd_length
        )
        return cwd

    def generate_iwd(self, rng):
        """
        Generate the information word
        :param rng: random number generator instance
        :return: information word (np.array of length k, type np.uint8)
        """
        return rng.integers(low=0, high=2, size=self.inf_bits, dtype=np.uint8)
