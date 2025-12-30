"""
LDPC codecs implementation
"""

import os
import sys
import numpy as np

from simulator_awgn_python.tools import load_json, read_array

from lbc_encoder.lbc_encoder import LBCEncoder
from lbc_encoder.lbc_encoder import lib_compile as lbc_compile

from ldpc_soft_py.bin_ldpc_soft import Alist
from ldpc_soft_py.bin_ldpc_soft import BinLdpcSoftDecoder, BinGldpcSoftDecoder
from ldpc_soft_py.bin_ldpc_soft import lib_compile as ldpc_compile


def get_array(config, field):
    """
    Try to read array from the configuration
    return None if field is missing
    """
    field_val = config.get(field, None)
    if field_val is None:
        return np.array([])
    return read_array(field_val)


class BinaryCodecBase:
    """
    Binary (G)LDPC codec base class. To specify the codec,
    the following parameters must be provided:
      - Source directory containing a general code data
      - Path to json file (relative to the source directory) that contains:
        * file specifying the parity check matrix (alist format), obligatory parameter
        * file specifying generator matrix (txt format, loaded by np.loadtxt(). If not specified,
          the simulation will be performed on all-zero codewords (AZCW)
        * List of punctured bits, described by a string (MATLAB-like syntax supported) or by a list
        * List of shortened bits
    Coding rate will be calculated in accordance with shortened and punctured structure.
    """
    def __init__(self, **kwargs):
        src_dir = kwargs.get('src_dir')
        code_filename = kwargs.get('code')
        if not (isinstance(src_dir, str) and isinstance(code_filename, str)):
            raise ValueError('Please specify \'src_dir\' and \'code\' parameters')
        config = load_json(os.path.join(src_dir, code_filename))
        # Load generator and parity check matrices

        # Get a unique name for a dedicated filename
        self.name = config.get('name', '')

        # Punctured and shortened positions
        self.punc_idx = get_array(config, 'punc_idx').astype(np.int32)

        # List of information bits indices. If empty, estimate BER given a whole codeword
        self.inf_bits = get_array(config, 'inf_bits')

        # Read the shape of the parity check matrix
        self.alist_path = os.path.join(src_dir, config.get('pcm'))
        self.pcm_shape = Alist.read_shape(self.alist_path)
        if self.inf_bits.size > 0 and len(self.inf_bits) != self.get_inf_bits_count():
            raise ValueError(
                f'Information bits mismatch: code dimension is {self.get_inf_bits_count()}' +
                f', length of information bits sequence is {len(self.inf_bits)}.'
            )

        # Read the generator matrix and initialize the encoder:
        generator_path = config.get('generator', None)
        if generator_path is not None:
            self.encoder = LBCEncoder(os.path.join(src_dir, generator_path))
            if self.pcm_shape[1] != self.encoder.cwd_length:
                raise ValueError('Generator / parity check matrices shape mismatch')

        self.decoder_impl = None  # To be instantiated by subclass

    def generate(self, rng):
        """
        Generate the information word and codeword
        return: codeword
        """
        if hasattr(self, 'encoder'):
            iwd = self.encoder.generate_iwd(rng)
            return self.encoder.encode(iwd)
        return np.zeros((self.decoder_impl.block_len,), dtype=np.uint8)

    def post_channel(self, llr_channel):
        """
        Apply shortening and puncturing
        """
        if self.punc_idx.size > 0:
            llr_channel[self.punc_idx] = 0
        return llr_channel

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Must be implemented by a subclass,
        information bits are calculated differently for LDPC and GLDPC
        """
        raise NotImplementedError('Must be implemented by subclass')

    def decode(self, llr_in):
        """
        Run decoder implementation
        :param llr_in input LLR vector with puncturing being applied (if needed)
        :return output LLRs and the number of decoding iterations
        """
        raise NotImplementedError('Must be implemented by subclass')

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        if self.name:
            template = self.name + '_'
        else:
            template = ''
        return template + f'k{self.get_inf_bits_count()}_n{self.get_block_length_str()[1]}_'

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return f'k = {self.get_inf_bits_count()}, n = {self.get_block_length_str()[0]}'

    def get_block_length_str(self):
        """
        Block length string and value with punctured/shortened bits taken into account
        """
        n_punctured = len(self.punc_idx)
        # Force conversion to int
        # https://github.com/numpy/numpy/issues/19294
        block_len_val = int(self.pcm_shape[1] - n_punctured)
        msg = f'{block_len_val}'
        if n_punctured:
            msg += f' ({n_punctured} punctured)'

        return msg, block_len_val

    def __str__(self):
        msg = 'Code parameters:\n'
        msg += f'  Alist path:               \'{self.alist_path}\'\n'
        msg += f'  Parity check matrix shape: {self.pcm_shape[0]} x {self.pcm_shape[1]}\n'
        msg += '  Generator matrix path:    '

        if hasattr(self, 'encoder'):
            generator_path = self.encoder.generator_path
            msg += f'\'{generator_path}\'\n'
            msg += '  Generator matrix shape:    '
            msg += f'{self.encoder.cwd_length} x {self.encoder.inf_bits}\n'
        else:
            msg += ' None. All-zero-codewords simulation\n'

        block_len_str, block_len_val = self.get_block_length_str()
        msg += f'  Block length:              {block_len_str} bits\n'
        msg += f'  Information bits count:    {self.get_inf_bits_count()}\n'
        msg += f'  Coding rate:               {self.get_inf_bits_count() / block_len_val:1.4f}\n'
        msg += '  Estimating FER by:         '
        if len(self.inf_bits):
            msg += 'information word (positions specified)'
        else:
            msg += 'a whole codeword'
        return msg


class BinarySoftCodecBase(BinaryCodecBase):
    """
    Binary (G)LDPC codec with soft decoding algorithm.
    The following additional parameters must be provided:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llr_scale = kwargs.get('llr_scale')
        self.llr_type = kwargs.get('llr_type')
        self.n_iterations = kwargs.get('n_iterations')
        self.algorithm = kwargs.get('algorithm')
        # Sanity checks
        if not isinstance(self.n_iterations, int) or self.n_iterations <= 0:
            raise ValueError('The number of decoding iterations must be positive')
        if not isinstance(self.llr_scale, float) or self.llr_scale < 0:
            raise ValueError('LLR scale must be non-negative')
        if self.llr_type not in ['float64', 'float32']:
            raise ValueError(f'LLR type {self.llr_type} is not supported')

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Must be implemented by a subclass,
        information bits are calculated differently for LDPC and GLDPC
        """
        raise NotImplementedError('Must be implemented by subclass')

    def decode(self, llr_in):
        """
        Run decoder implementation
        """
        raise NotImplementedError('Must be implemented by subclass')

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        template = super().get_filename_template()
        template += f'{self.algorithm}_iter_{self.n_iterations}_llr_{self.llr_type}'
        if self.algorithm != 'sum_product':
            template += f'_scale_{self.llr_scale:1.3f}'
        return template

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return super().get_title_template() + f', {self.algorithm}, {self.n_iterations} iterations'

    def get_decoder_instance(self, dec_type):
        """
        Instantiate particular decoder implementation
        """
        dec_instance = dec_type(
            alist_filename=self.alist_path,
            llr_type_str=self.llr_type,
            n_iterations=self.n_iterations,
            llr_scale=self.llr_scale
        )
        try:
            dec_fcn = getattr(dec_instance, self.algorithm)
        except AttributeError as exc:
            raise ValueError(
                f'Decoding algorithm {self.algorithm} ' +
                f'is not supported by {self.__class__.__name__}'
            ) from exc
        return dec_instance, dec_fcn

    def __str__(self):
        msg = super().__str__() + '\n'
        msg += 'Decoder-specific parameters:\n'
        msg += f'  Decoding algorithm:       \'{self.algorithm}\'\n'
        msg += f'  The number of iterations:  {self.n_iterations}\n'
        msg += f'  LLR type:                 \'{self.llr_type}\'\n'
        if self.algorithm != 'sum_product':
            msg += f'  LLR scale:                 {self.llr_scale}'
        return msg


class BinaryGldpcSoftCodec(BinarySoftCodecBase):
    """
    Binary GLDPC codec with soft decoding algorithm
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_impl, self.dec_fcn = self.get_decoder_instance(BinGldpcSoftDecoder)
        if hasattr(self, 'encoder'):
            raise ValueError(f'Encoding function is not supported by {self.__class__.__name__}')

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation.
        Warning: parity check matrix is assumed to be of full rank
        """
        return self.pcm_shape[1] - 2 * self.pcm_shape[0]

    def decode(self, llr_in):
        """
        Decoding function implementation
        """
        return self.dec_fcn(llr_in, True)

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        return 'bin_gldpc_' + super().get_filename_template()

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return 'Binary GLDPC, ' + super().get_title_template()

    def __str__(self):
        msg = 'Binary GLDPC (Cordaro-Wagner ext.) code with soft decoding algorithm\n'
        return msg + super().__str__()


class BinaryLdpcSoftCodec(BinarySoftCodecBase):
    """
    Binary LDPC codec with soft decoding algorithm
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_impl, self.dec_fcn = self.get_decoder_instance(BinLdpcSoftDecoder)

    def get_inf_bits_count(self):
        """
        Get the information bit count for correct coding rate evaluation
        Warning: parity check matrix is assumed to be of full rank
        """
        return self.pcm_shape[1] - self.pcm_shape[0]

    def decode(self, llr_in):
        """
        Decoding function implementation
        """
        is_azcw = not hasattr(self, 'encoder')
        return self.dec_fcn(llr_in, is_azcw)

    def get_filename_template(self):
        """
        A unique filename template that is generated from code parameters
        """
        # Encode information bit count and block length into the filename template
        return 'bin_ldpc_' + super().get_filename_template()

    def get_title_template(self):
        """
        Get title string template for live-plot
        """
        return 'Binary LDPC, ' + super().get_title_template()

    def __str__(self):
        msg = 'Binary LDPC code with soft decoding algorithm\n'
        return msg + super().__str__()


def instantiate_codec(**kwargs):
    """
    Generate codec instance from configuration provided by experiment file
    """
    codec_type = kwargs.pop('type')
    class_type = getattr(sys.modules[__name__], codec_type)
    return class_type(**kwargs)
