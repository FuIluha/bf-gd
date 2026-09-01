"""
LDPC simulation and postprocessing workflow
"""

import os
import dataclasses
import numpy as np

from codec_impl import instantiate_codec


from simulator_awgn_python.data_storage import DataEntry
from simulator_awgn_python.channel import AwgnQAMChannel
from simulator_awgn_python.tools import dir_exists
from simulator_awgn_python.settings import HLINE_STR


@dataclasses.dataclass
class LdpcDataEntry(DataEntry):
    """
    Data entry extension to keep track of iteration count distribution
    """
    n_iter: int  # actual number of decoding iterations
    iter_pdf: np.array

    def __str__(self):
        msg = super().__str__()
        avg_iter = self.n_iter / self.tests if self.tests else 0
        msg += f', {avg_iter:1.3f} avg. iterations'
        return msg


@dataclasses.dataclass
class LdpcExperimentSettings:
    """
    LDPC experiment parameters
    """
    # Channel settings
    #  - Modulation (a required parameter for simulations)
    modulation: str
    #  - Channel output ('soft' or 'hard')
    channel_output: str
    # Codec settings (to be parsed further)
    codec: dict  # Full settings (to be passed to a specific decoder)

    # Parameters to be filled at post_init
    data_dir: str = 'data'  # Directory where the simulation results will be saved
    filename: str = ''  # To save the simulated data
    title: str = ''  # To generate a live-plot title
    inf_bits_count: int = 0  # To estimate the simulation bit-rate
    codec_info: str = ''  # Human-readable codec information

    def __post_init__(self):
        # Check that data directory is correct
        dir_exists(self.data_dir)
        # Check channel parameters
        if self.channel_output not in ['soft', 'hard', 'received']:
            raise ValueError(f'Channel output {self.channel_output} is not supported')
        codec_instance = instantiate_codec(**self.codec)

        self.filename = os.path.join(
            self.data_dir,
            codec_instance.get_filename_template() + '_' +
            self.modulation + '_' + self.channel_output + '.pickle'
        )
        self.title = (
            codec_instance.get_title_template() +
            ', modulation ' + self.modulation + '-' + self.channel_output
        )
        self.codec_info = codec_instance.__str__()
        self.inf_bits_count = codec_instance.get_inf_bits_count()

    def __str__(self):
        msg = HLINE_STR + '\n'
        msg += 'Channel parameters:\n'
        msg += f'  Modulation:                        {self.modulation.upper()}\n'
        msg += f'  Channel output:                    {self.channel_output}\n'
        msg += HLINE_STR + '\n'
        msg += self.codec_info + '\n'
        msg += HLINE_STR + '\n'
        msg += f'Output filename: {self.filename}'
        return msg


class LdpcExperimentInstance:
    """
    Run LDPC decoder with AWGN channel.
    Has a lot of bulky data, is not an Experiment instance required by the simulator.
    """
    def __init__(self, settings):
        self.settings = settings
        # Instantiate codec
        self.codec = instantiate_codec(**settings.codec)
        # Initialize channel
        placeholders_shape = (self.codec.block_length(),)
        dtype = getattr(np, self.codec.llr_type)
        self.tx_bits = np.zeros(placeholders_shape, dtype=np.uint8)
        if not self.codec.is_azcw():
            self.iwd = np.zeros((self.codec.get_inf_bits_count(),), dtype=np.uint8)
        self.llr_in = np.zeros(placeholders_shape, dtype=dtype)
        self.llr_out = np.zeros(placeholders_shape, dtype=dtype)
        self.channel = AwgnQAMChannel(
            self.settings.modulation, # Modulation
            self.tx_bits[self.codec.punctured:], # transmitted bits placeholder
            self.llr_in[self.codec.punctured:], # LLR placeholder
            self.codec.is_azcw()  # Check whether an adapter required
            )
        self.is_channel_hard = self.settings.channel_output == 'hard'
        # Initialize encoder instance

    def run_channel(self, snr_db, rng):
        """
        Run AWGN channel
        """
        in_ber, in_ser = self.channel.run(snr_db, rng)
        if self.settings.channel_output == 'hard':
            self.llr_in[:] = np.sign(self.llr_in)

        elif self.settings.channel_output == 'received':
            sigma_noise = self.channel.modulation.sigma_noise(snr_db)
            self.llr_in[:] *= sigma_noise ** 2 / 2
        return in_ber, in_ser

    def run(self, snr_db, rng):
        """
        Perform single experiment trial
        """
        if not self.codec.is_azcw():
            self.codec.generate(rng, self.iwd, self.tx_bits)
        in_ber, in_ser = self.run_channel(snr_db, rng)

        n_iter = self.codec.decode(self.llr_in, self.llr_out, rng=rng)
        out_ber = self.output_ber(n_iter)

        return LdpcDataEntry(
            in_be_cum=in_ber,
            in_se_cum=in_ser,
            be_cum=out_ber,
            fe_cum=out_ber > 0,
            n_iter=n_iter,
            iter_pdf=self.one_hot(n_iter),
            tests=1
        )

    def output_ber(self, n_iter):
        """
        Calculate output BER. Avoid passing data to ctypes if early termination happened
        """
        if self.codec.is_azcw() and n_iter < self.codec.n_iterations:
            return 0.0
        return self.codec.decoder_impl.output_ber(self.llr_out, self.tx_bits, n_iter)

    def one_hot(self, n_iter):
        """
        One-hot encoding for iteration count PDF
        """
        vec = np.zeros(self.codec.n_iterations + 1, dtype=np.int32)
        vec[n_iter] = 1
        return vec
