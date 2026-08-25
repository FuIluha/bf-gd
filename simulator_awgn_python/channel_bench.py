"""
Channel benchmarking and testing
"""

# This file is part of the simulator_awgn_python distribution
# https://github.com/and-kirill/simulator_awgn_python/.
# Copyright (c) 2023 Kirill Andreev.
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

# To compare LLR values with MATLAB, run the following:
# MATLAB code below:
# >> snr_db = 8; sigma_noise = sqrt(1 / 10^(snr_db / 10)); ES = 10;
# >> qamdemod(-5:5, 16, 'OutputType', 'llr', 'NoiseVariance', ES * sigma_noise^2)
# Python code providing the same result:
# >>> from commpy import QAMModem
# >>> modem = QAMModem(16)
# >>> snr_db = 8; sigma_noise = np.sqrt(1 / 10 ** (snr_db / 10)) * np.sqrt(modem.Es)
# >>> print(-modem.demodulate(np.arange(-5, 6), 'soft', sigma_noise ** 2))


import time
import numpy as np

from commpy import QAMModem

from .channel import AwgnQAMChannel, Modulation, random_bits, lib_compile


def run_commpy_channel(modulation, tx_bits, snr_db, rng):
    """
    Run channel from scikit-commpy module
    """
    if modulation == 'qam-16':
        modem = QAMModem(16)
    elif modulation == 'pam-4':
        modem = QAMModem(16)
        snr_db = snr_db + 10 * np.log10(2)
    elif modulation == 'qpsk':
        modem = QAMModem(4)
    elif modulation == 'bpsk':
        modem = QAMModem(4)
        snr_db = snr_db + 10 * np.log10(2)
    else:
        raise ValueError('Unsupported modulation', modulation)

    tx_symb = modem.modulate(tx_bits)
    sigma_noise = np.sqrt(1 / 10 ** (snr_db / 10)) * np.sqrt(modem.Es)

    n_noise_samples = 2 * len(tx_symb)
    noise = sigma_noise * \
        rng.standard_normal(size=n_noise_samples, ) / np.sqrt(2)
    # Reshape the noise to match the result
    noise = -noise.reshape(-1, 2).T.reshape(-1)
    noise = noise[:len(tx_symb)] + 1j * noise[len(tx_symb):]

    return -modem.demodulate(tx_symb + noise, 'soft', sigma_noise ** 2)


def compare(modulation, snr_db):
    """
    Compare results. Evaluate bit-rate and the norm of LLR difference vector
    """
    dtype = np.float64
    n_bits = 400000
    llr = np.zeros((n_bits,), dtype=dtype)
    bits = np.zeros((n_bits,), dtype=np.uint8)
    channel = AwgnQAMChannel(modulation, bits, llr, False)

    bit_rng = np.random.default_rng(seed=2)

    t_start = time.time()
    random_bits(bits, bit_rng)
    t_end = time.time()

    print('Bit generator rate: ', len(bits) /
          (t_end - t_start) / 1e6, 'Mbit/s')
    print(f'Compare LLR values for {modulation.upper()}')

    t_start = time.time()
    llr_commpy = run_commpy_channel(
        modulation,
        bits,
        snr_db,
        np.random.default_rng(
            seed=1))
    t_end = time.time()
    print('Commpy:         ', len(bits) / (t_end - t_start) / 1e6, 'Mbit/s')

    t_start = time.time()
    channel.run(snr_db, np.random.default_rng(seed=1))
    t_end = time.time()

    print(f'AWGN channel:   {len(bits) / (t_end - t_start) / 1e6:1.3f} Mbit/s')
    print('LLR difference: ', np.linalg.norm(llr - llr_commpy))


def perf(modulation, dtype, snr_db):
    """
    Performance tests on all-zero codeword
    """
    n_tests = 10000
    n_bits = 10000
    llr = np.zeros((n_bits,), dtype=dtype)
    bits = np.zeros((n_bits,), dtype=np.uint8)

    rng = np.random.default_rng()
    channel = AwgnQAMChannel(modulation, bits, llr, False)
    t_start = time.time()
    for _ in range(n_tests):
        channel.run(snr_db, rng)
    t_end = time.time()
    rate_wo_adapter = n_bits * n_tests / (t_end - t_start) / 1024 / 1024

    channel = AwgnQAMChannel(modulation, bits, llr, True)

    t_start = time.time()
    # Collect BER and SER statistics and compare with reference values
    ber_cum = 0
    ser_cum = 0
    for _ in range(n_tests):
        ber, ser = channel.run(snr_db, rng)
        ber_cum += ber
        ser_cum += ser
    t_end = time.time()
    rate_wh_adapter = n_bits * n_tests / (t_end - t_start) / 1024 / 1024

    print(f'{modulation} channel bitrate (AZCW, without/with adapter, {dtype}):', end='')
    print(f'{rate_wo_adapter:1.2f} / {rate_wh_adapter:1.2f} Mbit/s. ', end='')
    ber_ref = Modulation(modulation).get_ber(snr_db)
    print(f'BER (got/ref): {ber_cum / n_tests:1.3e}/{ber_ref:1.3e}. SER: {ser_cum / n_tests:1.3e}.')


def main():
    """
    Main function: compare LLRs and evaluate single-core bitrate
    """
    lib_compile()
    print('Compare LLR values with respect to commpy')
    compare('qam-16', 8.0)
    compare('pam-4', 5.0)
    compare('qpsk', 1.0)
    compare('bpsk', -2.0)
    print('Performance benchmarking (sequence of runs):')
    for modulation in ['bpsk', 'qpsk', 'pam-4', 'qam-16']:
        perf(modulation, np.float64, 1.0)
        perf(modulation, np.float32, 1.0)


if __name__ == '__main__':
    main()
