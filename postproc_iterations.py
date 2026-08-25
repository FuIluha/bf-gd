""""
Postprocessing for some specific number of iterations
"""

import sys
import pickle
import os.path

import pandas as pd
import numpy as np

from simulator_awgn_python.postprocessing import PostProcessing
from simulator_awgn_python.settings import PostprocessingSettings


def extract_fer(data, n_iter):
    """
    Extract FER for specific number of iterations from pickle file given
    a histogram of iterations
    """
    snr_range = np.array(list(data.keys()))
    n_errors = np.zeros_like(snr_range)
    n_tests = np.zeros_like(snr_range)
    for i, snr_db in enumerate(snr_range):
        entry = data[snr_db]
        iter_pdf = entry['iter_pdf']
        n_errors[i] = np.sum(iter_pdf[n_iter + 1:])
        n_tests[i] = entry['tests']

    # Filter zero error count out
    snr_range = snr_range[n_errors > 0]
    n_errors = n_errors[n_errors > 0]
    n_errors = n_errors[n_errors > 0]
    return snr_range, n_errors, n_tests


def load_data(filename):
    """
    Load pickle file and handle exceptions
    """
    try:
        with open(filename, 'rb') as fhandle:
            data = pickle.load(fhandle)
        return data
    except FileNotFoundError:
        print(f'File {filename} not found')
        sys.exit(1)
    except pickle.UnpicklingError:
        print('Broken pickle file')
        sys.exit(1)


def pickle2txt(filename, n_iter, settings):
    """
    Generate fit and confidence intervals in accordance with settings
    """
    data = load_data(filename)

    print(f'Input filename: {filename}')
    print('-' * 25, 'Postprocessing settings: ', '-' * 25)
    print(settings)
    print('-' * 77)

    snr_range, n_errors, n_tests = extract_fer(data, n_iter)

    pp_instance = PostProcessing(filename, 'BPSK', settings)
    fer_fit = pp_instance.get_bernoulli_fit(snr_range, n_errors, n_tests)
    fer = n_errors / n_tests

    pe_minus = np.zeros(n_errors.shape)
    pe_plus = np.zeros(n_errors.shape)
    for i, err_count in enumerate(n_errors):
        pe_minus[i], pe_plus[i] = PostProcessing(filename, 'BPSK', settings).bernoulli_confidence(
            err_count,  # Number of errors
            int(n_tests[i])  # Number of tests
        )
    # snr tests fe_cum fer_e_minus fer_e_plus fer fer_fit
    txt_file = os.path.splitext(filename)[0] + f'_iter_{n_iter}' + '.txt'
    pd.DataFrame({
        'snr': snr_range,
        'tests': n_tests,
        'fe_cum': n_errors,
        'fer_e_minus': pe_minus,
        'fer_e_plus': pe_plus,
        'fer': fer,
        'fer_fit': fer_fit
    }).to_csv(txt_file, sep=' ', float_format='%1.6e', index=False)
    print(f'Data saved to {txt_file}')


def main():
    """
    Main function. Pickle filename will be converted to the text file.
    Note that the bit error rate data will be lost when reducing the number of iterations
    """
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <filename.pickle> <n_iter>')
        sys.exit(0)
    try:
        n_iter = int(sys.argv[2])
    except ValueError:
        print('The number of iterations must be integer')
        sys.exit(1)

    settings = PostprocessingSettings(
        regression_type='polynomial',
        max_degree=15,
        max_degree_ratio=3,
        confidence_level=0.95,
        pe_threshold=0.99,
    )
    pickle2txt(sys.argv[1], n_iter, settings)


if __name__ == '__main__':
    main()

