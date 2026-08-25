"""
This module implements postprocessing module with the following functionality:
 - Evaluating error bars for simulated frame error rate
 - Plot fit curves for frame error rate and bit error rate based on Bernoulli likelihood function
 - Two types of regression supported:
      - linear over polynomial features (in log-probability domain)
      - spline: a cubic spline is plotted through a set of adjustable reference points
                Final curve is such that the Bernoulli likelihood is maximum
 - Writing simulated data to text file (for further processing by plot tools, see berfit_plot.tex)
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

import multiprocessing
import os

import pandas as pd
import numpy as np

from scipy.optimize import minimize
from scipy.stats import binomtest
from scipy.interpolate import CubicSpline
from .channel import Modulation
from .tools import load_pickle


class PostProcessing:
    """
    Generate text/pandas data frame outputs from raw simulation results
    """
    def __init__(self, filename, modulation, params):
        # Parameters without default values
        self.modulation = modulation  # Channel modulation
        self.filename = filename

        # Parameters not subject to change
        # Confidence level for error bars
        self.params = params
        # Data update lock
        self.update_lock = multiprocessing.Lock()
        # Cached data access lock
        self.cache_lock = multiprocessing.Lock()

        # Cached data
        self.pickle_cache = {}
        self.data = None

    def txt_filename(self):
        """
        Get output textfile name
        """
        return os.path.splitext(self.filename)[0] + '.txt'

    def get(self):
        """
        Get the up-to-date pandas data frame.
        This data-frame will also be saved to txt file.
        """
        if not os.path.isfile(self.filename):
            self.__reset()
            return None

        txt_file = self.txt_filename()
        if os.path.isfile(txt_file):
            pickle_updated = os.path.getmtime(txt_file) < os.path.getmtime(self.filename)
            if not (pickle_updated or self.data is None):
                return self.__data_get()

        if self.update_lock.acquire(block=False):
            # Update cached data
            self.__update_pickle_cache()
            # Generate dataframe from the cached data
            data = self.__get_dataframe()
            self.update_lock.release()
            self.__data_set(data)
            return data

        # Lock was not acquired. Parallel postprocessing in progress. Return data as is.
        return self.__data_get()

    def __reset(self):
        """
        Reset cache and remove post-processed text file
        """
        with self.update_lock:
            self.pickle_cache = {}

        txt_file = os.path.splitext(self.filename)[0] + '.txt'
        with self.cache_lock:
            os.system(f'rm -f {txt_file}')
            self.data = None

    def __data_get(self) -> pd.DataFrame:
        """
        Get data with lock
        """
        with self.cache_lock:
            return self.data

    def __data_set(self, data):
        """
        Save data to file (with lock)
        """
        txt_file = os.path.splitext(self.filename)[0] + '.txt'
        with self.cache_lock:
            data.to_csv(txt_file, sep=' ', float_format='%1.6e', index=False)
            # Save dataframe for fast-return if pickle file was not changed
            self.data = data

    def __update_pickle_cache(self):
        """
        Load pickle and update cache
        """
        data_pickle = load_pickle(self.filename)
        # Update pickle cache: re-evaluate confidence intervals for updated entries
        for snr, entry in data_pickle.items():
            if entry['tests'] == 0:
                continue
            if snr not in self.pickle_cache or entry['tests'] != self.pickle_cache[snr]['tests']:
                self.pickle_cache[snr] = data_pickle[snr]
                # Error bars it the most expensive postprocessing procedure.
                # Do it only if the data was updated
                pe_minus, pe_plus = self.bernoulli_confidence(
                    entry['fe_cum'],  # Number of errors
                    entry['tests']  # Number of tests
                )
                self.pickle_cache[snr]['fer_e_minus'] = pe_minus
                self.pickle_cache[snr]['fer_e_plus'] = pe_plus
        # Sort pickle cache by SNR
        self.pickle_cache = dict(sorted(self.pickle_cache.items()))

    def __get_dataframe(self):
        """
        Generate pandas Data Frame from cached data
        """
        data = dict2pandas(self.pickle_cache)
        # Add theoretical BER values to immediately detect any bug from plots
        data['in_ber_ref'] = Modulation(self.modulation).get_ber(data.snr)

        # Generate smoothed BER/FER curves
        data['fer_fit'] = self.get_bernoulli_fit(
            data.snr.to_numpy(),
            data.fe_cum.to_numpy(),
            data.tests.to_numpy()
        )
        # Typically, errors in individual bits may be dependent.
        # Thus:
        # 1. One can generate fit without exact number of bits and
        #    use down-scaled values by the number of bits per single test
        # 2. Bit error rate confidence intervals may be useless
        data['ber_fit'] = self.get_bernoulli_fit(
            data.snr.to_numpy(),
            data.be_cum.to_numpy(),
            data.tests.to_numpy()
        )

        return data

    def bernoulli_confidence(self, n_errors: float, n_tests: int):
        """
        Calculate upper and lower confidence intervals for Bernoulli random variable
        """
        p_err = n_errors / n_tests
        result = binomtest(k=np.int64(np.round(n_errors)), n=n_tests, p=p_err)
        conf_interval = result.proportion_ci(
            confidence_level=self.params.confidence_level,
            method='wilson'  # Do not use 'exact' method when the number of tests is large
        )
        return p_err - conf_interval.low, conf_interval.high - p_err  # Error 'minus', error 'plus'

    def get_bernoulli_fit(self, snr_range: np.array, n_errors: np.array, n_tests: np.array):
        """
        Try multiple get_bernoulli_fit attempts with different degrees of freedom and
        output the best fit in terms of resulting loss
        :param snr_range        SNR points
        :param n_errors         The number of errors
        :param n_tests          The number of tests conducted
        """
        pe_raw = n_errors / n_tests

        # Skip points with error probabilities close to one

        idx_skip = np.argwhere(pe_raw > self.params.pe_threshold)
        skip_indx = 0
        if len(idx_skip):
            skip_indx = np.max(idx_skip) + 1

        if len(snr_range[skip_indx:]) < self.params.max_degree_ratio:
            return pe_raw
        # Estimate degree: minimum between requested degrees and a fraction of collected points
        degree = min(
            self.params.max_degree + 1,
            np.int32(np.round(len(snr_range) / self.params.max_degree_ratio))
        )
        # Perform fit
        if self.params.regression_type == 'polynomial':
            fit_type = BerFitPolynomial
        elif self.params.regression_type == 'spline':
            fit_type = BerFitSpline
        else:
            raise ValueError('Unknown BER/FER fit type')

        fit_instance = fit_type(
            snr_range[skip_indx:],  # SNR points
            n_errors[skip_indx:],  # The number of errors
            n_tests[skip_indx:],  # The number of experiments
            degree,  # Regression degree
        )
        # Polynomial regression
        pe_raw[skip_indx:] = fit_instance.fit()[0]
        return pe_raw


class BerFit:
    """
    Generate smooth bit error rate curves based on the linear regression with
    Bernoulli likelihood loss function for selected regression degree
    """
    def __init__(self, snr_range, n_errors, n_tests):
        # Normalize the SNR range
        self.snr_normalized = BerFit.normalize_snr_range(snr_range)
        # Save number of success and errors for Bernoulli likelihood function
        self.errors = n_errors
        self.success = n_tests - n_errors
        self.n_tests = n_tests

    @staticmethod
    def normalize_snr_range(snr_range):
        """
        Shift the SNR range to [0, 1] interval
        """
        snr_span = np.max(snr_range) - np.min(snr_range)
        return (snr_range - np.min(snr_range)) / snr_span

    def loss_from_points(self, log_pe):
        """
        Get Bernoulli loss given a set of fit values
        Bernoulli-loss function is p^k * (1 - p) ^ (n - k) * binom(n, k)
        Note that binom (n, k) does not change as the dataset is fixed, one can ignore it
        return: negative log-likelihood
        """
        if np.sum(log_pe > 0):
            return np.inf
        # Calculate log(1 - p) given log(p)
        log_p1 = np.log1p(-np.exp(log_pe))
        return -np.sum(self.errors * log_pe + self.success * log_p1)


class BerFitPolynomial(BerFit):
    """
    Linear regression over polynomial features in log-probability domain
    """
    def __init__(self, snr_range, n_errors, n_tests, fit_degree):
        super().__init__(snr_range, n_errors, n_tests)
        self.feature_matrix = np.vstack([
            self.snr_normalized ** i for i in range(fit_degree + 1)
        ]).T

    def fit(self):
        """
        Apply a polynomial regression on log(p) with Bernoulli loss function
        """
        # Suppress 'divide by zero encountered in _binom_cdf'
        np.seterr(divide='ignore')
        # Optimal solution with a Gaussian loss
        p_e = self.errors / self.n_tests
        # Note that Gaussian fit does not allow zero probability of error,
        # while the Bernoulli fit allows
        log_pe = np.log(p_e)
        features = np.linalg.pinv(self.feature_matrix) @ log_pe
        # Fine-tuned solution for Bernoulli loss
        sol = minimize(self.loss, features, jac=self.jac, method='Newton-CG')
        return np.exp(self.feature_matrix @ sol.x), sol.fun

    def loss(self, features):
        """
        Get loss value given regression coefficients
        """
        return self.loss_from_points(self.feature_matrix @ features)

    def jac(self, features):
        """
        Loss function gradient for polynomial regression
        """
        p_err = np.exp(self.feature_matrix @ features)
        return ((self.n_tests * p_err - self.errors) / (1 - p_err)) @ self.feature_matrix


class BerFitSpline(BerFit):
    """
    Cubic spline in log-probability domain. Degree specifies the number of reference points.
    These are placed such that the likelihood function is maximal.
    Note that gradient methods are not available here.
    """
    def __init__(self, snr_range, n_errors, n_tests, fit_degree):
        super().__init__(snr_range, n_errors, n_tests)
        self.n_points = fit_degree + 2  # Add two end-points

    def fit(self):
        """
        Apply a linear regression on log(p) with Bernoulli loss function
        """
        sol = minimize(self.loss, self.get_initial_spline(), method='Nelder-Mead')
        points = sol.x.reshape(2, -1)
        spline = CubicSpline(points[0, :], points[1, :], extrapolate=True)
        log_pe_fit = spline(self.snr_normalized)
        return np.exp(log_pe_fit), sol.fun

    def get_initial_spline(self):
        """
        Get initial points for spline regression
        """
        # Select points with positive error count
        all_idx = np.argwhere(self.errors > 0).reshape(-1)
        indices = np.round(np.linspace(0, len(all_idx) - 1, self.n_points)).astype(np.int32)
        idx = all_idx[indices]
        # Initial points:
        # 1. set of arguments
        # 2. set of values.
        # The optimization procedure will change both coordinates
        log_pe = np.log(self.errors / self.n_tests)
        return np.hstack([self.snr_normalized[idx], log_pe[idx]])

    def loss(self, features):
        """
        Get loss value given spline reference points
        """
        n_points = int(len(features) / 2)
        assert 2 * n_points == len(features)
        if np.min(features[:n_points]) < 0 or np.max(features[:n_points]) > 1.0:
            return np.inf
        try:
            log_pe = CubicSpline(features[:n_points], features[n_points:])(self.snr_normalized)
        except ValueError:
            return np.inf
        return self.loss_from_points(log_pe)


def dict2pandas(data_dict):
    """
    Convert dictionary loaded for postprocessing to pandas dataframe
    """
    entries = list(data_dict.values())
    snr_range = np.array([float(k) for k in list(data_dict.keys())])

    # Create Pandas dataframe
    data = pd.DataFrame()
    # Signal-to-noise ratio range
    data['snr'] = snr_range
    # Copy cumulative statistics
    fields = ['tests', 'fe_cum', 'be_cum', 'in_be_cum', 'fer_e_minus', 'fer_e_plus']
    for field in fields:
        data[field] = np.array([e[field] for e in entries])

    # Drop points having zero tests
    data = data.drop(data[data.tests == 0].index)
    # Drop points having zero frame errors
    data = data.drop(data[data.fe_cum == 0].index)

    # Derive values requiring division by the number of tests
    data['fer'] = data.fe_cum / data.tests
    data['ber'] = data.be_cum / data.tests
    data['in_ber'] = data.in_be_cum / data.tests

    # Drop points having zero tests
    data = data.drop(data[data.tests == 0].index)
    # Drop points having zero frame errors
    data = data.drop(data[data.fe_cum == 0].index)

    # Derive values requiring division by the number of tests
    data['fer'] = data.fe_cum / data.tests
    data['ber'] = data.be_cum / data.tests
    data['in_ber'] = data.in_be_cum / data.tests

    return data
