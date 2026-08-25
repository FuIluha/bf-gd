"""
Some useful tools to be performed over binary parity check matrices
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
 

import numpy as np


def invert_permutation(perm_forward):
    """
    Construct the inverse permutation.
    Note that matlab-like style of the inverse permutation p[idx]=p may not work for numpy
    """
    perm_reverse = np.empty(perm_forward.size, perm_forward.dtype)
    perm_reverse[perm_forward] = np.arange(perm_forward.size)
    return perm_reverse


def generator_from_pcm(pcm_rdonly):
    """
    Construct the generator matrix from the parity check matrix
    return: generator matrix (np.array, uint8) and information bits indices
    """
    pcm = pcm_rdonly.copy()  # Make a copy, row combinations will be performed
    n_rows, n_cols = pcm.shape
    col_ind = 0
    row_ind = 0
    eye_idx = []
    while row_ind < n_rows:
        # Find the first non-zero entry in column below <index> value
        nz_idx = np.argwhere(pcm[row_ind:, col_ind] == 1).reshape(-1)
        if not len(nz_idx):  # pylint C1802 is not applicable for np.array()
            col_ind += 1
            continue
        # Swap rows
        swap_ind = nz_idx[0] + row_ind
        pcm[[row_ind, swap_ind], :] = pcm[[swap_ind, row_ind], :]

        for i in np.argwhere(pcm[:, col_ind] == 1).reshape(-1):
            if row_ind == i:
                continue
            pcm[i, :] = np.mod(pcm[i, :] + pcm[row_ind, :], 2)
        eye_idx.append(col_ind)
        row_ind += 1
        col_ind = row_ind

    pc_idx = np.setdiff1d(np.arange(n_cols), eye_idx)  # Parity check indices
    all_idx = np.hstack([eye_idx, pc_idx])  # All indices (permuted)
    gen_mtx = np.hstack([pcm.copy()[:, pc_idx].T, np.eye(n_cols - n_rows).astype(np.uint8)])
    return gen_mtx[:, invert_permutation(all_idx)], all_idx[n_rows:]


def lift_pcm(pcm_base, factor, expand_seed=None):
    """
    Expand a base graph
    :param pcm_base: base graph (binary)
    :param factor: lifting index
    :param expand_seed: seed index to generate reproducible lifts
    :return: lifted parity check matrix
    """
    if expand_seed is None:
        expand_rng = np.random.default_rng()
    else:
        expand_rng = np.random.default_rng(seed=expand_seed)
    pcm_exp = pcm_base * expand_rng.integers(low=1, high=factor + 1, size=pcm_base.shape) - 1
    return expand_pcm(factor, pcm_exp)


def expand_pcm(factor, pcm_exp):
    """
    Expand the base parity check matrix. Taken from LDPC simulator, 5G module
    """
    # Only binary codes supported
    n_checks, blocklen_base = pcm_exp.shape
    pcm = []
    for i in range(n_checks):
        layer = []
        for j in range(blocklen_base):
            shift = pcm_exp[i, j]
            if shift == -1:
                layer.append(np.zeros((factor, factor)))
            else:
                layer.append(np.roll(np.eye(factor), shift, axis=1))
        pcm.append(np.hstack(layer))
    return np.vstack(pcm)


def disturb_pcm(pcm, n_ones):
    """
    Disturb the base-graph parity check matrix
    :param pcm base parity check matrix, np.array(dtype=np.uint8)
    :param n_ones is the number of changes to be injected
    """
    n_rows, n_cols = pcm.shape
    active_cols = np.arange(n_cols)
    n_elem = n_rows * active_cols.shape[0]

    disturbance = np.zeros(n_elem,).astype(np.uint8)
    disturbance[np.random.choice(np.arange(n_elem), n_ones, replace=False)] = 1
    disturbance = disturbance.reshape(n_rows, active_cols.shape[0])
    pcm_out = np.copy(pcm)
    pcm_out[:, active_cols] = np.mod(pcm_out[:, active_cols] + disturbance, 2)
    return pcm_out
