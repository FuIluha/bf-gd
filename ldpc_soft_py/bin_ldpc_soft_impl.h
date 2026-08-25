/*
 * This file is part of the simulator_awgn_python distribution
 * https://github.com/and-kirill/ldpc_soft_py/.
 * Copyright (c) 2023 Kirill Andreev, Alexey Frolov
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DECODER_H_
#define DECODER_H_

#include <limits>
#include <vector>
#include <cmath>

#include "./matrix.h"
#include "./bin_tanner_graph.h"
#include "./bin_ldpc_soft_settings.h"

#ifdef OFFSETS_ENABLED
#define OFFSET(A, B) ((A)-(B)) * ((A) > (B))
#define SIGN(A) (((A) > 0) ? 1 : -1)
#endif // ifdef OFFSETS_ENABLED

/// Load Tanner graph from the alist file
void* load_alist(const char *filename,
                 uint32_t    blocklen,
                 uint32_t    n_checks);


template<typename TL, typename TI>
bool check_syndrome(const TannerGraph<TI>    & ldpc,
                    const DecoderSettings<TL>& settings,
                    const TL                  *llr_out) {
  if (!settings.early_termination) {
    return false;
  }

  std::vector<uint8_t> llr_signs = std::vector<uint8_t>(ldpc.n, 0);

  for (TI vni = 0; vni < ldpc.n; vni++) {
    // No erasures allowed for syndrome check
    if (llr_out[vni] == 0) {
      return false;
    }
    llr_signs[vni] = llr_out[vni] < 0;

    if (settings.is_azcw && llr_signs[vni]) {
      // LLR < 0 is enough to fail the syndrome check if all-zero codeword
      return false;
    }
  }

  // All LLRs are positive, return true is all zero codeword assumed
  if (settings.is_azcw) {
    return true;
  }

  const TI *row_weights = ldpc.row_weight.data();

  // Check the syndrome under arbitrary codeword condition
  for (TI j = 0; j < ldpc.m; j++) {
    const TI  row_weight_j = row_weights[j];
    const TI *col_idx      = ldpc.col_idx_matrix.row(j);
    int pc_failed          = 0;

    for (TI k = 0; k < row_weight_j; k++) {
      const TI col_id = col_idx[k];
      pc_failed ^= llr_signs[col_id];
    } // Loop over single parity check

    if (pc_failed) {
      return false;
    }
  } // Loop over parity checks
  return true;
}

/**
 * Horizontally Layered Min-sum decoder. Layered processing assumes that
 * output LLRs are updated after each parity check processing.
 * The layered decoding algorithm is the following:
 *  1. Iterate over decoding iterations
 *  2. Iterate over all parity checks
 *  3. Process single parity check
 * There are also scaling and offset coefficients represented as vectors
   Each weight set is a vector of block length
 * The output message magnitudes from check to variable nodes
 * are scaled and offset
 * @param tng       Pointer to the Tanner graph structure
 * @param settings  LDPC decoder settings:
 *        - The number of decoding iterations
 *        - Early termination (exit if syndrome is 0)
 *        - Scales array
 *        - Offsets array
 *        - Row sequence array
 * @param llr_in vector of channel log likelihood ratios
 * @param row_seq is a parity checks processing schedule:
 *        array of check node indices
 * @param scales multiplicative scales
 * @param offsets applies relu(x - offset) for each message magnitude
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */
template<typename TL, typename TI>
unsigned int ldpc_hl_noms(const TannerGraph<TI>    & tng,
                          const DecoderSettings<TL>& settings,
                          const TL                  *llr_in,
                          TL                        *llr_out) {
  Matrix<TL> r_msg   = tng.template matrix_r<TL>(0);
  const TL   max_val = std::numeric_limits<TL>::max();

  // Precompute pointers for faster access
  const TI *row_weights = tng.row_weight.data();

  // Copy input - use memcpy for better vectorization
  std::copy(llr_in, llr_in + tng.n, llr_out);

  for (unsigned int loop = 0; loop < settings.n_iterations; ++loop) {
    if (check_syndrome(tng, settings, llr_out)) {
      return loop;
    }

    for (TI j = 0; j < tng.m; j++) {
      const TI row_weight_j = row_weights[j];

      // Get row pointers once
      TL *r_msg_row     = r_msg.row(j);
      const TI *col_idx = tng.col_idx_matrix.row(j);

      TL  first_min  = max_val;
      TL  second_min = max_val;
      TI  min_index  = 0;
      int sum_sign   = 0;

      // First pass: find mins and sign sum
      for (TI k = 0; k < row_weight_j; k++) {
        const TI col_id    = col_idx[k];
        const TL q_msg     = llr_out[col_id] - r_msg_row[k];
        const TL abs_q_msg = std::abs(q_msg);

        // Update min tracking - branchless version
        const bool update_first = (abs_q_msg < first_min);
        second_min = update_first ? first_min :
                     (abs_q_msg < second_min) ? abs_q_msg : second_min;
        first_min = update_first ? abs_q_msg : first_min;
        min_index = update_first ? k : min_index;

        sum_sign ^= (q_msg < 0);
      }

      // Second pass: update messages
      for (TI k = 0; k < row_weight_j; k++) {
        const TI col_id = col_idx[k];
        const TL q_msg  = llr_out[col_id] - r_msg_row[k];

        // Calculate new r_msg - branchless
        const int sign    = sum_sign ^ (q_msg < 0);
        const TL  min_val = (k == min_index) ? second_min : first_min;

        // Use multiplication instead of conditional
#ifdef OFFSETS_ENABLED
        TL scale  = settings.scale_array[col_id];
        TL offset = settings.offset_array[col_id];
        r_msg_row[k] = (1 - 2 * sign) * scale * OFFSET(min_val, offset);
#else // ifdef OFFSETS_ENABLED
        TL scale = settings.scale_array[0];
        r_msg_row[k] = (1 - 2 * sign) * scale * min_val;
#endif // ifdef OFFSETS_ENABLED

        // Update llr_out
        llr_out[col_id] = q_msg + r_msg_row[k];
      }
    }
  }
  return settings.n_iterations;
}

// Vertically layered normalized offset min-sum
template<typename TL, typename TI>
unsigned int ldpc_vl_noms(const TannerGraph<TI>    & tng,
                          const DecoderSettings<TL>& settings,
                          const TL                  *llr_in,
                          TL                        *llr_out) {
  Matrix<TL> r_msg   = tng.template matrix_r<TL>(0);
  const TL   max_val = std::numeric_limits<TL>::max();

  // Precompute pointers for faster access
  const TI *row_weights = tng.row_weight.data();
  const TI *col_weights = tng.col_weight.data();

  for (TI i = 0; i < tng.n; i++) {
    llr_out[i] = llr_in[i];
  }

  for (unsigned int loop = 0; loop < settings.n_iterations; ++loop) {
    // Start with the syndrome checking
    if (check_syndrome(tng, settings, llr_out)) {
      return loop;
    }

    // Start with a loop over variable nodes
    for (TI vni = 0; vni < tng.n; vni++) {
      TL  acc_change = 0;
      TI *col_idx_vn = tng.row_idx_matrix.row(vni);

      // For each variable node (VN), process all check nodes connected to it
      for (TI i = 0; i < col_weights[vni]; i++) {
        // Find check node (CN) index that is connected to the considered VN
        TI cni       = col_idx_vn[i];
        TI vn_chosen = 0; // Index of chosen VN in the parity check

        // Process check node. Need to find just one minimum
        TL  min_val  = max_val;
        int sum_sign = 0;

        // Get row pointers once
        const TI *col_idx = tng.col_idx_matrix.row(cni);
        TL *r_msg_row     = r_msg.row(cni);

        for (TI k = 0; k < row_weights[cni]; k++) {
          TI col_id = col_idx[k];

          if (vni == col_id) {
            vn_chosen = k;
            continue;
          }
          TL q_msg     = llr_out[col_id] - r_msg_row[k];
          TL abs_q_msg = std::abs(q_msg);

          // Update minimum value
          min_val = abs_q_msg < min_val ? abs_q_msg : min_val;

          // Update sign
          sum_sign ^= q_msg < 0;
        }
#ifdef OFFSETS_ENABLED
        TI coef_index = tng.col_idx_matrix.row(cni)[vn_chosen];
        TL scale      = settings.scale_array[coef_index];
        TL offset     = settings.offset_array[coef_index];
        TL update     = (1 - 2 * sum_sign) * scale *
                        OFFSET(min_val, offset);
#else // ifdef OFFSETS_ENABLED
        TL scale  = settings.scale_array[0];
        TL update = (1 - 2 * sum_sign) * scale * min_val;
#endif // ifdef OFFSETS_ENABLED
        acc_change          += update - r_msg_row[vn_chosen];
        r_msg_row[vn_chosen] = update;
      }
      llr_out[vni] += acc_change;
    } // Loop over parity checks
  }   // Loop over iterations
  return settings.n_iterations;
}

/**
 * Normalized offset Min-sum decoder.
 * Non-layered processing:
 * 1. Initialization
 * 2. For each decoding iteration:
 *  - update R-messages: from check to variable nodes
 *  - Update Q-messages: from variable to check nodes
 *  - Update output LLRs.
 * @param tng     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param scales multiplicative scales
 * @param offsets applies relu(x - offset) for each message magnitude
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */
template<typename TL, typename TI>
unsigned int ldpc_noms(const TannerGraph<TI>    & tng,
                       const DecoderSettings<TL>& settings,
                       const TL                  *llr_in,
                       TL                        *llr_out) {
#if 0

  // Unlikely case, needed just to match old implementation output
  if (settings.terminate_syndrome && check_syndrome(ldpc, llr_in, is_azcw)) {
    std::copy(llr_in, llr_in + ldpc.n, llr_out);
    return 0;
  }
#endif // if 0

  // Auxiliary matrices
  // messages from check to variable
  Matrix<TL> r_msg = tng.template matrix_r<TL> ();

  // messages from variable to check
  Matrix<TL> q_msg = tng.template matrix_r<TL> ();

  const TL max_val = std::numeric_limits<TL>::max();

  // Precompute pointers for faster access
  const TI *row_weights = tng.row_weight.data();

  TL *r_buf         = r_msg.buf();
  TL *q_buf         = q_msg.buf();
  const TI *col_idx = tng.col_idx_matrix.buf();

  // Initialization
  for (unsigned int j = 0; j < tng.nzc; j++) {
    TI col_id = col_idx[j];
    q_buf[j] = llr_in[col_id];
  }

  for (TI loop = 0; loop < settings.n_iterations; ++loop) {
    std::copy(llr_in, llr_in + tng.n, llr_out);

    // Update R messages
    for (TI j = 0; j < tng.m; j++) {
      const TI row_weight_j = row_weights[j];
      TL  first_min         = max_val;
      TL  second_min        = max_val;
      TI  min_index         = 0;
      int sum_sign          = 0;

      // Get row pointers once
      TL *r_msg_row     = r_msg.row(j);
      TL *q_msg_row     = q_msg.row(j);
      const TI *col_idx = tng.col_idx_matrix.row(j);

      for (TI k = 0; k < row_weight_j; k++) {
        TL abs_q_msg = std::abs(q_msg_row[k]);

        // Update min tracking - branchless version
        const bool update_first = (abs_q_msg < first_min);
        second_min = update_first ? first_min :
                     (abs_q_msg < second_min) ? abs_q_msg : second_min;
        first_min = update_first ? abs_q_msg : first_min;
        min_index = update_first ? k : min_index;

        sum_sign ^= (q_msg_row[k] < 0);
      } // Loop over check nodes

      for (TI k = 0; k < row_weight_j; k++) {
        int sign = sum_sign;
        sign ^= (q_msg_row[k] < 0);

        const TL min_val = (k == min_index) ? second_min : first_min;
#ifdef OFFSETS_ENABLED
        const TI col_id = col_idx[k];
        TL scale        = settings.scale_array[col_id];
        TL offset       = settings.offset_array[col_id];
        r_msg_row[k] = (1 - 2 * sign) * scale * OFFSET(min_val, offset);
#else // ifdef OFFSETS_ENABLED
        TL scale = settings.scale_array[0];
        r_msg_row[k] = (1 - 2 * sign) * scale * min_val;
#endif // ifdef OFFSETS_ENABLED
      }

      // Update output LLRs affected by current parity check
      for (TI k = 0; k < row_weight_j; k++) {
        const TI col_id = col_idx[k];
        llr_out[col_id] += r_msg_row[k];
      }
    }

    // Early stop if the syndrome is OK
    if (check_syndrome(tng, settings, llr_out)) {
      return loop + 1;
    }

    // Update Q-messages
    for (unsigned int j = 0; j < tng.nzc; j++) {
      TI col_id = col_idx[j];
      q_buf[j] = llr_out[col_id] - r_buf[j];
    }
  } // Loop over decoding iterations
  return settings.n_iterations;
}

template<typename TL>
static TL logtanh(TL x) {
  static const TL MAX_ARG = 12.5;
  static const TL MIN_ARG = -log(tanh(MAX_ARG / 2));

  static const TL MAX_VAL = log(std::numeric_limits<TL>::max() / 2);
  static const TL MIN_VAL = 2 * std::numeric_limits<TL>::min();

  if (x <= MIN_VAL) {
    return MAX_VAL;
  }

  // Use approximation for small argument
  if (x < MIN_ARG) {
    return -log(x / 2) + x * x * (TL(1.0) / TL(12.0));
  }

  // Use approximation for large values
  if (x > MAX_ARG) {
    TL val       = exp(-x);
    TL val_cubed = val * val * val;
    return TL(2) * val + (TL(2) / TL(3)) * val_cubed;
  }
  return -log(tanh(x * TL(0.5)));
}

/**
 * Sum-product decoder. To avoid a product of hyperbolic tangents,
 * the calculations are performed over the logarithm of the LLR magnitudes.
 * @param ldpc     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */
template<typename TL, typename TI>
unsigned int ldpc_sp(const TannerGraph<TI>    & tng,
                     const DecoderSettings<TL>& settings,
                     const TL                  *llr_in,
                     TL                        *llr_out) {
  // Messages from check to variable nodes
  Matrix<TL> r_msg = tng.template matrix_r<TL> ();

  // Messages from variable to check nodes: signs and log-magnitudes are stored
  // separately. Note that log-magnitudes are required to avoid products of
  // hyperbolic tangents
  Matrix<TL> q_ltanh      = tng.template matrix_r<TL> ();
  Matrix<uint8_t> q_signs = tng.template matrix_r<uint8_t> ();

  // Precompute pointers for faster access
  const TI *row_weights = tng.row_weight.data();

  TL *r_buf             = r_msg.buf();
  TL *q_ltanh_buf       = q_ltanh.buf();
  uint8_t  *q_signs_buf = q_signs.buf();
  const TI *col_idx     = tng.col_idx_matrix.buf();

  // Initialize input LLRs in the form of logtanh(|x|), sign(x)
  // Initialize: Compute
  std::vector<TL> llr_in_log      = std::vector<TL>(tng.n, 0);
  std::vector<uint8_t> llr_in_sgn = std::vector<uint8_t>(tng.n, 0);

  for (TI i = 0; i < tng.n; i++) {
    llr_in_log[i] = logtanh<TL>(std::abs(llr_in[i]));
    llr_in_sgn[i] = llr_in[i] < 0;
  }

  // Initialize: Propagate
  for (unsigned int j = 0; j < tng.nzc; j++) {
    TI col_id = col_idx[j];
    q_signs_buf[j] = llr_in_sgn[col_id];
    q_ltanh_buf[j] = llr_in_log[col_id];
  }

  // Decoding iterations
  for (unsigned int loop = 0; loop < settings.n_iterations; ++loop) {
    std::copy(llr_in, llr_in + tng.n, llr_out);

    // Update R messages

    for (TI j = 0; j < tng.m; j++) {
      const TI row_weight_j = row_weights[j];
      TL sum_ltanh          = 0;
      uint8_t sum_sign      = 0;

      // Get row pointers once
      TL *r_msg_row         = r_msg.row(j);
      TL *q_ltanh_row       = q_ltanh.row(j);
      uint8_t  *q_signs_row = q_signs.row(j);
      const TI *col_idx     = tng.col_idx_matrix.row(j);

      // Calculate R-message updates
      for (TI k = 0; k < tng.row_weight[j]; k++) {
        sum_ltanh += q_ltanh_row[k];
        sum_sign  ^= q_signs_row[k];
      }

      for (TI k = 0; k < row_weight_j; k++) {
        uint8_t r_sign = sum_sign ^ q_signs_row[k];
        TL r_val       = logtanh<TL>(sum_ltanh - q_ltanh_row[k]);
        r_msg_row[k] = (1 - 2.0 * r_sign) * r_val;

        // Update output LLRs affected by current parity check
        const TI col_id = col_idx[k];
        llr_out[col_id] += r_msg_row[k];
      }
    } // Loop over check nodes

    // Early stop if the syndrome is OK
    if (check_syndrome(tng, settings, llr_out)) {
      return loop + 1;
    }

    // Update Q messages
    for (unsigned int j = 0; j < tng.nzc; j++) {
      TI col_id = col_idx[j];
      TL val    = llr_out[col_id] - r_buf[j];
      q_signs_buf[j] = val < 0;
      q_ltanh_buf[j] = logtanh(std::abs(val));
    }
  } // Loop over decoding iterations
  return settings.n_iterations;
}

/*
 * !EXPERIMENTAL
 * Layered min-sum decoder of generalized LDPC codes generated by Cordaro-Wagner
 * codes
 */

// Normalized offset horizontally layered min-sum
template<typename TL, typename TI>
unsigned int gldpc_hl_noms(const TannerGraph<TI>    & tng,
                           const DecoderSettings<TL>& settings,
                           const TL                  *llr_in,
                           TL                        *llr_out) {
  static const TI N_GROPUS = 3; // Cordaro-Wagner: three groups of nodes
  // Messages from check to variable nodes
  Matrix<TL> r_msg   = tng.template matrix_r<TL>(0);
  const TL   max_val = std::numeric_limits<TL>::max();

  // Precompute pointers for faster access
  const TI *row_weights = tng.row_weight.data();

  // Copy input - use memcpy for better vectorization
  std::copy(llr_in, llr_in + tng.n, llr_out);

  // Decoding iterations
  for (unsigned int loop = 0; loop < settings.n_iterations; ++loop) {
    // NOTE: Check syndrome works only under all-zero codeword assumption
    if (check_syndrome(tng, settings, llr_out)) {
      return loop;
    }

    // Update R messages
    for (TI j = 0; j < tng.m; j++) {
      const TI row_weight_j = row_weights[j];

      // Get row pointers once
      TL *r_msg_row     = r_msg.row(j);
      const TI *col_idx = tng.col_idx_matrix.row(j);

      std::vector<TL> first_min(N_GROPUS, max_val);
      std::vector<TL> second_min(N_GROPUS, max_val);
      std::vector<uint8_t> sign_sums(N_GROPUS, 0);
      std::vector<TL> min_index(N_GROPUS, 0);

      // Find minimum values for each group of variable nodes
      for (TI k = 0; k < row_weight_j; k++) {
        const TI col_id = col_idx[k];
        TI group        = k % N_GROPUS;
        TL q_msg        = llr_out[col_id] - r_msg_row[k];
        TL abs_q_msg    = std::abs(q_msg);

        const bool update_first = (abs_q_msg < first_min[group]);
        second_min[group] = update_first ? first_min[group] :
                            (abs_q_msg < second_min[group]) ?
                            abs_q_msg : second_min[group];
        first_min[group] = update_first ? abs_q_msg : first_min[group];
        min_index[group] = update_first ? k : min_index[group];

        sign_sums[group] ^= (q_msg < 0);
      }

      for (TI k = 0; k < row_weight_j; k++) {
        const TI col_id  = col_idx[k];
        TI group         = k % N_GROPUS;
        TL q_msg         = llr_out[col_id] - r_msg_row[k];
        TL sign          = 1 - 2.0 * sign_sums[group];
        TL min_val_group = (k == min_index[group]) ?
                           second_min[group] : first_min[group];

        // Calculate the output LLR update
        TL llr_A  = 0; // In-group LLR
        TL llr_BC = 0; // Out-of-group LLR

        for (unsigned int g_id = 0; g_id < N_GROPUS; g_id++) {
          if (group == g_id) {
            llr_A += min_val_group * sign  * (1 - 2 * (TL)(q_msg < 0));
          } else {
            // Out-of-group. Always consider the first minimum here
            llr_BC += (1 - 2 * (TL)sign_sums[g_id]) * first_min[g_id];
          }
        }

        // GLDPC min-sum rule
        TL update = std::max<TL>(llr_BC + llr_A, 0) -
                    std::max<TL>(llr_BC, llr_A);
#ifdef OFFSETS_ENABLED
        TL scale  = settings.scale_array[col_id];
        TL offset = settings.offset_array[col_id];
        r_msg_row[k] = scale * SIGN(update) *
                       OFFSET(std::abs(update), offset);
#else // ifdef OFFSETS_ENABLED
        TL scale = settings.scale_array[0];
        r_msg_row[k] = scale * update;
#endif // ifdef OFFSETS_ENABLED
        llr_out[col_id] = q_msg + r_msg_row[k];
      }
    } // Loop over check nodes
  }   // Loop over decoding iterations
  return settings.n_iterations;
}

/// Generalized sum-product check node operation
template<typename TL>
std::pair<TL, int>gldpc_sp_cnop(TL log_x, int sign_x, TL log_y, int sign_y) {
  TL value = (1 - 2 * sign_x) * logtanh(log_x) +
             (1 - 2 * sign_y) * logtanh(log_y);

  return std::make_pair(logtanh(std::abs(value)), value < 0 ? 1 : 0);
}

/**
 * !EXPERIMENTAL
 * Sum-product GLDPC decoder. GLDPC are based on Cordaro-Wagner codes,
 * as previously specified NMS.
 * @param tng     Pointer to the Tanner graph structure
 * @param llr_in vector of channel log likelihood ratios
 * @param n_iter  The number of decoding iterations
 * @param llr_out is a raw buffer to keep output log-likelihood ratios
 */
template<typename TL, typename TI>
unsigned int gldpc_sp(const TannerGraph<TI>    & tng,
                      const DecoderSettings<TL>& settings,
                      const TL                  *llr_in,
                      TL                        *llr_out) {
  static const TI N_GROPUS = 3; // Cordaro-Wagner: three groups of nodes
  // Messages from check to variable nodes
  Matrix<TL> r_msg = tng.template matrix_r<TL> ();

  // Messages from variable to check nodes: signs and log-magnitudes are stored
  // separately. Note that log-magnitudes are required to avoid products of
  // hyperbolic tangents
  Matrix<TL> q_ltanh      = tng.template matrix_r<TL> ();
  Matrix<uint8_t> q_signs = tng.template matrix_r<uint8_t> ();

  std::vector<TL> sum_ltanh(N_GROPUS, 0);
  std::vector<uint8_t> sum_sign(N_GROPUS, 0);

  TL *r_buf             = r_msg.buf();
  TL *q_ltanh_buf       = q_ltanh.buf();
  uint8_t  *q_signs_buf = q_signs.buf();
  const TI *col_idx     = tng.col_idx_matrix.buf();

  // Initialize input LLRs in the form of logtanh(|x|), sign(x)
  // Initialize: Compute
  std::vector<TL> llr_in_log      = std::vector<TL>(tng.n, 0);
  std::vector<uint8_t> llr_in_sgn = std::vector<uint8_t>(tng.n, 0);

  for (TI i = 0; i < tng.n; i++) {
    llr_in_log[i] = logtanh<TL>(std::abs(llr_in[i]));
    llr_in_sgn[i] = llr_in[i] < 0;
  }

  // Initialize: Propagate
  for (unsigned int j = 0; j < tng.nzc; j++) {
    TI col_id = col_idx[j];
    q_signs_buf[j] = llr_in_sgn[col_id];
    q_ltanh_buf[j] = llr_in_log[col_id];
  }

  // Decoding iterations
  for (unsigned int loop = 0; loop < settings.n_iterations; ++loop) {
    std::copy(llr_in, llr_in + tng.n, llr_out);

    // Update R messages
    for (TI j = 0; j < tng.m; j++) {
      for (TI i = 0; i < N_GROPUS; i++ ) {
        sum_ltanh[i] = 0;
        sum_sign[i]  = 0;
      }

      // Get row pointers once
      TL *r_msg_row         = r_msg.row(j);
      TL *q_ltanh_row       = q_ltanh.row(j);
      uint8_t  *q_signs_row = q_signs.row(j);
      const TI *col_idx     = tng.col_idx_matrix.row(j);

      // LLR sign sums must be equal in all three groups
      std::vector<int> pc_signs(N_GROPUS, 0);

      for (TI k = 0; k < tng.row_weight[j]; k++) {
        TI group = k % N_GROPUS;

        sum_ltanh[group] += q_ltanh_row[k];
        sum_sign[group]  ^= q_signs_row[k];
      }

      // Out-of-group values (logtanh(atanh(Y) + atanh(Z)))
      std::vector<TL>  sum_ltanh_og(N_GROPUS, 0);
      std::vector<int> sum_sign_og(N_GROPUS, 0);

      for (TI i = 0; i < N_GROPUS; i++) {
        std::pair<TL, TI> og_term = gldpc_sp_cnop(
          sum_ltanh[(i + 1) % N_GROPUS], sum_sign[(i + 1) % N_GROPUS],
          sum_ltanh[(i + 2) % N_GROPUS], sum_sign[(i + 2) % N_GROPUS]
          );
        sum_ltanh_og[i] = og_term.first;
        sum_sign_og[i]  = og_term.second;
      }

      for (TI k = 0; k < tng.row_weight[j]; k++) {
        TI group = k % N_GROPUS;

        // Write an update properly here!
        uint8_t sign = sum_sign[group] ^ q_signs_row[k];
        sign ^= sum_sign_og[group];
        TL log_value = sum_ltanh[group] + sum_ltanh_og[group] - q_ltanh_row[k];
        r_msg_row[k] = (1 - 2 * sign) * logtanh<TL>(log_value);

        // Update output LLRs affected by current parity check
        const TI col_id = col_idx[k];
        llr_out[col_id] += r_msg_row[k];
      }
    } // Loop over check nodes

    // Early stop if the syndrome is OK
    // NOTE: Check syndrome works only under all-zero codeword assumption
    if (check_syndrome(tng, settings, llr_out)) {
      return loop + 1;
    }

    // Update Q messages
    for (unsigned int j = 0; j < tng.nzc; j++) {
      TI col_id = col_idx[j];
      TL val    = llr_out[col_id] - r_buf[j];
      q_signs_buf[j] = val < 0;
      q_ltanh_buf[j] = logtanh(std::abs(val));
    }
  } // Loop over decoding iterations
  return settings.n_iterations;
}

// Output bit error rate estimation
#ifdef OUTPUT_SANITY_CHECK
template<typename TL>
void llr_sanity_check(const DecoderSettings<TL>& settings,
                      const TL                  *llr_out) {
  // Check NaN and infinite values
  for (unsigned int i = 0; i < settings.block_length; i++) {
    if (std::isinf(llr_out[i]) || std::isnan(llr_out[i])) {
      std::cerr << "Infinite or NaN values detected in the decoder output.";
      std::cerr << std::endl;
      return;
    }
  }
}

#endif // ifdef OUTPUT_SANITY_CHECK

template<typename TL>
double output_ber(const DecoderSettings<TL>& settings,
                  const TL                  *llr_out,
                  const uint8_t             *tx_bits,
                  unsigned int               got_iter) {
#ifdef OUTPUT_SANITY_CHECK
  llr_sanity_check(settings, llr_out);
#endif // ifdef OUTPUT_SANITY_CHECK

  if (settings.is_azcw && (got_iter < settings.n_iterations)) {
    // For all-zero codeword, early convergence means no error
    // Check the remaining part of the codeword
    return 0.0;
  }

  // Define limits
  unsigned int limit = settings.block_length;

  if (settings.is_systematic) {
    limit = settings.get_inf_bits_count();
  }

  unsigned int be_cum = 0;

  for (unsigned int i = 0; i < limit; i++) {
    // Zero LLR meas no convergence
    be_cum += (llr_out[i] == 0) || ((llr_out[i] < 0) != tx_bits[i]);
  }
  return double(be_cum) / double(limit);
}

#endif  // DECODER_H_
