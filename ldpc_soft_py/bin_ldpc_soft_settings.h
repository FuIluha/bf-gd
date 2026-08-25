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

#ifndef LDPC_SETTINGS_H_
#define LDPC_SETTINGS_H_
#include <cstdint>


template<typename TL>
struct DecoderSettings {
  /// Block length, required to define array boundaries
  uint32_t block_length;

  /// The number of parity checks, required to define array boundaries
  uint32_t n_checks;

  /**
   * If true, the code is systematic, information bits are the first k bits.
   * In this case, the bit error rate is evaluated using the first k bits
   *  */
  bool is_systematic;

  // Flag indicating early termination (syndrome convergence)
  bool early_termination;

  /**
   * Flag indicating that the simulations assume AZCW' assumption
   * 'AZCW: all-zeros codeword */

  bool is_azcw;

  // Maximum number of decoding iterations
  uint32_t n_iterations;

  /**
   * Per-column scale array (for min-sum).
   * By default, only the first number is used,
   * and the scale is shared between all variable nodes (columns).
   * To use full-support, compile with -DOFFSETS_ENABLED flag.
   */
  TL *scale_array;

  /// Per-column offset array (used with -DOFFSETS_ENABLED only)
  TL *offset_array;

  unsigned int get_inf_bits_count() const {
    return block_length - n_checks;
  };
};


#endif // ifndef LDPC_SETTINGS_H_
