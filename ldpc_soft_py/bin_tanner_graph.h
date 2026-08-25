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

#ifndef TANNER_GRAPH_H_
#define TANNER_GRAPH_H_
#include <vector>
#include <utility>
#include <numeric>

#include "./matrix.h"

// Tanner graph representation is templated by index type. For block lengths
// shorter than 0xffff it is recommended to use uint16_t index types
template<typename T>
struct TannerGraph {
  /**
   * Initialize an empty Tanner graph representation
   * The constructor generates empty matrices (filled with zeros),
   * which are filled by load_alist() function.
   */
  TannerGraph(const std::vector<unsigned int>& row_weights,
              const std::vector<unsigned int>& col_weights) :
    n(col_weights.size()),
    m(row_weights.size()),
    nzc(std::accumulate(row_weights.begin(), row_weights.end(), 0)),

    row_weight(std::vector<T>(m)),
    row_offset(get_cumsum(row_weights)),

    // To be filled later
    col_idx_matrix(matrix_r<T>(0)),

    col_weight(std::vector<T>(n)),
    col_offset(get_cumsum(col_weights)),

    // To be filled later:
    row_idx_matrix(matrix_c<T>(0)) {
    // Need assign to convert types
    col_weight.assign(col_weights.begin(), col_weights.end());
    row_weight.assign(row_weights.begin(), row_weights.end());
  };

  std::vector<unsigned int>get_cumsum(
    const std::vector<unsigned int>vals) const {
    unsigned int length              = vals.size();
    std::vector<unsigned int> cumsum = std::vector<unsigned int>(length);
    unsigned int cumval              = 0;

    for (unsigned int i = 0; i < length; i++) {
      cumsum[i] = cumval;
      cumval   += vals[i];
    }
    return cumsum;
  }

  // Matrix factory. Row-expanded and column-expanded
  template<typename MT>
  Matrix<MT>matrix_r() const {
    return Matrix<MT>(m, nzc, row_offset);
  }

  template<typename MT>
  Matrix<MT>matrix_r(MT fill_val) const {
    return Matrix<MT>(m, nzc, row_offset, fill_val);
  }

  template<typename MT>
  Matrix<MT>matrix_c() const {
    return Matrix<MT>(n, nzc, col_offset);
  }

  template<typename MT>
  Matrix<MT>matrix_c(MT fill_val) const {
    return Matrix<MT>(n, nzc, col_offset, fill_val);
  }

  /// Block length (the number of parity check matrix columns)
  unsigned int n;

  /// The number of parity checks
  unsigned int m;

  /// The number of nonzero elements in the Tanner graph
  unsigned int nzc; // Non-zero count

  /// Row weights, a vector of length m
  std::vector<T>row_weight;

  /// Row-pointer offsets for row-unrolled matrix
  std::vector<unsigned int>row_offset;

  /// Column indices for each parity check. Indexing (#check, #nz-index)
  Matrix<T>col_idx_matrix;

  // Fields below are used only in vertical layering

  /// Column weights, a vector of length n
  std::vector<T>col_weight;

  /// Column weights, a vector of length n
  std::vector<unsigned int>col_offset;

  /// Row-pointer offsets for column-unrolled matrix
  Matrix<T>row_idx_matrix;
};
#endif // ifndef TANNER_GRAPH_H_
