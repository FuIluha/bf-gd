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

#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <utility>

/**
 * Matrix class. Used to perform indexing the messages passed over Tanner graph.
 * Given a parity check matrix, non-zero elements (and correpsonding messages)
 * are indexed by row or column index and the number of nonzero position in the
 * consdered row/column.
 */

// to be able to make a single loop over messages when updateing Q-messages
template<typename T>
class Matrix {
public:

  /**
   * Initialize sparse matrix with the number of rows, columns
   * Specify a default value to fill all elements with
   * @param nrows The number of rows in the matrix
   * @param nz    The number of nonzero in the matrix
   * @param row_offsets Row weightx (cumulative)
   */
  Matrix(unsigned int                     nrows,
         unsigned int                     nz,
         const std::vector<unsigned int>& row_offsets) :
    m_nrows(nrows),
    m_nz(nz),
    m_buf(new T[nz])
  {
    m_row_ptrs = new T *[m_nrows];

    for (unsigned int i = 0; i < m_nrows; i++) {
      m_row_ptrs[i] = m_buf + row_offsets[i];
    }
  }

  /**
   * Constructor with default value
   */
  Matrix(unsigned int                     nrows,
         unsigned int                     nz,
         const std::vector<unsigned int>& row_offsets,
         T                                fill_val) :
    m_nrows(nrows),
    m_nz(nz),
    m_buf(new T[nz]) {
    m_row_ptrs = new T *[m_nrows];

    for (unsigned int i = 0; i < m_nrows; i++) {
      m_row_ptrs[i] = m_buf + row_offsets[i];
    }
    fill_with(fill_val);
  }

  virtual ~Matrix() {
    delete[] m_buf;
    delete[] m_row_ptrs;
  }

  /// Fill all elements of the matrix with given value
  void fill_with(T fill_val) {
    for (unsigned int i = 0; i < m_nz; i++) {
      m_buf[i] = fill_val;
    }
  }

  T* row(unsigned int row_ind) const {
    return m_row_ptrs[row_ind];
  }

  T* buf() const {
    return m_buf;
  }

private:

  /// The number of rows
  unsigned int m_nrows;

  /// The number of nonzero elements
  unsigned int m_nz;

  /// Raw buffer of length nrows X ncols
  T *m_buf;

  // Row start pointers
  T **m_row_ptrs;
};
#endif // ifndef MATRIX_H_
