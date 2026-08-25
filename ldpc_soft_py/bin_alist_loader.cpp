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

#include <cmath>
#include <vector>
#include <cstdint>

#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "./bin_tanner_graph.h"

// Alist loader implementation
std::vector<unsigned int>read_line(std::ifstream& fp,
                                   unsigned int   expected_len,
                                   std::string    action) {
  std::string line;

  if (!std::getline(fp, line)) {
    std::cerr << "File corrupted. Failed at " << action << ". Exiting." <<
      std::endl;
    return std::vector<unsigned int>();
  }
  std::vector<unsigned int> line_vec;
  std::istringstream ss(line);
  unsigned int num;

  while (ss >> num) {
    line_vec.push_back(num);
  }

  if (line_vec.size() != expected_len) {
    std::cerr << "Line length mismatch at " << action << "." << std::endl;
    return std::vector<unsigned int>();
  }
  return line_vec;
}

template<typename TI>
void* fill_alist(unsigned int   n,
                 unsigned int   m,
                 std::ifstream& fp) {
  std::vector<unsigned int> data = read_line(fp, 2,
                                             "reading row/column max weights");

  if (!data.size()) return 0;
  TI cmax = data[0];
  TI rmax = data[1];

  std::vector<unsigned int> col_weights = read_line(fp, n,
                                                    "reading column weights");

  if (!col_weights.size()) return 0;

  std::vector<unsigned int> row_weights = read_line(fp, m,
                                                    "reading row weights");

  if (!row_weights.size()) return 0;

  TannerGraph<TI> *tng = new TannerGraph<TI>(row_weights, col_weights);

  // Per-column representation is skipped
  for (unsigned int i = 0; i < n; i++) {
    data = read_line(fp, cmax, "reading matrix column");

    if (!data.size()) return 0;
  }

  // Get data from per-row matrix representation
  std::vector<TI> count = std::vector<TI>(n);
  std::fill(count.begin(), count.end(), 0);

  for (unsigned int i = 0; i < m; i++) {
    data = read_line(fp, rmax, "reading matrix row");

    if (!data.size()) return 0;

    for (unsigned int j = 0; j < tng->row_weight[i]; j++) {
      TI v = data[j] - 1;
      tng->col_idx_matrix.row(i)[j]        = v;
      tng->row_idx_matrix.row(v)[count[v]] = i;
      count[v]++;
    }
  }
  return tng;
}

void* load_alist(const char *filename, uint32_t n, uint32_t m) {
  std::ifstream fp(filename);

  if (fp.fail()) {
    std::cout << "ERROR: Cannot open file " << filename << "." << std::endl;
    return 0;
  }
  std::vector<unsigned int> data = read_line(fp, 2, "reading matrix size");

  if (!data.size()) return 0;

  if (n != data[0]) {
    std::cerr << "Column count differs from expected value" << std::endl;
  }

  if (m != data[1]) {
    std::cerr << "Row count differs from expected value" << std::endl;
  }

  // Start type split at this point!
  if (n < std::numeric_limits<uint16_t>::max()) {
    return fill_alist<uint16_t>(n, m, fp);
  } else {
    return fill_alist<uint32_t>(n, m, fp);
  }
}
