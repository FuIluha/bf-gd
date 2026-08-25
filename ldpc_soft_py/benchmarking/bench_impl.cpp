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

// Template scripts for benchmarking further optimizatons

#include <vector>
#include <random>
#include <cstdio>
#include <cmath>
#include <chrono>

#include "./bench_impl.h"
#ifdef NEW_IMPL
#include "./bin_tanner_graph_fast.h"
#include "./bin_ldpc_soft_impl_fast.h"
#else
#include "./bin_tanner_graph.h"
#include "./bin_ldpc_soft_impl.h"
#endif

// Defined in bin alist loader
void* load_alist(const char *filename, unsigned int& n, unsigned int& m);

template<typename T>
std::vector<T> generate_llr_samples(unsigned int seed, size_t n) {
    std::vector<T> samples;
    samples.reserve(n);
    
    // Random number generator
    // std::random_device rd;
    std::mt19937 gen(seed);
    
    // Standard normal distribution (mean=0, stddev=1)
    std::normal_distribution<T> dist(0.0f, 1.0f);
    
    // Generate samples
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(dist(gen));
    }
    // Define LLRs from samples
        for (uint16_t i = 0; i < n; i ++){
        samples[i] = 2 * (1 + SIGMA_NOISE * samples[i]) / pow(SIGMA_NOISE, 2);
    }
    return samples;
}

void debug_print(float * llr_out, unsigned int blklen, unsigned int n_iter) {
    printf("%d iter: ", n_iter);
    for (unsigned int i = 0; i < blklen; i ++) {
        printf("%+1.4f ", llr_out[i]);
    }
    printf("\n");
}

unsigned int
single_test(unsigned int seed, const TannerGraph<uint16_t> & tng, const DecParams<float, uint16_t> & params, int dec) {
    // Use floa32 for performance tests
    std::vector<float> llr_in = generate_llr_samples<float> (seed, tng.n);
    float * llr_out = new float[tng.n];
    unsigned int n_iter = 0;
#ifndef GLDPC
    if (dec == 0) {
      n_iter = ldpc_hl_noms<float, uint16_t> (tng, params, llr_in.data(), true, llr_out);
    } else if (dec == 1) {
      n_iter = ldpc_noms<float, uint16_t> (tng, params, llr_in.data(), true, llr_out);
    } else if (dec == 2) {
      n_iter = ldpc_sp<float, uint16_t> (tng, params, llr_in.data(), true, llr_out);
    } else if (dec == 3) {
        n_iter = ldpc_vl_noms<float, uint16_t> (tng, params, llr_in.data(), true, llr_out);
    } 
#else
    uint16_t rmax = *std::min_element(tng.row_weight.begin(), tng.row_weight.end());
    if (dec == 4) {
      if ((tng.n >= 2 * tng.m) && (rmax >= 3)) {
        n_iter = gldpc_hl_noms<float, uint16_t> (tng, params, llr_in.data(), llr_out);
      }
      else {
        n_iter = 0;
        for (unsigned int i = 0; i < tng.n; i ++) {
            llr_out[i] = 0.0;
        }
      }
    } else if (dec == 5) {
      if ((tng.n >= 2 * tng.m) && (rmax >= 3)) {
        n_iter = gldpc_sp<float, uint16_t> (tng, params, llr_in.data(), llr_out);
      } 
      else {
        printf("Can not generate GLDPC output\n");
        n_iter = 0;
        for (unsigned int i = 0; i < tng.n; i ++) {
            llr_out[i] = 0.0;
        }
      }
    }
#endif
    // Uncomment to check diff
#ifdef PRINT_OUTPUT
    debug_print(llr_out, tng.n, n_iter);
#endif
    delete [] llr_out;
    llr_in.clear();
    return n_iter;
}

void bench_decoder(unsigned int                       decoder, 
                   const TannerGraph<uint16_t>      & tng,
                   const DecParams<float, uint16_t> & params,
                   const char                        *msg) {
  std::vector<unsigned int> n_iterations = std::vector<unsigned int> (N_TESTS);
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < N_TESTS; i ++) {
    n_iterations[i] = single_test(i, tng, params, decoder);
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  double iter_mean = 0;
  for (unsigned int i = 0; i < N_TESTS; i ++) {
    iter_mean += (double)n_iterations[i];
  }
  iter_mean /= N_TESTS;
  printf("%s %1.4f microseconds / test. Avg. iterations: %1.3f\n", msg, (double) duration.count() / N_TESTS, iter_mean);
}


void bench(const char * alist_path) {
    unsigned int n;
    unsigned int m;
    printf("Benchmarking %s\n", alist_path);
    void * raw_ptr = load_alist(alist_path, n, m);
    // We are sure that the index type is uint16, because the tested code is rather short
    TannerGraph<uint16_t> * tng = static_cast<TannerGraph<uint16_t> *> (raw_ptr);
    // Generate decosing parameters
    std::vector<float> scales = std::vector<float> (tng->n);
    std::vector<float> offsets = std::vector<float> (tng->n);
    std::vector<uint16_t> row_seq = std::vector<uint16_t> (tng->m);
    for (uint16_t i = 0; i < tng->m; i ++) {
        row_seq[i] = i;
    }
    for (uint16_t i = 0; i < tng->n; i ++) {
        offsets[i] = 0;
        scales[i] = 0.75;
    }

    DecParams<float, uint16_t>  * params = new DecParams<float, uint16_t>(
        50, // The number of decoding iterations
        true, // Terminate if syndrome converged,
        scales,
        offsets,
        row_seq
    );
    printf("Loaded alist file with n = %d, m = %d\n", tng->n, tng->m);
#ifndef GLDPC
    bench_decoder(0, *tng, *params, "Horizonral layered min sum:");
    bench_decoder(1, *tng, *params, "Flooding min sum:          ");
    bench_decoder(2, *tng, *params, "Flooding sum product:      ");
    bench_decoder(3, *tng, *params, "Vertically layered min sum:");
#else
    bench_decoder(4, *tng, *params, "GLDPC Horiz. layered NOMS: ");
    bench_decoder(5, *tng, *params, "GLDPC flooding sum prod.:  ");
#endif
    delete tng;
    delete params;
}


int main() {
#ifndef GLDPC
    bench("./codes/ldpc_5g_k120_n600_pcm.alist");
    bench("./codes/ldpc_5g_k960_n1920_pcm.alist");
    bench("./codes/ldpc_5g_k960_n4800_pcm.alist");
    bench("./codes/ldpc_5g_k1600_n1920_pcm.alist");
#else
    bench("codes/pcm_12x48_punc4_factor120.alist");
#endif
}
