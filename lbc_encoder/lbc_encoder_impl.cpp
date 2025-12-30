#include <cstdint>
#include <iostream>

// Compress all uint8_t bit vectors to uint32_t bit mask
// All indices are compressed by a factor 32 ( 1 << 5)
#define COMPRESS(X) (((X) >> 5))
#define OFFSET(X) (((X) & 0x0000001f))
#define REDUCED_SIZE(X) ((COMPRESS(X) + (OFFSET(X) ? 1 : 0)))

/// Specify the number of ones in the binary representation of a number
uint32_t bit_width(uint32_t x) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0f0f0f0f;
  x = x + (x >> 8);
  x = x + (x >> 16);
  return x & 0x0000003f;
}

extern "C"
void compress_generator(
  uint8_t  *gen_in,  // Input binary generator matrix
  uint32_t *gen_out, // Output compressed generator matrix
  uint32_t  k,       // The number of information bits
  uint32_t  n        // Code length
  ) {
  // Evaluate the size of compressed matrix
  uint32_t k_reduced = REDUCED_SIZE(k);

  // Perform compression (bitmask instead of array)
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k; j++) {
      gen_out[i * k_reduced + COMPRESS(j)] |= gen_in[i * k + j] << OFFSET(j);
    }
  }
}

extern "C"
void generator_multiply(
  uint32_t *gen, // Compressed generator matrix
  uint8_t  *iwd, // information word (binary vector of length n)
  uint8_t  *cwd, // Output codeword
  uint32_t  k,   // Information word length
  uint32_t  n    // Codeword length
  ) {
  uint32_t k_reduced = REDUCED_SIZE(k);

  // Allocate memory for compressed information word and extended codeword
  uint32_t *iwd_c = new uint32_t[k_reduced];

  for (unsigned int i = 0; i < k_reduced; i++) {
    iwd_c[i] = 0;
  }
  uint32_t *cwd_ext = new uint32_t[n];

  for (unsigned int i = 0; i < n; i++) {
    cwd_ext[i] = 0;
  }

  // Compress information word
  for (uint32_t i = 0; i < k; i++) {
    iwd_c[COMPRESS(i)] |= iwd[i] << OFFSET(i);
  }

  // Perform generator matrix multiplication
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < k_reduced; j++) {
      cwd_ext[i] ^= gen[i * k_reduced + j] & iwd_c[j];
    }
  }

  // Copy extended codeword to uint8_t* vector
  for (uint32_t i = 0; i < n; i++) {
    cwd[i] = bit_width(cwd_ext[i]) & 1;
  }

  // Clean allocated memory
  delete[] iwd_c;
  delete[] cwd_ext;
}
