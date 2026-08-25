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

#include <limits>
#include <cstdint>
#include <iostream>
#include "./bin_ldpc_soft_impl.h"
#include "./bin_ldpc_soft_settings.h"

// Specify LDPC decoding commands:
#define DECODE_LDPC_SPA      0 // Sum-product
#define DECODE_LDPC_NOMS     2 // Normalized offset min-sum (NOMS)
#define DECODE_LDPC_NOMS_HL  3 // NOMS, horizontally layered
#define DECODE_LDPC_NOMS_VL  4 // NOMS, vertically layered

// Specify GLDPC decoding commands:
#define DECODE_GLDPC_NOMS_HL 5 // NOMS, horizontally layered
#define DECODE_GLDPC_SPA     6 // Sum-product


/// Structure to be returned to python via ctypes
template<typename TL>
struct Decoder {
  Decoder(const char *alist_path, const DecoderSettings<TL> *settings) :
    settings(settings)
  {
    tng = load_alist(alist_path, settings->block_length, settings->n_checks);
  }

  ~Decoder() {
    if (settings->block_length < std::numeric_limits<uint16_t>::max()) {
      delete static_cast<TannerGraph<uint16_t> *>(tng);
    } else {
      delete static_cast<TannerGraph<uint32_t> *>(tng);
    }

    // LDPC settings are provided externally, do not remove the underlying
    // arrays (initialized by numpy)
  }

  /// Decoder settings
  const DecoderSettings<TL> *settings;

  /// Pointer to Tanner graph representation
  void *tng;
};

void warn_gldpc(bool is_azcw) {
  // Nonzero codeword is not supported by GLDPC
  if (!is_azcw) {
    std::cerr << "WARNING: Non-zero codeword is not supported by GLDPC";
    std::cerr << std::endl;
  }
}

template<typename TL, typename TI>
unsigned int do_decode_soft(unsigned int       command,
                            const Decoder<TL> *ldpc_ptr,
                            TL                *llr_in,
                            TL                *llr_out) {
  TannerGraph<TI> *tng =
    static_cast<TannerGraph<TI> *>(ldpc_ptr->tng);
  const DecoderSettings<TL> *settings = ldpc_ptr->settings;

  switch (command) {
    case DECODE_LDPC_SPA:
      return ldpc_sp<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_LDPC_NOMS:
      return ldpc_noms<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_LDPC_NOMS_HL:
      return ldpc_hl_noms<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_LDPC_NOMS_VL:
      return ldpc_vl_noms<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_GLDPC_NOMS_HL:
      warn_gldpc(settings->is_azcw);
      return gldpc_hl_noms<TL, TI>(*tng, *settings, llr_in, llr_out);
    case DECODE_GLDPC_SPA:
      warn_gldpc(settings->is_azcw);
      return gldpc_sp<TL, TI>(*tng, *settings, llr_in, llr_out);
  }
  std::cerr << "WARNING: Unknown decoding command: " << command << std::endl;
  return 0;
}

template<typename TL>
unsigned int decode_soft(unsigned int       command,
                         const Decoder<TL> *ldpc_ptr,
                         TL                *llr_in,
                         TL                *llr_out) {
  if (ldpc_ptr->settings->block_length > std::numeric_limits<uint16_t>::max()) {
    return do_decode_soft<TL, uint32_t>(command, ldpc_ptr, llr_in, llr_out);
  } else {
    return do_decode_soft<TL, uint16_t>(command, ldpc_ptr, llr_in, llr_out);
  }
}

// float32 external API
extern "C"
void* init_ldpc_float32(const char                   *alist_path,
                        const DecoderSettings<float> *settings) {
  return new Decoder<float>(alist_path, settings);
}

extern "C"
void free_ldpc_float32(void *ldpc_ptr) {
  delete (static_cast<Decoder<float> *>(ldpc_ptr));
}

extern "C"
unsigned int decode_siso_float32(unsigned int command,
                                 void        *ldpc_void_p,
                                 float       *llr_in,
                                 float       *llr_out) {
  Decoder<float> *ldpc_ptr = static_cast<Decoder<float> *>(ldpc_void_p);

  return decode_soft<float>(command, ldpc_ptr, llr_in, llr_out);
}

extern "C"
double output_ber_float32(const DecoderSettings<float> *settings,
                          const float                  *llr_out,
                          const uint8_t                *tx_bits,
                          unsigned int                  got_iter) {
  return output_ber<float>(*settings, llr_out, tx_bits, got_iter);
}

// float64 external API
extern "C"
void* init_ldpc_float64(char                          *alist_path,
                        const DecoderSettings<double> *settings) {
  return new Decoder<double>(alist_path, settings);
}

extern "C"
void free_ldpc_float64(void *ldpc_ptr) {
  delete (static_cast<Decoder<double> *>(ldpc_ptr));
}

extern "C"
unsigned int decode_siso_float64(unsigned int command,
                                 void        *ldpc_void_p,
                                 double      *llr_in,
                                 double      *llr_out) {
  Decoder<double> *ldpc_ptr = static_cast<Decoder<double> *>(ldpc_void_p);

  return decode_soft<double>(command, ldpc_ptr, llr_in, llr_out);
}

extern "C"
double output_ber_float64(const DecoderSettings<double> *settings,
                          const double                  *llr_out,
                          const uint8_t                 *tx_bits,
                          unsigned int                   got_iter) {
  return output_ber<double>(*settings, llr_out, tx_bits, got_iter);
}
