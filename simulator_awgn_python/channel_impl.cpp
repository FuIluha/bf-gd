#include <cstdint>
#include <iostream>
#include <cmath>


#define BPSK_MOD(X) ((1 - 2.0 * (X)))

// Supported channel models (real degrees of freedom),
// modulation and demodulation

template<typename TL>
struct BpskChannel {
  TL s2i; // 2.0 / pow(sigma, 2)
  TL si;  // 2.0 / sigma
  BpskChannel(TL sigma) :
    s2i(2.0 / sigma / sigma),
    si(2.0 / sigma)
  {}

  inline static TL modulate(const uint8_t& bit) {
    return 1 - 2.0 * bit;
  }

  inline void run_sumbol(const uint8_t *bits, const TL& gng_sample,
                         TL *llr) const {
    *llr = s2i * modulate(*bits) + si * gng_sample;
  }
};

template<typename TL>
struct Pam4Channel {
  TL s2i; // 2.0 / pow(sigma, 2)
  TL si;  // 2.0 / sigma
  TL dl;  // Delta for LLR, equals to - 4.0 / pow(sigma, 2)
  TL dle; // Exponent of the delta above
  Pam4Channel(TL sigma) :
    s2i(2.0 / sigma / sigma),
    si(2.0 / sigma),
    dl(-4 / sigma / sigma),
    dle(exp(dl))
  {}

  /**
     Gray-coded PAM-4 modulation for a single symbol
   */
  inline static TL modulate(const uint8_t *bits) {
    return (1 - 2.0 * bits[0]) * (3 - 2.0 * bits[1]);
  }

  /*
     Run AWGN channel for a single symbol.
     LLRs are derived using total probaility formula
   */
  inline void run_sumbol(const uint8_t *bits, const TL& gng_sample,
                         TL *llr) const {
    TL lbps = s2i * modulate(bits) + si * gng_sample;
    TL lexp = exp(lbps);
    llr[0] = lbps +      log((1 + dle * lexp)     / (1 + dle / lexp));
    llr[1] = lbps + dl + log((1 + exp(-3 * lbps)) / (1 + 1   / lexp));
  }
};


// Output sttistics: bit error rate (BER) and symbol error rate (SER)
template<typename TL>
inline void
symbol_stats_4bps(const uint8_t *tx_bits,
                  const TL      *llr,
                  uint32_t     & ber_cum,
                  uint32_t     & ser_cum) {
  uint32_t e1 = tx_bits[0] != (llr[0] < 0);
  uint32_t e2 = tx_bits[1] != (llr[1] < 0);
  uint32_t e3 = tx_bits[2] != (llr[2] < 0);
  uint32_t e4 = tx_bits[3] != (llr[3] < 0);

  ber_cum += e1 + e2 + e3 + e4;
  ser_cum += (e1 | e2 | e3 | e4);
}

template<typename TL>
inline void
symbol_stats_2bps(const uint8_t *tx_bits,
                  const TL      *llr,
                  uint32_t     & ber_cum,
                  uint32_t     & ser_cum) {
  uint32_t e1 = tx_bits[0] != (llr[0] < 0);
  uint32_t e2 = tx_bits[1] != (llr[1] < 0);

  ber_cum += e1 + e2;
  ser_cum += (e1 | e2);
}

// Real-valued channels

template<typename TL>
void run_bpsk_channel(
  const uint8_t *tx_bits,
  const TL      *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  TL            *llr,
  TL            *stats
  ) {
  BpskChannel<TL> chan = BpskChannel<TL>(sigma_noise);
  uint32_t ber_cum     = 0;

  for (unsigned int i = 0; i < nbits; i++)
  {
    chan.run_sumbol(tx_bits + i, gvals[i], llr + i);
    ber_cum += ((llr[i] < 0) != tx_bits[i]);
  }
  stats[0] = (TL)ber_cum / nbits;
  stats[1] = stats[0];
}

template<typename TL>
void run_pam4_channel(
  const uint8_t *tx_bits,
  const TL      *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  TL            *llr_channel,
  TL            *stats
  ) {
  // QAM-16 is equal to the orthogonal product of two PAM-4 modulations
  uint32_t ber_cum      = 0;
  uint32_t ser_cum      = 0;
  unsigned int    bpdof = 2; // Bits per degree of freedom
  Pam4Channel<TL> chan  = Pam4Channel<TL>(sigma_noise);

  for (unsigned int i = 0; i < nbits / bpdof; i++)
  {
    unsigned int idx = bpdof * i;
    chan.run_sumbol(tx_bits + idx, gvals[i], llr_channel + idx);
    symbol_stats_2bps<TL>(tx_bits + idx, llr_channel + idx, ber_cum, ser_cum);
  }
  stats[0] = (TL)ber_cum / nbits;
  stats[1] = (TL)ser_cum / (nbits / 2);
}

// Complex-valued channels

template<typename TL>
void run_qpsk_channel(
  const uint8_t *tx_bits,
  const TL      *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  TL            *llr,
  TL            *stats
  ) {
  unsigned int    bps  = 2; // Bits per symbol
  BpskChannel<TL> chan = BpskChannel<TL>(sigma_noise);

  // Statistics
  uint32_t ber_cum = 0;
  uint32_t ser_cum = 0;

  for (unsigned int i = 0; i < nbits; i += 2)
  {
    // I-component
    chan.run_sumbol(tx_bits + i,     gvals[i],     llr + i);

    // Q-component
    chan.run_sumbol(tx_bits + i + 1, gvals[i + 1], llr + i + 1);

    // SER for both I and Q
    symbol_stats_2bps<TL>(tx_bits + i, llr + i, ber_cum, ser_cum);
  }
  stats[0] = (TL)ber_cum / nbits;
  stats[1] = (TL)ser_cum / (nbits / bps);
}

template<typename TL>
void run_qam16_channel(
  const uint8_t *tx_bits,
  const TL      *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  TL            *llr,
  TL            *stats
  ) {
  // QAM-16 is equal to the orthogonal product of two PAM-4 modulations
  Pam4Channel<TL> chan  = Pam4Channel<TL>(sigma_noise);
  unsigned int    bpdof = 2; // Bits per degree of freedom
  unsigned int    bps   = 4; // Bits per symbol

  // Statistics
  uint32_t ber_cum = 0;
  uint32_t ser_cum = 0;

  for (unsigned int i = 0; i < nbits / bpdof; i += 2)
  {
    unsigned int idx = bpdof * i;

    // I-component
    chan.run_sumbol(tx_bits + idx,         gvals[i],     llr + idx);

    // Q-component
    chan.run_sumbol(tx_bits + idx + bpdof, gvals[i + 1], llr + idx + bpdof);

    // SER for both I and Q
    symbol_stats_4bps<TL>(tx_bits + idx, llr + idx, ber_cum, ser_cum);
  }
  stats[0] = (TL)ber_cum / nbits;
  stats[1] = (TL)ser_cum / (nbits / bps);
}

// API:


extern "C"
void run_bpsk_channel_f32(
  const uint8_t *tx_bits,
  const float   *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  float         *llr,
  float         *stats
  ) {
  run_bpsk_channel<float>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_pam4_channel_f32(
  const uint8_t *tx_bits,
  const float   *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  float         *llr,
  float         *stats
  ) {
  run_pam4_channel<float>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_qpsk_channel_f32(
  const uint8_t *tx_bits,
  const float   *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  float         *llr,
  float         *stats
  ) {
  run_qpsk_channel<float>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_qam16_channel_f32(
  const uint8_t *tx_bits,
  const float   *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  float         *llr,
  float         *stats
  ) {
  run_qam16_channel<float>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_bpsk_channel_f64(
  const uint8_t *tx_bits,
  const double  *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  double        *llr,
  double        *stats
  ) {
  run_bpsk_channel<double>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_pam4_channel_f64(
  const uint8_t *tx_bits,
  const double  *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  double        *llr,
  double        *stats
  ) {
  run_pam4_channel<double>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_qpsk_channel_f64(
  const uint8_t *tx_bits,
  const double  *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  double        *llr,
  double        *stats
  ) {
  run_qpsk_channel<double>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}

extern "C"
void run_qam16_channel_f64(
  const uint8_t *tx_bits,
  const double  *gvals,
  unsigned int   nbits,
  double         sigma_noise,
  double        *llr,
  double        *stats
  ) {
  run_qam16_channel<double>(tx_bits, gvals, nbits, sigma_noise, llr, stats);
}
