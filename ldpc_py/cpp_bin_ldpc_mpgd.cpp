#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Same two synchronous projected phases as BinLdpcMpgdDecoder.
class MpgdDecoder {
 public:
  MpgdDecoder(uint32_t n, uint32_t m, uint32_t iterations, double alpha,
              double eta_r, double eta_v, double tau_r, double tau_v,
              const uint32_t* variables, const uint32_t* offsets)
      : n_(n), m_(m), iterations_(iterations), alpha_(alpha), eta_r_(eta_r),
        eta_v_(eta_v), tau_r_(tau_r), tau_v_(tau_v),
        variables_(variables, variables + offsets[m]), offsets_(offsets, offsets + m + 1),
        s_(offsets[m]), u_(offsets[m]), prefix_(offsets[m]), gradient_(offsets[m]),
        channel_(n), x_(n), totals_(n) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
    for (uint32_t v = 0; v < n_; ++v) {
      channel_[v] = alpha_ * static_cast<double>(input[v]);
      if (!std::isfinite(channel_[v])) return Invalid();
      x_[v] = static_cast<double>(input[v]);
    }
    for (uint32_t e = 0; e < s_.size(); ++e) {
      s_[e] = Clip(std::tanh(channel_[variables_[e]] / 2.0));
    }
    for (uint32_t iteration = 0; iteration < iterations_; ++iteration) {
      if (Satisfied()) {
        Write(output);
        return iteration;
      }

      // Prefix/suffix products avoid division by a recipient or a zero input.
      for (uint32_t c = 0; c < m_; ++c) {
        double prefix = 1.0;
        for (uint32_t e = offsets_[c]; e < offsets_[c + 1]; ++e) {
          prefix_[e] = prefix;
          prefix *= s_[e];
        }
        double suffix = 1.0;
        for (uint32_t e = offsets_[c + 1]; e-- > offsets_[c];) {
          gradient_[e] = prefix_[e] * suffix;
          suffix *= s_[e];
        }
      }
      std::fill(totals_.begin(), totals_.end(), 0.0);
      for (uint32_t e = 0; e < s_.size(); ++e) {
        const double direction = gradient_[e]
            - (tau_r_ == 0.0 ? 0.0 : tau_r_ * std::atanh(s_[e]));
        const double next = s_[e] + eta_r_ * direction;
        if (!std::isfinite(next)) return Invalid();
        u_[e] = Clip(next);
        totals_[variables_[e]] += u_[e];
      }
      // All VN directions read the same completed CN phase.
      for (uint32_t e = 0; e < s_.size(); ++e) {
        const double direction = channel_[variables_[e]] + totals_[variables_[e]] - u_[e]
            - (tau_v_ == 0.0 ? 0.0 : tau_v_ * std::atanh(u_[e]));
        const double next = u_[e] + eta_v_ * direction;
        if (!std::isfinite(next)) return Invalid();
        s_[e] = Clip(next);
      }
      std::fill(x_.begin(), x_.end(), 0.0);
      for (uint32_t e = 0; e < s_.size(); ++e) x_[variables_[e]] += s_[e];
      for (uint32_t v = 0; v < n_; ++v) {
        if (x_[v] == 0.0) x_[v] = input[v] >= 0 ? 1.0 : -1.0;
      }
    }
    Write(output);
    return iterations_;
  }

 private:
  static uint32_t Invalid() { return std::numeric_limits<uint32_t>::max(); }
  static double Clip(double v) { return std::max(-0.999999, std::min(0.999999, v)); }
  bool Satisfied() const {
    for (uint32_t c = 0; c < m_; ++c) {
      bool odd = false;
      for (uint32_t e = offsets_[c]; e < offsets_[c + 1]; ++e) {
        odd ^= x_[variables_[e]] < 0.0;
      }
      if (odd) return false;
    }
    return true;
  }
  template <typename Float>
  void Write(Float* output) const {
    for (uint32_t v = 0; v < n_; ++v) output[v] = static_cast<Float>(x_[v]);
  }

  uint32_t n_, m_, iterations_;
  double alpha_, eta_r_, eta_v_, tau_r_, tau_v_;
  std::vector<uint32_t> variables_, offsets_;
  std::vector<double> s_, u_, prefix_, gradient_, channel_, x_, totals_;
};

extern "C" void* cpp_mpgd_create(
    uint32_t n, uint32_t m, uint32_t iterations, double alpha,
    double eta_r, double eta_v, double tau_r, double tau_v,
    const uint32_t* variables, const uint32_t* offsets) {
  try {
    return new MpgdDecoder(n, m, iterations, alpha, eta_r, eta_v, tau_r, tau_v, variables, offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_mpgd_decode_float32(void* decoder, const float* input, float* output) {
  return static_cast<MpgdDecoder*>(decoder)->Decode(input, output);
}
extern "C" uint32_t cpp_mpgd_decode_float64(void* decoder, const double* input, double* output) {
  return static_cast<MpgdDecoder*>(decoder)->Decode(input, output);
}
extern "C" void cpp_mpgd_free(void* decoder) { delete static_cast<MpgdDecoder*>(decoder); }
