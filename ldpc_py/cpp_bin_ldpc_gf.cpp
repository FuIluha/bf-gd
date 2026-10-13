#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Classic gradient flow (GF) decoder, Euler discretization:
//   x <- clip(x - eta * (x - y + gamma * grad h(x)), -theta, theta),
//   x(0) = 0, y or sign(y)
//   h(x) = alpha * sum_j (x_j^2 - 1)^2 + beta * sum_i (prod_{j in A(i)} x_j - 1)^2
class CppGfDecoder {
 public:
  // Initial point x(0); the order matches INIT_MODES in bin_ldpc_gf.py.
  enum InitMode : uint32_t { kInitZero = 0, kInitY = 1, kInitHard = 2 };

  CppGfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double alpha,
      double beta,
      double gamma,
      double eta,
      uint32_t init_mode,
      double theta,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        alpha_(alpha),
        beta_(beta),
        gamma_(gamma),
        eta_(eta),
        init_mode_(init_mode),
        theta_(theta),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        gradient_(block_length) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      const double value = static_cast<double>(input[variable]);
      if (init_mode_ == kInitY) {
        x_[variable] = value;
      } else if (init_mode_ == kInitHard) {
        x_[variable] = value >= 0.0 ? 1.0 : -1.0;
      } else {
        x_[variable] = 0.0;
      }
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      CalculateGradient();
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        x_[variable] -= eta_ * (
            x_[variable] - static_cast<double>(input[variable])
            + gamma_ * gradient_[variable]);
        // std::clamp keeps NaN, so a broken state is still caught below.
        x_[variable] = std::clamp(x_[variable], -theta_, theta_);
      }
      if (!ValuesFit<Float>()) {
        return std::numeric_limits<uint32_t>::max();
      }
    }

    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      output[variable] = static_cast<Float>(x_[variable]);
    }
    return n_iterations_;
  }

 private:
  // Gradient of the code potential energy h(x).
  // The product over a check without one bit is computed without division,
  // so zero components (e.g. the initial point x = 0) are handled exactly.
  void CalculateGradient() {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      const double value = x_[variable];
      gradient_[variable] = 4.0 * alpha_ * (value * value - 1.0) * value;
    }

    for (uint32_t check = 0; check < n_checks_; ++check) {
      uint32_t zero_count = 0;
      double nonzero_product = 1.0;
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const double value = x_[edge_vn_[edge]];
        if (value == 0.0) {
          ++zero_count;
        } else {
          nonzero_product *= value;
        }
      }
      const double full_product = zero_count == 0 ? nonzero_product : 0.0;

      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        const double value = x_[variable];
        double exclusive_product = 0.0;
        if (zero_count == 0) {
          exclusive_product = nonzero_product / value;
        } else if (zero_count == 1 && value == 0.0) {
          exclusive_product = nonzero_product;
        }
        gradient_[variable] += 2.0 * beta_ * (full_product - 1.0) * exclusive_product;
      }
    }
  }

  template <typename Float>
  bool ValuesFit() const {
    const double limit =
        static_cast<double>(std::numeric_limits<Float>::max());
    for (const double value : x_) {
      if (!std::isfinite(value) || std::abs(value) > limit) {
        return false;
      }
    }
    return true;
  }

  uint32_t block_length_;
  uint32_t n_checks_;
  uint32_t n_iterations_;
  double alpha_;
  double beta_;
  double gamma_;
  double eta_;
  uint32_t init_mode_;
  double theta_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<double> x_;
  std::vector<double> gradient_;
};

extern "C" void* cpp_gf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double alpha,
    double beta,
    double gamma,
    double eta,
    uint32_t init_mode,
    double theta,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  if (init_mode > CppGfDecoder::kInitHard || !(theta >= 1.0)) {
    return nullptr;
  }
  try {
    return new CppGfDecoder(
        block_length,
        n_checks,
        n_iterations,
        alpha,
        beta,
        gamma,
        eta,
        init_mode,
        theta,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_gf_decode_float32(
    void* decoder,
    const float* input,
    float* output) {
  return static_cast<CppGfDecoder*>(decoder)->Decode(input, output);
}

extern "C" uint32_t cpp_gf_decode_float64(
    void* decoder,
    const double* input,
    double* output) {
  return static_cast<CppGfDecoder*>(decoder)->Decode(input, output);
}

extern "C" void cpp_gf_free(void* decoder) {
  delete static_cast<CppGfDecoder*>(decoder);
}
