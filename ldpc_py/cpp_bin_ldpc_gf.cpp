#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Classic gradient flow (GF) decoder, Euler discretization:
//   x <- x - eta * (x - y + gamma * grad h(x)),  x(0) = 0
//   h(x) = alpha * sum_j (x_j^2 - 1)^2 + beta * sum_i (prod_{j in A(i)} x_j - 1)^2
class CppGfDecoder {
 public:
  CppGfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double alpha,
      double beta,
      double gamma,
      double eta,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        alpha_(alpha),
        beta_(beta),
        gamma_(gamma),
        eta_(eta),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        gradient_(block_length) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
    std::fill(x_.begin(), x_.end(), 0.0);

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      CalculateGradient();
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        x_[variable] -= eta_ * (
            x_[variable] - static_cast<double>(input[variable])
            + gamma_ * gradient_[variable]);
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
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppGfDecoder(
        block_length,
        n_checks,
        n_iterations,
        alpha,
        beta,
        gamma,
        eta,
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
