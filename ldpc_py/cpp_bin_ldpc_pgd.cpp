#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

class CppPgdDecoder {
 public:
  CppPgdDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double learning_rate,
      double learning_rate_decay,
      double momentum,
      double alpha,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        learning_rate_(learning_rate),
        learning_rate_decay_(learning_rate_decay),
        momentum_(momentum),
        alpha_(alpha),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        bipolar_probabilities_(block_length),
        gradient_(block_length),
        velocity_(block_length),
        prefix_products_(check_offsets[n_checks] + 1),
        suffix_products_(check_offsets[n_checks] + 1) {}

  template <typename Float>
  uint32_t Decode(
      const Float* input,
      Float* output,
      double sigma_noise) {
    if (!(sigma_noise > 0.0) || !std::isfinite(sigma_noise)) {
      return std::numeric_limits<uint32_t>::max();
    }

    sigma_squared_ = sigma_noise * sigma_noise;
    std::fill(velocity_.begin(), velocity_.end(), 0.0);
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      x_[variable] = static_cast<double>(input[variable]);
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      if (ParityChecksSatisfied()) {
        WriteOutput(output);
        return iteration;
      }

      CalculateGradient(input);
      const double current_learning_rate =
          learning_rate_ /
          std::sqrt(1.0 + learning_rate_decay_ * iteration);
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        velocity_[variable] =
            momentum_ * velocity_[variable] +
            (1.0 - momentum_) * gradient_[variable];
        x_[variable] += current_learning_rate * velocity_[variable];
      }
      if (!ValuesFit<Float>()) {
        return std::numeric_limits<uint32_t>::max();
      }
    }

    WriteOutput(output);
    return n_iterations_;
  }

 private:
  bool ParityChecksSatisfied() const {
    for (uint32_t check = 0; check < n_checks_; ++check) {
      int syndrome = 1;
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        syndrome *= x_[edge_vn_[edge]] >= 0.0 ? 1 : -1;
      }
      if (syndrome != 1) {
        return false;
      }
    }
    return true;
  }

  template <typename Float>
  void CalculateGradient(const Float* input) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      bipolar_probabilities_[variable] =
          std::tanh(x_[variable] / sigma_squared_);
      gradient_[variable] =
          alpha_ * static_cast<double>(input[variable]);
    }

    for (uint32_t check = 0; check < n_checks_; ++check) {
      const uint32_t begin = check_offsets_[check];
      const uint32_t end = check_offsets_[check + 1];

      prefix_products_[begin] = 1.0;
      for (uint32_t edge = begin; edge < end; ++edge) {
        prefix_products_[edge + 1] =
            prefix_products_[edge] *
            bipolar_probabilities_[edge_vn_[edge]];
      }

      suffix_products_[end] = 1.0;
      for (uint32_t edge = end; edge > begin; --edge) {
        const uint32_t current_edge = edge - 1;
        suffix_products_[current_edge] =
            bipolar_probabilities_[edge_vn_[current_edge]] *
            suffix_products_[current_edge + 1];
      }

      for (uint32_t edge = begin; edge < end; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        const double extrinsic_product =
            prefix_products_[edge] * suffix_products_[edge + 1];
        gradient_[variable] +=
            (1.0 - bipolar_probabilities_[variable] *
                       bipolar_probabilities_[variable]) /
            sigma_squared_ * extrinsic_product;
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

  template <typename Float>
  void WriteOutput(Float* output) const {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      output[variable] = static_cast<Float>(x_[variable]);
    }
  }

  uint32_t block_length_;
  uint32_t n_checks_;
  uint32_t n_iterations_;
  double learning_rate_;
  double learning_rate_decay_;
  double momentum_;
  double alpha_;
  double sigma_squared_ = 0.0;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<double> x_;
  std::vector<double> bipolar_probabilities_;
  std::vector<double> gradient_;
  std::vector<double> velocity_;
  std::vector<double> prefix_products_;
  std::vector<double> suffix_products_;
};

extern "C" void* cpp_pgd_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double learning_rate,
    double learning_rate_decay,
    double momentum,
    double alpha,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppPgdDecoder(
        block_length,
        n_checks,
        n_iterations,
        learning_rate,
        learning_rate_decay,
        momentum,
        alpha,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_pgd_decode_float32(
    void* decoder,
    const float* input,
    float* output,
    double sigma_noise) {
  return static_cast<CppPgdDecoder*>(decoder)->Decode(
      input, output, sigma_noise);
}

extern "C" uint32_t cpp_pgd_decode_float64(
    void* decoder,
    const double* input,
    double* output,
    double sigma_noise) {
  return static_cast<CppPgdDecoder*>(decoder)->Decode(
      input, output, sigma_noise);
}

extern "C" void cpp_pgd_free(void* decoder) {
  delete static_cast<CppPgdDecoder*>(decoder);
}
