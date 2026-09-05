#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

class CppSoftGdbfDecoder {
 public:
  CppSoftGdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double learning_rate,
      double learning_rate_decay,
      double momentum,
      double regularization,
      double alpha,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        learning_rate_(learning_rate),
        learning_rate_decay_(learning_rate_decay),
        momentum_(momentum),
        regularization_(regularization),
        alpha_(alpha),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        gradient_(block_length),
        velocity_(block_length),
        check_signs_(n_checks),
        first_minima_(n_checks),
        second_minima_(n_checks),
        first_minimum_counts_(n_checks) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
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
      gradient_[variable] =
          alpha_ * static_cast<double>(input[variable]) -
          regularization_ * x_[variable];
    }

    for (uint32_t check = 0; check < n_checks_; ++check) {
      int sign_product = 1;
      double first_minimum = std::numeric_limits<double>::infinity();
      double second_minimum = std::numeric_limits<double>::infinity();
      uint32_t first_minimum_count = 0;

      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const double value = x_[edge_vn_[edge]];
        sign_product *= value < 0.0 ? -1 : 1;
        const double magnitude = std::abs(value);
        if (magnitude < first_minimum) {
          second_minimum = first_minimum;
          first_minimum = magnitude;
          first_minimum_count = 1;
        } else if (magnitude == first_minimum) {
          ++first_minimum_count;
        } else if (magnitude < second_minimum) {
          second_minimum = magnitude;
        }
      }

      check_signs_[check] = sign_product;
      first_minima_[check] = first_minimum;
      second_minima_[check] = second_minimum;
      first_minimum_counts_[check] = first_minimum_count;
    }

    for (uint32_t check = 0; check < n_checks_; ++check) {
      const double first_minimum = first_minima_[check];
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        const double value = x_[variable];
        const bool unique_first_minimum =
            std::abs(value) == first_minimum &&
            first_minimum_counts_[check] == 1;
        const double magnitude = unique_first_minimum
            ? second_minima_[check]
            : first_minimum;
        const int extrinsic_sign =
            check_signs_[check] * (value < 0.0 ? -1 : 1);
        gradient_[variable] += extrinsic_sign * magnitude;
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
  double regularization_;
  double alpha_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<double> x_;
  std::vector<double> gradient_;
  std::vector<double> velocity_;
  std::vector<int> check_signs_;
  std::vector<double> first_minima_;
  std::vector<double> second_minima_;
  std::vector<uint32_t> first_minimum_counts_;
};

extern "C" void* cpp_soft_gdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double learning_rate,
    double learning_rate_decay,
    double momentum,
    double regularization,
    double alpha,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppSoftGdbfDecoder(
        block_length,
        n_checks,
        n_iterations,
        learning_rate,
        learning_rate_decay,
        momentum,
        regularization,
        alpha,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_soft_gdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output) {
  return static_cast<CppSoftGdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" uint32_t cpp_soft_gdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output) {
  return static_cast<CppSoftGdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" void cpp_soft_gdbf_free(void* decoder) {
  delete static_cast<CppSoftGdbfDecoder*>(decoder);
}
