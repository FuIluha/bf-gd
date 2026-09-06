#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

class CppSpGdbfDecoder {
 public:
  CppSpGdbfDecoder(
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
        prefix_messages_(check_offsets[n_checks] + 1),
        suffix_messages_(check_offsets[n_checks] + 1) {}

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
  static double BoxPlus(double first, double second) {
    if (std::isinf(first)) {
      return std::signbit(first) ? -second : second;
    }
    if (std::isinf(second)) {
      return std::signbit(second) ? -first : first;
    }

    const double first_magnitude = std::abs(first);
    const double second_magnitude = std::abs(second);
    const double minimum = std::min(first_magnitude, second_magnitude);
    const double magnitude =
        minimum +
        std::log1p(std::exp(-(first_magnitude + second_magnitude))) -
        std::log1p(std::exp(-std::abs(first_magnitude - second_magnitude)));
    const bool negative = std::signbit(first) != std::signbit(second);
    return negative ? -magnitude : magnitude;
  }

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

    const double identity = std::numeric_limits<double>::infinity();
    for (uint32_t check = 0; check < n_checks_; ++check) {
      const uint32_t begin = check_offsets_[check];
      const uint32_t end = check_offsets_[check + 1];

      prefix_messages_[begin] = identity;
      for (uint32_t edge = begin; edge < end; ++edge) {
        prefix_messages_[edge + 1] = BoxPlus(
            prefix_messages_[edge],
            x_[edge_vn_[edge]]);
      }

      suffix_messages_[end] = identity;
      for (uint32_t edge = end; edge > begin; --edge) {
        const uint32_t current_edge = edge - 1;
        suffix_messages_[current_edge] = BoxPlus(
            x_[edge_vn_[current_edge]],
            suffix_messages_[current_edge + 1]);
      }

      for (uint32_t edge = begin; edge < end; ++edge) {
        const double check_message = BoxPlus(
            prefix_messages_[edge],
            suffix_messages_[edge + 1]);
        gradient_[edge_vn_[edge]] += check_message;
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
  std::vector<double> prefix_messages_;
  std::vector<double> suffix_messages_;
};

extern "C" void* cpp_sp_gdbf_create(
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
    return new CppSpGdbfDecoder(
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

extern "C" uint32_t cpp_sp_gdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output) {
  return static_cast<CppSpGdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" uint32_t cpp_sp_gdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output) {
  return static_cast<CppSpGdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" void cpp_sp_gdbf_free(void* decoder) {
  delete static_cast<CppSpGdbfDecoder*>(decoder);
}
