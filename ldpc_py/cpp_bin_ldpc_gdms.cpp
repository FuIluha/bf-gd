#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

// GDMS/MGDMS with extrinsic edge states, an AWGN channel-distance term, a
// smooth bipolar penalty, optional momentum, and an optional sign-only check
// update for E-GDBF V4.
class CppGdmsDecoder {
 public:
  CppGdmsDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double learning_rate,
      double learning_rate_decay,
      double channel_weight,
      double bipolar_weight,
      double momentum,
      bool sign_only,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        learning_rate_(learning_rate),
        learning_rate_decay_(learning_rate_decay),
        channel_weight_(channel_weight),
        bipolar_weight_(bipolar_weight),
        momentum_(momentum),
        sign_only_(sign_only),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        prev_x_(block_length),
        next_x_(block_length),
        gradient_(block_length),
        variable_messages_(check_offsets[n_checks]),
        prev_variable_messages_(check_offsets[n_checks]),
        next_variable_messages_(check_offsets[n_checks]),
        check_messages_(check_offsets[n_checks]),
        check_signs_(n_checks),
        first_minima_(n_checks),
        second_minima_(n_checks),
        first_minimum_counts_(n_checks) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
    std::fill(check_messages_.begin(), check_messages_.end(), 0.0);
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      x_[variable] = static_cast<double>(input[variable]);
      prev_x_[variable] = x_[variable];
    }

    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      variable_messages_[edge] = static_cast<double>(input[edge_vn_[edge]]);
      prev_variable_messages_[edge] = variable_messages_[edge];
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      if (ParityChecksSatisfied()) {
        WriteOutput(output);
        return iteration;
      }

      CalculateCheckMessages();
      const double current_learning_rate =
          learning_rate_ /
          std::sqrt(1.0 + learning_rate_decay_ * iteration);
      for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
        const uint32_t variable = edge_vn_[edge];
        const double value = variable_messages_[edge];
        const double channel_direction =
            channel_weight_ * (static_cast<double>(input[variable]) - value);
        const double bipolar_direction =
            -4.0 * bipolar_weight_ * value * (value * value - 1.0);
        next_variable_messages_[edge] = variable_messages_[edge]
            + current_learning_rate * (
                channel_direction + gradient_[variable] - check_messages_[edge]
                + bipolar_direction)
            + momentum_ * (variable_messages_[edge] - prev_variable_messages_[edge]);
      }
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        const double value = x_[variable];
        const double channel_direction =
            channel_weight_ * (static_cast<double>(input[variable]) - value);
        const double bipolar_direction =
            -4.0 * bipolar_weight_ * value * (value * value - 1.0);
        next_x_[variable] = x_[variable]
            + current_learning_rate * (
                channel_direction + gradient_[variable] + bipolar_direction)
            + momentum_ * (x_[variable] - prev_x_[variable]);
      }
      prev_variable_messages_.swap(variable_messages_);
      variable_messages_.swap(next_variable_messages_);
      prev_x_.swap(x_);
      x_.swap(next_x_);
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

  void CalculateCheckMessages() {
    std::fill(gradient_.begin(), gradient_.end(), 0.0);

    for (uint32_t check = 0; check < n_checks_; ++check) {
      int sign_product = 1;
      double first_minimum = std::numeric_limits<double>::infinity();
      double second_minimum = std::numeric_limits<double>::infinity();
      uint32_t first_minimum_count = 0;

      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const double value = variable_messages_[edge];
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
        const double value = variable_messages_[edge];
        const bool unique_first_minimum =
            std::abs(value) == first_minimum &&
            first_minimum_counts_[check] == 1;
        const double magnitude = sign_only_
            ? 1.0
            : unique_first_minimum ? second_minima_[check] : first_minimum;
        const int extrinsic_sign =
            check_signs_[check] * (value < 0.0 ? -1 : 1);
        check_messages_[edge] = extrinsic_sign * magnitude;
        gradient_[variable] += check_messages_[edge];
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
    for (const double value : variable_messages_) {
      if (!std::isfinite(value)) return false;
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
  double channel_weight_;
  double bipolar_weight_;
  double momentum_;
  bool sign_only_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<double> x_;
  std::vector<double> prev_x_;
  std::vector<double> next_x_;
  std::vector<double> gradient_;
  std::vector<double> variable_messages_;
  std::vector<double> prev_variable_messages_;
  std::vector<double> next_variable_messages_;
  std::vector<double> check_messages_;
  std::vector<int> check_signs_;
  std::vector<double> first_minima_;
  std::vector<double> second_minima_;
  std::vector<uint32_t> first_minimum_counts_;
};

extern "C" void* cpp_gdms_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double learning_rate,
    double learning_rate_decay,
    double channel_weight,
    double bipolar_weight,
    double momentum,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppGdmsDecoder(
        block_length,
        n_checks,
        n_iterations,
        learning_rate,
        learning_rate_decay,
        channel_weight,
        bipolar_weight,
        momentum,
        false,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void* cpp_egdbf_v4_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double learning_rate,
    double learning_rate_decay,
    double channel_weight,
    double bipolar_weight,
    double momentum,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppGdmsDecoder(
        block_length,
        n_checks,
        n_iterations,
        learning_rate,
        learning_rate_decay,
        channel_weight,
        bipolar_weight,
        momentum,
        true,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_gdms_decode_float32(
    void* decoder,
    const float* input,
    float* output) {
  return static_cast<CppGdmsDecoder*>(decoder)->Decode(input, output);
}

extern "C" uint32_t cpp_gdms_decode_float64(
    void* decoder,
    const double* input,
    double* output) {
  return static_cast<CppGdmsDecoder*>(decoder)->Decode(input, output);
}

extern "C" void cpp_gdms_free(void* decoder) {
  delete static_cast<CppGdmsDecoder*>(decoder);
}
