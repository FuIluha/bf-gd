#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <random>
#include <vector>

class CppSoftGdbfDecoder {
 public:
  CppSoftGdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double learning_rate,
      double update_probability,
      double beta1,
      double beta2,
      double adam_epsilon,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        learning_rate_(learning_rate),
        update_probability_(update_probability),
        beta1_(beta1),
        beta2_(beta2),
        adam_epsilon_(adam_epsilon),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        gradient_(block_length),
        penalized_gradient_(block_length),
        gradient_history_(block_length),
        second_moment_(block_length),
        check_sign_products_(n_checks),
        check_log_magnitude_sums_(n_checks),
        check_zero_counts_(n_checks) {}

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output, uint64_t seed) {
    std::fill(gradient_history_.begin(), gradient_history_.end(), 0.0);
    std::fill(second_moment_.begin(), second_moment_.end(), 0.0);
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      x_[variable] = static_cast<double>(input[variable]);
    }

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      if (ParityChecksSatisfied()) {
        WriteHardDecision(output);
        return iteration;
      }

      CalculateGradient(input);
      const double second_moment_correction =
          1.0 - std::pow(beta2_, static_cast<double>(iteration + 1));

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        const double penalized =
            gradient_[variable] - beta1_ * gradient_history_[variable];
        penalized_gradient_[variable] = penalized;
        second_moment_[variable] =
            beta2_ * second_moment_[variable] +
            (1.0 - beta2_) * penalized * penalized;
      }

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        const bool update = uniform(generator) < update_probability_;
        gradient_history_[variable] *= beta1_;
        if (!update) {
          continue;
        }

        const double corrected_second_moment =
            second_moment_[variable] / second_moment_correction;
        x_[variable] +=
            learning_rate_ * penalized_gradient_[variable] /
            (std::sqrt(corrected_second_moment) + adam_epsilon_);
        gradient_history_[variable] +=
            (1.0 - beta1_) * gradient_[variable];
      }
    }

    WriteHardDecision(output);
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
    for (uint32_t check = 0; check < n_checks_; ++check) {
      double sign_product = 1.0;
      double log_magnitude_sum = 0.0;
      uint32_t zero_count = 0;

      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const double value = x_[edge_vn_[edge]];
        if (value == 0.0) {
          ++zero_count;
        } else {
          sign_product *= value > 0.0 ? 1.0 : -1.0;
          log_magnitude_sum += std::log(std::abs(value));
        }
      }

      check_sign_products_[check] = sign_product;
      check_log_magnitude_sums_[check] = log_magnitude_sum;
      check_zero_counts_[check] = zero_count;
    }

    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      gradient_[variable] = static_cast<double>(input[variable]);
    }

    for (uint32_t check = 0; check < n_checks_; ++check) {
      const uint32_t zero_count = check_zero_counts_[check];
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        const double value = x_[variable];
        double extrinsic_product = 0.0;

        if (zero_count == 0) {
          extrinsic_product =
              check_sign_products_[check] * (value > 0.0 ? 1.0 : -1.0) *
              std::exp(
                  check_log_magnitude_sums_[check] -
                  std::log(std::abs(value)));
        } else if (zero_count == 1 && value == 0.0) {
          extrinsic_product =
              check_sign_products_[check] *
              std::exp(check_log_magnitude_sums_[check]);
        }

        gradient_[variable] += extrinsic_product;
      }
    }
  }

  template <typename Float>
  void WriteHardDecision(Float* output) const {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      output[variable] = static_cast<Float>(
          x_[variable] >= 0.0 ? 1.0 : -1.0);
    }
  }

  uint32_t block_length_;
  uint32_t n_checks_;
  uint32_t n_iterations_;
  double learning_rate_;
  double update_probability_;
  double beta1_;
  double beta2_;
  double adam_epsilon_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<double> x_;
  std::vector<double> gradient_;
  std::vector<double> penalized_gradient_;
  std::vector<double> gradient_history_;
  std::vector<double> second_moment_;
  std::vector<double> check_sign_products_;
  std::vector<double> check_log_magnitude_sums_;
  std::vector<uint32_t> check_zero_counts_;
};

extern "C" void* cpp_soft_gdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double learning_rate,
    double update_probability,
    double beta1,
    double beta2,
    double adam_epsilon,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppSoftGdbfDecoder(
        block_length,
        n_checks,
        n_iterations,
        learning_rate,
        update_probability,
        beta1,
        beta2,
        adam_epsilon,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_soft_gdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output,
    uint64_t seed) {
  return static_cast<CppSoftGdbfDecoder*>(decoder)->Decode(
      input, output, seed);
}

extern "C" uint32_t cpp_soft_gdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output,
    uint64_t seed) {
  return static_cast<CppSoftGdbfDecoder*>(decoder)->Decode(
      input, output, seed);
}

extern "C" void cpp_soft_gdbf_free(void* decoder) {
  delete static_cast<CppSoftGdbfDecoder*>(decoder);
}
