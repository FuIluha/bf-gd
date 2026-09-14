#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

// Hard message passing with persistent variable-to-check signs.
class CppEgdbfDecoder {
 public:
  CppEgdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double alpha,
      const double* rho,
      uint32_t momentum_length,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        alpha_(alpha),
        momentum_length_(momentum_length),
        rho_(rho, rho + momentum_length),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        channel_signs_(block_length),
        hard_word_(block_length),
        variable_messages_(check_offsets[n_checks]),
        new_variable_messages_(check_offsets[n_checks]),
        check_messages_(check_offsets[n_checks]),
        ages_(check_offsets[n_checks]),
        incoming_sums_(block_length),
        posterior_gradients_(block_length) {
    // rho[L] is used before an edge message has changed for the first time.
    rho_.push_back(0.0);
  }

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      channel_signs_[variable] = input[variable] >= 0 ? 1 : -1;
    }
    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      variable_messages_[edge] = channel_signs_[edge_vn_[edge]];
      ages_[edge] = momentum_length_ + 1;
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      CalculateCheckMessages();
      CalculatePosterior(input);
      if (HardWordSatisfiesChecks()) {
        WriteOutput(output);
        return iteration;
      }

      UpdateVariableMessages(input);
    }

    CalculateCheckMessages();
    CalculatePosterior(input);
    WriteOutput(output);
    return n_iterations_;
  }

 private:
  void CalculateCheckMessages() {
    std::fill(incoming_sums_.begin(), incoming_sums_.end(), 0.0);
    for (uint32_t check = 0; check < n_checks_; ++check) {
      int8_t product = 1;
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        product *= variable_messages_[edge];
      }
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        check_messages_[edge] = product * variable_messages_[edge];
        incoming_sums_[variable] += check_messages_[edge];
      }
    }
  }

  template <typename Float>
  void CalculatePosterior(const Float* input) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      posterior_gradients_[variable] =
          alpha_ * static_cast<double>(input[variable]) +
          incoming_sums_[variable];
      if (posterior_gradients_[variable] > 0.0) {
        hard_word_[variable] = 1;
      } else if (posterior_gradients_[variable] < 0.0) {
        hard_word_[variable] = -1;
      } else {
        hard_word_[variable] = channel_signs_[variable];
      }
    }
  }

  bool HardWordSatisfiesChecks() const {
    for (uint32_t check = 0; check < n_checks_; ++check) {
      int8_t syndrome = 1;
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        syndrome *= hard_word_[edge_vn_[edge]];
      }
      if (syndrome != 1) return false;
    }
    return true;
  }

  template <typename Float>
  void UpdateVariableMessages(const Float* input) {
    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      ages_[edge] = std::min(ages_[edge], momentum_length_) + 1;
      const uint32_t variable = edge_vn_[edge];
      const double extrinsic_gradient =
          alpha_ * static_cast<double>(input[variable]) +
          incoming_sums_[variable] -
          static_cast<double>(check_messages_[edge]) +
          rho_[ages_[edge] - 1] * variable_messages_[edge];
      if (extrinsic_gradient > 0.0) {
        new_variable_messages_[edge] = 1;
      } else if (extrinsic_gradient < 0.0) {
        new_variable_messages_[edge] = -1;
      } else {
        new_variable_messages_[edge] = variable_messages_[edge];
      }
    }

    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      if (new_variable_messages_[edge] != variable_messages_[edge]) {
        ages_[edge] = 0;
      }
      variable_messages_[edge] = new_variable_messages_[edge];
    }
  }

  template <typename Float>
  void WriteOutput(Float* output) const {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      output[variable] = static_cast<Float>(hard_word_[variable]);
    }
  }

  uint32_t block_length_;
  uint32_t n_checks_;
  uint32_t n_iterations_;
  double alpha_;
  uint32_t momentum_length_;
  std::vector<double> rho_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<int8_t> channel_signs_;
  std::vector<int8_t> hard_word_;
  std::vector<int8_t> variable_messages_;
  std::vector<int8_t> new_variable_messages_;
  std::vector<int8_t> check_messages_;
  std::vector<uint32_t> ages_;
  std::vector<double> incoming_sums_;
  std::vector<double> posterior_gradients_;
};

extern "C" void* cpp_egdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double alpha,
    const double* rho,
    uint32_t momentum_length,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppEgdbfDecoder(
        block_length,
        n_checks,
        n_iterations,
        alpha,
        rho,
        momentum_length,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_egdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output) {
  return static_cast<CppEgdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" uint32_t cpp_egdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output) {
  return static_cast<CppEgdbfDecoder*>(decoder)->Decode(input, output);
}

extern "C" void cpp_egdbf_free(void* decoder) {
  delete static_cast<CppEgdbfDecoder*>(decoder);
}
