#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

// Thresholded hard message passing with persistent edge signs.
class CppEgdbfDecoder {
 public:
  CppEgdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double alpha,
      double delta,
      double probability,
      const double* rho,
      uint32_t momentum_length,
      bool use_mean_threshold,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        alpha_(alpha),
        delta_(delta),
        probability_(probability),
        momentum_length_(momentum_length),
        use_mean_threshold_(use_mean_threshold),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        channel_signs_(block_length),
        hard_word_(block_length),
        variable_messages_(check_offsets[n_checks]),
        check_messages_(check_offsets[n_checks]),
        edge_energies_(check_offsets[n_checks]),
        ages_(momentum_length ? check_offsets[n_checks] : 0),
        incoming_sums_(block_length),
        posterior_scores_(block_length),
        variable_degrees_(block_length, 0),
        bit_energy_sums_(block_length) {
    for (uint32_t variable : edge_vn_) ++variable_degrees_[variable];
    if (momentum_length_) {
      rho_.assign(rho, rho + momentum_length_);
      // rho[L] is used before an edge message has changed for the first time.
      rho_.push_back(0.0);
    }
  }

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output, uint64_t seed) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      channel_signs_[variable] = input[variable] >= 0 ? 1 : -1;
    }
    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      variable_messages_[edge] = channel_signs_[edge_vn_[edge]];
      if (momentum_length_) ages_[edge] = momentum_length_ + 1;
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      CalculateCheckMessages();
      CalculatePosterior(input);
      if (HardWordSatisfiesChecks()) {
        WriteOutput(output);
        return iteration;
      }

      UpdateVariableMessages(input, iteration, seed);
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
      posterior_scores_[variable] =
          alpha_ * static_cast<double>(input[variable]) +
          incoming_sums_[variable];
      if (posterior_scores_[variable] > 0.0) {
        hard_word_[variable] = 1;
      } else if (posterior_scores_[variable] < 0.0) {
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
  void UpdateVariableMessages(const Float* input, uint32_t iteration,
                              uint64_t seed) {
    double minimum_energy = std::numeric_limits<double>::infinity();
    if (use_mean_threshold_) {
      std::fill(bit_energy_sums_.begin(), bit_energy_sums_.end(), 0.0);
    }
    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      const uint32_t variable = edge_vn_[edge];
      const double extrinsic_score =
          alpha_ * static_cast<double>(input[variable]) +
          incoming_sums_[variable] -
          static_cast<double>(check_messages_[edge]);
      double energy = variable_messages_[edge] * extrinsic_score;
      if (momentum_length_) {
        ages_[edge] = std::min(ages_[edge], momentum_length_) + 1;
        energy += rho_[ages_[edge] - 1];
      }
      edge_energies_[edge] = energy;
      if (use_mean_threshold_) {
        bit_energy_sums_[variable] += energy;
      } else {
        minimum_energy = std::min(minimum_energy, energy);
      }
    }

    if (use_mean_threshold_) {
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        minimum_energy = std::min(
            minimum_energy,
            bit_energy_sums_[variable] / variable_degrees_[variable]);
      }
    }

    const double threshold = minimum_energy + delta_;
    for (uint32_t edge = 0; edge < edge_vn_.size(); ++edge) {
      if (edge_energies_[edge] <= threshold &&
          FlipAccepted(seed, iteration, edge)) {
        variable_messages_[edge] = -variable_messages_[edge];
        if (momentum_length_) ages_[edge] = 0;
      }
    }
  }

  bool FlipAccepted(uint64_t seed, uint32_t iteration, uint32_t edge) const {
    if (probability_ == 1.0) return true;
    if (probability_ == 0.0) return false;
    const uint64_t counter =
        static_cast<uint64_t>(iteration) * edge_vn_.size() + edge + 1;
    uint64_t value = seed + 0x9E3779B97F4A7C15ULL * counter;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    value ^= value >> 31;
    const double uniform = static_cast<double>(value >> 11) * 0x1.0p-53;
    return uniform < probability_;
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
  double delta_;
  double probability_;
  uint32_t momentum_length_;
  bool use_mean_threshold_;
  std::vector<double> rho_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<int8_t> channel_signs_;
  std::vector<int8_t> hard_word_;
  std::vector<int8_t> variable_messages_;
  std::vector<int8_t> check_messages_;
  std::vector<double> edge_energies_;
  std::vector<uint32_t> ages_;
  std::vector<double> incoming_sums_;
  std::vector<double> posterior_scores_;
  std::vector<uint32_t> variable_degrees_;
  std::vector<double> bit_energy_sums_;
};

extern "C" void* cpp_egdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double alpha,
    double delta,
    double probability,
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
        delta,
        probability,
        rho,
        momentum_length,
        false,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void* cpp_egdbf_v2_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double alpha,
    double delta,
    double probability,
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
        delta,
        probability,
        rho,
        momentum_length,
        true,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_egdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output,
    uint64_t seed) {
  return static_cast<CppEgdbfDecoder*>(decoder)->Decode(input, output, seed);
}

extern "C" uint32_t cpp_egdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output,
    uint64_t seed) {
  return static_cast<CppEgdbfDecoder*>(decoder)->Decode(input, output, seed);
}

extern "C" void cpp_egdbf_free(void* decoder) {
  delete static_cast<CppEgdbfDecoder*>(decoder);
}
