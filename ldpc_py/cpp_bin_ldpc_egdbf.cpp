#include <algorithm>
#include <cstdint>
#include <new>
#include <random>
#include <vector>

// Edge-wise gradient descent bit-flipping with binary extrinsic messages.
class CppEgdbfDecoder {
 public:
  CppEgdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double delta,
      double alpha,
      double probability,
      const double* rho,
      uint32_t momentum_length,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        delta_(delta),
        alpha_(alpha),
        probability_(probability),
        momentum_length_(momentum_length),
        rho_(rho, rho + momentum_length),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        x_(block_length),
        ages_(block_length),
        check_syndromes_(n_checks),
        check_messages_(check_offsets[n_checks]),
        edge_energies_(check_offsets[n_checks]),
        posterior_energies_(block_length) {
    // The final zero implements rho(l)=0 before a bit has been flipped.
    rho_.push_back(0.0);
  }

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output, uint64_t seed) {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      x_[variable] = input[variable] >= 0 ? 1 : -1;
      ages_[variable] = momentum_length_ + 1;
    }
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      if (CalculateSyndromes()) {
        WriteOutput(output);
        return iteration;
      }

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        ages_[variable] =
            std::min(ages_[variable], momentum_length_) + 1;
        posterior_energies_[variable] = 0.0;
      }

      CalculateEdgeEnergies(input);
      const double threshold =
          *std::min_element(
              posterior_energies_.begin(), posterior_energies_.end()) +
          delta_;

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        const bool selected = uniform(generator) < probability_;
        if (posterior_energies_[variable] <= threshold && selected) {
          x_[variable] *= -1;
          ages_[variable] = 0;
        }
      }
    }

    WriteOutput(output);
    return n_iterations_;
  }

 private:
  bool CalculateSyndromes() {
    bool all_satisfied = true;
    for (uint32_t check = 0; check < n_checks_; ++check) {
      int8_t syndrome = 1;
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        syndrome *= x_[edge_vn_[edge]];
      }
      check_syndromes_[check] = syndrome;
      all_satisfied = all_satisfied && syndrome == 1;
    }
    return all_satisfied;
  }

  template <typename Float>
  void CalculateEdgeEnergies(const Float* input) {
    for (uint32_t check = 0; check < n_checks_; ++check) {
      for (uint32_t edge = check_offsets_[check];
           edge < check_offsets_[check + 1]; ++edge) {
        const uint32_t variable = edge_vn_[edge];
        // Since x_i is +/-1, c_a*x_i is the product excluding i.
        check_messages_[edge] = check_syndromes_[check] * x_[variable];
        const double intrinsic_and_momentum =
            alpha_ * static_cast<double>(x_[variable]) *
                static_cast<double>(input[variable]) +
            rho_[ages_[variable] - 1];
        edge_energies_[edge] =
            static_cast<double>(x_[variable] * check_messages_[edge]) +
            intrinsic_and_momentum;
        posterior_energies_[variable] += edge_energies_[edge];
      }
    }
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
  double delta_;
  double alpha_;
  double probability_;
  uint32_t momentum_length_;
  std::vector<double> rho_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<int8_t> x_;
  std::vector<uint32_t> ages_;
  std::vector<int8_t> check_syndromes_;
  std::vector<int8_t> check_messages_;
  std::vector<double> edge_energies_;
  std::vector<double> posterior_energies_;
};

extern "C" void* cpp_egdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double delta,
    double alpha,
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
        delta,
        alpha,
        probability,
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
