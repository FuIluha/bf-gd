#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  double Uniform() {
    uint64_t value = Next();
    return static_cast<double>(value >> 11) * 0x1.0p-53;
  }

 private:
  uint64_t Next() {
    uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  uint64_t state_;
};

class CppEpmgdbfDecoder {
 public:
  CppEpmgdbfDecoder(
      uint32_t block_length,
      uint32_t n_checks,
      uint32_t n_iterations,
      double delta,
      double delta_e,
      double alpha,
      double probability,
      double init_erasure_threshold,
      const float* rho,
      uint32_t momentum_length,
      const uint32_t* edge_vn,
      const uint32_t* check_offsets)
      : block_length_(block_length),
        n_checks_(n_checks),
        n_iterations_(n_iterations),
        delta_(delta),
        delta_e_(delta_e),
        alpha_(alpha),
        probability_(probability),
        init_erasure_threshold_(init_erasure_threshold),
        momentum_length_(momentum_length),
        edge_vn_(edge_vn, edge_vn + check_offsets[n_checks]),
        check_offsets_(check_offsets, check_offsets + n_checks + 1),
        rho_(rho, rho + momentum_length),
        x_(block_length),
        momentum_age_(block_length),
        check_nonzero_products_(n_checks),
        check_erasure_counts_(n_checks),
        check_syndromes_(n_checks),
        incident_syndrome_sums_(block_length),
        energies_(block_length),
        selected_(block_length),
        update_(block_length),
        recovery_(block_length),
        new_erasure_(block_length),
        recovery_sums_(block_length) {
    rho_.push_back(0.0F);
  }

  template <typename Float>
  uint32_t Decode(const Float* input, Float* output, uint64_t seed) {
    SplitMix64 random(seed);

    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      const Float received = input[variable];
      const Float threshold = static_cast<Float>(init_erasure_threshold_);
      x_[variable] = received >= threshold
                         ? 1
                         : (received <= -threshold ? -1 : 0);
      momentum_age_[variable] = momentum_length_ + 1;
    }

    for (uint32_t iteration = 0; iteration < n_iterations_; ++iteration) {
      bool all_checks_satisfied = true;

      for (uint32_t check = 0; check < n_checks_; ++check) {
        int8_t nonzero_product = 1;
        uint16_t erasure_count = 0;

        for (uint32_t edge = check_offsets_[check];
             edge < check_offsets_[check + 1]; ++edge) {
          const int8_t value = x_[edge_vn_[edge]];
          if (value == 0) {
            ++erasure_count;
          } else {
            nonzero_product = static_cast<int8_t>(nonzero_product * value);
          }
        }

        check_nonzero_products_[check] = nonzero_product;
        check_erasure_counts_[check] = erasure_count;
        check_syndromes_[check] = erasure_count == 0 ? nonzero_product : 0;
        all_checks_satisfied &= check_syndromes_[check] == 1;
      }

      if (all_checks_satisfied) {
        CopyOutput(output);
        return iteration;
      }

      std::fill(
          incident_syndrome_sums_.begin(),
          incident_syndrome_sums_.end(),
          0);
      for (uint32_t check = 0; check < n_checks_; ++check) {
        const int16_t syndrome = check_syndromes_[check];
        for (uint32_t edge = check_offsets_[check];
             edge < check_offsets_[check + 1]; ++edge) {
          incident_syndrome_sums_[edge_vn_[edge]] += syndrome;
        }
      }

      double minimum_energy = std::numeric_limits<double>::infinity();
      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        momentum_age_[variable] =
            std::min(momentum_age_[variable], momentum_length_) + 1;
        const double energy =
            alpha_ * static_cast<double>(x_[variable]) *
                static_cast<double>(input[variable]) +
            static_cast<double>(incident_syndrome_sums_[variable]) +
            static_cast<double>(rho_[momentum_age_[variable] - 1]);
        energies_[variable] = energy;
        minimum_energy = std::min(minimum_energy, energy);
      }

      const double update_threshold = minimum_energy + delta_;
      const double erasure_threshold = minimum_energy + delta_e_;
      bool has_recovery = false;

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        selected_[variable] = random.Uniform() < probability_;
        update_[variable] =
            selected_[variable] && energies_[variable] <= update_threshold;
        recovery_[variable] = update_[variable] && x_[variable] == 0;
        new_erasure_[variable] =
            selected_[variable] && x_[variable] != 0 &&
            energies_[variable] > update_threshold &&
            energies_[variable] <= erasure_threshold;
        has_recovery |= recovery_[variable];
      }

      if (has_recovery) {
        std::fill(recovery_sums_.begin(), recovery_sums_.end(), 0);
        for (uint32_t check = 0; check < n_checks_; ++check) {
          if (check_erasure_counts_[check] != 1) {
            continue;
          }
          for (uint32_t edge = check_offsets_[check];
               edge < check_offsets_[check + 1]; ++edge) {
            const uint32_t variable = edge_vn_[edge];
            if (recovery_[variable] && x_[variable] == 0) {
              recovery_sums_[variable] += check_nonzero_products_[check];
              break;
            }
          }
        }
      }

      for (uint32_t variable = 0; variable < block_length_; ++variable) {
        if (update_[variable]) {
          if (x_[variable] != 0) {
            x_[variable] = static_cast<int8_t>(-x_[variable]);
          } else if (recovery_sums_[variable] > 0) {
            x_[variable] = 1;
          } else if (recovery_sums_[variable] < 0) {
            x_[variable] = -1;
          }
          momentum_age_[variable] = 0;
        }
        if (new_erasure_[variable]) {
          x_[variable] = 0;
        }
      }
    }

    for (int8_t& value : x_) {
      if (value == 0) {
        value = -1;
      }
    }
    CopyOutput(output);
    return n_iterations_;
  }

 private:
  template <typename Float>
  void CopyOutput(Float* output) const {
    for (uint32_t variable = 0; variable < block_length_; ++variable) {
      output[variable] = static_cast<Float>(x_[variable]);
    }
  }

  uint32_t block_length_;
  uint32_t n_checks_;
  uint32_t n_iterations_;
  double delta_;
  double delta_e_;
  double alpha_;
  double probability_;
  double init_erasure_threshold_;
  uint32_t momentum_length_;
  std::vector<uint32_t> edge_vn_;
  std::vector<uint32_t> check_offsets_;
  std::vector<float> rho_;

  std::vector<int8_t> x_;
  std::vector<uint32_t> momentum_age_;
  std::vector<int8_t> check_nonzero_products_;
  std::vector<uint16_t> check_erasure_counts_;
  std::vector<int8_t> check_syndromes_;
  std::vector<int16_t> incident_syndrome_sums_;
  std::vector<double> energies_;
  std::vector<uint8_t> selected_;
  std::vector<uint8_t> update_;
  std::vector<uint8_t> recovery_;
  std::vector<uint8_t> new_erasure_;
  std::vector<int16_t> recovery_sums_;
};

}  // namespace

extern "C" void* cpp_epmgdbf_create(
    uint32_t block_length,
    uint32_t n_checks,
    uint32_t n_iterations,
    double delta,
    double delta_e,
    double alpha,
    double probability,
    double init_erasure_threshold,
    const float* rho,
    uint32_t momentum_length,
    const uint32_t* edge_vn,
    const uint32_t* check_offsets) {
  try {
    return new CppEpmgdbfDecoder(
        block_length,
        n_checks,
        n_iterations,
        delta,
        delta_e,
        alpha,
        probability,
        init_erasure_threshold,
        rho,
        momentum_length,
        edge_vn,
        check_offsets);
  } catch (...) {
    return nullptr;
  }
}

extern "C" uint32_t cpp_epmgdbf_decode_float32(
    void* decoder,
    const float* input,
    float* output,
    uint64_t seed) {
  return static_cast<CppEpmgdbfDecoder*>(decoder)->Decode(input, output, seed);
}

extern "C" uint32_t cpp_epmgdbf_decode_float64(
    void* decoder,
    const double* input,
    double* output,
    uint64_t seed) {
  return static_cast<CppEpmgdbfDecoder*>(decoder)->Decode(input, output, seed);
}

extern "C" void cpp_epmgdbf_free(void* decoder) {
  delete static_cast<CppEpmgdbfDecoder*>(decoder);
}
