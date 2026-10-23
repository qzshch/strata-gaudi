#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace strata::hpu {

struct RoutedExpert {
    uint32_t expert = 0;
    float weight = 0;
};

// GLM-5.3-Flash's fixed 288-expert, top-8, single-group routing profile.
// Inputs are already computed FP32 router logits and the FP32 correction bias.
// Sigmoid(logit)+bias selects the experts; only the ORIGINAL sigmoid scores enter
// the normalized weights, multiplied by 2.5. This does not perform the router
// linear, shared-expert calculation, result merging or HPU execution.
//
// Results are ordered by decreasing corrected score, breaking exact ties by
// increasing expert ID. Other frameworks need not choose the same IDs at ties.
// Arithmetic is FP32, including sigmoid saturation and sum+1e-20 normalization.
// Finite inputs and exactly 288 elements in each input are required. No allocation.
std::array<RoutedExpert, 8> select_glm53_experts(std::span<const float> logits,
                                             std::span<const float> correction_bias);

}  // namespace strata::hpu
