#include "strata/hpu/routing_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
struct Candidate { uint32_t expert; float score, corrected; };
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "GLM router reference requires IEEE-754 binary32");
}

std::array<RoutedExpert, 8> select_glm53_experts(std::span<const float> logits,
                                             std::span<const float> correction_bias) {
    if (logits.size() != 288 || correction_bias.size() != 288)
        throw std::invalid_argument("GLM-5.3-Flash routing requires 288 logits and correction biases");
    std::array<Candidate, 288> candidates;
    for (uint32_t i = 0; i < candidates.size(); ++i) {
        if (!std::isfinite(logits[i]) || !std::isfinite(correction_bias[i]))
            throw std::invalid_argument("GLM routing inputs must be finite");
        // Deliberately keep binary32 exp/divide, including sigmoid's saturation
        // to zero when exp(-logit) overflows. This is not a higher precision sigmoid.
        const float score = 1.0f / (1.0f + std::exp(-logits[i]));
        candidates[i] = {i, score, score + correction_bias[i]};
    }
    std::partial_sort(candidates.begin(), candidates.begin() + 8, candidates.end(),
        [](const Candidate& left, const Candidate& right) {
            return left.corrected != right.corrected ? left.corrected > right.corrected
                                                     : left.expert < right.expert;
        });
    float denominator = 0;
    for (size_t i = 0; i < 8; ++i) denominator += candidates[i].score;
    denominator += 1e-20f;
    std::array<RoutedExpert, 8> result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = {candidates[i].expert, (candidates[i].score / denominator) * 2.5f};
    return result;
}

}  // namespace strata::hpu
