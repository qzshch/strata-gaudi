#include "strata/hpu/mhc_reference.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
namespace {
constexpr uint32_t kGlmHc = 4;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "GLM mHC reference requires IEEE-754 binary32");

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("mHC reference size overflow");
    return a * b;
}

void demand_finite(std::span<const float> plane, const char* what) {
    for (const float value : plane)
        if (!std::isfinite(value)) throw std::invalid_argument(what);
}

size_t token_count(std::span<const float> plane, size_t per_token, const char* what) {
    if (per_token == 0 || plane.size() % per_token != 0)
        throw std::invalid_argument(what);
    return plane.size() / per_token;
}
}  // namespace

void mhc_pre_reference(std::span<const float> streams, std::span<const float> fn,
                       std::span<const float> base, std::span<const float> scale,
                       std::span<float> pre, std::span<float> post, std::span<float> comb,
                       std::span<float> collapsed, const MhcProfile& profile, uint32_t hidden) {
    if (profile.hc != kGlmHc)
        throw std::invalid_argument("mHC reference pins the audited 4-stream geometry");
    if (profile.hc == 0 || hidden == 0 || profile.sinkhorn_iters == 0)
        throw std::invalid_argument("mHC profile is degenerate");
    if (!std::isfinite(profile.rms_eps) || profile.rms_eps < 0 ||
        !std::isfinite(profile.hc_eps) || profile.hc_eps < 0)
        throw std::invalid_argument("mHC profile epsilons must be finite and nonnegative");
    const size_t hc = profile.hc;
    const size_t stream_values = checked_mul(hc, hidden);
    const size_t mix_values = checked_mul(2 + hc, hc);
    const size_t tokens = token_count(streams, stream_values,
                                      "mHC streams are not whole flattened token planes");
    if (fn.size() != checked_mul(mix_values, stream_values) || base.size() != mix_values ||
        scale.size() != 3)
        throw std::invalid_argument("mHC parameters do not match the profile");
    if (pre.size() != checked_mul(tokens, hc) || post.size() != pre.size() ||
        comb.size() != checked_mul(tokens, checked_mul(hc, hc)) ||
        collapsed.size() != checked_mul(tokens, hidden))
        throw std::invalid_argument("mHC weight plane sizes do not match the token count");
    if (tokens == 0) throw std::invalid_argument("mHC requires at least one token");
    demand_finite(streams, "mHC streams must be finite");
    demand_finite(fn, "mHC fn must be finite");
    demand_finite(base, "mHC base must be finite");
    demand_finite(scale, "mHC scale must be finite");

    for (size_t t = 0; t < tokens; ++t) {
        const float* row = streams.data() + t * stream_values;
        float mean_square = 0;
        for (size_t i = 0; i < stream_values; ++i) mean_square += row[i] * row[i];
        const float inverse = 1.0f / std::sqrt(mean_square / static_cast<float>(stream_values) +
                                               profile.rms_eps);

        // Mix projection on the normalized plane, left-to-right accumulation.
        static_assert((2 + kGlmHc) * kGlmHc == 24);
        float mix_row[24];
        for (size_t m = 0; m < mix_values; ++m) {
            const float* weights = fn.data() + m * stream_values;
            float acc = 0;
            for (size_t i = 0; i < stream_values; ++i) acc = std::fma(weights[i], row[i], acc);
            mix_row[m] = acc * inverse;
        }
        for (size_t m = 0; m < mix_values; ++m)
            if (!std::isfinite(mix_row[m])) throw std::overflow_error("mHC mix overflowed");

        float* pre_t = pre.data() + t * hc;
        float* post_t = post.data() + t * hc;
        for (size_t n = 0; n < hc; ++n) {
            const float pre_logit = mix_row[n] * scale[0] + base[n];
            const float post_logit = mix_row[hc + n] * scale[1] + base[hc + n];
            pre_t[n] = sigmoid(pre_logit) + profile.hc_eps;
            post_t[n] = 2.0f * sigmoid(post_logit);
        }

        // comb[src][dst], flat checkpoint layout comb[dst + hc*src]; softmax
        // over dst per src row, +eps, then the pinned Sinkhorn order.
        float* comb_t = comb.data() + t * hc * hc;
        for (size_t src = 0; src < hc; ++src) {
            float logits[4], maximum = 0;
            for (size_t dst = 0; dst < hc; ++dst) {
                const size_t flat = 2 * hc + dst + hc * src;
                logits[dst] = mix_row[flat] * scale[2] + base[flat];
                if (dst == 0 || logits[dst] > maximum) maximum = logits[dst];
            }
            float denominator = 0;
            float exps[4];
            for (size_t dst = 0; dst < hc; ++dst) {
                exps[dst] = std::exp(logits[dst] - maximum);
                denominator += exps[dst];
            }
            for (size_t dst = 0; dst < hc; ++dst)
                comb_t[src * hc + dst] = exps[dst] / denominator + profile.hc_eps;
        }
        auto column_normalize = [&]() {   // each dst sums to 1 over src
            for (size_t dst = 0; dst < hc; ++dst) {
                float sum = 0;
                for (size_t src = 0; src < hc; ++src) sum += comb_t[src * hc + dst];
                const float inverse_sum = 1.0f / (sum + profile.hc_eps);
                for (size_t src = 0; src < hc; ++src)
                    comb_t[src * hc + dst] *= inverse_sum;
            }
        };
        auto row_normalize = [&]() {      // each src sums to 1 over dst
            for (size_t src = 0; src < hc; ++src) {
                float sum = 0;
                for (size_t dst = 0; dst < hc; ++dst) sum += comb_t[src * hc + dst];
                const float inverse_sum = 1.0f / (sum + profile.hc_eps);
                for (size_t dst = 0; dst < hc; ++dst)
                    comb_t[src * hc + dst] *= inverse_sum;
            }
        };
        column_normalize();
        for (uint32_t round = 1; round < profile.sinkhorn_iters; ++round) {
            row_normalize();
            column_normalize();
        }

        float* collapsed_t = collapsed.data() + t * hidden;
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0;
            for (size_t n = 0; n < hc; ++n) acc = std::fma(pre_t[n], row[n * hidden + d], acc);
            collapsed_t[d] = acc;
        }
        for (size_t n = 0; n < hc; ++n)
            if (!std::isfinite(pre_t[n]) || !std::isfinite(post_t[n]))
                throw std::overflow_error("mHC gate overflowed");
        for (size_t i = 0; i < hc * hc; ++i)
            if (!std::isfinite(comb_t[i])) throw std::overflow_error("mHC comb overflowed");
        for (size_t d = 0; d < hidden; ++d)
            if (!std::isfinite(collapsed_t[d])) throw std::overflow_error("mHC collapse overflowed");
    }
}

void mhc_post_reference(std::span<const float> block, std::span<const float> residual,
                        std::span<const float> post, std::span<const float> comb,
                        std::span<float> out, const MhcProfile& profile, uint32_t hidden) {
    const size_t hc = profile.hc;
    const size_t stream_values = checked_mul(hc, hidden);
    const size_t tokens = token_count(residual, stream_values,
                                      "mHC residual is not whole flattened token planes");
    if (block.size() != checked_mul(tokens, hidden) || post.size() != checked_mul(tokens, hc) ||
        comb.size() != checked_mul(tokens, checked_mul(hc, hc)) || out.size() != residual.size())
        throw std::invalid_argument("mHC post plane sizes do not match the token count");
    if (tokens == 0) throw std::invalid_argument("mHC requires at least one token");
    demand_finite(block, "mHC block output must be finite");
    demand_finite(residual, "mHC residual must be finite");
    demand_finite(post, "mHC post must be finite");
    demand_finite(comb, "mHC comb must be finite");

    // Per token: compute the whole output row, validate, then publish, so that
    // an out aliasing residual never mixes old and new stream values.
    std::vector<float> staged;
    staged.resize(stream_values);
    for (size_t t = 0; t < tokens; ++t) {
        const float* block_t = block.data() + t * hidden;
        const float* residual_t = residual.data() + t * stream_values;
        const float* post_t = post.data() + t * hc;
        const float* comb_t = comb.data() + t * hc * hc;
        for (size_t dst = 0; dst < hc; ++dst) {
            for (size_t d = 0; d < hidden; ++d) {
                float mix = 0;
                for (size_t src = 0; src < hc; ++src)
                    mix = std::fma(comb_t[src * hc + dst], residual_t[src * hidden + d], mix);
                staged[dst * hidden + d] = std::fma(post_t[dst], block_t[d], mix);
            }
        }
        for (size_t i = 0; i < stream_values; ++i)
            if (!std::isfinite(staged[i])) throw std::overflow_error("mHC post overflowed");
        float* out_t = out.data() + t * stream_values;
        for (size_t i = 0; i < stream_values; ++i) out_t[i] = staged[i];
    }
}

void mhc_head_reference(std::span<const float> streams, std::span<float> out,
                        const MhcProfile& profile, uint32_t hidden) {
    const size_t hc = profile.hc;
    const size_t stream_values = checked_mul(hc, hidden);
    const size_t tokens = token_count(streams, stream_values,
                                      "mHC streams are not whole flattened token planes");
    if (out.size() != checked_mul(tokens, hidden))
        throw std::invalid_argument("mHC head output size mismatch");
    if (tokens == 0) throw std::invalid_argument("mHC requires at least one token");
    demand_finite(streams, "mHC streams must be finite");
    const float inverse_count = 1.0f / static_cast<float>(hc);
    for (size_t t = 0; t < tokens; ++t) {
        const float* row = streams.data() + t * stream_values;
        float* out_t = out.data() + t * hidden;
        // hidden-major accumulation keeps an out alias safe: element d reads
        // only the d-th column of every stream.
        for (size_t d = 0; d < hidden; ++d) {
            float acc = 0;
            for (size_t n = 0; n < hc; ++n) acc += row[n * hidden + d];
            const float mean = acc * inverse_count;
            if (!std::isfinite(mean)) throw std::overflow_error("mHC head overflowed");
            out_t[d] = mean;
        }
    }
}

}  // namespace strata::hpu
