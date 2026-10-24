#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

inline constexpr const char* kMhcReferenceProfile = "glm53_mhc_fp32";

// GLM-5.3-Flash's audited mHC geometry: 4 hyper-connection streams, 20 Sinkhorn
// iterations, epsilon 1e-6 on every gate/denominator and the unweighted RMS
// norm over the flattened streams (eps 1e-5). hc is pinned to 4 by both
// production references; the parameterization only keeps test sizes honest.
struct MhcProfile {
    uint32_t hc;
    float rms_eps;
    float hc_eps;
    uint32_t sinkhorn_iters;
};
inline constexpr MhcProfile kGlm53MhcProfile{4, 1e-5f, 1e-6f, 20};

// Size of one token's flattened stream plane / mix vector for a profile.
[[nodiscard]] constexpr size_t mhc_stream_values(const MhcProfile& p, uint32_t hidden) {
    return size_t(p.hc) * hidden;
}
[[nodiscard]] constexpr size_t mhc_mix_values(const MhcProfile& p) {
    return (2 + p.hc) * p.hc;  // pre | post | comb, in this flat order
}

// Weight generation for one mHC site (attn or ffn; parameters are per-site).
// streams is [tokens][hc*hidden] FP32; fn is [mix_values][hc*hidden] row-major,
// base [mix_values] and scale exactly [pre, post, comb]. All arithmetic FP32:
// the flattened streams get the UNWEIGHTED RMS norm (eps inside the sqrt), the
// mix projection is a plain left-to-right FP32 dot, pre = sigmoid+scaled by
// pre_scale + base then +hc_eps, post = 2*sigmoid(...), and comb is built from
// the flat layout comb[dst + hc*src] as a softmax over dst (each src row)
// followed by Sinkhorn: one column normalization (sum over src, +hc_eps), then
// sinkhorn_iters-1 rounds of {row over dst, column over src}, each +hc_eps.
//
// pre/post/comb are [tokens][hc]/[tokens][hc]/[tokens][hc*hc] in that same
// comb[src,dst] order; collapsed is [tokens][hidden] = sum_n pre[n]*streams[n]
// rounded from the FP32 accumulation. Inputs must be finite (validated); each
// token's planes publish once all of its values validate, so an adversarial
// overflow leaves only earlier tokens written.
void mhc_pre_reference(std::span<const float> streams, std::span<const float> fn,
                       std::span<const float> base, std::span<const float> scale,
                       std::span<float> pre, std::span<float> post, std::span<float> comb,
                       std::span<float> collapsed, const MhcProfile&, uint32_t hidden);

// Stream update after the block: out[dst] = post[dst]*block_output +
// sum_src comb[src,dst] * residual[src]. block is [tokens][hidden]; residual
// and out are [tokens][hc*hidden] FP32 and may alias each other (publication
// is per token after the whole token validates). pre/post/comb come from
// mhc_pre_reference on that same residual.
void mhc_post_reference(std::span<const float> block, std::span<const float> residual,
                        std::span<const float> post, std::span<const float> comb,
                        std::span<float> out, const MhcProfile&, uint32_t hidden);

// Final head: the UNWEIGHTED mean of the streams, [tokens][hc*hidden] ->
// [tokens][hidden]. out may alias the input.
void mhc_head_reference(std::span<const float> streams, std::span<float> out,
                        const MhcProfile&, uint32_t hidden);

}  // namespace strata::hpu
