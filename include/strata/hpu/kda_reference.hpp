#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

inline constexpr const char* kKdaReferenceProfile = "glm53_kda_fp32";

// GLM-5.3-Flash's audited linear-attention geometry: 34 KDA layers, 64 heads of
// 128, causal short conv of 4 taps and the -5.0 safe gate lower bound
// (glm53-config evidence, 2026-10-23 header audit).
struct KdaProfile {
    uint32_t heads;
    uint32_t head_dim;
    uint32_t d_conv;
    float gate_lower_bound;
};
inline constexpr KdaProfile kGlm53KdaProfile{64, 128, 4, -5.0f};

// Bounded scratch requirement of kda_gated_delta_reference, in floats: one full
// state image for the validate-then-commit state update.
[[nodiscard]] constexpr size_t kda_state_values(const KdaProfile& profile) {
    return size_t(profile.heads) * profile.head_dim * profile.head_dim;
}

// Total scratch of kda_gated_delta_reference: the state image followed by one
// output image, both committed only after the whole scan validates.
[[nodiscard]] constexpr size_t kda_scan_scratch_values(const KdaProfile& profile, size_t tokens) {
    return kda_state_values(profile) + tokens * size_t(profile.heads) * profile.head_dim;
}

// Causal per-channel short conv followed by SiLU. x is [tokens][d_inner] with
// d_inner = heads*head_dim, weights [d_inner][d_conv], conv_state [d_conv-1][d_inner]
// oldest tap first. Each out element is published once it is finite; conv_state
// is advanced to the last d_conv-1 columns of [conv_state | x] only after every
// element succeeded. No projections, activation gating or HPU dispatch.
void kda_conv_reference(std::span<const float> x, std::span<const float> weights,
                        std::span<float> conv_state, std::span<float> out, const KdaProfile&);

// Safe-path decay gate in LOG space: out[t][h*head_dim+i] =
// gate_lower_bound * sigmoid(exp(a_log[h]) * g[t][h*head_dim+i]), i.e. values in
// (gate_lower_bound, 0] that the scan exponentiates. g is the already-summed
// f_b(f_a(x)) + dt_bias plane; A_log is per head. FP32 sigmoid and exp, no
// softplus fallback: GLM-5.3-Flash fixes the safe path.
void kda_decay_gate_reference(std::span<const float> g, std::span<const float> a_log,
                              std::span<float> out, const KdaProfile&);

// Gated delta rule scan over tokens. q/k/v are post-conv SiLU [tokens][d_inner];
// q and k get FLA l2 norm (eps 1e-6 inside the sqrt) and q additionally the
// head_dim^-0.5 scale, both in FP32, matching the fused kernel's in-kernel path.
// gate is the kda_decay_gate_reference output (log space, decay applied as
// exp(gate) on the KEY axis); beta is ALREADY sigmoided [tokens][heads].
// state is [heads][head_dim key][head_dim value], row = key axis; scratch must
// hold kda_scan_scratch_values(profile, tokens) floats (one state image then
// one output image) and must not alias any plane. out is [tokens][d_inner];
// out and state change together, only after the whole scan validates (a
// non-finite intermediate rejects the call leaving both untouched). This is
// the recurrent algorithm only; chunk prefill is a separate future module.
void kda_gated_delta_reference(std::span<const float> q, std::span<const float> k,
                               std::span<const float> v, std::span<const float> gate,
                               std::span<const float> beta, std::span<float> state,
                               std::span<float> out, std::span<float> scratch, const KdaProfile&);

// Output gate: per token and head, o is RMS-normalized over head_dim (eps 1e-6,
// FP32), multiplied by norm_w and by sigmoid(g2), writing [tokens][d_inner].
// o and out may alias: publication happens only after all values validate.
void kda_out_gate_reference(std::span<const float> o, std::span<const float> g2,
                            std::span<const float> norm_w, std::span<float> out,
                            const KdaProfile&);

}  // namespace strata::hpu
