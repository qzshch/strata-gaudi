#include "strata/hpu/kda_reference.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
constexpr float kL2Eps = 1e-6f;   // transformers l2norm/glm kernels add eps inside the sqrt
constexpr float kNormEps = 1e-6f; // Glm5NextTextRMSNormGated default eps
// Per-head staging bound: the reference head slice is 128 floats; twice that
// keeps an accidental 256-dim profile testable instead of silently corrupt.
constexpr size_t kMaxHeadDim = 256;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "GLM KDA reference requires IEEE-754 binary32");

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("KDA reference size overflow");
    return a * b;
}

void demand_finite(std::span<const float> plane, const char* what) {
    for (const float value : plane)
        if (!std::isfinite(value)) throw std::invalid_argument(what);
}

size_t token_count(std::span<const float> plane, size_t d_inner, const char* what) {
    if (d_inner == 0 || plane.size() % d_inner != 0)
        throw std::invalid_argument(what);
    return plane.size() / d_inner;
}
}  // namespace

void kda_conv_reference(std::span<const float> x, std::span<const float> weights,
                        std::span<float> conv_state, std::span<float> out, const KdaProfile& profile) {
    if (profile.d_conv < 2) throw std::invalid_argument("KDA conv requires at least two taps");
    const size_t d_inner = checked_mul(profile.heads, profile.head_dim);
    const size_t hist = profile.d_conv - 1;
    const size_t tokens = token_count(x, d_inner, "KDA conv input is not a whole token plane");
    if (weights.size() != checked_mul(d_inner, profile.d_conv))
        throw std::invalid_argument("KDA conv weights do not match the profile");
    if (conv_state.size() != checked_mul(hist, d_inner))
        throw std::invalid_argument("KDA conv state does not match the profile");
    if (out.size() != x.size()) throw std::invalid_argument("KDA conv output size mismatch");
    if (tokens == 0) throw std::invalid_argument("KDA conv requires at least one token");
    demand_finite(x, "KDA conv input must be finite");
    demand_finite(weights, "KDA conv weights must be finite");
    demand_finite(conv_state, "KDA conv state must be finite");

    // The virtual window [state | x]: sample s maps to state row s+hist for s<0
    // and to x[s] otherwise. Each output element is published as soon as it is
    // finite; the state is only advanced after every element succeeded.
    const auto sample = [&](long long s, size_t channel) -> float {
        return s >= 0 ? x[static_cast<size_t>(s) * d_inner + channel]
                      : conv_state[static_cast<size_t>(s + static_cast<long long>(hist)) * d_inner + channel];
    };
    for (size_t t = 0; t < tokens; ++t)
        for (size_t channel = 0; channel < d_inner; ++channel) {
            float acc = 0;
            for (size_t tap = 0; tap < profile.d_conv; ++tap) {
                const float w = weights[channel * profile.d_conv + tap];
                acc = std::fma(w, sample(static_cast<long long>(t) - static_cast<long long>(hist) +
                                         static_cast<long long>(tap), channel), acc);
            }
            if (!std::isfinite(acc)) throw std::overflow_error("KDA conv tap sum overflowed");
            const float silu = acc * sigmoid(acc);
            if (!std::isfinite(silu)) throw std::overflow_error("KDA conv SiLU overflowed");
            out[t * d_inner + channel] = silu;
        }
    for (size_t channel = 0; channel < d_inner; ++channel) {
        // Stage first: with tokens < hist the new state still reads old rows
        // that earlier writes of this loop would already have replaced.
        std::array<float, 64> replacement{};
        if (hist > replacement.size()) throw std::invalid_argument("KDA conv history exceeds the staging bound");
        for (size_t tap = 0; tap < hist; ++tap) {
            const long long s = static_cast<long long>(tokens) - static_cast<long long>(hist) +
                                static_cast<long long>(tap);
            replacement[tap] = sample(s, channel);
        }
        for (size_t tap = 0; tap < hist; ++tap) conv_state[tap * d_inner + channel] = replacement[tap];
    }
}

void kda_decay_gate_reference(std::span<const float> g, std::span<const float> a_log,
                              std::span<float> out, const KdaProfile& profile) {
    const size_t d_inner = checked_mul(profile.heads, profile.head_dim);
    const size_t tokens = token_count(g, d_inner, "KDA gate input is not a whole token plane");
    if (a_log.size() != profile.heads)
        throw std::invalid_argument("KDA gate requires one A_log per head");
    if (out.size() != g.size()) throw std::invalid_argument("KDA gate output size mismatch");
    if (!std::isfinite(profile.gate_lower_bound) || profile.gate_lower_bound >= 0)
        throw std::invalid_argument("KDA safe gate lower bound must be finite and negative");
    demand_finite(g, "KDA gate input must be finite");
    demand_finite(a_log, "KDA gate A_log must be finite");
    for (size_t t = 0; t < tokens; ++t)
        for (uint32_t head = 0; head < profile.heads; ++head) {
            const float decay_rate = std::exp(a_log[head]);
            for (uint32_t i = 0; i < profile.head_dim; ++i) {
                const size_t index = t * d_inner + size_t(head) * profile.head_dim + i;
                out[index] = profile.gate_lower_bound * sigmoid(decay_rate * g[index]);
            }
        }
    // The safe-path result lies in (lower_bound, 0] by construction; the loop
    // above cannot produce a non-finite value from finite inputs because sigmoid
    // saturates to 0 or 1 and lower_bound is finite.
}

void kda_gated_delta_reference(std::span<const float> q, std::span<const float> k,
                               std::span<const float> v, std::span<const float> gate,
                               std::span<const float> beta, std::span<float> state,
                               std::span<float> out, std::span<float> scratch, const KdaProfile& profile) {
    const size_t d_inner = checked_mul(profile.heads, profile.head_dim);
    const size_t dk = profile.head_dim;
    const size_t state_values = kda_state_values(profile);
    const size_t tokens = token_count(q, d_inner, "KDA scan query is not a whole token plane");
    if (k.size() != q.size() || v.size() != q.size() || gate.size() != q.size())
        throw std::invalid_argument("KDA scan k/v/gate planes must match the query plane");
    if (beta.size() != checked_mul(tokens, profile.heads))
        throw std::invalid_argument("KDA scan beta must hold one value per token and head");
    if (state.size() != state_values)
        throw std::invalid_argument("KDA scan state does not match the profile");
    if (out.size() != q.size()) throw std::invalid_argument("KDA scan output size mismatch");
    if (scratch.size() != state_values + out.size())
        throw std::invalid_argument("KDA scan scratch must hold one state and one output image");
    if (tokens == 0) throw std::invalid_argument("KDA scan requires at least one token");
    for (const auto* plane : {&q, &k, &v, &gate, &beta}) demand_finite(*plane, "KDA scan inputs must be finite");
    demand_finite(state, "KDA scan state must be finite");

    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    float* work = scratch.data();                       // state image, then output image
    float* staged = scratch.data() + state_values;
    for (size_t i = 0; i < state_values; ++i) work[i] = state[i];

    for (size_t t = 0; t < tokens; ++t)
        for (uint32_t head = 0; head < profile.heads; ++head) {
            const float* qt = q.data() + t * d_inner + size_t(head) * dk;
            const float* kt = k.data() + t * d_inner + size_t(head) * dk;
            const float* vt = v.data() + t * d_inner + size_t(head) * dk;
            const float* gt = gate.data() + t * d_inner + size_t(head) * dk;
            const float b = beta[t * profile.heads + head];
            float* S = work + size_t(head) * dk * dk;   // row = key axis

            float q_norm = kL2Eps, k_norm = kL2Eps;     // eps inside the sqrt keeps a zero vector finite
            for (size_t i = 0; i < dk; ++i) {
                q_norm = std::fma(qt[i], qt[i], q_norm);
                k_norm = std::fma(kt[i], kt[i], k_norm);
            }
            q_norm = std::sqrt(q_norm); k_norm = std::sqrt(k_norm);
            if (dk > kMaxHeadDim) throw std::invalid_argument("KDA scan head_dim exceeds the staging bound");
            std::array<float, kMaxHeadDim> qh{}, kh{};
            for (size_t i = 0; i < dk; ++i) {
                qh[i] = (qt[i] / q_norm) * scale;
                kh[i] = kt[i] / k_norm;
            }
            for (size_t i = 0; i < dk; ++i) {           // decay applies to the key axis
                const float decay = std::exp(gt[i]);
                for (size_t j = 0; j < dk; ++j) S[i * dk + j] *= decay;
            }
            std::array<float, kMaxHeadDim> u{}, delta{};
            for (size_t j = 0; j < dk; ++j) {
                float acc = 0;
                for (size_t i = 0; i < dk; ++i) acc = std::fma(S[i * dk + j], kh[i], acc);
                u[j] = acc;
            }
            for (size_t j = 0; j < dk; ++j) delta[j] = b * (vt[j] - u[j]);
            for (size_t i = 0; i < dk; ++i)
                for (size_t j = 0; j < dk; ++j) S[i * dk + j] = std::fma(kh[i], delta[j], S[i * dk + j]);
            float* ot = staged + t * d_inner + size_t(head) * dk;
            for (size_t j = 0; j < dk; ++j) {
                float acc = 0;
                for (size_t i = 0; i < dk; ++i) acc = std::fma(S[i * dk + j], qh[i], acc);
                ot[j] = acc;
            }
            for (size_t i = 0; i < dk * dk; ++i)
                if (!std::isfinite(S[i])) throw std::overflow_error("KDA scan state overflowed");
            for (size_t j = 0; j < dk; ++j)
                if (!std::isfinite(ot[j])) throw std::overflow_error("KDA scan output overflowed");
        }
    for (size_t i = 0; i < state_values; ++i) state[i] = work[i];
    for (size_t i = 0; i < out.size(); ++i) out[i] = staged[i];
}

void kda_out_gate_reference(std::span<const float> o, std::span<const float> g2,
                            std::span<const float> norm_w, std::span<float> out,
                            const KdaProfile& profile) {
    const size_t dk = profile.head_dim;
    const size_t d_inner = checked_mul(profile.heads, dk);
    const size_t tokens = token_count(o, d_inner, "KDA out-gate input is not a whole token plane");
    if (g2.size() != o.size()) throw std::invalid_argument("KDA out-gate gate plane mismatch");
    if (norm_w.size() != dk) throw std::invalid_argument("KDA out-gate weight must hold head_dim values");
    if (out.size() != o.size()) throw std::invalid_argument("KDA out-gate output size mismatch");
    demand_finite(o, "KDA out-gate input must be finite");
    demand_finite(g2, "KDA out-gate gate must be finite");
    demand_finite(norm_w, "KDA out-gate weight must be finite");

    // Publication is per element after it validates; o and out may alias, and
    // each output element only reads its own head slice, so an alias is safe.
    for (size_t t = 0; t < tokens; ++t)
        for (uint32_t head = 0; head < profile.heads; ++head) {
            const float* ot = o.data() + t * d_inner + size_t(head) * dk;
            const float* gt = g2.data() + t * d_inner + size_t(head) * dk;
            float mean = 0;
            for (size_t i = 0; i < dk; ++i) mean = std::fma(ot[i], ot[i], mean);
            mean /= static_cast<float>(dk);
            const float inv = 1.0f / std::sqrt(mean + kNormEps);
            for (size_t i = 0; i < dk; ++i) {
                const float value = ot[i] * inv * norm_w[i] * sigmoid(gt[i]);
                if (!std::isfinite(value)) throw std::overflow_error("KDA out-gate overflowed");
                out[t * d_inner + size_t(head) * dk + i] = value;
            }
        }
}

}  // namespace strata::hpu
