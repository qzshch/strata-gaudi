#include "strata/hpu/kda_reference.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;

namespace {

// Deterministic fixture: exact values in binary32 keep oracle tolerances tight.
float source(uint32_t i) {
    return static_cast<float>(static_cast<int>(i % 19u) - 9) / 4.0f;
}

// ---------------------------------------------------------------------------
// Independent long-double oracle assembled from the FLA naive formulation
// (S *= exp(g); S += (beta*k) outer (v - S^T k); o = S^T q with q pre-scaled),
// deliberately written in a different association order than the C++ loop.
struct ScanOracle {
    long double sq(long double v) const { return v * v; }
    long double l2(const float* x) const {
        long double sum = 1e-6L;
        for (uint32_t i = 0; i < profile.head_dim; ++i) sum += sq(x[i]);
        return std::sqrt(sum);
    }
    explicit ScanOracle(KdaProfile p) : profile(p) {}

    void token(const float* q, const float* k, const float* v, const float* g, long double beta,
               std::vector<long double>& S, float* o) const {
        const uint32_t dk = profile.head_dim;
        const long double scale = 1.0L / std::sqrt(static_cast<long double>(dk));
        std::vector<long double> qh(dk), kh(dk);
        for (uint32_t i = 0; i < dk; ++i) {
            qh[i] = q[i] / l2(q) * scale;
            kh[i] = k[i] / l2(k);
        }
        std::vector<long double> decay(dk);
        for (uint32_t i = 0; i < dk; ++i) decay[i] = std::exp(static_cast<long double>(g[i]));
        // S is [key][value]; FLA writes S = S*exp(g) then the rank-one update.
        for (uint32_t i = 0; i < dk; ++i)
            for (uint32_t j = 0; j < dk; ++j) S[static_cast<size_t>(i) * dk + j] *= decay[i];
        for (uint32_t j = 0; j < dk; ++j) {
            long double u = 0;
            for (uint32_t i = 0; i < dk; ++i) u += kh[i] * S[static_cast<size_t>(i) * dk + j];
            const long double delta = beta * (v[j] - u);
            for (uint32_t i = 0; i < dk; ++i) S[static_cast<size_t>(i) * dk + j] += kh[i] * delta;
        }
        for (uint32_t j = 0; j < dk; ++j) {
            long double out = 0;
            for (uint32_t i = 0; i < dk; ++i) out += qh[i] * S[static_cast<size_t>(i) * dk + j];
            o[j] = static_cast<float>(out);
        }
    }

    KdaProfile profile;
};

void compare(const std::vector<float>& actual, const std::vector<long double>& expected,
             long double tolerance, const char* what) {
    CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const long double reference = expected[i] == 0 ? 0 : expected[i];
        const long double error = std::abs(static_cast<long double>(actual[i]) - reference);
        CHECK(error <= tolerance * std::max<long double>(1, std::abs(reference)));
        if (!(error <= tolerance * std::max<long double>(1, std::abs(reference))))
            std::fprintf(stderr, "%s[%zu]=%e vs %Le\n", what, i, actual[i], reference);
    }
}

// ---------------------------------------------------------------------------
void conv_oracle_and_continuity() {
    const KdaProfile profile{3, 4, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    const size_t hist = profile.d_conv - 1;

    std::vector<float> weights(d_inner * profile.d_conv);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = source(static_cast<uint32_t>(i) + 1);
    std::vector<float> state(hist * d_inner);
    for (size_t i = 0; i < state.size(); ++i) state[i] = source(static_cast<uint32_t>(i) * 3 + 5) / 2;

    for (const size_t tokens : {size_t{1}, size_t{2}, size_t{5}}) {
        std::vector<float> x(tokens * d_inner);
        for (size_t i = 0; i < x.size(); ++i) x[i] = source(static_cast<uint32_t>(i) + 7);

        std::vector<float> one_out(tokens * d_inner, -1e30f), one_state = state;
        kda_conv_reference(x, weights, one_state, one_out, profile);

        // Independent oracle over the virtual [state | x] window.
        std::vector<float> expected(tokens * d_inner);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t c = 0; c < d_inner; ++c) {
                long double acc = 0;
                for (uint32_t tap = 0; tap < profile.d_conv; ++tap) {
                    const long long s = static_cast<long long>(t) - static_cast<long long>(hist) + tap;
                    const long double sample = s >= 0 ? x[static_cast<size_t>(s) * d_inner + c]
                                                      : state[static_cast<size_t>(s + static_cast<long long>(hist)) * d_inner + c];
                    acc += weights[c * profile.d_conv + tap] * sample;
                }
                const long double silu = acc / (1.0L + std::exp(-acc));
                expected[t * d_inner + c] = static_cast<float>(silu);
            }
        compare(one_out, std::vector<long double>(expected.begin(), expected.end()), 3e-6L, "conv out");

        // One-shot must equal token-by-token, bit for bit, including the state.
        std::vector<float> split_out, split_state = state;
        for (size_t t = 0; t < tokens; ++t) {
            std::vector<float> token(d_inner), out(d_inner);
            for (size_t c = 0; c < d_inner; ++c) token[c] = x[t * d_inner + c];
            kda_conv_reference(token, weights, split_state, out, profile);
            split_out.insert(split_out.end(), out.begin(), out.end());
        }
        CHECK(one_out == split_out);
        CHECK(one_state == split_state);
    }
}

// ---------------------------------------------------------------------------
void gate_matches_kernel_formula() {
    const KdaProfile profile{2, 3, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    const size_t tokens = 2;
    std::vector<float> g(tokens * d_inner), a_log(profile.heads), out(tokens * d_inner);
    for (size_t i = 0; i < g.size(); ++i) g[i] = source(static_cast<uint32_t>(i) + 3);
    a_log[0] = -0.5f; a_log[1] = 0.25f;
    kda_decay_gate_reference(g, a_log, out, profile);
    for (size_t t = 0; t < tokens; ++t)
        for (uint32_t h = 0; h < profile.heads; ++h) {
            const long double rate = std::exp(static_cast<long double>(a_log[h]));
            for (uint32_t i = 0; i < profile.head_dim; ++i) {
                const size_t index = t * d_inner + size_t(h) * profile.head_dim + i;
                const long double expected = profile.gate_lower_bound /
                    (1.0L + std::exp(-rate * g[index]));
                CHECK(std::abs(static_cast<long double>(out[index]) - expected) < 1e-6L);
                CHECK(out[index] < 0 && out[index] > profile.gate_lower_bound);
            }
        }
}

// ---------------------------------------------------------------------------
void scan_oracle_split_and_state_contract() {
    const KdaProfile profile{2, 4, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    const size_t state_values = kda_state_values(profile);
    const size_t tokens = 5;

    std::vector<float> q(tokens * d_inner), k(tokens * d_inner), v(tokens * d_inner),
        gate(tokens * d_inner), beta(tokens * profile.heads);
    for (size_t i = 0; i < tokens * d_inner; ++i) {
        q[i] = source(static_cast<uint32_t>(i) * 2 + 1);
        k[i] = source(static_cast<uint32_t>(i) * 2 + 2);
        v[i] = source(static_cast<uint32_t>(i) * 2 + 3) / 2;
        gate[i] = source(static_cast<uint32_t>(i) * 2 + 4) / 3;
    }
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.25f + 0.5f * (static_cast<float>(i % 3) / 3.0f);

    std::vector<float> state(state_values), out(tokens * d_inner, -1e30f),
        scratch(kda_scan_scratch_values(profile, tokens));
    for (size_t i = 0; i < state.size(); ++i) state[i] = source(static_cast<uint32_t>(i)) / 5;
    const std::vector<float> initial_state = state;

    kda_gated_delta_reference(q, k, v, gate, beta, state, out, scratch, profile);

    // Independent oracle per head, straight from the FLA naive recurrence.
    const ScanOracle oracle(profile);
    std::vector<float> expected(tokens * d_inner);
    for (uint32_t h = 0; h < profile.heads; ++h) {
        std::vector<long double> S(state_values);
        for (size_t i = 0; i < state_values; ++i) S[i] = initial_state[h * profile.head_dim * profile.head_dim + i];
        for (size_t t = 0; t < tokens; ++t) {
            const size_t base = t * d_inner + size_t(h) * profile.head_dim;
            oracle.token(&q[base], &k[base], &v[base], &gate[base], beta[t * profile.heads + h], S,
                         &expected[base]);
        }
    }
    compare(out, std::vector<long double>(expected.begin(), expected.end()), 2e-5L, "scan out");
    // 20^2 accumulations over 5 tokens stay far from FP32 limits, but the state
    // must equal running the oracle's FP32 rounding, checked via the split run.

    // Token-by-token must reproduce the one-shot run bit for bit, including state.
    std::vector<float> split_state = initial_state, split_out;
    std::vector<float> one(q.begin(), q.begin() + d_inner), onek(k.begin(), k.begin() + d_inner),
        onev(v.begin(), v.begin() + d_inner), oneg(gate.begin(), gate.begin() + d_inner);
    std::vector<float> token_scratch(kda_scan_scratch_values(profile, 1)), token_out(d_inner);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t c = 0; c < d_inner; ++c) {
            one[c] = q[t * d_inner + c]; onek[c] = k[t * d_inner + c];
            onev[c] = v[t * d_inner + c]; oneg[c] = gate[t * d_inner + c];
        }
        std::vector<float> token_beta(beta.begin() + t * profile.heads,
                                      beta.begin() + (t + 1) * profile.heads);
        kda_gated_delta_reference(one, onek, onev, oneg, token_beta, split_state, token_out,
                                  token_scratch, profile);
        split_out.insert(split_out.end(), token_out.begin(), token_out.end());
    }
    CHECK(out == split_out);
    CHECK(state == split_state);

    // Fresh state stays zero-preserving: q=k=v=0 must leave the state untouched
    // apart from decay and write o = decayed state times zero query... instead
    // assert the practical invariant: zero inputs leave o finite and state decayed.
    std::vector<float> zeros(tokens * d_inner, 0), zero_beta(tokens * profile.heads, 0),
        zero_state = initial_state, zero_out(tokens * d_inner, -1e30f);
    kda_gated_delta_reference(zeros, zeros, zeros, gate, zero_beta, zero_state, zero_out, scratch, profile);
    CHECK(zero_out == std::vector<float>(tokens * d_inner, 0));  // q=0 => o=0 exactly
}

// ---------------------------------------------------------------------------
void scan_failure_keeps_planes() {
    const KdaProfile profile{1, 2, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    std::vector<float> q(d_inner, 1), k(d_inner, 1), v(d_inner, 1), gate(d_inner, -1),
        beta(profile.heads, 0.5f), state(kda_state_values(profile), 0),
        out(d_inner, -7), scratch(kda_scan_scratch_values(profile, 1));
    // Non-finite input is rejected before anything is modified. A finite huge
    // magnitude is NOT rejected at the gate: the l2 norm maps it toward zero,
    // matching the FP32 kernel behaviour.
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    q[0] = nan;
    try {
        kda_gated_delta_reference(q, k, v, gate, beta, state, out, scratch, profile);
        throw std::runtime_error("NaN scan input accepted");
    } catch (const std::invalid_argument&) {
    }
    q[0] = std::numeric_limits<float>::max();
    k[1] = std::numeric_limits<float>::max();
    kda_gated_delta_reference(q, k, v, gate, beta, state, out, scratch, profile);
    CHECK(state == std::vector<float>(kda_state_values(profile), 0));  // k normalizes away
    q[0] = 1; k[1] = 1;
    // Arithmetic overflow: an out-of-range (still finite) beta multiplies the
    // delta rule past binary32 range mid-scan; both planes must stay untouched.
    // Unit-normalized q/k with in-range beta alone cannot overflow a dk=2 head
    // (outputs stay below 0.71*FLT_MAX), so the guard is what stands between a
    // caller-side sigmoid bug and a silently corrupt state.
    CHECK(out != std::vector<float>(d_inner, -7));  // the run above published
    std::fill(out.begin(), out.end(), -7.0f);
    v[0] = std::numeric_limits<float>::max(); v[1] = std::numeric_limits<float>::max();
    beta[0] = 4;
    try {
        kda_gated_delta_reference(q, k, v, gate, beta, state, out, scratch, profile);
        throw std::runtime_error("overflowing scan accepted");
    } catch (const std::overflow_error&) {
    }
    CHECK(out == std::vector<float>(d_inner, -7));
    CHECK(state == std::vector<float>(kda_state_values(profile), 0));
}

// ---------------------------------------------------------------------------
void out_gate_oracle_and_alias() {
    const KdaProfile profile{2, 3, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    const size_t tokens = 3;
    std::vector<float> o(tokens * d_inner), g2(tokens * d_inner), norm_w(profile.head_dim);
    for (size_t i = 0; i < o.size(); ++i) o[i] = source(static_cast<uint32_t>(i) + 11) / 3;
    for (size_t i = 0; i < g2.size(); ++i) g2[i] = source(static_cast<uint32_t>(i) + 13) / 2;
    for (size_t i = 0; i < norm_w.size(); ++i) norm_w[i] = 1 + source(static_cast<uint32_t>(i)) / 9;

    const std::vector<float> input = o;
    kda_out_gate_reference(o, g2, norm_w, o, profile);  // in-place alias must work
    for (size_t t = 0; t < tokens; ++t)
        for (uint32_t h = 0; h < profile.heads; ++h) {
            const size_t base = t * d_inner + size_t(h) * profile.head_dim;
            long double mean = 0;
            for (uint32_t i = 0; i < profile.head_dim; ++i)
                mean += static_cast<long double>(input[base + i]) * input[base + i];
            mean /= profile.head_dim;
            const long double inv = 1.0L / std::sqrt(mean + 1e-6L);
            for (uint32_t i = 0; i < profile.head_dim; ++i) {
                const long double expected = input[base + i] * inv * norm_w[i] /
                    (1.0L + std::exp(-static_cast<long double>(g2[base + i])));
                CHECK(std::abs(static_cast<long double>(o[base + i]) - expected) < 1e-5L);
            }
        }
}

// ---------------------------------------------------------------------------
void glm53_dimensions_smoke() {
    // The real 64x128 profile at small T: bit-level split continuity plus the
    // audited constant set, without a long-double oracle over the full geometry.
    CHECK(kGlm53KdaProfile.heads == 64 && kGlm53KdaProfile.head_dim == 128);
    CHECK(kGlm53KdaProfile.d_conv == 4 && kGlm53KdaProfile.gate_lower_bound == -5.0f);
    const KdaProfile profile = kGlm53KdaProfile;
    const size_t d_inner = size_t(profile.heads) * profile.head_dim;
    const size_t tokens = 4;
    std::vector<float> q(tokens * d_inner), k(tokens * d_inner), v(tokens * d_inner),
        gate(tokens * d_inner), beta(tokens * profile.heads), state(kda_state_values(profile)),
        out(tokens * d_inner), scratch(kda_scan_scratch_values(profile, tokens));
    for (size_t i = 0; i < tokens * d_inner; ++i) {
        q[i] = source(static_cast<uint32_t>(i));
        k[i] = source(static_cast<uint32_t>(i) + 1);
        v[i] = source(static_cast<uint32_t>(i) + 2);
        gate[i] = source(static_cast<uint32_t>(i) + 3) / 4;
    }
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.5f;
    kda_gated_delta_reference(q, k, v, gate, beta, state, out, scratch, profile);
    std::vector<float> split_state(kda_state_values(profile)),
        split_out, token_scratch(kda_scan_scratch_values(profile, 1));
    for (size_t t = 0; t < tokens; ++t) {
        std::vector<float> token_out(d_inner);
        kda_gated_delta_reference(std::span(q).subspan(t * d_inner, d_inner),
                                  std::span(k).subspan(t * d_inner, d_inner),
                                  std::span(v).subspan(t * d_inner, d_inner),
                                  std::span(gate).subspan(t * d_inner, d_inner),
                                  std::span(beta).subspan(t * profile.heads, profile.heads),
                                  split_state, token_out, token_scratch, profile);
        split_out.insert(split_out.end(), token_out.begin(), token_out.end());
    }
    CHECK(out == split_out);
    CHECK(state == split_state);
}

// ---------------------------------------------------------------------------
template<class F> void rejects(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid KDA reference input accepted");
}

void invalid_inputs() {
    const KdaProfile profile{2, 3, 4, -5.0f};
    const size_t d_inner = profile.heads * profile.head_dim;
    std::vector<float> plane(d_inner), weights(d_inner * profile.d_conv),
        conv_state((profile.d_conv - 1) * d_inner), out(d_inner);
    rejects([&] { (void)kda_conv_reference(std::span(plane).first(d_inner - 1), weights, conv_state, out, profile); });
    rejects([&] { (void)kda_conv_reference(plane, std::span(weights).first(weights.size() - 1), conv_state, out, profile); });
    rejects([&] { (void)kda_conv_reference(plane, weights, std::span(conv_state).first(conv_state.size() - 1), out, profile); });
    const auto bad = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> poisoned = plane; poisoned[0] = bad;
    rejects([&] { (void)kda_conv_reference(poisoned, weights, conv_state, out, profile); });

    std::vector<float> g(d_inner), a_log(profile.heads), gate_out(d_inner);
    rejects([&] { (void)kda_decay_gate_reference(g, std::span(a_log).first(1), gate_out, profile); });
    std::vector<float> poisoned_log = a_log; poisoned_log[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)kda_decay_gate_reference(g, poisoned_log, gate_out, profile); });

    std::vector<float> state(kda_state_values(profile)), beta(profile.heads),
        scratch(kda_scan_scratch_values(profile, 1));
    rejects([&] { (void)kda_gated_delta_reference(std::span(plane).first(d_inner - 1), plane, plane, plane,
                                                  beta, state, out, scratch, profile); });
    rejects([&] { (void)kda_gated_delta_reference(plane, plane, plane, plane, std::span(beta).first(1),
                                                  state, out, scratch, profile); });
    rejects([&] { (void)kda_gated_delta_reference(plane, plane, plane, plane, beta,
                                                  std::span(state).first(state.size() - 1), out, scratch, profile); });
    rejects([&] { (void)kda_gated_delta_reference(plane, plane, plane, plane, beta, state, out,
                                                  std::span(scratch).first(scratch.size() - 1), profile); });
    std::vector<float> poisoned_gate = plane; poisoned_gate[1] = bad;
    rejects([&] { (void)kda_gated_delta_reference(plane, plane, plane, poisoned_gate, beta, state, out,
                                                  scratch, profile); });

    std::vector<float> norm_w(profile.head_dim);
    rejects([&] { (void)kda_out_gate_reference(plane, std::span(plane).first(d_inner - 1), norm_w, out, profile); });
    rejects([&] { (void)kda_out_gate_reference(plane, plane, std::span(norm_w).first(1), out, profile); });
    std::vector<float> poisoned_o = plane; poisoned_o[2] = bad;
    rejects([&] { (void)kda_out_gate_reference(poisoned_o, plane, norm_w, out, profile); });
}

}  // namespace

int main() {
    try {
        conv_oracle_and_continuity();
        gate_matches_kernel_formula();
        scan_oracle_split_and_state_contract();
        scan_failure_keeps_planes();
        out_gate_oracle_and_alias();
        glm53_dimensions_smoke();
        invalid_inputs();
        std::puts("GLM KDA conv/gate/delta/out-gate references match independent oracles");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
