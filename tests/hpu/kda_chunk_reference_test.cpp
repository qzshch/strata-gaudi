#include "strata/hpu/kda_chunk_reference.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;

namespace {

float source(uint32_t i) {
    return static_cast<float>(static_cast<int>(i % 19u) - 9) / 4.0f;
}

struct Planes {
    KdaProfile profile{2, 4, 4, -5.0f};
    size_t tokens = 8, d_inner, state_values;
    std::vector<float> q, k, v, gate, beta, state;
    explicit Planes(size_t t = 8) : tokens(t) {
        d_inner = size_t(profile.heads) * profile.head_dim;
        state_values = kda_state_values(profile);
        q.resize(tokens * d_inner); k.resize(tokens * d_inner); v.resize(tokens * d_inner);
        gate.resize(tokens * d_inner);
        beta.resize(tokens * profile.heads);
        for (size_t i = 0; i < tokens * d_inner; ++i) {
            q[i] = source(static_cast<uint32_t>(i) * 2 + 1);
            k[i] = source(static_cast<uint32_t>(i) * 2 + 2);
            v[i] = source(static_cast<uint32_t>(i) * 2 + 3) / 2;
            gate[i] = source(static_cast<uint32_t>(i) * 2 + 4) / 3 - 0.25f;   // log-space, <= 0-ish
        }
        for (size_t i = 0; i < beta.size(); ++i) beta[i] = 0.25f + 0.5f * (i % 3) / 3.0f;
        state.resize(state_values);
        for (size_t i = 0; i < state.size(); ++i) state[i] = source(static_cast<uint32_t>(i)) / 5;
    }
};

// Independent long-double oracle: the RECURRENT math (not the chunk algorithm),
// so agreement proves the chunk reorganization, not a shared bug.
struct RecurrentOracle {
    explicit RecurrentOracle(const KdaProfile& p) : profile(p) {}
    long double l2(const float* x) const {
        long double sum = 1e-6L;
        for (uint32_t i = 0; i < profile.head_dim; ++i) sum += x[i] * static_cast<long double>(x[i]);
        return std::sqrt(sum);
    }
    void run(const Planes& planes, std::vector<float>& out, std::vector<float>& final_state) const {
        const uint32_t dk = profile.head_dim;
        const size_t d_inner = size_t(profile.heads) * dk;
        const long double scale = 1.0L / std::sqrt(static_cast<long double>(dk));
        std::vector<long double> S(planes.state_values);
        for (size_t i = 0; i < planes.state_values; ++i) S[i] = planes.state[i];
        out.assign(planes.tokens * d_inner, 0);
        for (size_t t = 0; t < planes.tokens; ++t)
            for (uint32_t h = 0; h < profile.heads; ++h) {
                const size_t base = t * d_inner + size_t(h) * dk;
                const float* qt = &planes.q[base]; const float* kt = &planes.k[base];
                const float* vt = &planes.v[base]; const float* gt = &planes.gate[base];
                const long double qn = l2(qt), kn = l2(kt);
                std::vector<long double> qh(dk), kh(dk);
                for (uint32_t i = 0; i < dk; ++i) {
                    qh[i] = qt[i] / qn * scale;
                    kh[i] = kt[i] / kn;
                }
                long double* Sh = S.data() + size_t(h) * dk * dk;
                for (uint32_t i = 0; i < dk; ++i) {
                    const long double decay = std::exp(static_cast<long double>(gt[i]));
                    for (uint32_t j = 0; j < dk; ++j) Sh[static_cast<size_t>(i) * dk + j] *= decay;
                }
                for (uint32_t j = 0; j < dk; ++j) {
                    long double u = 0;
                    for (uint32_t i = 0; i < dk; ++i)
                        u += kh[i] * Sh[static_cast<size_t>(i) * dk + j];
                    const long double delta =
                        planes.beta[t * profile.heads + h] * (vt[j] - u);
                    for (uint32_t i = 0; i < dk; ++i)
                        Sh[static_cast<size_t>(i) * dk + j] += kh[i] * delta;
                }
                for (uint32_t j = 0; j < dk; ++j) {
                    long double acc = 0;
                    for (uint32_t i = 0; i < dk; ++i)
                        acc += qh[i] * Sh[static_cast<size_t>(i) * dk + j];
                    out[base + j] = static_cast<float>(acc);
                }
            }
        final_state.assign(planes.state_values, 0);
        for (size_t i = 0; i < planes.state_values; ++i)
            final_state[i] = static_cast<float>(S[i]);
    }
    KdaProfile profile;
};

void compare_span(const std::vector<float>& got, const std::vector<long double>& want,
                  long double tolerance, const char* what) {
    CHECK(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const long double error = std::abs(static_cast<long double>(got[i]) - want[i]);
        if (!(error <= tolerance * std::max<long double>(1, std::abs(want[i]))))
            std::fprintf(stderr, "%s[%zu]=%.9g want %.9Le\n", what, i, got[i], want[i]);
        CHECK(error <= tolerance * std::max<long double>(1, std::abs(want[i])));
    }
}

void chunk_matches_oracle_and_recurrent() {
    const Planes planes;
    const RecurrentOracle oracle(planes.profile);

    std::vector<float> oracle_out, oracle_state;
    oracle.run(planes, oracle_out, oracle_state);

    // The chunk algorithm at several widths, all against the oracle.
    for (const uint32_t chunk : {uint32_t{2}, uint32_t{4}, uint32_t{8}}) {
        std::vector<float> state = planes.state, out(planes.tokens * planes.d_inner, -1e30f),
            scratch(kda_chunk_scratch_values(planes.profile, planes.tokens, chunk));
        kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta, state, out,
                            scratch, planes.profile, chunk);
        compare_span(out, {oracle_out.begin(), oracle_out.end()}, 2e-4L, "chunk out");
        compare_span(state, {oracle_state.begin(), oracle_state.end()}, 2e-4L, "chunk state");
    }

    // The fp32 recurrent reference on the same inputs: both algorithms are
    // FP32 with different associations; they must agree to the same band.
    std::vector<float> recurrent_state = planes.state, recurrent_out(planes.tokens * planes.d_inner),
        recurrent_scratch(kda_scan_scratch_values(planes.profile, planes.tokens));
    kda_gated_delta_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                              recurrent_state, recurrent_out, recurrent_scratch, planes.profile);
    std::vector<float> chunk_state = planes.state, chunk_out(planes.tokens * planes.d_inner),
        chunk_scratch(kda_chunk_scratch_values(planes.profile, planes.tokens, 4));
    kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta, chunk_state,
                        chunk_out, chunk_scratch, planes.profile, 4);
    compare_span(chunk_out, {recurrent_out.begin(), recurrent_out.end()}, 5e-4L, "chunk vs recurrent out");
    compare_span(chunk_state, {recurrent_state.begin(), recurrent_state.end()}, 5e-4L,
                 "chunk vs recurrent state");

    // Continuation: 4 tokens chunked, then 4 more chunked must equal the
    // 8-token single call to the same tolerance band.
    Planes split(8);
    std::vector<float> split_state = planes.state, split_out;
    for (size_t part = 0; part < 2; ++part) {
        std::vector<float> part_out(4 * planes.d_inner),
            part_scratch(kda_chunk_scratch_values(planes.profile, 4, 4));
        const size_t offset = part * 4 * planes.d_inner;
        kda_chunk_reference(std::span(planes.q).subspan(offset, 4 * planes.d_inner),
                            std::span(planes.k).subspan(offset, 4 * planes.d_inner),
                            std::span(planes.v).subspan(offset, 4 * planes.d_inner),
                            std::span(planes.gate).subspan(offset, 4 * planes.d_inner),
                            std::span(planes.beta).subspan(part * 4 * planes.profile.heads, 4 * planes.profile.heads),
                            split_state, part_out, part_scratch, planes.profile, 4);
        split_out.insert(split_out.end(), part_out.begin(), part_out.end());
    }
    compare_span(split_out, {chunk_out.begin(), chunk_out.end()}, 5e-4L, "split vs whole out");
    compare_span(split_state, {chunk_state.begin(), chunk_state.end()}, 5e-4L, "split vs whole state");
}

void zero_padding_rows_are_inert() {
    // The documented tail contract: padding rows carry g=0 and q/k/v/beta=0
    // and must not disturb real outputs or the state - bitwise.
    Planes real(4);
    std::vector<float> state_a = real.state, out_a(4 * real.d_inner),
        scratch_a(kda_chunk_scratch_values(real.profile, 4, 4));
    kda_chunk_reference(real.q, real.k, real.v, real.gate, real.beta, state_a, out_a, scratch_a,
                        real.profile, 4);

    Planes padded(8);   // rewrite the first 4 tokens from `real`
    for (size_t i = 0; i < 4 * real.d_inner; ++i) {
        padded.q[i] = real.q[i]; padded.k[i] = real.k[i];
        padded.v[i] = real.v[i]; padded.gate[i] = real.gate[i];
    }
    for (size_t i = 0; i < 4 * real.profile.heads; ++i) padded.beta[i] = real.beta[i];
    for (size_t i = 4 * real.d_inner; i < 8 * real.d_inner; ++i) {
        padded.q[i] = 0; padded.k[i] = 0; padded.v[i] = 0; padded.gate[i] = 0;
    }
    for (size_t i = 4 * real.profile.heads; i < 8 * real.profile.heads; ++i) padded.beta[i] = 0;
    std::vector<float> state_b = real.state, out_b(8 * real.d_inner),
        scratch_b(kda_chunk_scratch_values(real.profile, 8, 4));
    kda_chunk_reference(padded.q, padded.k, padded.v, padded.gate, padded.beta, state_b, out_b,
                        scratch_b, real.profile, 4);
    CHECK(state_b == state_a);
    for (size_t t = 0; t < 4; ++t)
        for (size_t i = 0; i < real.d_inner; ++i)
            CHECK(out_b[t * real.d_inner + i] == out_a[t * real.d_inner + i]);
    for (size_t i = 4 * real.d_inner; i < 8 * real.d_inner; ++i) CHECK(out_b[i] == 0.0f);
}

void glm53_dimensions_smoke() {
    static_assert(kGlm53KdaChunk == 64, "production chunk width is part of the audited profile");
    const KdaProfile profile = kGlm53KdaProfile;
    const size_t tokens = 128, d_inner = size_t(profile.heads) * profile.head_dim;
    Planes planes(128);
    planes.profile = profile;
    planes.d_inner = d_inner;
    planes.q.resize(tokens * d_inner); planes.k.resize(tokens * d_inner);
    planes.v.resize(tokens * d_inner); planes.gate.resize(tokens * d_inner);
    planes.beta.resize(tokens * profile.heads);
    planes.state.assign(kda_state_values(profile), 0);
    for (size_t i = 0; i < tokens * d_inner; ++i) {
        planes.q[i] = source(static_cast<uint32_t>(i));
        planes.k[i] = source(static_cast<uint32_t>(i) + 1);
        planes.v[i] = source(static_cast<uint32_t>(i) + 2) / 2;
        planes.gate[i] = source(static_cast<uint32_t>(i) + 3) / 8 - 0.5f;
    }
    for (size_t i = 0; i < planes.beta.size(); ++i) planes.beta[i] = 0.5f;

    std::vector<float> chunk_state = planes.state, chunk_out(tokens * d_inner),
        chunk_scratch(kda_chunk_scratch_values(profile, tokens, 64));
    kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta, chunk_state,
                        chunk_out, chunk_scratch, profile, 64);
    std::vector<float> recurrent_state = planes.state, recurrent_out(tokens * d_inner),
        recurrent_scratch(kda_scan_scratch_values(profile, tokens));
    kda_gated_delta_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                              recurrent_state, recurrent_out, recurrent_scratch, profile);
    compare_span(chunk_out, {recurrent_out.begin(), recurrent_out.end()}, 3e-3L,
                 "glm53 chunk vs recurrent out");
    compare_span(chunk_state, {recurrent_state.begin(), recurrent_state.end()}, 3e-3L,
                 "glm53 chunk vs recurrent state");
}

template<class F> void rejects(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid KDA chunk input accepted");
}

void invalid_inputs() {
    const Planes planes(8);
    std::vector<float> state = planes.state, out(planes.tokens * planes.d_inner);
    std::vector<float> scratch4(kda_chunk_scratch_values(planes.profile, 8, 4));
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                                            state, out, scratch4, planes.profile, 3); });   // 8 % 3 != 0
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                                            state, out, scratch4, planes.profile, 0); });
    rejects([&] { (void)kda_chunk_reference(std::span(planes.q).first(planes.q.size() - 1), planes.k,
                                            planes.v, planes.gate, planes.beta, state, out,
                                            scratch4, planes.profile, 4); });
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate,
                                            std::span(planes.beta).first(3), state, out,
                                            scratch4, planes.profile, 4); });
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                                            std::span(state).first(state.size() - 1), out,
                                            scratch4, planes.profile, 4); });
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, planes.gate, planes.beta,
                                            state, out, std::span(scratch4).first(scratch4.size() - 1),
                                            planes.profile, 4); });
    auto poisoned = planes.gate;
    poisoned[5] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)kda_chunk_reference(planes.q, planes.k, planes.v, poisoned, planes.beta,
                                            state, out, scratch4, planes.profile, 4); });
}

}  // namespace

int main() {
    try {
        chunk_matches_oracle_and_recurrent();
        zero_padding_rows_are_inert();
        glm53_dimensions_smoke();
        invalid_inputs();
        std::puts("GLM KDA chunk reference matches recurrent/oracle at every chunk width");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
