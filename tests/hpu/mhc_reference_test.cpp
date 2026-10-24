#include "strata/hpu/mhc_reference.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;

namespace {

float source(uint32_t i) {
    return static_cast<float>(static_cast<int>(i % 17u) - 8) / 8.0f;
}

template<class F> void rejects(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid mHC input accepted");
}

struct Fixture {
    MhcProfile profile = kGlm53MhcProfile;
    uint32_t hidden = 12;
    size_t tokens = 3, hc, stream_values, mix_values;
    std::vector<float> streams, fn, base, scale;
    explicit Fixture(uint32_t h = 12, size_t t = 3) : hidden(h), tokens(t) {
        hc = profile.hc;
        stream_values = hc * hidden;
        mix_values = (2 + hc) * hc;
        streams.resize(tokens * stream_values);
        fn.resize(mix_values * stream_values);
        base.resize(mix_values);
        scale = {0.7f, 0.5f, 0.9f};
        for (size_t i = 0; i < streams.size(); ++i) streams[i] = source(static_cast<uint32_t>(i) + 5) / 4;
        for (size_t i = 0; i < fn.size(); ++i) fn[i] = source(static_cast<uint32_t>(i) * 3 + 1) / 16;
        for (size_t i = 0; i < base.size(); ++i) base[i] = source(static_cast<uint32_t>(i) * 5 + 3) / 8;
    }
};

// Independent long-double oracle, re-derived from the transformers module:
// unweighted RMS norm over the flattened streams, sigmoid/2x-sigmoid gates,
// softmax over dst per src row from the flat comb[dst + hc*src] layout, the
// pinned Sinkhorn order, and the layer update. Written in a different
// association order than the C++ loops.
struct Oracle {
    const Fixture& f;
    long double sigmoid(long double x) const { return 1.0L / (1.0L + std::exp(-x)); }

    void weights(size_t t, std::vector<long double>& pre, std::vector<long double>& post,
                 std::vector<long double>& comb) const {
        const size_t hc = f.hc, sv = f.stream_values;
        long double square_sum = 0;
        for (size_t i = 0; i < sv; ++i) {
            const long double v = f.streams[t * sv + i];
            square_sum += v * v;
        }
        const long double inv = 1.0L / std::sqrt(square_sum / static_cast<long double>(sv) + f.profile.rms_eps);
        std::vector<long double> mixes(f.mix_values);
        for (size_t m = 0; m < f.mix_values; ++m)
            for (size_t i = 0; i < sv; ++i)
                mixes[m] += static_cast<long double>(f.fn[m * sv + i]) * f.streams[t * sv + i];
        for (auto& m : mixes) m *= inv;

        pre.resize(hc); post.resize(hc); comb.assign(hc * hc, 0);
        for (size_t n = 0; n < hc; ++n) {
            pre[n] = sigmoid(mixes[n] * f.scale[0] + f.base[n]) + f.profile.hc_eps;
            post[n] = 2.0L * sigmoid(mixes[hc + n] * f.scale[1] + f.base[hc + n]);
        }
        for (size_t src = 0; src < hc; ++src) {
            long double denominator = 0;
            std::vector<long double> exps(hc);
            for (size_t dst = 0; dst < hc; ++dst) {
                const size_t flat = 2 * hc + dst + hc * src;
                exps[dst] = std::exp((mixes[flat] * f.scale[2] + f.base[flat]) -
                                     (mixes[2 * hc + 0 + hc * src] * f.scale[2] + f.base[2 * hc + 0 + hc * src]));
                denominator += exps[dst];
            }
            for (size_t dst = 0; dst < hc; ++dst)
                comb[src * hc + dst] = exps[dst] / denominator + f.profile.hc_eps;
        }
        auto col = [&]() {
            for (size_t dst = 0; dst < hc; ++dst) {
                long double sum = 0;
                for (size_t src = 0; src < hc; ++src) sum += comb[src * hc + dst];
                const long double inv_sum = 1.0L / (sum + f.profile.hc_eps);
                for (size_t src = 0; src < hc; ++src) comb[src * hc + dst] *= inv_sum;
            }
        };
        auto row = [&]() {
            for (size_t src = 0; src < hc; ++src) {
                long double sum = 0;
                for (size_t dst = 0; dst < hc; ++dst) sum += comb[src * hc + dst];
                const long double inv_sum = 1.0L / (sum + f.profile.hc_eps);
                for (size_t dst = 0; dst < hc; ++dst) comb[src * hc + dst] *= inv_sum;
            }
        };
        col();
        for (uint32_t r = 1; r < f.profile.sinkhorn_iters; ++r) { row(); col(); }
    }
};

void compare(float got, long double want, long double tolerance, const char* what) {
    const long double error = std::abs(static_cast<long double>(got) - want);
    if (!(error <= tolerance * std::max<long double>(1, std::abs(want))))
        std::fprintf(stderr, "%s: got %.9g want %.9Le\n", what, got, want);
    CHECK(error <= tolerance * std::max<long double>(1, std::abs(want)));
}

void pre_matches_oracle_and_invariants() {
    const Fixture f;
    const Oracle oracle{f};
    const size_t hc = f.hc;
    std::vector<float> pre(f.tokens * hc), post(f.tokens * hc), comb(f.tokens * hc * hc),
        collapsed(f.tokens * f.hidden);
    mhc_pre_reference(f.streams, f.fn, f.base, f.scale, pre, post, comb, collapsed, f.profile, f.hidden);

    for (size_t t = 0; t < f.tokens; ++t) {
        std::vector<long double> pre_o, post_o, comb_o;
        oracle.weights(t, pre_o, post_o, comb_o);
        long double pre_sum = 0;
        for (size_t n = 0; n < hc; ++n) {
            compare(pre[t * hc + n], pre_o[n], 2e-6L, "pre");
            compare(post[t * hc + n], post_o[n], 2e-6L, "post");
            pre_sum += pre_o[n];
        }
        CHECK(pre_sum < static_cast<long double>(hc) * 2 + 1);  // sigmoid+eps, NOT softmax
        CHECK(pre_sum > 1e-6L);
        for (size_t i = 0; i < hc * hc; ++i) compare(comb[t * hc * hc + i], comb_o[i], 2e-6L, "comb");
        // Sinkhorn target: columns (fixed dst over src) sum to 1 up to the eps slack.
        for (size_t dst = 0; dst < hc; ++dst) {
            float column = 0, row = 0;
            for (size_t src = 0; src < hc; ++src) column += comb[t * hc * hc + src * hc + dst];
            for (size_t d = 0; d < hc; ++d) row += comb[t * hc * hc + dst * hc + d];
            CHECK(std::abs(column - 1.0f) < 1e-4f);
            CHECK(std::abs(row - 1.0f) < 2e-3f);  // rows only approximately (eps + last-op asymmetry)
        }
        for (size_t d = 0; d < f.hidden; ++d) {
            long double acc = 0;
            for (size_t n = 0; n < hc; ++n)
                acc += pre_o[n] * f.streams[t * f.stream_values + n * f.hidden + d];
            compare(collapsed[t * f.hidden + d], acc, 2e-5L, "collapsed");
        }
    }
}

// The flat comb layout comb[dst + hc*src] is the easiest thing to transpose
// silently; a probe with one large src-specific logit pins it: that src row's
// softmax concentrates on the dst with the large base.
void comb_flat_layout_probe() {
    Fixture f(8, 1);
    // Make base[2hc + dst + hc*src] strongly favor (src=2, dst=1).
    for (size_t i = 0; i < f.base.size(); ++i) f.base[i] = -4;
    const size_t hot = 2 * f.hc + 1 + f.hc * 2;
    f.base[hot] = +8;
    std::vector<float> pre(f.hc), post(f.hc), comb(f.hc * f.hc), collapsed(f.hidden);
    mhc_pre_reference(f.streams, f.fn, f.base, f.scale, pre, post, comb, collapsed, f.profile, f.hidden);
    // Before Sinkhorn the src=2 row is ~one-hot on dst=1; Sinkhorn's first
    // move is a column normalization, so that row stays the row closest to a
    // one-hot on dst=1 across all src rows.
    size_t argmax = 0;
    float entropy_best = 1e30f;
    for (size_t src = 0; src < f.hc; ++src) {
        float entropy = 0;
        for (size_t dst = 0; dst < f.hc; ++dst) entropy -= comb[src * f.hc + dst] * std::log(comb[src * f.hc + dst]);
        if (entropy < entropy_best) { entropy_best = entropy; argmax = src; }
    }
    CHECK(argmax == 2);
    CHECK(comb[2 * f.hc + 1] > 0.9f);
}

void post_and_head_and_layer_chain() {
    const Fixture f;
    const Oracle oracle{f};
    const size_t hc = f.hc;
    std::vector<float> pre(f.tokens * hc), post(f.tokens * hc), comb(f.tokens * hc * hc),
        collapsed(f.tokens * f.hidden);
    mhc_pre_reference(f.streams, f.fn, f.base, f.scale, pre, post, comb, collapsed, f.profile, f.hidden);

    // A deterministic stand-in block output.
    std::vector<float> block(f.tokens * f.hidden);
    for (size_t i = 0; i < block.size(); ++i) block[i] = source(static_cast<uint32_t>(i) * 7 + 2) / 3;

    std::vector<float> out(f.tokens * f.stream_values, -1e30f);
    mhc_post_reference(block, f.streams, post, comb, out, f.profile, f.hidden);
    for (size_t t = 0; t < f.tokens; ++t) {
        std::vector<long double> pre_o, post_o, comb_o;
        oracle.weights(t, pre_o, post_o, comb_o);
        for (size_t dst = 0; dst < hc; ++dst)
            for (size_t d = 0; d < f.hidden; ++d) {
                long double mix = 0;
                for (size_t src = 0; src < hc; ++src)
                    mix += comb_o[src * hc + dst] * f.streams[t * f.stream_values + src * f.hidden + d];
                const long double expected = post_o[dst] * block[t * f.hidden + d] + mix;
                compare(out[t * f.stream_values + dst * f.hidden + d], expected, 2e-5L, "post");
            }
    }

    // Head: unweighted mean; in-place alias (output = prefix view of the same
    // buffer) must work because each element's reads precede its write.
    std::vector<float> streams_copy = f.streams;
    mhc_head_reference(streams_copy, std::span(streams_copy).first(f.tokens * f.hidden),
                       f.profile, f.hidden);
    for (size_t t = 0; t < f.tokens; ++t)
        for (size_t d = 0; d < f.hidden; ++d) {
            const float expected = (f.streams[t * f.stream_values + 0 * f.hidden + d] +
                                    f.streams[t * f.stream_values + 1 * f.hidden + d] +
                                    f.streams[t * f.stream_values + 2 * f.hidden + d] +
                                    f.streams[t * f.stream_values + 3 * f.hidden + d]) / 4.0f;
            CHECK(streams_copy[t * f.hidden + d] == expected);
        }
}

void sinkhorn_iteration_count_matters() {
    // iters=1 (single column normalization) vs 20 must differ on a general
    // input, pinning that the reference actually iterates.
    Fixture f(8, 1);
    std::vector<float> pre1(f.hc), post1(f.hc), comb1(f.hc * f.hc), col1(f.hidden);
    MhcProfile one = f.profile; one.sinkhorn_iters = 1;
    mhc_pre_reference(f.streams, f.fn, f.base, f.scale, pre1, post1, comb1, col1, one, f.hidden);
    std::vector<float> pre20(f.hc), post20(f.hc), comb20(f.hc * f.hc), col20(f.hidden);
    mhc_pre_reference(f.streams, f.fn, f.base, f.scale, pre20, post20, comb20, col20, f.profile, f.hidden);
    CHECK(pre1 == pre20 && post1 == post20);   // gates do not depend on iterations
    bool comb_differs = false;
    for (size_t i = 0; i < f.hc * f.hc; ++i) comb_differs = comb_differs || comb1[i] != comb20[i];
    CHECK(comb_differs);
    float column_error = 0;
    for (size_t dst = 0; dst < f.hc; ++dst) {
        float sum = 0;
        for (size_t src = 0; src < f.hc; ++src) sum += comb1[src * f.hc + dst];
        column_error = std::max(column_error, std::abs(sum - 1.0f));
    }
    CHECK(column_error < 1e-4f);  // iters=1 already fixes columns exactly once
}

void glm53_smoke_and_invalid() {
    // Real mix width: fn is 24x16384; geometry constants checked separately.
    CHECK(mhc_mix_values(kGlm53MhcProfile) == 24);
    CHECK(mhc_stream_values(kGlm53MhcProfile, 4096) == 16384);
    CHECK(kGlm53MhcProfile.sinkhorn_iters == 20 && kGlm53MhcProfile.hc_eps == 1e-6f);
    CHECK(kGlm53MhcProfile.rms_eps == 1e-5f);

    const Fixture g;
    std::vector<float> pre(g.hc), post(g.hc), comb(g.hc * g.hc), col(g.hidden);
    rejects([&] { (void)mhc_pre_reference(std::span(g.streams).first(g.streams.size() - 1), g.fn,
                                          g.base, g.scale, pre, post, comb, col, g.profile, g.hidden); });
    rejects([&] { (void)mhc_pre_reference(g.streams, std::span(g.fn).first(g.fn.size() - 1),
                                          g.base, g.scale, pre, post, comb, col, g.profile, g.hidden); });
    rejects([&] { (void)mhc_pre_reference(g.streams, g.fn, g.base, std::span(g.scale).first(2),
                                          pre, post, comb, col, g.profile, g.hidden); });
    rejects([&] { (void)mhc_pre_reference(g.streams, g.fn, g.base, g.scale, std::span(pre).first(3),
                                          post, comb, col, g.profile, g.hidden); });
    MhcProfile broken = g.profile; broken.hc = 5;
    rejects([&] { (void)mhc_pre_reference(g.streams, g.fn, g.base, g.scale, pre, post, comb, col,
                                          broken, g.hidden); });
    auto poisoned = g.streams;
    poisoned[3] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)mhc_pre_reference(poisoned, g.fn, g.base, g.scale, pre, post, comb, col,
                                          g.profile, g.hidden); });
    std::vector<float> block(g.hidden), out(g.stream_values);
    rejects([&] { (void)mhc_post_reference(std::span(block).first(g.hidden - 1), g.streams,
                                           post, comb, out, g.profile, g.hidden); });
    rejects([&] { (void)mhc_head_reference(g.streams, std::span(out).first(4), g.profile, g.hidden); });
}

}  // namespace

int main() {
    try {
        pre_matches_oracle_and_invariants();
        comb_flat_layout_probe();
        post_and_head_and_layer_chain();
        sinkhorn_iteration_count_matters();
        glm53_smoke_and_invalid();
        std::puts("GLM mHC pre/post/head references match independent oracles");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
