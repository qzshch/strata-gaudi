#include "strata/hpu/dsa_reference.hpp"

#include <algorithm>
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

void compare(float got, long double want, long double tolerance, const char* what) {
    const long double error = std::abs(static_cast<long double>(got) - want);
    if (!(error <= tolerance * std::max<long double>(1, std::abs(want))))
        std::fprintf(stderr, "%s: got %.9g want %.9Le\n", what, got, want);
    CHECK(error <= tolerance * std::max<long double>(1, std::abs(want)));
}

// ---------------------------------------------------------------------------
void pool_matches_oracle() {
    // Small probe profile: kpool 4 matches production; dim 5 keeps the oracle cheap.
    const DsaIndexerProfile profile{2, 5, 4, 8, true};
    const size_t tokens = 12, dim = profile.head_dim, kpool = profile.kpool;
    const size_t pools = tokens / kpool;
    std::vector<float> keys(tokens * dim), gates(tokens * dim), ape(kpool * dim), pooled(pools * dim);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = source(static_cast<uint32_t>(i) + 3);
        gates[i] = source(static_cast<uint32_t>(i) * 2 + 1) / 2;
    }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = source(static_cast<uint32_t>(i) * 5 + 7) / 4;
    dsa_pool_reference(keys, gates, ape, pooled, profile);

    for (size_t p = 0; p < pools; ++p)
        for (size_t e = 0; e < dim; ++e) {
            // independent long-double per-element softmax
            long double logits[4];
            for (size_t m = 0; m < kpool; ++m)
                logits[m] = gates[(p * kpool + m) * dim + e] + ape[m * dim + e];
            long double maximum = logits[0];
            for (size_t m = 1; m < kpool; ++m) maximum = std::max(maximum, logits[m]);
            long double denominator = 0, exps[4];
            for (size_t m = 0; m < kpool; ++m) {
                exps[m] = std::exp(logits[m] - maximum);
                denominator += exps[m];
            }
            long double expected = 0;
            for (size_t m = 0; m < kpool; ++m)
                expected += exps[m] / denominator * keys[(p * kpool + m) * dim + e];
            compare(pooled[p * dim + e], expected, 1e-6L, "pooled");
        }
}

// ---------------------------------------------------------------------------
void score_matches_oracle_and_relu_boundary() {
    const DsaIndexerProfile profile{3, 4, 4, 8, true};
    const size_t heads = profile.heads, dim = profile.head_dim;
    const size_t tokens = 3, pools = 2;
    std::vector<float> iq(tokens * heads * dim), weights(tokens * heads),
        pooled(pools * dim), scores(tokens * pools);
    for (size_t i = 0; i < iq.size(); ++i) iq[i] = source(static_cast<uint32_t>(i) + 1) / 2;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = source(static_cast<uint32_t>(i) * 3 + 2) / 2;
    for (size_t i = 0; i < pooled.size(); ++i) pooled[i] = source(static_cast<uint32_t>(i) * 7 + 5) / 3;
    dsa_score_reference(iq, weights, pooled, scores, profile);

    const long double dot_scale = std::sqrt(1.0L / dim);
    const long double head_scale = std::sqrt(1.0L / heads);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t p = 0; p < pools; ++p) {
            long double expected = 0;
            for (size_t h = 0; h < heads; ++h) {
                long double dot = 0;
                for (size_t e = 0; e < dim; ++e)
                    dot += static_cast<long double>(iq[t * heads * dim + h * dim + e]) *
                           pooled[p * dim + e];
                expected += std::max(dot * dot_scale, 0.0L) *
                            weights[t * heads + h] * head_scale;
            }
            compare(scores[t * pools + p], expected, 1e-6L, "score");
        }

    // ReLU boundary: an exactly-orthogonal head contributes nothing even with
    // a large weight; a negative dot also contributes nothing.
    std::fill(iq.begin(), iq.end(), 0.0f);
    std::fill(weights.begin(), weights.end(), 10.0f);
    for (size_t e = 0; e < dim; ++e) iq[0 * heads * dim + 0 * dim + e] = 1.0f;   // positive dot
    for (size_t e = 0; e < dim; ++e) iq[0 * heads * dim + 1 * dim + e] = -1.0f;  // negative dot
    dsa_score_reference(iq, weights, pooled, scores, profile);
    // head0 dot = sum(pooled), head1 = -sum(pooled): whichever sign is
    // positive passes the ReLU, the other is clipped to zero.
    long double sum = 0;
    for (size_t e = 0; e < dim; ++e) sum += pooled[e];
    const long double expected = std::max(std::abs(sum) * dot_scale, 0.0L) * 10.0f * head_scale;
    compare(scores[0], expected, 1e-6L, "relu boundary");
}

// ---------------------------------------------------------------------------
void select_rank_ties_tail_and_boundaries() {
    // top_k 8 -> 2 pools; width 8 + 3 = 11.
    const DsaIndexerProfile profile{1, 4, 4, 8, true};
    const size_t kpool = profile.kpool, top_pools = 2, width = 11;
    CHECK(dsa_selection_width(profile) == 11 && dsa_top_pools(profile) == 2);

    // 4 complete pools (16 tokens), query at position 17 -> n_vis = 4 > top 2.
    std::vector<float> scores{30, 20, 40, 10};   // pool ranks: 2 > 0 > 1 > 3
    std::vector<uint32_t> pos{17};
    std::vector<int32_t> cells(width);
    dsa_select_reference(scores, pos, cells, profile);
    // rank0 pool2 -> slots 0..3 = cells 8..11; rank1 pool0 -> slots 4..7 = 0..3;
    // tail: n_vis*kpool=16..17 -> slots 8..10 = {16,17,-1}
    const std::vector<int32_t> expected{8, 9, 10, 11, 0, 1, 2, 3, 16, 17, -1};
    CHECK((std::vector<int32_t>(cells.begin(), cells.end()) == expected));

    // Exact ties break by lower pool index, stably.
    std::vector<float> tied{5, 5, 5, 1};
    dsa_select_reference(tied, pos, cells, profile);
    CHECK(cells[0] == 0 && cells[4] == 4);   // pools 0 then 1 in rank order

    // top_pools == 0 visible pools (query at position 1): tail only + padding.
    std::vector<uint32_t> early{1};
    dsa_select_reference(scores, early, cells, profile);
    const std::vector<int32_t> tail_only{-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1};
    CHECK((std::vector<int32_t>(cells.begin(), cells.end()) == tail_only));

    // Every tail width: positions 0..3 -> n_tail 1..4/0.
    for (uint32_t p = 0; p < 8; ++p) {
        std::vector<uint32_t> one{p};
        dsa_select_reference(scores, one, cells, profile);
        const size_t n_tail = (p + 1) % kpool;
        for (size_t m = 0; m + 1 < kpool; ++m) {
            const int32_t want = m < n_tail ? static_cast<int32_t>(((p + 1) / kpool) * kpool + m) : -1;
            CHECK(cells[top_pools * kpool + m] == want);
        }
    }

    // All visible pools fit: n_vis <= top_pools selects everything, in rank order.
    std::vector<uint32_t> mid{7};   // n_vis = 2 (pools 0,1 visible)
    dsa_select_reference(scores, mid, cells, profile);
    CHECK(cells[0] == 0 && cells[4] == 4);   // pools 0 (30) then 1 (20)
    CHECK(cells[8] == -1 && cells[9] == -1 && cells[10] == -1);   // tail starts at 8 > pos 7: none
}

// ---------------------------------------------------------------------------
// Independent dense (materialized) attention oracle for the eager cross-check.
struct DenseOracle {
    const DsaIndexerProfile& indexer;
    const DsaMlaProfile& mla;
    // materialized per-head attention in long double over the selected cells
    // attention restricted to the query's visible set.
    std::vector<float> run(const std::vector<float>& q_abs, const std::vector<float>& latents,
                           const std::vector<int32_t>& cells, const std::vector<float>& wv_b,
                           size_t tokens) {
        const size_t lora = mla.kv_lora, heads = mla.heads, v_head = mla.v_head;
        const size_t width = dsa_selection_width(indexer);
        std::vector<float> out(tokens * heads * v_head, 0);
        const float scale = 1.0f / std::sqrt(static_cast<float>(mla.qk_nope));
        for (size_t t = 0; t < tokens; ++t)
            for (size_t h = 0; h < heads; ++h) {
                // scores over all selected cells, fp64
                std::vector<long double> probs;
                long double maximum = 0;
                for (size_t s = 0; s < width; ++s) {
                    const int32_t cell = cells[t * width + s];
                    if (cell < 0) { probs.push_back(-1e300L); continue; }
                    long double dot = 0;
                    for (size_t e = 0; e < lora; ++e)
                        dot += static_cast<long double>(q_abs[t * heads * lora + h * lora + e]) *
                               latents[static_cast<size_t>(cell) * lora + e];
                    const long double score = dot * scale;
                    probs.push_back(score);
                    maximum = std::max(maximum, score);
                }
                long double denominator = 0;
                for (auto& value : probs)
                    if (value > -1e299L) { value = std::exp(value - maximum); denominator += value; }
                std::vector<long double> ctx(lora, 0);
                for (size_t s = 0; s < width; ++s)
                    if (cells[t * width + s] >= 0)
                        for (size_t e = 0; e < lora; ++e)
                            ctx[e] += probs[s] / denominator *
                                      latents[static_cast<size_t>(cells[t * width + s]) * lora + e];
                for (size_t v = 0; v < v_head; ++v) {
                    long double acc = 0;
                    for (size_t c = 0; c < lora; ++c)
                        acc += static_cast<long double>(wv_b[c + lora * v + lora * v_head * h]) * ctx[c];
                    out[t * heads * v_head + h * v_head + v] = static_cast<float>(acc);
                }
            }
        return out;
    }
};

void attention_matches_dense_oracle_and_full_selection() {
    const DsaIndexerProfile indexer{1, 4, 4, 8, true};
    const DsaMlaProfile mla{6, 2, 4, 3};   // tiny geometry, same structure
    const size_t tokens = 2, cache = 8, width = dsa_selection_width(indexer);
    std::vector<float> q_abs(tokens * mla.heads * mla.kv_lora),
        latents(cache * mla.kv_lora), wv_b(mla.kv_lora * mla.v_head * mla.heads);
    for (size_t i = 0; i < q_abs.size(); ++i) q_abs[i] = source(static_cast<uint32_t>(i) + 1) / 2;
    for (size_t i = 0; i < latents.size(); ++i) latents[i] = source(static_cast<uint32_t>(i) * 3 + 2) / 3;
    for (size_t i = 0; i < wv_b.size(); ++i) wv_b[i] = source(static_cast<uint32_t>(i) * 7 + 1) / 4;

    // Hand-built selection: token0 sees cells {0,2,5} (+padding), token1 sees {7,1}.
    std::vector<int32_t> cells(tokens * width, -1);
    cells[0] = 0; cells[1] = 2; cells[2] = 5;
    cells[width + 0] = 7; cells[width + 1] = 1;

    std::vector<float> out(tokens * mla.heads * mla.v_head);
    dsa_attention_reference(q_abs, latents, cells, wv_b, out, indexer, mla, cache);
    const auto expected = DenseOracle{indexer, mla}.run(q_abs, latents, cells, wv_b, tokens);
    for (size_t i = 0; i < out.size(); ++i) compare(out[i], expected[i], 1e-5L, "attention");

    // FULL selection (every cache cell, no padding except the tail slots):
    // the sparse path must equal a plain dense softmax over the same cells.
    std::vector<int32_t> full(tokens * width, -1);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t c = 0; c < cache && c < width; ++c) full[t * width + c] = static_cast<int32_t>(c);
    dsa_attention_reference(q_abs, latents, full, wv_b, out, indexer, mla, cache);
    const auto dense = DenseOracle{indexer, mla}.run(q_abs, latents, full, wv_b, tokens);
    for (size_t i = 0; i < out.size(); ++i) compare(out[i], dense[i], 1e-5L, "full selection");

    // Duplicate cells are legal (selections can overlap pools/tail in none of
    // the production paths, but the reference must not double-count vs oracle).
    std::vector<int32_t> dup(tokens * width, -1);
    dup[0] = 3; dup[1] = 3; dup[2] = 4;
    dup[width + 0] = 5; dup[width + 1] = 5;
    dsa_attention_reference(q_abs, latents, dup, wv_b, out, indexer, mla, cache);
    const auto dup_expected = DenseOracle{indexer, mla}.run(q_abs, latents, dup, wv_b, tokens);
    for (size_t i = 0; i < out.size(); ++i) compare(out[i], dup_expected[i], 1e-5L, "duplicate cells");
}

// End-to-end: pool -> score -> select -> attention on a mid-size profile with
// the dense equivalence guaranteed at n_vis*kpool+tail <= top_k.
void end_to_end_dense_equivalence() {
    const DsaIndexerProfile indexer{3, 8, 4, 8, true};
    const DsaMlaProfile mla{16, 2, 8, 6};
    const size_t tokens = 8, cache = tokens, dim = indexer.head_dim, heads = indexer.heads;
    const size_t pools = tokens / indexer.kpool, width = dsa_selection_width(indexer);

    std::vector<float> keys(tokens * dim), gates(tokens * dim), ape(indexer.kpool * dim),
        pooled(pools * dim), iq(tokens * heads * dim), weights(tokens * heads),
        scores(tokens * pools);
    std::vector<uint32_t> pos(tokens);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = source(static_cast<uint32_t>(i) + 2);
        gates[i] = source(static_cast<uint32_t>(i) * 3 + 1) / 2;
    }
    for (size_t i = 0; i < ape.size(); ++i) ape[i] = source(static_cast<uint32_t>(i) * 5 + 3) / 4;
    for (size_t i = 0; i < iq.size(); ++i) iq[i] = source(static_cast<uint32_t>(i) + 7) / 2;
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = source(static_cast<uint32_t>(i) * 3 + 5) / 2;
    for (size_t t = 0; t < tokens; ++t) pos[t] = static_cast<uint32_t>(t);
    std::vector<float> q_abs(tokens * mla.heads * mla.kv_lora),
        latents(cache * mla.kv_lora), wv_b(mla.kv_lora * mla.v_head * mla.heads);
    for (size_t i = 0; i < q_abs.size(); ++i) q_abs[i] = source(static_cast<uint32_t>(i) * 2 + 3) / 3;
    for (size_t i = 0; i < latents.size(); ++i) latents[i] = source(static_cast<uint32_t>(i) + 4) / 3;
    for (size_t i = 0; i < wv_b.size(); ++i) wv_b[i] = source(static_cast<uint32_t>(i) * 7 + 2) / 4;

    dsa_pool_reference(keys, gates, ape, pooled, indexer);
    dsa_score_reference(iq, weights, pooled, scores, indexer);
    std::vector<int32_t> cells(tokens * width);
    dsa_select_reference(scores, pos, cells, indexer);

    // Every token sees <= 8 cells total (n_vis pools of 4 + tail of 3 <= top_k 8?
    // top_k/kpool = 2 pools = 8 slots; the selection may drop early pools for
    // late tokens, so compare against the DENSE-over-selected-cells oracle,
    // and separately assert the <= (top_k + tail) budget.
    std::vector<float> out(tokens * mla.heads * mla.v_head);
    dsa_attention_reference(q_abs, latents, cells, wv_b, out, indexer, mla, cache);
    const auto expected = DenseOracle{indexer, mla}.run(q_abs, latents, cells, wv_b, tokens);
    for (size_t i = 0; i < out.size(); ++i) compare(out[i], expected[i], 1e-5L, "end-to-end");
    for (size_t t = 0; t < tokens; ++t) {
        size_t live = 0;
        for (size_t s = 0; s < width; ++s) live += cells[t * width + s] >= 0;
        CHECK(live <= size_t(indexer.top_k) + indexer.kpool - 1);
        CHECK(live <= pos[t] + 1);   // causality: never more cells than tokens so far
    }
}

// ---------------------------------------------------------------------------
template<class F> void rejects(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid DSA input accepted");
}

void invalid_inputs() {
    const DsaIndexerProfile indexer{1, 4, 4, 8, true};
    const DsaMlaProfile mla{6, 2, 4, 3};
    const size_t tokens = 4, dim = 4, pools = 1, width = dsa_selection_width(indexer);
    std::vector<float> keys(tokens * dim), gates(tokens * dim), ape(4 * dim), pooled(pools * dim);
    rejects([&] { (void)dsa_pool_reference(std::span(keys).first(tokens * dim - 1), gates, ape, pooled, indexer); });
    rejects([&] { (void)dsa_pool_reference(keys, gates, std::span(ape).first(3 * dim), pooled, indexer); });
    rejects([&] { (void)dsa_pool_reference(std::span(keys).first(3 * dim), gates, ape, pooled, indexer); });  // not a kpool multiple
    auto poisoned = gates;
    poisoned[2] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { (void)dsa_pool_reference(keys, poisoned, ape, pooled, indexer); });

    std::vector<float> iq(tokens * dim), weights(tokens), scores(tokens * pools);
    rejects([&] { (void)dsa_score_reference(std::span(iq).first(iq.size() - 1), weights, pooled, scores, indexer); });
    rejects([&] { (void)dsa_score_reference(iq, std::span(weights).first(tokens - 1), pooled, scores, indexer); });

    std::vector<float> one{1, 2, 3, 4};
    std::vector<uint32_t> pos{99};   // sees 25 pools > 4 scored
    std::vector<int32_t> cells(width);
    rejects([&] { (void)dsa_select_reference(one, pos, cells, indexer); });
    std::vector<float> nan_scores{1, std::numeric_limits<float>::quiet_NaN(), 3, 4};
    std::vector<uint32_t> small_pos{3};
    rejects([&] { (void)dsa_select_reference(nan_scores, small_pos, cells, indexer); });

    std::vector<float> q_abs(mla.heads * mla.kv_lora), latents(4 * mla.kv_lora),
        wv_b(mla.kv_lora * mla.v_head * mla.heads), out(mla.heads * mla.v_head);
    std::vector<int32_t> row(width, -1);
    row[0] = 4;   // == cache size: out of range
    rejects([&] { (void)dsa_attention_reference(q_abs, latents, row, wv_b, out, indexer, mla, 4); });
    std::vector<int32_t> empty(width, -1);
    rejects([&] { (void)dsa_attention_reference(q_abs, latents, empty, wv_b, out, indexer, mla, 4); });
    row[0] = 1;
    auto bad_wv = wv_b;
    bad_wv[0] = std::numeric_limits<float>::infinity();
    rejects([&] { (void)dsa_attention_reference(q_abs, latents, row, bad_wv, out, indexer, mla, 4); });
}

void glm53_constants() {
    CHECK(kGlm53DsaIndexerProfile.heads == 32 && kGlm53DsaIndexerProfile.head_dim == 128);
    CHECK(kGlm53DsaIndexerProfile.kpool == 4 && kGlm53DsaIndexerProfile.top_k == 2048);
    CHECK(kGlm53DsaIndexerProfile.select_tail);
    CHECK(dsa_selection_width(kGlm53DsaIndexerProfile) == 2051);
    CHECK(dsa_top_pools(kGlm53DsaIndexerProfile) == 512);
    CHECK(kGlm53DsaMlaProfile.kv_lora == 512 && kGlm53DsaMlaProfile.heads == 64);
    CHECK(kGlm53DsaMlaProfile.qk_nope == 256 && kGlm53DsaMlaProfile.v_head == 256);
}

}  // namespace

int main() {
    try {
        glm53_constants();
        pool_matches_oracle();
        score_matches_oracle_and_relu_boundary();
        select_rank_ties_tail_and_boundaries();
        attention_matches_dense_oracle_and_full_selection();
        end_to_end_dense_equivalence();
        invalid_inputs();
        std::puts("GLM DSA indexer pool/score/select and absorbed MLA match independent oracles");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
