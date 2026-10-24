#include "strata/hpu/dsa_reference.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "GLM DSA reference requires IEEE-754 binary32");

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("DSA reference size overflow");
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

void dsa_pool_reference(std::span<const float> keys, std::span<const float> gates,
                        std::span<const float> ape, std::span<float> pooled,
                        const DsaIndexerProfile& profile) {
    const size_t dim = profile.head_dim;
    const size_t kpool = profile.kpool;
    if (dim == 0 || kpool == 0 || profile.top_k == 0 || profile.top_k % kpool)
        throw std::invalid_argument("DSA indexer profile is degenerate");
    const size_t tokens = token_count(keys, dim, "DSA keys are not whole token planes");
    if (gates.size() != keys.size() || ape.size() != checked_mul(kpool, dim))
        throw std::invalid_argument("DSA pool gate/ape plane mismatch");
    if (tokens == 0 || tokens % kpool)
        throw std::invalid_argument("DSA pooling requires tokens to be a positive multiple of kpool");
    const size_t pools = tokens / kpool;
    if (pooled.size() != checked_mul(pools, dim))
        throw std::invalid_argument("DSA pooled plane does not match the pool count");
    demand_finite(keys, "DSA keys must be finite");
    demand_finite(gates, "DSA gates must be finite");
    demand_finite(ape, "DSA ape must be finite");

    for (size_t p = 0; p < pools; ++p)
        for (size_t e = 0; e < dim; ++e) {
            float logits[8], maximum = 0;
            if (kpool > 8) throw std::invalid_argument("DSA pooling pins kpool at 8 members or fewer");
            for (size_t m = 0; m < kpool; ++m) {
                logits[m] = gates[(p * kpool + m) * dim + e] + ape[m * dim + e];
                if (m == 0 || logits[m] > maximum) maximum = logits[m];
            }
            float denominator = 0, exps[8];
            for (size_t m = 0; m < kpool; ++m) {
                exps[m] = std::exp(logits[m] - maximum);
                denominator += exps[m];
            }
            const float inverse = 1.0f / denominator;
            float acc = 0;
            for (size_t m = 0; m < kpool; ++m)
                acc = std::fma(exps[m] * inverse, keys[(p * kpool + m) * dim + e], acc);
            if (!std::isfinite(acc)) throw std::overflow_error("DSA pooling overflowed");
            pooled[p * dim + e] = acc;
        }
}

void dsa_score_reference(std::span<const float> iq, std::span<const float> weights,
                         std::span<const float> pooled, std::span<float> scores,
                         const DsaIndexerProfile& profile) {
    const size_t heads = profile.heads;
    const size_t dim = profile.head_dim;
    const size_t head_plane = checked_mul(heads, dim);
    const size_t tokens = token_count(iq, head_plane, "DSA iq is not whole token planes");
    const size_t pools = token_count(pooled, dim, "DSA pooled is not whole pool planes");
    if (weights.size() != checked_mul(tokens, heads))
        throw std::invalid_argument("DSA weights must hold one value per token and head");
    if (scores.size() != checked_mul(tokens, pools))
        throw std::invalid_argument("DSA score plane does not match tokens x pools");
    if (tokens == 0) throw std::invalid_argument("DSA scoring requires at least one token");
    demand_finite(iq, "DSA iq must be finite");
    demand_finite(weights, "DSA weights must be finite");
    demand_finite(pooled, "DSA pooled must be finite");

    const float dot_scale = 1.0f / std::sqrt(static_cast<float>(dim));
    const float head_scale = 1.0f / std::sqrt(static_cast<float>(heads));
    for (size_t t = 0; t < tokens; ++t)
        for (size_t p = 0; p < pools; ++p) {
            float acc = 0;
            for (size_t h = 0; h < heads; ++h) {
                const float* q = iq.data() + t * head_plane + h * dim;
                const float* key = pooled.data() + p * dim;
                float dot = 0;
                for (size_t e = 0; e < dim; ++e) dot = std::fma(q[e], key[e], dot);
                acc = std::fma(std::fmax(dot * dot_scale, 0.0f),
                               weights[t * heads + h] * head_scale, acc);
            }
            if (!std::isfinite(acc)) throw std::overflow_error("DSA scoring overflowed");
            scores[t * pools + p] = acc;
        }
}

void dsa_select_reference(std::span<const float> scores, std::span<const uint32_t> pos,
                          std::span<int32_t> cells, const DsaIndexerProfile& profile) {
    const size_t kpool = profile.kpool;
    const size_t tokens = pos.size();
    const size_t top_pools = dsa_top_pools(profile);
    const size_t width = dsa_selection_width(profile);
    if (tokens == 0) throw std::invalid_argument("DSA selection requires at least one token");
    if (scores.size() % tokens != 0)
        throw std::invalid_argument("DSA score plane does not match the position count");
    const size_t pools = scores.size() / tokens;
    if (cells.size() != checked_mul(tokens, width))
        throw std::invalid_argument("DSA cell plane does not match the selection width");
    if (pools == 0) throw std::invalid_argument("DSA selection requires at least one scored pool");
    demand_finite(scores, "DSA scores must be finite");
    for (const auto position : pos)
        if ((position + 1) / kpool > pools)
            throw std::invalid_argument("DSA position sees more pools than were scored");

    // 2051 is the production width; a generous bound keeps narrower probe
    // profiles testable without dynamic allocation.
    constexpr size_t kMaxSelectionWidth = 4096;
    if (width > kMaxSelectionWidth)
        throw std::invalid_argument("DSA selection width exceeds the staging bound");
    for (size_t t = 0; t < tokens; ++t) {
        const size_t visible = (pos[t] + 1) / kpool;
        int32_t row[kMaxSelectionWidth];
        for (size_t s = 0; s < width; ++s) row[s] = -1;

        for (size_t p = 0; p < visible; ++p) {
            const float value = scores[t * pools + p];
            size_t rank = 0;
            for (size_t q = 0; q < visible; ++q) {
                const float other = scores[t * pools + q];
                if (other > value) ++rank;
                else if (q < p && other == value) ++rank;
            }
            if (rank < top_pools)
                for (size_t m = 0; m < kpool; ++m) row[rank * kpool + m] = static_cast<int32_t>(p * kpool + m);
        }
        if (profile.select_tail)
            for (size_t m = 0; m + 1 < kpool; ++m) {
                const size_t cell = visible * kpool + m;
                if (cell <= pos[t]) row[top_pools * kpool + m] = static_cast<int32_t>(cell);
            }
        for (size_t s = 0; s < width; ++s) cells[t * width + s] = row[s];
    }
}

void dsa_attention_reference(std::span<const float> q_abs, std::span<const float> latents,
                             std::span<const int32_t> cells, std::span<const float> wv_b,
                             std::span<float> out, const DsaIndexerProfile& indexer,
                             const DsaMlaProfile& mla, size_t cache_tokens) {
    const size_t lora = mla.kv_lora;
    const size_t heads = mla.heads;
    const size_t v_head = mla.v_head;
    const size_t head_plane = checked_mul(lora, heads);
    const size_t tokens = token_count(q_abs, head_plane, "DSA q_abs is not whole token planes");
    const size_t width = dsa_selection_width(indexer);
    if (cells.size() != checked_mul(tokens, width))
        throw std::invalid_argument("DSA cells do not match tokens x selection width");
    if (latents.size() != checked_mul(cache_tokens, lora))
        throw std::invalid_argument("DSA latents do not match the cache token count");
    if (wv_b.size() != checked_mul(checked_mul(lora, v_head), heads))
        throw std::invalid_argument("DSA wv_b does not match the GGUF geometry");
    if (out.size() != checked_mul(tokens, checked_mul(heads, v_head)))
        throw std::invalid_argument("DSA output does not match tokens x heads x v_head");
    if (tokens == 0 || cache_tokens == 0)
        throw std::invalid_argument("DSA attention requires tokens and cache");
    demand_finite(q_abs, "DSA q_abs must be finite");
    demand_finite(latents, "DSA latents must be finite");
    demand_finite(wv_b, "DSA wv_b must be finite");
    for (const auto cell : cells)
        if (cell < -1 || cell >= static_cast<int64_t>(cache_tokens))
            throw std::invalid_argument("DSA cell is outside the cache");

    const float scale = 1.0f / std::sqrt(static_cast<float>(mla.qk_nope));
    std::vector<float> probabilities(width), context(lora), staged(heads * v_head);
    for (size_t t = 0; t < tokens; ++t) {
        const int32_t* row = cells.data() + t * width;
        for (size_t s = 0; s < width; ++s) probabilities[s] = 0.0f;
        for (size_t h = 0; h < heads; ++h) {
            const float* q = q_abs.data() + t * head_plane + h * lora;
            float maximum = 0;
            bool any = false;
            for (size_t s = 0; s < width; ++s) {
                if (row[s] < 0) continue;
                const float* latent = latents.data() + static_cast<size_t>(row[s]) * lora;
                float dot = 0;
                for (size_t e = 0; e < lora; ++e) dot = std::fma(q[e], latent[e], dot);
                probabilities[s] = dot * scale;
                if (!any || probabilities[s] > maximum) { maximum = probabilities[s]; any = true; }
            }
            if (!any) throw std::invalid_argument("DSA attention row has no valid cell");
            float denominator = 0;
            for (size_t s = 0; s < width; ++s)
                if (row[s] >= 0) {
                    probabilities[s] = std::exp(probabilities[s] - maximum);
                    denominator += probabilities[s];
                }
            const float inverse = 1.0f / denominator;
            for (size_t e = 0; e < lora; ++e) context[e] = 0.0f;
            for (size_t s = 0; s < width; ++s)
                if (row[s] >= 0) {
                    const float* latent = latents.data() + static_cast<size_t>(row[s]) * lora;
                    const float weight = probabilities[s] * inverse;
                    for (size_t e = 0; e < lora; ++e) context[e] = std::fma(weight, latent[e], context[e]);
                }
            // out[t][h][v] = sum_c wv_b[c + lora*v + lora*v_head*h] * context[c]
            for (size_t v = 0; v < v_head; ++v) {
                const float* column = wv_b.data() + static_cast<size_t>(lora * v + lora * v_head * h);
                float acc = 0;
                for (size_t c = 0; c < lora; ++c) acc = std::fma(column[c], context[c], acc);
                if (!std::isfinite(acc)) throw std::overflow_error("DSA attention overflowed");
                staged[h * v_head + v] = acc;
            }
        }
        for (size_t i = 0; i < staged.size(); ++i) out[t * heads * v_head + i] = staged[i];
    }
}

}  // namespace strata::hpu
