#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

inline constexpr const char* kDsaReferenceProfile = "glm53_dsa_fp32";

// GLM-5.3-Flash's audited DSA indexer geometry: 32 index heads of 128, pools
// of 4 tokens, a 2048-token selection budget (512 pools) plus the mandatory
// 3-slot incomplete tail, giving the fixed 2051-wide selection rows.
struct DsaIndexerProfile {
    uint32_t heads;
    uint32_t head_dim;
    uint32_t kpool;
    uint32_t top_k;
    bool select_tail;
};
inline constexpr DsaIndexerProfile kGlm53DsaIndexerProfile{32, 128, 4, 2048, true};

[[nodiscard]] constexpr uint32_t dsa_top_pools(const DsaIndexerProfile& p) {
    return p.top_k / p.kpool;  // 2048/4 = 512
}
[[nodiscard]] constexpr size_t dsa_selection_width(const DsaIndexerProfile& p) {
    return p.select_tail ? size_t(p.top_k) + p.kpool - 1 : size_t(p.top_k);
}

// Pool compression over COMPLETE pools: pool p covers tokens [p*kpool,
// p*kpool+kpool). pooled[p][e] = sum_m softmax_m(gates[p*kpool+m][e] +
// ape[m][e]) * keys[p*kpool+m][e] where the softmax runs SEPARATELY PER
// KEY-DIM ELEMENT over the pool members - the parity-pinned quirk; a whole-row
// softmax is the natural wrong reading. Stable max-subtracted softmax in FP32.
// tokens must be a positive multiple of kpool (the caller keeps the tail);
// ape is [kpool][head_dim]. Everything validates before any output is written.
void dsa_pool_reference(std::span<const float> keys, std::span<const float> gates,
                        std::span<const float> ape, std::span<float> pooled,
                        const DsaIndexerProfile&);

// index_scores[t][p] = sum_h relu(iq[t][h] . pooled[p] * head_dim^-0.5) *
// (weights[t][h] * heads^-0.5), all FP32. The two scale factors follow the
// transformers placement (inside the ReLU / on the head weights); llama.cpp
// and Maya pre-multiply 1/sqrt(head_dim*heads) onto the weights instead,
// which is the same product in exact arithmetic and differs only in rounding.
// iq is [tokens][heads][head_dim], weights [tokens][heads].
void dsa_score_reference(std::span<const float> iq, std::span<const float> weights,
                         std::span<const float> pooled, std::span<float> scores,
                         const DsaIndexerProfile&);

// Selection for one sequence of aligned pools: token t at GLOBAL position
// pos[t] sees the first n_vis = (pos[t]+1)/kpool completed pools (pool p is
// visible iff its last member p*kpool+kpool-1 <= pos[t]; the tail includes the
// query token itself). Visible pools are ranked stable-descending with exact
// ties broken by lower pool index; the pool at rank r occupies slots
// [r*kpool, r*kpool+kpool) with its member cells in position order (slot
// order IS score order). The incomplete tail (cells n_vis*kpool .. pos[t])
// then fills [top_pools*kpool, ...) in position order when select_tail.
// Every unused slot is -1. Output rows have the fixed dsa_selection_width.
// pools is the completed-pool count of scores' second axis; n_vis must stay
// <= pools. scores/pos/cells inputs must be finite/valid; cells publish per
// token after the whole row validates.
void dsa_select_reference(std::span<const float> scores, std::span<const uint32_t> pos,
                          std::span<int32_t> cells, const DsaIndexerProfile&);

// GLM-5.3-Flash's audited absorbed-MLA geometry: 512 latent, 64 heads, 256
// nope score dim (also the softmax scale base) and 256 value dim. NoPE only.
struct DsaMlaProfile {
    uint32_t kv_lora;
    uint32_t heads;
    uint32_t qk_nope;
    uint32_t v_head;
};
inline constexpr DsaMlaProfile kGlm53DsaMlaProfile{512, 64, 256, 256};

// Absorbed sparse attention over one layer's selection rows.
// q_abs is [tokens][heads][kv_lora] (queries already absorbed through
// W_K_B: q_abs = W_K_B_h^T q_h), latents [cache][kv_lora] (the only cached
// per-token state; no per-head K/V exists), cells [tokens][selection_width]
// with -1 padding from dsa_select_reference, and wv_b the GGUF tensor
// {kv_lora, v_head, heads}: wv_b[c + kv_lora*v + kv_lora*v_head*h].
// Per token and head: score_s = (q_abs . latent_{cells[s]}) * qk_nope^-0.5
// (the model constant 256^-0.5, NOT kv_lora^-0.5); softmax EXCLUDES the -1
// slots entirely (numerically the graph's -inf additive mask); ctx accumulates
// only over valid slots; out[t][h][v] = sum_c wv_b[c,v,h] * ctx[c]. Output
// is [tokens][heads][v_head]. Inputs validate first; out publishes per token
// once the whole token's values are finite.
void dsa_attention_reference(std::span<const float> q_abs, std::span<const float> latents,
                             std::span<const int32_t> cells, std::span<const float> wv_b,
                             std::span<float> out, const DsaIndexerProfile&,
                             const DsaMlaProfile&, size_t cache_tokens);

}  // namespace strata::hpu
