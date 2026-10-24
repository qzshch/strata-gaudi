#include "strata/hpu/kda_chunk_reference.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
constexpr float kL2Eps = 1e-6f;

size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::overflow_error("KDA chunk reference size overflow");
    return a * b;
}

void demand_finite(std::span<const float> plane, const char* what) {
    for (const float value : plane)
        if (!std::isfinite(value)) throw std::invalid_argument(what);
}
}  // namespace

void kda_chunk_reference(std::span<const float> q, std::span<const float> k,
                         std::span<const float> v, std::span<const float> gate,
                         std::span<const float> beta, std::span<float> state,
                         std::span<float> out, std::span<float> scratch, const KdaProfile& profile,
                         uint32_t chunk) {
    const size_t d_inner = checked_mul(profile.heads, profile.head_dim);
    const size_t dk = profile.head_dim;
    const size_t state_values = kda_state_values(profile);
    if (d_inner == 0 || q.size() % d_inner != 0)
        throw std::invalid_argument("KDA chunk query is not a whole token plane");
    const size_t tokens = q.size() / d_inner;
    if (chunk == 0 || chunk > kKdaMaxChunk || tokens % chunk != 0)
        throw std::invalid_argument("KDA chunk requires tokens to be a positive multiple of chunk");
    if (k.size() != q.size() || v.size() != q.size() || gate.size() != q.size())
        throw std::invalid_argument("KDA chunk k/v/gate planes must match the query plane");
    if (beta.size() != checked_mul(tokens, profile.heads))
        throw std::invalid_argument("KDA chunk beta must hold one value per token and head");
    if (state.size() != state_values)
        throw std::invalid_argument("KDA chunk state does not match the profile");
    if (out.size() != q.size()) throw std::invalid_argument("KDA chunk output size mismatch");
    if (scratch.size() != kda_chunk_scratch_values(profile, tokens, chunk))
        throw std::invalid_argument("KDA chunk scratch does not match the requirement");
    if (tokens == 0) throw std::invalid_argument("KDA chunk requires at least one token");
    for (const auto* plane : {&q, &k, &v, &gate, &beta})
        demand_finite(*plane, "KDA chunk inputs must be finite");
    demand_finite(state, "KDA chunk state must be finite");

    const size_t c = chunk;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    float* work = scratch.data();                        // state image ...
    float* staged = work + state_values;                 // ... then output image ...
    float* plane = staged + out.size();                  // ... then the per-head work plane
    float* gsum = plane;                                 // [c][dk]
    float* q_normed = gsum + c * dk;                     // [c][dk], scaled by head_dim^-0.5
    float* k_normed = q_normed + c * dk;                 // [c][dk]
    float* t_matrix = k_normed + c * dk;                 // [c][c]
    float* w_rows = t_matrix + c * c;                    // [c][dk]
    float* u_rows = w_rows + c * dk;                     // [c][dk]
    float* v_fixed = u_rows + c * dk;                    // [c][dk]
    float* staging = v_fixed + c * dk;                   // [c] substitution deltas / Aqk row

    for (size_t i = 0; i < state_values; ++i) work[i] = state[i];
    const size_t chunks = tokens / c;

    for (size_t chunk_index = 0; chunk_index < chunks; ++chunk_index)
        for (uint32_t head = 0; head < profile.heads; ++head) {
            const size_t base = chunk_index * c * d_inner + size_t(head) * dk;
            const size_t beta_base = chunk_index * c * profile.heads + head;  // + row*heads reaches row token

            for (size_t row = 0; row < c; ++row)
                for (size_t d = 0; d < dk; ++d) {
                    const float value = gate[base + row * d_inner + d];
                    gsum[row * dk + d] = value + (row ? gsum[(row - 1) * dk + d] : 0.0f);
                }
            for (size_t row = 0; row < c; ++row) {
                float q_square = kL2Eps, k_square = kL2Eps;
                for (size_t d = 0; d < dk; ++d) {
                    const float qv = q[base + row * d_inner + d];
                    const float kv = k[base + row * d_inner + d];
                    q_square = std::fma(qv, qv, q_square);
                    k_square = std::fma(kv, kv, k_square);
                }
                const float q_inv = 1.0f / std::sqrt(q_square);
                const float k_inv = 1.0f / std::sqrt(k_square);
                for (size_t d = 0; d < dk; ++d) {
                    q_normed[row * dk + d] = (q[base + row * d_inner + d] * q_inv) * scale;
                    k_normed[row * dk + d] = k[base + row * d_inner + d] * k_inv;
                }
            }

            // Decayed key Gram from differences only. Entries are
            // [row = later token i][col = earlier token j], j < i, with the
            // exponent gsum[i] - gsum[j] <= 0 (gsum is non-increasing), so no
            // positive exponent is ever computed. The LATER token's beta
            // scales the Gram row; exact forward substitution then produces
            // (I+L)^-1 - I on the strict lower triangle.
            for (size_t i = 0; i < c; ++i)
                for (size_t j = 0; j < c; ++j) t_matrix[i * c + j] = 0.0f;
            for (size_t i = 1; i < c; ++i)
                for (size_t j = 0; j < i; ++j) {
                    float acc = 0;
                    for (size_t d = 0; d < dk; ++d)
                        acc = std::fma(k_normed[i * dk + d] *
                                           std::exp(gsum[i * dk + d] - gsum[j * dk + d]),
                                       k_normed[j * dk + d], acc);
                    t_matrix[i * c + j] = -beta[beta_base + i * profile.heads] * acc;
                }
            for (size_t i = 1; i < c; ++i) {
                for (size_t j = 0; j < i; ++j) {
                    float acc = 0;
                    for (size_t m = 0; m < i; ++m) acc += t_matrix[i * c + m] * t_matrix[m * c + j];
                    staging[j] = acc;
                }
                for (size_t j = 0; j < i; ++j) t_matrix[i * c + j] += staging[j];
            }
            // u = (T+I) @ (beta_key * v), w = (T+I) @ (beta_key * e^gsum * k):
            // the diagonal of (T+I) is beta of the own token, the strict-lower
            // columns carry the EARLIER (column) token's beta.
            for (size_t row = 0; row < c; ++row)
                for (size_t d = 0; d < dk; ++d) {
                    const float b_diag = beta[beta_base + row * profile.heads];
                    float w_acc = b_diag * k_normed[row * dk + d] * std::exp(gsum[row * dk + d]);
                    float u_acc = b_diag * v[base + row * d_inner + d];
                    for (size_t col = 0; col < row; ++col) {
                        const float t = t_matrix[row * c + col] * beta[beta_base + col * profile.heads];
                        w_acc = std::fma(t, k_normed[col * dk + d] * std::exp(gsum[col * dk + d]), w_acc);
                        u_acc = std::fma(t, v[base + col * d_inner + d], u_acc);
                    }
                    w_rows[row * dk + d] = w_acc;
                    u_rows[row * dk + d] = u_acc;
                }

            float* S = work + size_t(head) * dk * dk;
            for (size_t row = 0; row < c; ++row) {
                for (size_t d = 0; d < dk; ++d) {
                    float acc = 0;
                    for (size_t kd = 0; kd < dk; ++kd)
                        acc = std::fma(w_rows[row * dk + kd], S[kd * dk + d], acc);
                    v_fixed[row * dk + d] = u_rows[row * dk + d] - acc;
                }
                // Aqk row with diagonal kept, differences only.
                for (size_t col = 0; col <= row; ++col) {
                    float gram = 0;
                    for (size_t d = 0; d < dk; ++d)
                        gram = std::fma(q_normed[row * dk + d] *
                                            std::exp(gsum[row * dk + d] - gsum[col * dk + d]),
                                        k_normed[col * dk + d], gram);
                    staging[col] = gram;
                }
                float* o_row = staged + base + row * d_inner;
                for (size_t d = 0; d < dk; ++d) {
                    float acc = 0;
                    for (size_t kd = 0; kd < dk; ++kd)
                        acc = std::fma(q_normed[row * dk + kd] * std::exp(gsum[row * dk + kd]),
                                       S[kd * dk + d], acc);
                    for (size_t col = 0; col <= row; ++col)
                        acc = std::fma(staging[col], v_fixed[col * dk + d], acc);
                    o_row[d] = acc;
                }
            }
            // State: rows decay by the chunk's last cumsum, then the rank-c
            // update with decay-shifted keys and the corrected values.
            for (size_t kd = 0; kd < dk; ++kd) {
                const float decay = std::exp(gsum[(c - 1) * dk + kd]);
                for (size_t d = 0; d < dk; ++d) S[kd * dk + d] *= decay;
            }
            for (size_t kd = 0; kd < dk; ++kd)
                for (size_t d = 0; d < dk; ++d) {
                    float acc = 0;
                    for (size_t row = 0; row < c; ++row)
                        acc = std::fma(k_normed[row * dk + kd] *
                                           std::exp(gsum[(c - 1) * dk + kd] - gsum[row * dk + kd]),
                                       v_fixed[row * dk + d], acc);
                    S[kd * dk + d] += acc;
                }
        }

    for (size_t i = 0; i < state_values; ++i)
        if (!std::isfinite(work[i])) throw std::overflow_error("KDA chunk state overflowed");
    for (size_t i = 0; i < out.size(); ++i)
        if (!std::isfinite(staged[i])) throw std::overflow_error("KDA chunk output overflowed");
    for (size_t i = 0; i < state_values; ++i) state[i] = work[i];
    for (size_t i = 0; i < out.size(); ++i) out[i] = staged[i];
}

}  // namespace strata::hpu
