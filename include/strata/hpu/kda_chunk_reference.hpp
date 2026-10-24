#pragma once

#include "strata/hpu/kda_reference.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

inline constexpr const char* kKdaChunkReferenceProfile = "glm53_kda_chunk_fp32";

// Production chunk width of the reference stack (FLA/abotsis default).
inline constexpr uint32_t kGlm53KdaChunk = 64;
// Staging bound: 64 production, 256 keeps accidental wider profiles testable.
inline constexpr uint32_t kKdaMaxChunk = 256;

// Total scratch of kda_chunk_reference in floats: the staged state image and
// output image (validate-then-commit, same contract as the recurrent scan)
// plus one reusable per-head work plane of chunk*(chunk + 6*head_dim + 1)
// values (rebased cumsum, normalized q/k, T, u, w, corrected values - six
// chunk x head_dim planes - plus the forward-substitution/Aqk staging row).
[[nodiscard]] constexpr size_t kda_chunk_scratch_values(const KdaProfile& profile,
                                                         size_t tokens, uint32_t chunk) {
    const size_t per_head = size_t(chunk) * (chunk + 6 * size_t(profile.head_dim) + 1);
    return kda_scan_scratch_values(profile, tokens) + per_head;
}

// Chunked prefill over whole chunks: the FLA naive chunk algorithm with the
// audited GLM-5.3 semantics. Inputs match kda_gated_delta_reference exactly
// (q/k/v post-conv SiLU planes, gate in log space from the decay-gate
// reference, already-sigmoided beta); tokens must be a multiple of chunk (the
// caller pads tails with g=0/q=k=v=beta=0 rows, which contribute nothing).
// q/k get the in-kernel l2 norm (eps 1e-6) and q the head_dim^-0.5 scale.
//
// Per chunk and head: a per-chunk rebased cumsum of the log gate; the decayed
// key Gram taken from DIFFERENCES exp(gsum[c]-gsum[i]) (never the factored
// product form, which overflows for long chunks); the unit-diagonal strict
// lower system solved by exact forward substitution (no Neumann iteration);
// beta of the earlier token scales the Gram rows and beta of the own token
// scales the v/w operands; Aqk keeps its diagonal; the state affine update
// carries the chunk recurrence. All arithmetic FP32 with /fp:strict.
//
// state and out commit together only after every value validates, exactly like
// the recurrent scan; scratch must hold kda_chunk_scratch_values and must not
// alias any plane. This is the reference chunk algorithm only: no varlen
// packing, no HPU execution, no performance claim.
void kda_chunk_reference(std::span<const float> q, std::span<const float> k,
                         std::span<const float> v, std::span<const float> gate,
                         std::span<const float> beta, std::span<float> state,
                         std::span<float> out, std::span<float> scratch, const KdaProfile&,
                         uint32_t chunk);

}  // namespace strata::hpu
