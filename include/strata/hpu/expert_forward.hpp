#pragma once

#include "strata/hpu/expert_plan.hpp"
#include "strata/hpu/tile_stream.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

inline constexpr const char* kExpertReferenceProfile = "packed_fp32_fp64_accum";

struct ExpertForwardRequirements {
    size_t packed_tile_bytes = 0;
    size_t decoded_tile_values = 0;
    size_t activation_values = 0;
    size_t candidate_values = 0;
    size_t scratch_values = 0;
    // Payload only: add the caller's stream.resident_bytes(). File-stream/allocator metadata
    // and OS page cache remain part of the application's separate Host reserve.
    size_t scratch_bytes = 0;
    // Explicit float expansion only; packed block structs and ordinary call-stack overhead
    // stay in the application's Host reserve, not an asserted exact process-stack bound.
    size_t codec_stack_bytes = 256 * sizeof(float);
};

ExpertForwardRequirements expert_forward_requirements(const PackedExpert&, uint64_t tile_rows);

struct ExpertForwardOptions {
    uint64_t tile_rows = 64;
    float swiglu_limit = 10.f;
    size_t payload_budget_bytes = 0;  // caller must explicitly budget stream + scratch + codec stack
};

struct ExpertForwardStats {
    uint64_t requested_file_bytes = 0;
    uint64_t tiles = 0;
    size_t payload_workspace_bytes = 0;
};

// Single token, one UNWEIGHTED routed expert. GGUF weights are decoded exactly to FP32.
// Each dot accumulates left-to-right in FP64, then casts to FP32. Activation is
// silu(min(gate,limit))*clamp(up,-limit,limit), evaluated in FP64 and stored as FP32.
// Down uses those FP32 activations and FP64 accumulation, then stores FP32 output.
// There is no Q8_K activation quantization, router scaling, BF16 or implicit model dispatch.
//
// Scratch and the stream are caller-owned and exclusively borrowed for this call. Scratch
// must not alias input/output. Input/output may overlap: publication occurs only after all
// reads and arithmetic succeed. On any failure the caller's output is unchanged; scratch
// is unspecified. No whole matrix/expert is decoded or copied into RAM.
ExpertForwardStats expert_forward_reference(const PackedExpert&, std::span<const float> input,
                                            std::span<float> output, PackedTileStream&,
                                            std::span<float> scratch, const ExpertForwardOptions&);

}  // namespace strata::hpu
