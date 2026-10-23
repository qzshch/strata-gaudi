#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace strata::hpu {

// Explicit codec capability: these are the five routed-expert representations observed in
// the GLM UD-IQ3_XXS artifact, not an inference/TPC/MME capability declaration.
bool packed_decode_supported(uint32_t ggml_type) noexcept;
uint16_t fp32_to_bf16_rne(float value) noexcept;

// Row-major [rows, columns], no transpose or gate/up fusion. Input must contain exactly the
// requested packed rows; columns must be a whole number of GGML 256-value blocks. Output
// storage is supplied by the caller and must not overlap the input. Internally only one
// 256-float stack block is expanded.
// Extra output capacity is left untouched. No entire expert/layer or FP32 tile is allocated.
size_t decode_packed_bf16(uint32_t ggml_type, std::span<const uint8_t> packed,
                          uint64_t rows, uint64_t columns, std::span<uint16_t> output);

// Same scalar arithmetic before BF16 rounding, for independent numerical validation.
size_t decode_packed_fp32(uint32_t ggml_type, std::span<const uint8_t> packed,
                          uint64_t rows, uint64_t columns, std::span<float> output);

}  // namespace strata::hpu
