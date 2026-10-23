#include "strata/hpu/packed_decode.hpp"
#include "strata/artifact/dequant.hpp"

// Keep the exact existing codebooks and block geometry. This header is already vendored by
// Maya from ggml 3cf03257; its MIT notice is in third_party/ggml/LICENSE.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strata::hpu {
static_assert(std::endian::native == std::endian::little,
              "GGUF packed decoding currently requires a little-endian Host");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Packed codec reference requires IEEE-754 binary32");
namespace {

// Host expansion of Maya's existing dq_iq2_s / dq_iq3_s CUDA lane mappings in
// src/kernels/cuda/iq_kernels.cu. The grid entries and scales are unchanged; run all
// 32 logical lanes for one block. memcpy allows arbitrary alignment of a packed tile.
void iq2_s(const uint8_t* source, float* output) {
    block_iq2_s block;
    std::memcpy(&block, source, sizeof(block));
    for (int lane = 0; lane < 32; ++lane) {
        const int il = lane / 8, ib = lane % 8;
        const uint64_t grid = iq2s_grid[block.qs[4 * ib + il] | ((block.qh[ib] << (8 - 2 * il)) & 0x300)];
        const float scale = strata::fp16_to_fp32(block.d) *
                            (0.5f + ((block.scales[ib] >> (4 * (il / 2))) & 15)) * 0.25f;
        const uint8_t signs = block.qs[QK_K / 8 + 4 * ib + il];
        for (int j = 0; j < 8; ++j)
            output[32 * ib + 8 * il + j] = scale * static_cast<uint8_t>(grid >> (8 * j)) *
                                           (signs & kmask_iq2xs[j] ? -1.f : 1.f);
    }
}
void iq3_s(const uint8_t* source, float* output) {
    block_iq3_s block;
    std::memcpy(&block, source, sizeof(block));
    for (int lane = 0; lane < 32; ++lane) {
        const int il = lane / 8, ib = lane % 8;
        const uint8_t* qs = block.qs + 8 * ib;
        const uint32_t grid1 = iq3s_grid[qs[2 * il] | ((block.qh[ib] << (8 - 2 * il)) & 256)];
        const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((block.qh[ib] << (7 - 2 * il)) & 256)];
        const float scale = strata::fp16_to_fp32(block.d) *
                            (1 + 2 * ((block.scales[ib / 2] >> (4 * (ib % 2))) & 15));
        const uint8_t signs = block.signs[4 * ib + il];
        for (int j = 0; j < 4; ++j) {
            output[32 * ib + 8 * il + j] = scale * static_cast<uint8_t>(grid1 >> (8 * j)) *
                                          (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            output[32 * ib + 8 * il + j + 4] = scale * static_cast<uint8_t>(grid2 >> (8 * j)) *
                                              (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
    }
}

// ggml 3cf03257 ggml/src/ggml-quants.c::dequantize_row_q2_K. Unlike Q2_0 this
// representation has separate scale/min nibbles, bit planes and d/dmin at bytes 80/82.
void q2_k(const uint8_t* source, float* output) {
    const float d = strata::fp16_to_fp32(strata::read_u16(source + 80));
    const float minimum = strata::fp16_to_fp32(strata::read_u16(source + 82));
    for (int half = 0; half < 2; ++half) {
        for (int group = 0; group < 4; ++group) {
            for (int sub = 0; sub < 2; ++sub) {
                const uint8_t scales = source[8 * half + 2 * group + sub];
                const float dl = d * (scales & 15), ml = minimum * (scales >> 4);
                for (int j = 0; j < 16; ++j) {
                    const int code = (source[16 + 32 * half + 16 * sub + j] >> (2 * group)) & 3;
                    output[128 * half + 32 * group + 16 * sub + j] = dl * code - ml;
                }
            }
        }
    }
}

void decode_block(uint32_t type, const uint8_t* source, float* output) {
    switch (type) {
    case 10: q2_k(source, output); return;
    case 11: strata::dequantize_q3_K(source, output); return;
    case 21: iq3_s(source, output); return;
    case 22: iq2_s(source, output); return;
    case 23: strata::dequantize_iq4_xs(source, output); return;
    default: throw std::invalid_argument("HPU packed decoder does not support this GGUF type");
    }
}

template<class Output>
size_t decode(uint32_t type, std::span<const uint8_t> packed, uint64_t rows,
              uint64_t columns, std::span<Output> output) {
    if (!packed_decode_supported(type)) throw std::invalid_argument("HPU packed decoder does not support this GGUF type");
    if (!rows || !columns || columns % 256 || columns > SIZE_MAX || rows > SIZE_MAX / columns)
        throw std::invalid_argument("HPU packed decoder requires bounded rows and whole 256-value columns");
    int elements = 0, bytes = 0;
    strata::block_geometry(type, elements, bytes);
    const auto values = static_cast<size_t>(rows * columns);
    if (values > SIZE_MAX / sizeof(Output)) throw std::overflow_error("HPU packed decoder output size overflow");
    if (values / 256 > SIZE_MAX / static_cast<size_t>(bytes)) throw std::overflow_error("HPU packed decoder size overflow");
    const auto expected = values / 256 * static_cast<size_t>(bytes);
    if (packed.size() != expected || output.size() < values)
        throw std::invalid_argument("HPU packed decoder input/output extent mismatch");
    const auto input_address = reinterpret_cast<uintptr_t>(packed.data());
    const auto output_address = reinterpret_cast<uintptr_t>(output.data());
    if (input_address <= output_address ? output_address - input_address < packed.size()
                                       : input_address - output_address < values * sizeof(Output))
        throw std::invalid_argument("HPU packed decoder requires non-overlapping input/output storage");
    std::array<float, 256> block;
    for (size_t i = 0; i < values / 256; ++i) {
        decode_block(type, packed.data() + i * static_cast<size_t>(bytes), block.data());
        for (size_t j = 0; j < block.size(); ++j) {
            if constexpr (std::is_same_v<Output, uint16_t>) output[256 * i + j] = fp32_to_bf16_rne(block[j]);
            else output[256 * i + j] = block[j];
        }
    }
    return values;
}
}  // namespace

bool packed_decode_supported(uint32_t type) noexcept {
    return type == 10 || type == 11 || type == 21 || type == 22 || type == 23;
}
uint16_t fp32_to_bf16_rne(float value) noexcept {
    auto bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<uint16_t>((bits >> 16) | 0x40U);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}
size_t decode_packed_bf16(uint32_t type, std::span<const uint8_t> packed, uint64_t rows,
                         uint64_t columns, std::span<uint16_t> output) {
    return decode(type, packed, rows, columns, output);
}
size_t decode_packed_fp32(uint32_t type, std::span<const uint8_t> packed, uint64_t rows,
                         uint64_t columns, std::span<float> output) {
    return decode(type, packed, rows, columns, output);
}
}  // namespace strata::hpu
