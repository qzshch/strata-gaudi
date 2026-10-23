#include "strata/hpu/expert_forward.hpp"
#include "strata/hpu/packed_decode.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
size_t add(size_t a, size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("expert reference workspace overflow");
    return a + b;
}
size_t multiply(uint64_t a, uint64_t b) {
    if (a > SIZE_MAX || b > SIZE_MAX || (a && b > SIZE_MAX / a))
        throw std::overflow_error("expert reference extent overflow");
    return static_cast<size_t>(a * b);
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    if (!an || !bn) return false;
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}
float stored(double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::runtime_error("expert reference arithmetic exceeds finite FP32");
    const auto result = static_cast<float>(value);
    if (!std::isfinite(result)) throw std::runtime_error("expert reference arithmetic is not finite FP32");
    return result;
}
float activation(float gate, float up, float limit) {
    const double g = std::min(gate, limit), u = std::clamp(up, -limit, limit);
    // Equivalent SiLU, with a stable negative branch. Large negative gates have no lower
    // clamp: their limiting output is signed zero, rather than an overflow in exp(-gate).
    const double sigmoid = g >= 0 ? 1.0 / (1.0 + std::exp(-g))
                                  : std::exp(g) / (1.0 + std::exp(g));
    return stored((g * sigmoid) * u);
}
struct ReadyTile {
    PackedTileStream& stream;
    PackedTileStream::Ticket ticket;
    ~ReadyTile() { stream.discard(ticket); }
};

template<class Consume>
void matrix_rows(const PackedMatrix& matrix, std::span<const float> input,
                  std::span<float> decoded, PackedTileStream& stream,
                  uint64_t tile_rows, uint64_t& tiles, Consume consume) {
    for (uint64_t first = 0; first < matrix.rows;) {
        const auto rows = std::min(tile_rows, matrix.rows - first);
        const auto ticket = stream.stage(matrix, first, rows);
        if (!ticket) throw std::logic_error("expert reference staging stream is busy");
        ReadyTile lease{stream, *ticket};
        decode_packed_fp32(matrix.type, stream.bytes(*ticket), rows, matrix.columns, decoded);
        ++tiles;
        for (uint64_t row = 0; row < rows; ++row) {
            double sum = 0;
            const auto offset = multiply(row, matrix.columns);
            for (size_t column = 0; column < input.size(); ++column) {
                const float weight = decoded[offset + column];
                if (!std::isfinite(weight)) throw std::runtime_error("expert reference decoded a non-finite weight");
                sum += static_cast<double>(weight) * static_cast<double>(input[column]);
            }
            consume(static_cast<size_t>(first + row), stored(sum));
        }
        first += rows;
    }
}
}  // namespace

ExpertForwardRequirements expert_forward_requirements(const PackedExpert& expert, uint64_t tile_rows) {
    if (!tile_rows) throw std::invalid_argument("expert reference tile rows must be positive");
    const auto& gate = expert.matrices[0];
    const auto& up = expert.matrices[1];
    const auto& down = expert.matrices[2];
    if (!gate.rows || !gate.columns || up.rows != gate.rows || up.columns != gate.columns ||
        down.rows != gate.columns || down.columns != gate.rows)
        throw std::invalid_argument("expert reference requires gate/up[ff,hidden], down[hidden,ff]");
    ExpertForwardRequirements result;
    for (const auto& matrix : expert.matrices) {
        if (!packed_decode_supported(matrix.type) || matrix.columns % 256)
            throw std::invalid_argument("expert reference requires a supported whole-block packed format");
        int block = 0, bytes = 0;
        strata::block_geometry(matrix.type, block, bytes);
        const auto expected_row = multiply(matrix.columns / static_cast<uint64_t>(block), static_cast<uint64_t>(bytes));
        if (matrix.row_bytes != expected_row) throw std::invalid_argument("expert reference row bytes disagree with GGUF type");
        const auto size = multiply(matrix.rows, matrix.row_bytes);
        if (matrix.offset > UINT64_MAX - size) throw std::overflow_error("expert reference matrix file extent overflow");
        const auto rows = std::min(tile_rows, matrix.rows);
        result.packed_tile_bytes = std::max(result.packed_tile_bytes, multiply(rows, matrix.row_bytes));
        result.decoded_tile_values = std::max(result.decoded_tile_values, multiply(rows, matrix.columns));
    }
    result.activation_values = multiply(gate.rows, 1);
    result.candidate_values = multiply(down.rows, 1);
    result.scratch_values = add(add(result.decoded_tile_values, result.activation_values), result.candidate_values);
    result.scratch_bytes = multiply(result.scratch_values, sizeof(float));
    return result;
}

ExpertForwardStats expert_forward_reference(const PackedExpert& expert, std::span<const float> input,
                                            std::span<float> output, PackedTileStream& stream,
                                            std::span<float> scratch, const ExpertForwardOptions& options) {
    const auto required = expert_forward_requirements(expert, options.tile_rows);
    if (input.size() != expert.matrices[0].columns || output.size() < required.candidate_values ||
        scratch.size() < required.scratch_values)
        throw std::invalid_argument("expert reference input/output/scratch extent mismatch");
    if (!(options.swiglu_limit > 0) || !std::isfinite(options.swiglu_limit))
        throw std::invalid_argument("expert reference requires a finite positive SwiGLU limit");
    const auto payload = add(add(stream.resident_bytes(), required.scratch_bytes), required.codec_stack_bytes);
    if (payload > options.payload_budget_bytes || stream.slot_capacity_bytes() < required.packed_tile_bytes)
        throw std::invalid_argument("expert reference staging and scratch exceed the caller payload budget");
    if (stream.busy_slots()) throw std::invalid_argument("expert reference requires an exclusively borrowed idle stream");
    if (overlap(scratch.data(), scratch.size_bytes(), input.data(), input.size_bytes()) ||
        overlap(scratch.data(), scratch.size_bytes(), output.data(), output.size_bytes()))
        throw std::invalid_argument("expert reference scratch aliases input or output");
    for (const auto value : input) if (!std::isfinite(value)) throw std::invalid_argument("expert reference input must be finite");
    const auto decoded = scratch.first(required.decoded_tile_values);
    const auto hidden = scratch.subspan(required.decoded_tile_values, required.activation_values);
    const auto candidate = scratch.subspan(required.decoded_tile_values + required.activation_values, required.candidate_values);
    ExpertForwardStats stats;
    stats.payload_workspace_bytes = payload;
    const auto file_before = stream.requested_file_bytes();
    matrix_rows(expert.matrices[0], input, decoded, stream, options.tile_rows, stats.tiles,
                [&](size_t row, float value) { hidden[row] = value; });
    matrix_rows(expert.matrices[1], input, decoded, stream, options.tile_rows, stats.tiles,
                [&](size_t row, float value) { hidden[row] = activation(hidden[row], value, options.swiglu_limit); });
    matrix_rows(expert.matrices[2], hidden, decoded, stream, options.tile_rows, stats.tiles,
                [&](size_t row, float value) { candidate[row] = value; });
    // No earlier path writes the caller's output. In-place input/output is safe here too.
    std::copy(candidate.begin(), candidate.end(), output.begin());
    stats.requested_file_bytes = stream.requested_file_bytes() - file_before;
    return stats;
}
}  // namespace strata::hpu
