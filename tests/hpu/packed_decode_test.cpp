#include "strata/hpu/packed_decode.hpp"
#include "strata/hpu/tile_stream.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <vector>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;

static void exact(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    CHECK(stream.gcount() == static_cast<std::streamsize>(size));
}
static uint32_t u32(std::istream& stream) {
    std::array<uint8_t, 4> b;
    exact(stream, b.data(), b.size());
    return b[0] | (uint32_t{b[1]} << 8) | (uint32_t{b[2]} << 16) | (uint32_t{b[3]} << 24);
}
static uint16_t u16(std::istream& stream) {
    std::array<uint8_t, 2> b;
    exact(stream, b.data(), b.size());
    return static_cast<uint16_t>(b[0] | (uint16_t{b[1]} << 8));
}
static void refuses(const std::function<void()>& f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
}

static void rounding() {
    struct BitsCase { uint32_t bits; uint16_t expected; };
    const std::array<BitsCase, 15> pairs = {{
        {0x00000000, 0x0000}, {0x80000000, 0x8000},
        {0x3f808000, 0x3f80}, {0x3f818000, 0x3f82},
        {0x3f808001, 0x3f81}, {0x3f807fff, 0x3f80}, {0xbf808000, 0xbf80},
        {0x00008000, 0x0000}, {0x00018000, 0x0002},
        {0x7f800000, 0x7f80}, {0xff800000, 0xff80},
        {0x7f800001, 0x7fc0}, {0x7fffffff, 0x7fff},
        {0xff800001, 0xffc0}, {0x3f800000, 0x3f80},
    }};
    for (const auto [bits, expected] : pairs) CHECK(fp32_to_bf16_rne(std::bit_cast<float>(bits)) == expected);
}

static void vectors(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    CHECK(stream.good());
    std::array<char, 8> magic;
    exact(stream, magic.data(), magic.size());
    CHECK((magic == std::array<char, 8>{'S','T','H','P','U','D','Q','1'}));
    const auto cases = u32(stream);
    CHECK(cases == 10);
    size_t compared = 0;
    for (uint32_t i = 0; i < cases; ++i) {
        const auto type = u32(stream), rows = u32(stream), columns = u32(stream), bytes = u32(stream);
        CHECK(packed_decode_supported(type) && rows > 0 && rows < 10 && columns <= 4096 && bytes < 65536);
        const size_t values = static_cast<size_t>(rows) * columns;
        // Intentionally misalign input; the on-disk expert's offset need not align with a C struct.
        std::vector<uint8_t> storage(bytes + 1);
        const auto source_offset = static_cast<uint64_t>(stream.tellg());
        exact(stream, storage.data() + 1, bytes);
        const std::span<const uint8_t> packed{storage.data() + 1, bytes};
        std::vector<uint32_t> expected_fp32(values);
        std::vector<uint16_t> expected_bf16(values);
        for (auto& value : expected_fp32) value = u32(stream);
        for (auto& value : expected_bf16) value = u16(stream);
        constexpr float sentinel = 1234567.0f;
        std::vector<float> fp32(values + 2, sentinel);
        std::vector<uint16_t> bf16(values + 2, 0x55aa);
        CHECK(decode_packed_fp32(type, packed, rows, columns, {fp32.data() + 1, values}) == values);
        CHECK(decode_packed_bf16(type, packed, rows, columns, {bf16.data() + 1, values}) == values);
        CHECK(fp32.front() == sentinel && fp32.back() == sentinel);
        CHECK(bf16.front() == 0x55aa && bf16.back() == 0x55aa);
        for (size_t j = 0; j < values; ++j) {
            if (std::bit_cast<uint32_t>(fp32[j + 1]) != expected_fp32[j] || bf16[j + 1] != expected_bf16[j]) {
                std::fprintf(stderr, "oracle mismatch type %u case %u index %zu: fp32 %08x vs %08x, bf16 %04x vs %04x\n",
                             type, i, j, std::bit_cast<uint32_t>(fp32[j + 1]), expected_fp32[j],
                             bf16[j + 1], expected_bf16[j]);
                throw std::runtime_error("packed codec differs from pinned gguf-py numerical oracle");
            }
        }
        // Splitting by rows must preserve exactly the same output and never expand a full expert.
        const auto row_bytes = bytes / rows;
        std::vector<uint16_t> row(columns);
        const PackedMatrix matrix{path, source_offset, rows, columns, row_bytes, type};
        PackedTileStream tiles(2, row_bytes);
        for (uint32_t r = 0; r < rows; ++r) {
            const auto ticket = tiles.stage(matrix, r, 1).value();
            CHECK(decode_packed_bf16(type, tiles.bytes(ticket), 1, columns, row) == columns);
            CHECK(std::equal(row.begin(), row.end(), expected_bf16.begin() + static_cast<size_t>(r) * columns));
            tiles.discard(ticket);  // the synchronous CPU decoder has completely consumed it
        }
        CHECK(tiles.resident_bytes() == 2 * row_bytes && tiles.requested_file_bytes() == bytes);
        refuses([&] { decode_packed_bf16(type, packed.first(bytes - 1), rows, columns, bf16); });
        refuses([&] { decode_packed_bf16(type, packed, rows, columns, {bf16.data(), values - 1}); });
        refuses([&] { decode_packed_bf16(type, packed, rows, columns - 1, bf16); });
        refuses([&] { decode_packed_bf16(type, packed, 0, columns, bf16); });
        refuses([&] { decode_packed_bf16(18, packed, rows, columns, bf16); });
        const auto alias = std::span<const uint8_t>{reinterpret_cast<const uint8_t*>(bf16.data()), bytes};
        refuses([&] { decode_packed_bf16(type, alias, rows, columns, bf16); });
        compared += values;
    }
    CHECK(stream.peek() == std::char_traits<char>::eof());
    std::printf("Packed codec matches %zu FP32 values and BF16 bits from pinned gguf-py\n", compared);
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        rounding();
        vectors(argv[1]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
