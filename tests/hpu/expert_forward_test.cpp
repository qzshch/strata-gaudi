#include "strata/hpu/expert_forward.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winioctl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;
namespace fs = std::filesystem;

struct GoldenBlock {
    std::vector<uint8_t> packed;
    std::array<float, 256> values;
};
using Goldens = std::map<uint32_t, std::vector<GoldenBlock>>;
struct Fragment { size_t row = 0, column = 0; const GoldenBlock* block = nullptr; };
using Fragments = std::array<std::vector<Fragment>, 3>;

static uint32_t u32(std::istream& stream) {
    std::array<uint8_t, 4> bytes;
    stream.read(reinterpret_cast<char*>(bytes.data()), 4);
    CHECK(stream.gcount() == 4);
    return bytes[0] | (uint32_t{bytes[1]} << 8) | (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[3]} << 24);
}
static Goldens load_goldens(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    std::array<char, 8> magic;
    stream.read(magic.data(), 8);
    CHECK((magic == std::array<char, 8>{'S','T','H','P','U','D','Q','1'}));
    const auto cases = u32(stream);
    CHECK(cases == 10);
    Goldens result;
    for (uint32_t c = 0; c < cases; ++c) {
        const auto type = u32(stream), rows = u32(stream), columns = u32(stream), bytes = u32(stream);
        CHECK(rows && rows < 10 && columns <= 4096 && bytes < 65536);
        const auto blocks = static_cast<size_t>(rows) * columns / 256;
        const auto block_bytes = bytes / blocks;
        std::vector<GoldenBlock> data(blocks);
        for (auto& block : data) {
            block.packed.resize(block_bytes);
            stream.read(reinterpret_cast<char*>(block.packed.data()), static_cast<std::streamsize>(block_bytes));
            CHECK(stream.gcount() == static_cast<std::streamsize>(block_bytes));
        }
        for (auto& block : data) for (auto& value : block.values) value = std::bit_cast<float>(u32(stream));
        stream.seekg(static_cast<std::streamoff>(static_cast<size_t>(rows) * columns * 2), std::ios::cur);
        CHECK(stream.good());
        if (columns == 4096) result[type] = std::move(data);
    }
    CHECK(result.size() == 5 && stream.peek() == std::char_traits<char>::eof());
    return result;
}

// Real-shape matrices use sparse holes, not huge in-memory arrays or explicit zero writes.
// Every nonzero block comes from the already-independent gguf-py numerical goldens.
static uint64_t sparse_matrix(const fs::path& path, uint64_t extent,
                               const std::vector<Fragment>& fragments, size_t row_bytes,
                               size_t block_bytes, size_t data_offset) {
#ifdef _WIN32
    const HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(handle != INVALID_HANDLE_VALUE);
    try {
        DWORD returned = 0;
        CHECK(DeviceIoControl(handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr));
        LARGE_INTEGER end;
        end.QuadPart = static_cast<LONGLONG>(extent);
        CHECK(SetFilePointerEx(handle, end, nullptr, FILE_BEGIN) && SetEndOfFile(handle));
        for (const auto& fragment : fragments) {
            LARGE_INTEGER offset;
            offset.QuadPart = static_cast<LONGLONG>(data_offset + fragment.row * row_bytes + fragment.column / 256 * block_bytes);
            CHECK(SetFilePointerEx(handle, offset, nullptr, FILE_BEGIN));
            DWORD wrote = 0;
            CHECK(WriteFile(handle, fragment.block->packed.data(), static_cast<DWORD>(block_bytes), &wrote, nullptr));
            CHECK(wrote == block_bytes);
        }
        CHECK(CloseHandle(handle));
    } catch (...) { CloseHandle(handle); throw; }
    DWORD high = 0;
    SetLastError(NO_ERROR);
    const auto low = GetCompressedFileSizeW(path.c_str(), &high);
    CHECK(low != INVALID_FILE_SIZE || GetLastError() == NO_ERROR);
    return (uint64_t{high} << 32) | low;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    CHECK(fd >= 0);
    try {
        CHECK(::ftruncate(fd, static_cast<off_t>(extent)) == 0);
        for (const auto& fragment : fragments) {
            const auto offset = static_cast<off_t>(data_offset + fragment.row * row_bytes + fragment.column / 256 * block_bytes);
            CHECK(::pwrite(fd, fragment.block->packed.data(), block_bytes, offset) == static_cast<ssize_t>(block_bytes));
        }
        CHECK(::close(fd) == 0);
    } catch (...) { ::close(fd); throw; }
    struct stat status;
    CHECK(::stat(path.c_str(), &status) == 0);
    return static_cast<uint64_t>(status.st_blocks) * 512;
#endif
}

struct TemporarySources {
    fs::path directory;
    explicit TemporarySources(const fs::path& base) {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = fs::absolute(base / ("expert-forward-" + std::to_string(unique)));
        CHECK(fs::create_directory(directory));
    }
    ~TemporarySources() { std::error_code error; fs::remove_all(directory, error); }
};

static std::pair<PackedExpert, Fragments> build(const fs::path& base, const Goldens& goldens,
                                               size_t hidden, size_t ff, std::array<uint32_t, 3> types) {
    PackedExpert expert;
    Fragments fragments;
    for (size_t role = 0; role < 3; ++role) {
        const auto rows = role == 2 ? hidden : ff, columns = role == 2 ? ff : hidden;
        const auto& blocks = goldens.at(types[role]);
        const auto block_bytes = blocks[0].packed.size();
        const std::array<size_t, 5> active_rows = {0, 1, rows / 2, rows - 2, rows - 1};
        for (size_t i = 0; i < active_rows.size(); ++i) {
            fragments[role].push_back({active_rows[i], 0, &blocks[6 + (i + role) % 4]});
            if (columns > 256)
                fragments[role].push_back({active_rows[i], columns - 256, &blocks[6 + (i + role + 1) % 4]});
        }
        const auto path = base / ("matrix-" + std::to_string(role) + ".bin");
        const auto row_bytes = columns / 256 * block_bytes;
        const size_t data_offset = 128 + role * 32;
        const auto allocated = sparse_matrix(path, data_offset + rows * row_bytes, fragments[role], row_bytes, block_bytes, data_offset);
        CHECK(allocated < 1024 * 1024);  // large geometry still has bounded physical fixture storage
        expert.matrices[role] = {path.string(), data_offset, rows, columns, row_bytes, types[role]};
    }
    return {expert, fragments};
}

struct Oracle {
    std::vector<float> output;
    size_t gate_clip = 0, negative_gate = 0, up_clip = 0;
};
// No production decoder or forward function is called here. Expected FP32 weights are
// pinned gguf-py's independent results. Sparse FP64 sums model a separate matrix multiply;
// the SiLU expression is the direct scalar formula rather than the implementation's branch.
static Oracle oracle(const Fragments& fragments, const std::vector<float>& input, size_t ff, float limit) {
    std::vector<double> gate(ff, 0), up(ff, 0);
    const auto dots = [](const auto& pieces, const auto& x, auto& sums) {
        for (const auto& fragment : pieces)
            for (size_t j = 0; j < 256; ++j)
                sums[fragment.row] += static_cast<double>(fragment.block->values[j]) * x[fragment.column + j];
    };
    dots(fragments[0], input, gate);
    dots(fragments[1], input, up);
    Oracle result;
    std::vector<float> activation(ff, 0);
    for (size_t j = 0; j < ff; ++j) {
        const auto g_stored = static_cast<float>(gate[j]), u_stored = static_cast<float>(up[j]);
        result.gate_clip += g_stored > limit;
        result.negative_gate += g_stored < -limit;
        result.up_clip += std::abs(u_stored) > limit;
        const double g = std::min(g_stored, limit), u = std::clamp(u_stored, -limit, limit);
        activation[j] = static_cast<float>((g / (1.0 + std::exp(-g))) * u);
    }
    std::vector<double> down(input.size(), 0);
    dots(fragments[2], activation, down);
    for (const auto value : down) result.output.push_back(static_cast<float>(value));
    return result;
}

static void refuses_unchanged(const std::function<void()>& work, const std::vector<float>& output,
                               const std::vector<float>& original) {
    bool refused = false;
    try { work(); } catch (const std::exception&) { refused = true; }
    CHECK(refused && output == original);
}

static void run_case(const fs::path& base, const Goldens& goldens, size_t hidden, size_t ff,
                     std::array<uint32_t, 3> types, bool failures) {
    TemporarySources sources(base);
    const auto [expert, pieces] = build(sources.directory, goldens, hidden, ff, types);
    std::vector<float> input(hidden);
    for (size_t j = 0; j < hidden; ++j) input[j] = static_cast<float>(static_cast<int>(j % 19) - 9) * 0.013f;
    const auto expected = oracle(pieces, input, ff, 10.f);
    if (hidden == 256) CHECK(expected.gate_clip > 0 && expected.negative_gate > 0 && expected.up_clip > 0);
    CHECK(std::any_of(expected.output.begin(), expected.output.end(), [](float value) { return value != 0; }));
    std::vector<float> previous;
    for (const size_t tile_rows : {size_t{1}, size_t{7}, size_t{17}}) {
        if (hidden > 256 && tile_rows == 1) continue;
        const auto requirements = expert_forward_requirements(expert, tile_rows);
        PackedTileStream stream(1, requirements.packed_tile_bytes);
        std::vector<float> scratch(requirements.scratch_values + 2, 314159.f);
        std::vector<float> output(hidden + 2, 123456.f);
        const auto workspace = std::span<float>{scratch.data() + 1, requirements.scratch_values};
        const auto target = std::span<float>{output.data() + 1, hidden};
        ExpertForwardOptions options{tile_rows, 10.f, stream.resident_bytes() + requirements.scratch_bytes + requirements.codec_stack_bytes};
        const auto report = expert_forward_reference(expert, input, target, stream, workspace, options);
        CHECK(report.payload_workspace_bytes == options.payload_budget_bytes);
        CHECK(report.requested_file_bytes == expert.matrices[0].bytes() + expert.matrices[1].bytes() + expert.matrices[2].bytes());
        CHECK(report.tiles == 2 * ((ff + tile_rows - 1) / tile_rows) + (hidden + tile_rows - 1) / tile_rows);
        CHECK(stream.busy_slots() == 0);
        CHECK(scratch.front() == 314159.f && scratch.back() == 314159.f);
        CHECK(output.front() == 123456.f && output.back() == 123456.f);
        for (size_t j = 0; j < hidden; ++j) {
            const double error = std::abs(static_cast<double>(target[j]) - expected.output[j]);
            CHECK(error <= 1e-6 + 3e-6 * std::abs(static_cast<double>(expected.output[j])));
        }
        if (!previous.empty()) CHECK(std::equal(target.begin(), target.end(), previous.begin()));
        previous.assign(target.begin(), target.end());
        if (failures && tile_rows == 17) {
            const auto original = output;
            auto fail = [&](const auto& e, const auto& x, auto out, auto work, auto opt) {
                refuses_unchanged([&] { expert_forward_reference(e, x, out, stream, work, opt); }, output, original);
                CHECK(stream.busy_slots() == 0);
            };
            auto too_small = options; --too_small.payload_budget_bytes;
            fail(expert, input, target, workspace, too_small);
            PackedTileStream undersized_slots(2, requirements.packed_tile_bytes - 1);
            CHECK(undersized_slots.resident_bytes() >= requirements.packed_tile_bytes);
            auto adequate_budget = options;
            adequate_budget.payload_budget_bytes = undersized_slots.resident_bytes() + requirements.scratch_bytes + requirements.codec_stack_bytes;
            refuses_unchanged([&] { expert_forward_reference(expert, input, target, undersized_slots, workspace, adequate_budget); }, output, original);
            CHECK(undersized_slots.requested_file_bytes() == 0);
            fail(expert, input, target, workspace.first(workspace.size() - 1), options);
            fail(expert, std::span<const float>(input).first(hidden - 1), target, workspace, options);
            fail(expert, input, target.first(hidden - 1), workspace, options);
            auto bad = expert; ++bad.matrices[0].row_bytes;
            fail(bad, input, target, workspace, options);
            bad = expert; bad.matrices[1].type = 18;
            fail(bad, input, target, workspace, options);
            bad = expert; --bad.matrices[2].columns;
            fail(bad, input, target, workspace, options);
            auto bad_input = input; bad_input[0] = std::numeric_limits<float>::quiet_NaN();
            fail(expert, bad_input, target, workspace, options);
            bad_input = input; bad_input[0] = std::numeric_limits<float>::max();
            fail(expert, bad_input, target, workspace, options);
            too_small = options; too_small.swiglu_limit = 0;
            fail(expert, input, target, workspace, too_small);
            too_small = options; too_small.tile_rows = 0;
            fail(expert, input, target, workspace, too_small);
            bad = expert; bad.matrices[1].path += ".absent";
            fail(bad, input, target, workspace, options);
            fail(expert, std::span<const float>(workspace).first(hidden), target, workspace, options);
            fail(expert, input, workspace.first(hidden), workspace, options);
            std::vector<float> aliased_tail(hidden + requirements.scratch_values + 1, 271828.f);
            const auto tail_original = aliased_tail;
            const auto tail_work = std::span<float>(aliased_tail).subspan(hidden + 1);
            refuses_unchanged([&] { expert_forward_reference(expert, input, aliased_tail, stream, tail_work, options); }, aliased_tail, tail_original);
            // Successful in-place publication is safe after down has consumed every input.
            auto in_place = input;
            expert_forward_reference(expert, in_place, in_place, stream, workspace, options);
            CHECK(std::equal(in_place.begin(), in_place.end(), target.begin()));
            // A non-finite stored scale is rejected after decode, with its tile lease returned.
            const auto scale_offset = static_cast<std::streamoff>(expert.matrices[0].offset +
                (expert.matrices[0].type == 10 ? 80 : expert.matrices[0].type == 11 ? 108 : 0));
            std::array<char, 2> original_scale;
            {
                std::fstream file(expert.matrices[0].path, std::ios::binary | std::ios::in | std::ios::out);
                CHECK(file.good());
                file.seekg(scale_offset);
                file.read(original_scale.data(), 2);
                CHECK(file.gcount() == 2);
                file.seekp(scale_offset);
                const std::array<char, 2> nan_scale = {0, 0x7e};
                file.write(nan_scale.data(), 2);
                CHECK(file.good());
            }
            fail(expert, input, target, workspace, options);
            {
                std::fstream file(expert.matrices[0].path, std::ios::binary | std::ios::in | std::ios::out);
                file.seekp(scale_offset);
                file.write(original_scale.data(), 2);
                CHECK(file.good());
            }
            // The final down tile fails only after all gate/up and earlier down work; the
            // caller still sees its original output and the ready ticket is not leaked.
            fs::resize_file(expert.matrices[2].path, fs::file_size(expert.matrices[2].path) - 1);
            fail(expert, input, target, workspace, options);
        }
    }
    std::printf("Expert reference: hidden=%zu ff=%zu formats=%u/%u/%u matches independent sparse FP64 oracle\n",
                hidden, ff, types[0], types[1], types[2]);
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 3);
        const auto goldens = load_goldens(argv[1]);
        run_case(argv[2], goldens, 256, 256, {21, 22, 23}, true);
        run_case(argv[2], goldens, 256, 256, {10, 11, 21}, true);
        run_case(argv[2], goldens, 4096, 2048, {22, 21, 10}, false);
        std::puts("Bounded expert reference: mixed codecs, exact geometry, clip semantics and transactional output passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
