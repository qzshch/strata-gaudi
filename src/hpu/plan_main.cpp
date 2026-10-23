#include "strata/hpu/expert_plan.hpp"

#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {
uint64_t integer(const std::string& text) {
    uint64_t value = 0;
    const auto [p, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || p != text.data() + text.size()) throw std::invalid_argument("invalid integer: " + text);
    return value;
}
uint64_t gib(const std::string& text) {
    const auto n = integer(text);
    if (n > (UINT64_MAX >> 30)) throw std::invalid_argument("GiB argument overflows");
    return n << 30;
}
void usage() {
    std::puts("strata-hpu-plan [options] shard1.gguf [shard2.gguf ...]\n"
              "  --host-gib N             physical Host budget (default 64)\n"
              "  --hbm-gib N              available HBM budget (default 96)\n"
              "  --host-reserve-gib N     OS, CPU workers and other Host allocations (default 16)\n"
              "  --hbm-reserve-gib N      estimated dense/state/workspace/margin reserve (default 27)\n"
              "  --page-cache-gib N       Host allowance for file cache, not extra RAM (default 8)\n"
              "  --tile-rows N            maximum decoded rows at once (default 64)\n"
              "  --slots N                bounded in-flight transfer slots (default 2)\n"
              "  --placements FILE.csv    export each expert's unique owner and arena offset\n"
              "Reserves are user assumptions; packed nonexpert bytes do not prove runtime fit.\n"
              "No HPU allocation, inference or speed measurement is performed.");
}
}  // namespace

int main(int argc, char** argv) {
    using namespace strata::hpu;
    constexpr uint64_t G = uint64_t{1} << 30;
    MemoryBudget budget{96 * G, 27 * G, 64 * G, 16 * G, 8 * G, 64, 2, 64};
    std::vector<std::string> paths;
    std::string placements;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { usage(); return 0; }
            if (arg.starts_with("--")) {
                if (++i >= argc) throw std::invalid_argument("missing value for " + arg);
                const std::string value = argv[i];
                if (arg == "--host-gib") budget.host_total = gib(value);
                else if (arg == "--hbm-gib") budget.device_total = gib(value);
                else if (arg == "--host-reserve-gib") budget.host_reserve = gib(value);
                else if (arg == "--hbm-reserve-gib") budget.device_reserve = gib(value);
                else if (arg == "--page-cache-gib") budget.file_cache_allowance = gib(value);
                else if (arg == "--tile-rows") budget.tile_rows = integer(value);
                else if (arg == "--slots") budget.transfer_slots = integer(value);
                else if (arg == "--placements") placements = value;
                else throw std::invalid_argument("unknown option " + arg);
            } else paths.push_back(arg);
        }
        if (paths.empty()) { usage(); return 2; }
        const auto inventory = inspect_glm_gguf(paths);
        const auto plan = plan_experts(inventory, budget);
        if (!placements.empty()) {
            // A model is often hundreds of GiB: never truncate a source because a CSV
            // argument names it, including alternate paths, symlinks or hard links.
            const auto target = std::filesystem::absolute(placements).lexically_normal();
            for (const auto& source : paths) {
                std::error_code error;
                const auto input = std::filesystem::absolute(source).lexically_normal();
                if (target == input || std::filesystem::equivalent(target, input, error))
                    throw std::invalid_argument("placement output aliases a source GGUF shard: " + placements);
            }
            std::ofstream out(placements);
            if (!out) throw std::runtime_error("cannot create placement file " + placements);
            out << "layer,expert,owner,packed_bytes,allocation_bytes,arena_offset\n";
            for (const auto& p : plan.placements)
                out << p.id.layer << ',' << p.id.expert << ',' << tier_name(p.tier) << ','
                    << p.packed_bytes << ',' << p.allocation_bytes << ',' << p.arena_offset << '\n';
            if (!out) throw std::runtime_error("cannot write placement file " + placements);
        }
        std::printf("{\n  \"kind\": \"packed-expert-plan\",\n  \"hpu_inference_available\": false,\n"
                    "  \"trunk_layers\": %u,\n  \"experts_per_layer\": %u,\n"
                    "  \"nonexpert_packed_bytes\": %llu,\n  \"skipped_mtp_bytes\": %llu,\n"
                    "  \"host_staging_bytes\": %llu,\n  \"device_staging_bytes\": %llu,\n"
                    "  \"packed_tile_bytes\": %llu,\n  \"bf16_tile_bytes\": %llu,\n"
                    "  \"device_capacity_bytes\": %llu,\n  \"host_capacity_bytes\": %llu,\n"
                    "  \"device_allocated_bytes\": %llu,\n  \"host_allocated_bytes\": %llu,\n"
                    "  \"file_expert_bytes\": %llu,\n"
                    "  \"device_experts\": %llu,\n  \"host_experts\": %llu,\n  \"file_experts\": %llu\n}\n",
                    inventory.trunk_layers, inventory.experts_per_layer,
                    static_cast<unsigned long long>(inventory.nonexpert_packed_bytes),
                    static_cast<unsigned long long>(inventory.skipped_mtp_bytes),
                    static_cast<unsigned long long>(plan.host_staging_bytes),
                    static_cast<unsigned long long>(plan.device_staging_bytes),
                    static_cast<unsigned long long>(plan.packed_tile_bytes),
                    static_cast<unsigned long long>(plan.bf16_tile_bytes),
                    static_cast<unsigned long long>(plan.device_capacity),
                    static_cast<unsigned long long>(plan.host_capacity),
                    static_cast<unsigned long long>(plan.device_bytes),
                    static_cast<unsigned long long>(plan.host_bytes),
                    static_cast<unsigned long long>(plan.file_bytes),
                    static_cast<unsigned long long>(plan.device_count),
                    static_cast<unsigned long long>(plan.host_count),
                    static_cast<unsigned long long>(plan.file_count));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-hpu-plan: %s\n", e.what());
        return 1;
    }
}
