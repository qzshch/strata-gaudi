#include "strata/hpu/expert_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)

using namespace strata::hpu;

static void refuses(const std::function<void()>& f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
}

static ExpertInventory toy() {
    ExpertInventory inventory;
    // Each expert is 3 * 256 bytes; one-row staging is 64 packed + 64 BF16.
    for (uint32_t e = 0; e < 6; ++e) {
        PackedExpert expert;
        expert.id = {3, e};
        for (auto& m : expert.matrices) m = {"unused", 0, 4, 32, 64, 1};
        inventory.experts.push_back(expert);
    }
    return inventory;
}

static void budget_and_ownership() {
    const auto inventory = toy();
    MemoryBudget budget{2 * 768 + 256, 0, 2 * 768 + 128, 0, 0, 1, 2, 64};
    const auto plan = plan_experts(inventory, budget, {{3, 5}, {3, 1}});
    CHECK(plan.device_count == 2 && plan.host_count == 2 && plan.file_count == 2);
    CHECK(plan.device_bytes == 1536 && plan.host_bytes == 1536 && plan.file_bytes == 1536);
    CHECK(plan.device_capacity == 1536 && plan.host_capacity == 1536);
    CHECK(plan.host_staging_bytes == 128 && plan.device_staging_bytes == 256);
    CHECK(plan.placements[0].id.expert == 5 && plan.placements[1].id.expert == 1);
    CHECK(plan.placements[2].id.expert == 0 && plan.placements[2].tier == ExpertTier::host);
    std::set<ExpertId> owners;
    for (const auto& p : plan.placements) CHECK(owners.insert(p.id).second);
    CHECK(owners.size() == inventory.experts.size());
    refuses([&] { plan_experts(inventory, budget, {{3, 1}, {3, 1}}); });
    refuses([&] { plan_experts(inventory, budget, {{3, 9}}); });
    auto small = budget;
    small.device_total = 255;  // below two physical staging slots: fail before an allocation
    refuses([&] { plan_experts(inventory, small); });
    small = budget;
    small.file_cache_allowance = small.host_total;
    refuses([&] { plan_experts(inventory, small); });
    small = budget;
    small.tile_rows = 0;
    refuses([&] { plan_experts(inventory, small); });
    small = budget;
    small.transfer_slots = UINT64_MAX;
    refuses([&] { plan_experts(inventory, small); });

    // A byte short of an expert cannot create an overlapping or uncharged allocation.
    --budget.device_total;
    const auto short_plan = plan_experts(inventory, budget);
    CHECK(short_plan.device_count == 1 && short_plan.host_count == 2 && short_plan.file_count == 3);
    // Exhaustive small capacities exercise fit/no-fit and arena offsets independently of the
    // default 64/96 GiB machine, while the staging bytes remain physically charged.
    for (uint64_t device = 0; device < 4800; device += 31) {
        budget.device_total = 256 + device;
        for (uint64_t host = 0; host < 4800; host += 97) {
            budget.host_total = 128 + host;
            const auto p = plan_experts(inventory, budget);
            CHECK(p.device_bytes <= device && p.host_bytes <= host);
            CHECK(p.device_count + p.host_count + p.file_count == 6);
            uint64_t device_end = 0, host_end = 0;
            for (const auto& entry : p.placements) {
                if (entry.tier == ExpertTier::device) { CHECK(entry.arena_offset == device_end); device_end += entry.allocation_bytes; }
                if (entry.tier == ExpertTier::host) { CHECK(entry.arena_offset == host_end); host_end += entry.allocation_bytes; }
            }
            CHECK(device_end == p.device_bytes && host_end == p.host_bytes);
        }
    }
}

static void real_directory(const char* path) {
    const auto inventory = inspect_glm_gguf({path});
    CHECK(inventory.trunk_layers == 6 && inventory.dense_layers == 1);
    CHECK(inventory.experts_per_layer == 12 && inventory.experts.size() == 60);
    CHECK(inventory.skipped_mtp_bytes == 0);
    uint64_t bytes = 0;
    for (const auto& e : inventory.experts) {
        CHECK(e.id.layer >= 1 && e.id.layer < 6 && e.id.expert < 12);
        CHECK(e.bytes() == 3 * 128 * 64 * 4);
        CHECK(e.matrices[0].rows == 64 && e.matrices[0].columns == 128);
        CHECK(e.matrices[2].rows == 128 && e.matrices[2].columns == 64);
        for (const auto& m : e.matrices) CHECK(m.type == 0 && m.row_bytes == m.columns * 4);
        bytes += e.bytes();
    }
    CHECK(bytes == 5898240);
    // Matrix base is offset from the file directory, not inferred by adjacency to gate.
    CHECK(inventory.experts[1].matrices[0].offset - inventory.experts[0].matrices[0].offset == 32768);
    refuses([&] { inspect_glm_gguf({path, path}); });
    refuses([] { inspect_glm_gguf({}); });
    MemoryBudget b{uint64_t{96} << 30, uint64_t{27} << 30,
                   uint64_t{64} << 30, uint64_t{16} << 30, uint64_t{8} << 30, 64, 2, 64};
    const auto plan = plan_experts(inventory, b);
    CHECK(plan.device_count == 60 && plan.host_count == 0 && plan.file_count == 0);
    CHECK(plan.host_staging_bytes <= 2 * 64 * 128 * 4);
    CHECK(plan.device_staging_bytes < 1024 * 1024);
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        budget_and_ownership();
        real_directory(argv[1]);
        std::puts("HPU packed-expert planner: directory, ownership and budget checks passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
