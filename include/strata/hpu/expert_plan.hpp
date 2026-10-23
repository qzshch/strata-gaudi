#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hpu {

struct ExpertId {
    uint32_t layer = 0, expert = 0;
    auto operator<=>(const ExpertId&) const = default;
};

// A single expert matrix in its original GGUF representation. No BF16 materialization.
struct PackedMatrix {
    std::string path;
    uint64_t offset = 0, rows = 0, columns = 0, row_bytes = 0;
    uint32_t type = 0;
    uint64_t bytes() const;
};

struct PackedExpert {
    ExpertId id;
    std::array<PackedMatrix, 3> matrices;  // gate, up, down
    uint64_t bytes() const;
};

struct ExpertInventory {
    uint32_t trunk_layers = 0, dense_layers = 0, experts_per_layer = 0;
    uint64_t nonexpert_packed_bytes = 0, skipped_mtp_bytes = 0;
    std::vector<PackedExpert> experts;
};

// All shards must be supplied, including a metadata-only shard. Reads directories through the
// existing mmap reader, then closes each mapping; never touches the expert payload to plan it.
ExpertInventory inspect_glm_gguf(const std::vector<std::string>& paths);

enum class ExpertTier { device, host, file };
const char* tier_name(ExpertTier tier);

struct MemoryBudget {
    uint64_t device_total = 0, device_reserve = 0;
    uint64_t host_total = 0, host_reserve = 0, file_cache_allowance = 0;
    uint64_t tile_rows = 64, transfer_slots = 2, allocation_alignment = 64;
};

struct Placement {
    ExpertId id;
    ExpertTier tier = ExpertTier::file;
    uint64_t packed_bytes = 0, allocation_bytes = 0, arena_offset = 0;
};

struct ExpertPlan {
    uint64_t host_staging_bytes = 0, device_staging_bytes = 0;
    uint64_t packed_tile_bytes = 0, bf16_tile_bytes = 0;
    uint64_t device_capacity = 0, host_capacity = 0;
    uint64_t device_bytes = 0, host_bytes = 0, file_bytes = 0;
    uint64_t device_count = 0, host_count = 0, file_count = 0;
    std::vector<Placement> placements;
};

// Priority is optional and must contain unique, existing IDs. Device and host own disjoint
// compressed copies; a file owner consumes no anonymous resident bank, but its reads can use
// the OS page cache. The caller supplies the reserve for dense weights, states and workspaces.
ExpertPlan plan_experts(const ExpertInventory&, const MemoryBudget&,
                        const std::vector<ExpertId>& priority = {});

}  // namespace strata::hpu
