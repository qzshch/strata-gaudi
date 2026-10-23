#include "strata/hpu/expert_plan.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace strata::hpu {
namespace {
uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("HPU expert byte count overflow");
    return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("HPU expert byte count overflow");
    return a * b;
}
uint64_t aligned(uint64_t value, uint64_t alignment) {
    return mul(add(value, alignment - 1) / alignment, alignment);
}
uint64_t remaining(uint64_t total, uint64_t reserve) {
    if (reserve > total) throw std::invalid_argument("HPU reserve and staging exceed the memory budget");
    return total - reserve;
}
uint64_t tensor_bytes(const TensorInfo& t) {
    int block = 0, bytes = 0;
    if (!block_geometry(t.type, block, bytes))
        throw std::invalid_argument("HPU planner: unknown GGUF byte geometry for " + t.name);
    uint64_t n = 1;
    for (const auto d : t.shape) {
        if (!d) throw std::invalid_argument("HPU planner: zero tensor dimension: " + t.name);
        n = mul(n, d);
    }
    if (t.shape[0] % static_cast<uint64_t>(block))
        throw std::invalid_argument("HPU planner: partial quantization block in " + t.name);
    return mul(n / static_cast<uint64_t>(block), static_cast<uint64_t>(bytes));
}
bool layer_name(const std::string& name, uint32_t& layer, std::string& role) {
    if (!name.starts_with("blk.")) return false;
    const auto end = name.find('.', 4);
    if (end == std::string::npos) return false;
    const auto [p, ec] = std::from_chars(name.data() + 4, name.data() + end, layer);
    if (ec != std::errc{} || p != name.data() + end)
        throw std::invalid_argument("HPU planner: malformed block name " + name);
    role = name.substr(end + 1);
    return true;
}
struct TensorRecord { TensorInfo info; std::string path; uint64_t absolute = 0, bytes = 0; };
uint64_t nonnegative_integer(const MetaValue& value, const std::string& key) {
    switch (value.type) {
    case MetaType::U8: case MetaType::U16: case MetaType::U32: case MetaType::U64:
        return value.u;
    case MetaType::I8: case MetaType::I16: case MetaType::I32: case MetaType::I64:
        if (value.u <= INT64_MAX) return value.u;
        break;
    default: break;
    }
    throw std::invalid_argument("HPU planner: nonnegative integer required: " + key);
}
}  // namespace

uint64_t PackedMatrix::bytes() const { return mul(rows, row_bytes); }
uint64_t PackedExpert::bytes() const {
    uint64_t n = 0;
    for (const auto& m : matrices) n = add(n, m.bytes());
    return n;
}

ExpertInventory inspect_glm_gguf(const std::vector<std::string>& paths) {
    if (paths.empty()) throw std::invalid_argument("HPU planner requires every GGUF shard");
    std::map<std::string, TensorRecord> tensors;
    std::map<std::string, uint64_t> config;
    std::set<std::string> seen_paths;
    std::set<uint64_t> shard_numbers;
    bool architecture_seen = false;
    const char* keys[] = {"glm5-next.block_count", "glm5-next.trunk_block_count",
                          "glm5-next.nextn_predict_layers", "glm5-next.leading_dense_block_count",
                          "glm5-next.expert_count", "glm5-next.embedding_length", "split.count", "split.tensors.count"};
    for (const auto& path : paths) {
        if (!seen_paths.insert(std::filesystem::absolute(path).lexically_normal().string()).second)
            throw std::invalid_argument("HPU planner: repeated shard path " + path);
        GgufFile file(path);
        if (const auto* number = file.get("split.no")) {
            if (!shard_numbers.insert(nonnegative_integer(*number, "split.no")).second)
                throw std::invalid_argument("HPU planner: invalid or duplicate split.no");
        }
        if (const auto* arch = file.get("general.architecture")) {
            if (arch->type != MetaType::STRING || arch->s != "glm5-next")
                throw std::invalid_argument("HPU planner requires glm5-next, shard " + path);
            architecture_seen = true;
        }
        for (const auto* key : keys) if (const auto* value = file.get(key)) {
            const auto number = nonnegative_integer(*value, key);
            const auto [it, inserted] = config.emplace(key, number);
            if (!inserted && it->second != number)
                throw std::invalid_argument(std::string("HPU planner: inconsistent shard metadata: ") + key);
        }
        for (const auto& t : file.tensors()) {
            const auto bytes = tensor_bytes(t);
            const auto absolute = add(file.data_start(), t.offset);
            if (absolute > file.file_size() || bytes > file.file_size() - absolute)
                throw std::invalid_argument("HPU planner: tensor exceeds file: " + t.name);
            if (!tensors.emplace(t.name, TensorRecord{t, path, absolute, bytes}).second)
                throw std::invalid_argument("HPU planner: duplicate tensor across shards: " + t.name);
        }
    }
    auto required = [&](const char* key) {
        const auto it = config.find(key);
        if (it == config.end()) throw std::invalid_argument(std::string("HPU planner: missing metadata ") + key);
        return it->second;
    };
    if (config.contains("split.count") && config.at("split.count") != paths.size())
        throw std::invalid_argument("HPU planner: supply all shards, including the metadata-only shard");
    if (!architecture_seen) throw std::invalid_argument("HPU planner: missing general.architecture");
    if (config.contains("split.count")) {
        if (shard_numbers.size() != paths.size()) throw std::invalid_argument("HPU planner: each shard requires split.no");
        uint64_t expected = 0;
        for (const auto number : shard_numbers)
            if (number != expected++) throw std::invalid_argument("HPU planner: incomplete split.no sequence");
    } else if (!shard_numbers.empty() || paths.size() != 1) {
        throw std::invalid_argument("HPU planner: multiple shards require split metadata");
    }
    if (config.contains("split.tensors.count") && config.at("split.tensors.count") != tensors.size())
        throw std::invalid_argument("HPU planner: tensor count disagrees with split.tensors.count");
    const auto blocks = required("glm5-next.block_count");
    const auto nextn = config.contains("glm5-next.nextn_predict_layers") ? config.at("glm5-next.nextn_predict_layers") : 0;
    if (nextn > blocks) throw std::invalid_argument("HPU planner: NextN layers exceed block_count");
    const auto trunk = config.contains("glm5-next.trunk_block_count") ? config.at("glm5-next.trunk_block_count") : blocks - nextn;
    if (config.contains("glm5-next.trunk_block_count") && blocks != trunk && blocks != add(trunk, nextn))
        throw std::invalid_argument("HPU planner: inconsistent trunk/block/NextN counts");
    const auto dense = required("glm5-next.leading_dense_block_count");
    const auto experts = required("glm5-next.expert_count");
    const auto hidden = required("glm5-next.embedding_length");
    if (!trunk || trunk > blocks || trunk > UINT32_MAX || dense >= trunk || !experts || experts > UINT32_MAX || !hidden)
        throw std::invalid_argument("HPU planner: invalid GLM geometry");
    ExpertInventory inventory;
    inventory.trunk_layers = static_cast<uint32_t>(trunk);
    inventory.dense_layers = static_cast<uint32_t>(dense);
    inventory.experts_per_layer = static_cast<uint32_t>(experts);
    const std::array<std::string, 3> roles = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
    std::map<uint32_t, std::array<const TensorRecord*, 3>> layers;
    for (const auto& [name, record] : tensors) {
        uint32_t layer = 0;
        std::string role;
        if (layer_name(name, layer, role)) {
            if (layer >= trunk) {
                if (layer >= add(trunk, nextn))
                    throw std::invalid_argument("HPU planner: block outside trunk and declared NextN range");
                inventory.skipped_mtp_bytes = add(inventory.skipped_mtp_bytes, record.bytes);
                continue;
            }
            const auto pos = std::find(roles.begin(), roles.end(), role);
            if (pos != roles.end()) {
                if (layer < dense) throw std::invalid_argument("HPU planner: routed expert in dense lead");
                layers[layer][static_cast<size_t>(pos - roles.begin())] = &record;
                continue;
            }
        }
        inventory.nonexpert_packed_bytes = add(inventory.nonexpert_packed_bytes, record.bytes);
    }
    for (uint32_t l = inventory.dense_layers; l < inventory.trunk_layers; ++l) {
        const auto it = layers.find(l);
        if (it == layers.end() || std::any_of(it->second.begin(), it->second.end(), [](auto p) { return !p; }))
            throw std::invalid_argument("HPU planner: incomplete gate/up/down tensors for layer " + std::to_string(l));
        const auto& parts = it->second;
        for (const auto* p : parts)
            if (p->info.shape.size() != 3 || p->info.shape[2] != experts)
                throw std::invalid_argument("HPU planner: expert tensor must be [columns, rows, experts]");
        const auto& gate = parts[0]->info.shape;
        const auto& up = parts[1]->info.shape;
        const auto& down = parts[2]->info.shape;
        if (gate != up || gate[0] != hidden || down[0] != gate[1] || down[1] != hidden)
            throw std::invalid_argument("HPU planner: inconsistent expert matrix dimensions");
        for (uint32_t e = 0; e < inventory.experts_per_layer; ++e) {
            PackedExpert expert;
            expert.id = {l, e};
            for (size_t r = 0; r < parts.size(); ++r) {
                const auto& p = *parts[r];
                const auto& shape = p.info.shape;
                const auto expert_bytes = p.bytes / experts;
                const auto row_bytes = expert_bytes / shape[1];
                expert.matrices[r] = {p.path, add(p.absolute, mul(e, expert_bytes)), shape[1], shape[0], row_bytes, p.info.type};
            }
            inventory.experts.push_back(std::move(expert));
        }
    }
    return inventory;
}

const char* tier_name(ExpertTier tier) {
    switch (tier) {
    case ExpertTier::device: return "device";
    case ExpertTier::host: return "host";
    case ExpertTier::file: return "file";
    }
    throw std::invalid_argument("invalid expert tier");
}

ExpertPlan plan_experts(const ExpertInventory& inventory, const MemoryBudget& budget,
                        const std::vector<ExpertId>& priority) {
    if (!budget.tile_rows || !budget.transfer_slots || !budget.allocation_alignment)
        throw std::invalid_argument("HPU tile rows, transfer slots and alignment must be positive");
    ExpertPlan plan;
    std::map<ExpertId, const PackedExpert*> lookup;
    for (const auto& e : inventory.experts) {
        if (!lookup.emplace(e.id, &e).second) throw std::invalid_argument("HPU duplicate expert ID");
        for (const auto& m : e.matrices) {
            if (!m.rows || !m.columns || !m.row_bytes) throw std::invalid_argument("HPU empty expert matrix");
            const auto rows = std::min(budget.tile_rows, m.rows);
            plan.packed_tile_bytes = std::max(plan.packed_tile_bytes, mul(rows, m.row_bytes));
            plan.bf16_tile_bytes = std::max(plan.bf16_tile_bytes, mul(mul(rows, m.columns), 2));
        }
    }
    plan.packed_tile_bytes = aligned(plan.packed_tile_bytes, budget.allocation_alignment);
    plan.bf16_tile_bytes = aligned(plan.bf16_tile_bytes, budget.allocation_alignment);
    plan.host_staging_bytes = mul(budget.transfer_slots, plan.packed_tile_bytes);
    plan.device_staging_bytes = mul(budget.transfer_slots, add(plan.packed_tile_bytes, plan.bf16_tile_bytes));
    plan.device_capacity = remaining(budget.device_total, add(budget.device_reserve, plan.device_staging_bytes));
    plan.host_capacity = remaining(budget.host_total, add(add(budget.host_reserve, budget.file_cache_allowance), plan.host_staging_bytes));
    std::vector<const PackedExpert*> ordered;
    std::set<ExpertId> seen;
    for (const auto id : priority) {
        const auto it = lookup.find(id);
        if (it == lookup.end() || !seen.insert(id).second) throw std::invalid_argument("HPU invalid or duplicate priority expert");
        ordered.push_back(it->second);
    }
    for (const auto& [id, e] : lookup) if (seen.insert(id).second) ordered.push_back(e);
    for (const auto* e : ordered) {
        const auto packed = e->bytes();
        const auto charge = aligned(packed, budget.allocation_alignment);
        Placement placement{e->id, ExpertTier::file, packed, 0, 0};
        if (charge <= plan.device_capacity - plan.device_bytes) {
            placement.tier = ExpertTier::device;
            placement.allocation_bytes = charge;
            placement.arena_offset = plan.device_bytes;
            plan.device_bytes = add(plan.device_bytes, charge);
            ++plan.device_count;
        } else if (charge <= plan.host_capacity - plan.host_bytes) {
            placement.tier = ExpertTier::host;
            placement.allocation_bytes = charge;
            placement.arena_offset = plan.host_bytes;
            plan.host_bytes = add(plan.host_bytes, charge);
            ++plan.host_count;
        } else {
            plan.file_bytes = add(plan.file_bytes, packed);
            ++plan.file_count;
        }
        plan.placements.push_back(placement);
    }
    return plan;
}
}  // namespace strata::hpu
