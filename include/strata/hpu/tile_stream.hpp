#pragma once

#include "strata/hpu/expert_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace strata::hpu {

// Reads only complete packed rows into caller-owned storage. The count returned measures
// requested file bytes, not physical SSD traffic or page-cache misses. No dequantization.
size_t read_packed_rows(const PackedMatrix&, uint64_t first_row, uint64_t rows,
                        std::span<uint8_t> destination);

class PackedTileStream {
public:
    struct Ticket {
        const PackedTileStream* owner = nullptr;
        size_t slot = 0;
        uint64_t generation = 0;
    };

    // These are pageable Host buffers. SynapseAI registration and device scratch ownership
    // belong to the future runtime adapter; this class does not advertise pinned allocation.
    PackedTileStream(size_t slots, size_t packed_tile_bytes);
    ~PackedTileStream();
    PackedTileStream(const PackedTileStream&) = delete;
    PackedTileStream& operator=(const PackedTileStream&) = delete;

    // Backpressure returns nullopt. A read failure releases its slot and publishes no ticket.
    // Reads are serialized; device execution can overlap reads into the other slots.
    std::optional<Ticket> stage(const PackedMatrix&, uint64_t first_row, uint64_t rows);
    std::span<const uint8_t> bytes(Ticket) const;

    // The runtime retains this buffer until wait_for_completion returns. The callback must
    // wait for the actual consumer/DMA fence and must not throw; merely enqueuing work is
    // insufficient. Successful completion occurs once, by finish() or close(), before
    // reuse/free. If an explicit finish reports a wait exception, storage stays in flight
    // and the callback may be retried; an exception escaping destructor close is fatal.
    // Submit before handing a ticket to another thread; bytes() views need external lifetime
    // coordination until submit. A bare ready-ticket is not an asynchronous DMA lease.
    void submit(Ticket, std::function<void()> wait_for_completion);
    // Exactly one thread claims a consumer wait. If close() already claimed it, finish()
    // rejects the duplicate; close() itself still waits and completes normally.
    void finish(Ticket);
    void discard(Ticket);  // only a ready tile; cannot cancel an in-flight transfer

    // Stops new stages, discards unpublished ready tiles, waits for submitted consumers and
    // only then releases the fixed buffers. Destructor calls close().
    void close();
    size_t resident_bytes() const;
    size_t slot_capacity_bytes() const;
    size_t busy_slots() const;
    uint64_t requested_file_bytes() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace strata::hpu
