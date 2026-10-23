#include "strata/hpu/tile_stream.hpp"

#include <algorithm>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace strata::hpu {

size_t read_packed_rows(const PackedMatrix& matrix, uint64_t first_row, uint64_t rows,
                        std::span<uint8_t> destination) {
    if (!rows || !matrix.row_bytes || first_row > matrix.rows || rows > matrix.rows - first_row)
        throw std::invalid_argument("HPU packed tile row range is outside the matrix");
    if (rows > SIZE_MAX / matrix.row_bytes || first_row > UINT64_MAX / matrix.row_bytes)
        throw std::overflow_error("HPU packed tile size overflow");
    const auto bytes = static_cast<size_t>(rows * matrix.row_bytes);
    const auto displacement = first_row * matrix.row_bytes;
    if (displacement > UINT64_MAX - matrix.offset) throw std::overflow_error("HPU packed tile offset overflow");
    const auto offset = matrix.offset + displacement;
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("HPU packed tile exceeds the file I/O range");
    if (bytes > destination.size()) throw std::invalid_argument("HPU packed tile exceeds the fixed staging slot");
    std::ifstream file(std::filesystem::path(matrix.path), std::ios::binary);
    if (!file) throw std::runtime_error("HPU packed tile cannot open " + matrix.path);
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) throw std::runtime_error("HPU packed tile cannot seek " + matrix.path);
    file.read(reinterpret_cast<char*>(destination.data()), static_cast<std::streamsize>(bytes));
    if (file.gcount() != static_cast<std::streamsize>(bytes))
        throw std::runtime_error("HPU packed tile short read " + matrix.path);
    return bytes;
}

struct PackedTileStream::State {
    enum class Phase { free, ready, in_flight, waiting };
    struct Slot {
        std::vector<uint8_t> buffer;
        size_t length = 0;
        uint64_t generation = 0;
        Phase phase = Phase::free;
        std::function<void()> wait;
    };
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<Slot> slots;
    bool closing = false;
    uint64_t file_bytes = 0;

    Slot& checked(Ticket t, const PackedTileStream* owner) {
        if (t.owner != owner || t.slot >= slots.size()) throw std::invalid_argument("HPU tile ticket belongs to another stream");
        auto& slot = slots[t.slot];
        if (slot.generation != t.generation || slot.phase == Phase::free)
            throw std::invalid_argument("HPU stale tile ticket");
        return slot;
    }

    void wait_slot(Slot& slot, std::unique_lock<std::mutex>& lock) {
        slot.phase = Phase::waiting;
        auto wait = std::move(slot.wait);
        lock.unlock();
        try { wait(); }
        catch (...) {
            // Preserve the buffer for a retry. A failed device wait never publishes completion.
            lock.lock();
            slot.wait = std::move(wait);
            slot.phase = Phase::in_flight;
            changed.notify_all();
            throw;
        }
        lock.lock();
        slot.phase = Phase::free;
        slot.length = 0;
        changed.notify_all();
    }
};

PackedTileStream::PackedTileStream(size_t slots, size_t packed_tile_bytes) : state_(std::make_unique<State>()) {
    if (!slots || !packed_tile_bytes || slots > SIZE_MAX / packed_tile_bytes)
        throw std::invalid_argument("HPU staging requires a positive, bounded slot allocation");
    state_->slots.resize(slots);
    for (auto& slot : state_->slots) slot.buffer.resize(packed_tile_bytes);
}

PackedTileStream::~PackedTileStream() {
    // A completion callback throwing here is a runtime integration error: freeing storage
    // still referenced by DMA would be unsafe. Explicit finish() can surface an error sooner.
    try { close(); } catch (...) { std::terminate(); }
}

std::optional<PackedTileStream::Ticket> PackedTileStream::stage(const PackedMatrix& matrix,
                                                               uint64_t first_row, uint64_t rows) {
    std::lock_guard lock(state_->mutex);
    if (state_->closing) throw std::logic_error("HPU staging stream is closed");
    for (size_t i = 0; i < state_->slots.size(); ++i) {
        auto& slot = state_->slots[i];
        if (slot.phase != State::Phase::free) continue;
        if (slot.generation == UINT64_MAX) throw std::overflow_error("HPU tile generation exhausted");
        // The file read happens while the slot is unavailable to consumers; no ready state
        // or ticket is published until the exact byte count arrives.
        const auto length = read_packed_rows(matrix, first_row, rows, slot.buffer);
        if (length > UINT64_MAX - state_->file_bytes) throw std::overflow_error("HPU file byte counter overflow");
        slot.length = length;
        ++slot.generation;
        state_->file_bytes += length;
        slot.phase = State::Phase::ready;
        return Ticket{this, i, slot.generation};
    }
    return std::nullopt;
}

std::span<const uint8_t> PackedTileStream::bytes(Ticket ticket) const {
    std::lock_guard lock(state_->mutex);
    auto& slot = state_->checked(ticket, this);
    return {slot.buffer.data(), slot.length};
}

void PackedTileStream::submit(Ticket ticket, std::function<void()> wait_for_completion) {
    if (!wait_for_completion) throw std::invalid_argument("HPU transfer requires a completion wait");
    std::lock_guard lock(state_->mutex);
    auto& slot = state_->checked(ticket, this);
    if (slot.phase != State::Phase::ready) throw std::logic_error("HPU tile already submitted");
    slot.wait = std::move(wait_for_completion);
    slot.phase = State::Phase::in_flight;
}

void PackedTileStream::finish(Ticket ticket) {
    std::unique_lock lock(state_->mutex);
    auto& slot = state_->checked(ticket, this);
    if (slot.phase != State::Phase::in_flight) throw std::logic_error("HPU tile has no outstanding consumer");
    state_->wait_slot(slot, lock);
}

void PackedTileStream::discard(Ticket ticket) {
    std::lock_guard lock(state_->mutex);
    auto& slot = state_->checked(ticket, this);
    if (slot.phase != State::Phase::ready) throw std::logic_error("HPU cannot discard a submitted transfer before completion");
    slot.phase = State::Phase::free;
    slot.length = 0;
}

void PackedTileStream::close() {
    std::unique_lock lock(state_->mutex);
    state_->closing = true;
    for (;;) {
        size_t in_flight = state_->slots.size();
        bool waiting = false;
        for (size_t i = 0; i < state_->slots.size(); ++i) {
            auto& slot = state_->slots[i];
            if (slot.phase == State::Phase::ready) { slot.phase = State::Phase::free; slot.length = 0; }
            if (slot.phase == State::Phase::waiting) waiting = true;
            if (slot.phase == State::Phase::in_flight) { in_flight = i; break; }
        }
        if (in_flight != state_->slots.size()) {
            state_->wait_slot(state_->slots[in_flight], lock);
        } else if (waiting) {
            state_->changed.wait(lock);
        } else break;
    }
    for (auto& slot : state_->slots) std::vector<uint8_t>().swap(slot.buffer);
}

size_t PackedTileStream::resident_bytes() const {
    std::lock_guard lock(state_->mutex);
    size_t bytes = 0;
    for (const auto& slot : state_->slots) bytes += slot.buffer.size();
    return bytes;
}
size_t PackedTileStream::slot_capacity_bytes() const {
    std::lock_guard lock(state_->mutex);
    return state_->slots.empty() ? 0 : state_->slots.front().buffer.size();
}
size_t PackedTileStream::busy_slots() const {
    std::lock_guard lock(state_->mutex);
    return static_cast<size_t>(std::count_if(state_->slots.begin(), state_->slots.end(),
        [](const auto& slot) { return slot.phase != State::Phase::free; }));
}
uint64_t PackedTileStream::requested_file_bytes() const {
    std::lock_guard lock(state_->mutex);
    return state_->file_bytes;
}

}  // namespace strata::hpu
