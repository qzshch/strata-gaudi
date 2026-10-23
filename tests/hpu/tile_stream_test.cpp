#include "strata/hpu/tile_stream.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <thread>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;
using namespace std::chrono_literals;

static void refuses(const std::function<void()>& f) {
    bool refused = false;
    try { f(); } catch (const std::exception&) { refused = true; }
    CHECK(refused);
}

static void bytes_and_lifetime(const PackedMatrix& matrix) {
    PackedTileStream stream(2, static_cast<size_t>(matrix.row_bytes * 3));
    const auto resident = stream.resident_bytes();
    const auto a = stream.stage(matrix, 0, 3).value();
    const auto b = stream.stage(matrix, 3, 2).value();
    CHECK(a.slot != b.slot && stream.busy_slots() == 2);
    CHECK(!stream.stage(matrix, 5, 1));
    std::vector<uint8_t> independent(static_cast<size_t>(matrix.row_bytes * 3));
    // Independent whole-source seek/read checks the tile's exact byte range, including offset.
    std::ifstream source(std::filesystem::path(matrix.path), std::ios::binary);
    source.seekg(static_cast<std::streamoff>(matrix.offset));
    source.read(reinterpret_cast<char*>(independent.data()), static_cast<std::streamsize>(independent.size()));
    CHECK(source.gcount() == static_cast<std::streamsize>(independent.size()));
    CHECK(std::equal(stream.bytes(a).begin(), stream.bytes(a).end(), independent.begin()));
    CHECK(stream.bytes(b).size() == matrix.row_bytes * 2);
    PackedTileStream other(1, independent.size());
    refuses([&] { other.bytes(a); });
    refuses([&] { stream.submit(a, {}); });

    std::promise<void> consumed;
    const auto complete = consumed.get_future().share();
    std::promise<void> entered;
    const auto entered_wait = entered.get_future();
    std::atomic<int> waits = 0;
    stream.submit(a, [&] { ++waits; entered.set_value(); complete.wait(); });
    refuses([&] { stream.discard(a); });
    refuses([&] { stream.submit(a, [] {}); });
    auto finishing = std::async(std::launch::async, [&] { stream.finish(a); });
    entered_wait.wait();
    // No buffer becomes reusable before the real CPU consumer's completion signal.
    const auto was_blocked = finishing.wait_for(20ms) == std::future_status::timeout;
    const auto stayed_full = !stream.stage(matrix, 5, 1);
    consumed.set_value();  // always release the temporary consumer before any test failure
    finishing.get();
    CHECK(was_blocked && stayed_full && waits == 1);
    CHECK(stream.busy_slots() == 1);
    const auto next = stream.stage(matrix, 5, 1).value();
    CHECK(next.slot == a.slot && next.generation != a.generation);
    refuses([&] { stream.finish(a); });
    refuses([&] { stream.discard(a); });
    CHECK(stream.bytes(next).size() == matrix.row_bytes);
    stream.discard(next);
    stream.discard(b);
    CHECK(stream.resident_bytes() == resident);
    CHECK(stream.requested_file_bytes() == matrix.row_bytes * 6);

    // A runtime wait reporting failure must retain the storage instead of recycling it.
    const auto retry = stream.stage(matrix, 0, 1).value();
    int attempts = 0;
    stream.submit(retry, [&] { if (++attempts == 1) throw std::runtime_error("transient wait failure"); });
    refuses([&] { stream.finish(retry); });
    CHECK(stream.busy_slots() == 1 && stream.bytes(retry).size() == matrix.row_bytes);
    stream.finish(retry);
    CHECK(attempts == 2 && stream.busy_slots() == 0);

    // Failed reads cannot publish partially filled buffers or consume a slot forever.
    auto missing = matrix;
    missing.path += ".does-not-exist";
    refuses([&] { stream.stage(missing, 0, 1); });
    auto beyond_eof = matrix;
    beyond_eof.offset = std::filesystem::file_size(matrix.path);
    refuses([&] { stream.stage(beyond_eof, 0, 1); });
    refuses([&] { stream.stage(matrix, matrix.rows, 1); });
    refuses([&] { stream.stage(matrix, 0, 4); });
    CHECK(stream.busy_slots() == 0 && stream.requested_file_bytes() == matrix.row_bytes * 7);
    stream.close();
    CHECK(stream.resident_bytes() == 0);
    refuses([&] { stream.stage(matrix, 0, 1); });
}

static void close_waits_for_consumer(const PackedMatrix& matrix) {
    auto stream = std::make_unique<PackedTileStream>(1, static_cast<size_t>(matrix.row_bytes));
    const auto ticket = stream->stage(matrix, 0, 1).value();
    std::promise<void> consumed;
    const auto complete = consumed.get_future().share();
    std::promise<void> entered;
    const auto entered_wait = entered.get_future();
    std::atomic<int> waits = 0;
    stream->submit(ticket, [&] { ++waits; entered.set_value(); complete.wait(); });
    auto closing = std::async(std::launch::async, [&] { stream->close(); });
    entered_wait.wait();
    const auto blocked = closing.wait_for(20ms) == std::future_status::timeout;
    const auto bytes_held = stream->resident_bytes();
    bool duplicate_finish_rejected = false;
    try { stream->finish(ticket); } catch (const std::logic_error&) { duplicate_finish_rejected = true; }
    consumed.set_value();
    closing.get();
    CHECK(blocked && bytes_held == matrix.row_bytes && waits == 1 && duplicate_finish_rejected);
    CHECK(stream->resident_bytes() == 0);
    stream.reset();  // closed destructor must not invoke the callback a second time
    CHECK(waits == 1);

    // Destructor also waits, so an otherwise forgotten close cannot free submitted storage.
    stream = std::make_unique<PackedTileStream>(1, static_cast<size_t>(matrix.row_bytes));
    const auto final = stream->stage(matrix, 0, 1).value();
    std::promise<void> final_consumed;
    const auto final_complete = final_consumed.get_future().share();
    std::promise<void> final_entered;
    const auto final_entered_wait = final_entered.get_future();
    stream->submit(final, [&] { final_entered.set_value(); final_complete.wait(); });
    auto destroy = std::async(std::launch::async, [&] { stream.reset(); });
    final_entered_wait.wait();
    const auto destructor_blocked = destroy.wait_for(20ms) == std::future_status::timeout;
    final_consumed.set_value();
    destroy.get();
    CHECK(destructor_blocked);
}

static void concurrent_finish_and_close(const PackedMatrix& matrix) {
    PackedTileStream stream(1, static_cast<size_t>(matrix.row_bytes));
    const auto ticket = stream.stage(matrix, 0, 1).value();
    std::promise<void> entered, consumed;
    const auto entered_wait = entered.get_future();
    const auto complete = consumed.get_future().share();
    std::atomic<int> waits = 0;
    stream.submit(ticket, [&] { ++waits; entered.set_value(); complete.wait(); });
    auto finish = std::async(std::launch::async, [&] { stream.finish(ticket); });
    entered_wait.wait();
    auto close = std::async(std::launch::async, [&] { stream.close(); });
    const auto closing_blocked = close.wait_for(20ms) == std::future_status::timeout;
    const auto bytes_held = stream.resident_bytes();
    consumed.set_value();
    finish.get();
    close.get();
    CHECK(closing_blocked && bytes_held == matrix.row_bytes && waits == 1);
    CHECK(stream.resident_bytes() == 0);
}

int main(int argc, char** argv) {
    try {
        CHECK(argc == 2);
        const auto inventory = inspect_glm_gguf({argv[1]});
        const auto& matrix = inventory.experts.at(13).matrices[1];
        bytes_and_lifetime(matrix);
        close_waits_for_consumer(matrix);
        concurrent_finish_and_close(matrix);
        std::puts("HPU packed-tile reads, fixed buffers, stale tickets and completion lifetime passed");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
