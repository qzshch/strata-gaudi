#include "strata/hpu/routing_reference.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#define CHECK(expression) do { if (!(expression)) throw std::runtime_error("check failed: " #expression); } while (false)
using namespace strata::hpu;

static void oracle() {
    // Independent long-double real sigmoid and all-pairs rank oracle. The tested
    // selection boundaries have margins well above binary32 rounding; exact ties
    // and saturated FP32 sigmoid values are checked separately below.
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        std::array<float, 288> logits, bias;
        std::array<long double, 288> scores, choices;
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>(static_cast<int>(i) - 144) / 32.0f;
            bias[i] = static_cast<float>((i * (13 + seed) + seed * 23) % 71) / 17.0f;
            scores[i] = 1.L / (1.L + std::exp(-static_cast<long double>(logits[i])));
            choices[i] = scores[i] + static_cast<long double>(bias[i]);
        }
        std::array<uint32_t, 8> expected{};
        long double sum = 0;
        for (uint32_t i = 0; i < logits.size(); ++i) {
            size_t rank = 0;
            for (uint32_t j = 0; j < logits.size(); ++j)
                if (choices[j] > choices[i] || (choices[j] == choices[i] && j < i)) ++rank;
            if (rank < expected.size()) { expected[rank] = i; sum += scores[i]; }
        }
        const auto actual = select_glm53_experts(logits, bias);
        long double actual_sum = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            CHECK(actual[i].expert == expected[i]);
            const auto weight = scores[expected[i]] / (sum + 1e-20L) * 2.5L;
            CHECK(std::abs(static_cast<long double>(actual[i].weight) - weight) < 2e-6L);
            actual_sum += actual[i].weight;
        }
        CHECK(std::abs(actual_sum - 2.5L) < 2e-6L);
    }
}

static void semantics() {
    std::array<float, 288> logits{}, bias{};
    auto routes = select_glm53_experts(logits, bias);
    for (size_t i = 0; i < routes.size(); ++i) {
        CHECK(routes[i].expert == i);
        CHECK(routes[i].weight == 0.3125f);
    }
    // Correction bias forces a low-probability expert into the selected set.
    // Its probability must stay small; using the corrected score as a weight is wrong.
    logits[287] = -4; bias[287] = 10;
    routes = select_glm53_experts(logits, bias);
    CHECK(routes[0].expert == 287 && routes[0].weight < 0.02f);
    for (size_t i = 1; i < routes.size(); ++i) CHECK(routes[i].expert == i - 1);
    logits.fill(-50); bias.fill(0);
    routes = select_glm53_experts(logits, bias);
    const long double tiny_score = 1.L / (1.L + std::exp(50.L));
    const long double tiny_weight = tiny_score / (8 * tiny_score + 1e-20L) * 2.5L;
    for (const auto route : routes)
        CHECK(std::abs(static_cast<long double>(route.weight) - tiny_weight) < 1e-7L);
    logits.fill(-1000); bias.fill(0);
    routes = select_glm53_experts(logits, bias);
    for (size_t i = 0; i < routes.size(); ++i)
        CHECK(routes[i].expert == i && routes[i].weight == 0);
    // At very negative inputs, the FP32 exponential saturates before division.
    logits.fill(-100);
    routes = select_glm53_experts(logits, bias);
    for (const auto route : routes) CHECK(route.weight == 0);
    logits.fill(-std::numeric_limits<float>::max());
    for (size_t i = 0; i < 8; ++i) logits[280 + i] = std::numeric_limits<float>::max();
    routes = select_glm53_experts(logits, bias);
    for (size_t i = 0; i < routes.size(); ++i)
        CHECK(routes[i].expert == 280 + i && routes[i].weight == 0.3125f);
}

template<class F> static void refuses(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid router input accepted");
}
static void invalid() {
    std::array<float, 288> logits{}, bias{};
    refuses([&] { (void)select_glm53_experts(std::span(logits).first(287), bias); });
    refuses([&] { (void)select_glm53_experts(logits, std::span(bias).first(287)); });
    for (const auto bad : {std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN()}) {
        logits[100] = bad;
        refuses([&] { (void)select_glm53_experts(logits, bias); });
        logits[100] = 0; bias[25] = bad;
        refuses([&] { (void)select_glm53_experts(logits, bias); });
        bias[25] = 0;
    }
}

int main() {
    try {
        oracle(); semantics(); invalid();
        std::puts("GLM fixed-profile FP32 router matches independent rank/weight oracle");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
