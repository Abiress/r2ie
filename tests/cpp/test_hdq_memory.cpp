// test_hdq_memory.cpp
// Minimal assertion-based unit tests for the C++ HDQMemoryManager.
// No external test framework required. Returns 0 on success, 1 on failure.
#include <cstdio>
#include <cmath>
#include <vector>
#include "hdq_memory.hpp"

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        ++g_failures;
    } else {
        std::printf("ok:   %s\n", msg);
    }
}

float norm(const std::vector<float>& v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    return std::sqrt(s);
}

}  // namespace

int main() {
    using r2ie::HDQMemoryManager;

    // 1. accretion changes state and stays finite.
    {
        HDQMemoryManager m(16, 0.95f, 5.0f);
        float before = m.active_norm();
        std::vector<float> d(16, 0.1f);
        m.apply_accretion(d);
        float after = m.active_norm();
        check(after != before, "accretion changes active norm");
        check(std::isfinite(after), "active norm is finite after accretion");
    }

    // 2. norm stays bounded over many iterations (no explosion / NaN).
    {
        HDQMemoryManager m(16, 0.95f, 5.0f);
        bool bounded = true;
        for (int i = 0; i < 1000; ++i) {
            std::vector<float> d(16);
            for (int j = 0; j < 16; ++j) d[j] = ((i * 7 + j) % 10) / 10.0f - 0.45f;
            m.apply_accretion(d);
            float n = m.active_norm();
            if (!std::isfinite(n) || n > (4.0f * 5.0f + 1e-3f)) { bounded = false; break; }
        }
        check(bounded, "norm bounded over 1000 accretions (no NaN/explosion)");
        check(m.active_norm() > 0.0f, "mass accumulated non-trivially");
    }

    // 3. swap moves state: a recognizable vector survives a swap.
    {
        HDQMemoryManager m(8, 1.0f, 5.0f);  // decay=1 so no decay
        std::vector<float> seed(8);
        for (int i = 0; i < 8; ++i) seed[i] = static_cast<float>(i);
        m.apply_accretion(seed);
        float n0 = m.active_norm();
        // One more accretion with zero delta still swaps pointer.
        m.apply_accretion(std::vector<float>(8, 0.0f));
        float n1 = m.active_norm();
        check(std::fabs(n0 - n1) < 1e-5f, "swap preserves mass norm");
    }

    if (g_failures == 0) {
        std::printf("\nAll C++ HDQ tests passed.\n");
        return 0;
    }
    std::printf("\n%d C++ HDQ test(s) FAILED.\n", g_failures);
    return 1;
}
