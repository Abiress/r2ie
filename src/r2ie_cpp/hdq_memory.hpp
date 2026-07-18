// hdq_memory.hpp
// HDQ MassState double-buffered memory manager (from RALE.docx, Loop 1).
//
// Two physical buffers (A/B). The forward pass reads from the "active" buffer
// via an atomic pointer; the Condensation Loop writes accretion into the
// "background" buffer, then swaps the pointer (lock-free). This is the docx's
// "lock-free pointer swap" realized safely for the mass buffer.
//
// Bounds: every write is decayed and hard-clamped so the mass cannot diverge.

#pragma once

#include <atomic>
#include <cstddef>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace r2ie {

struct MassState {
    std::vector<float> weights;
    explicit MassState(size_t size) : weights(size, 0.0f) {}
    void zero() { std::fill(weights.begin(), weights.end(), 0.0f); }
    float norm() const {
        float s = 0.0f;
        for (float w : weights) s += w * w;
        return std::sqrt(s);
    }
};

class HDQMemoryManager {
public:
    HDQMemoryManager(size_t size, float decay = 0.95f, float clamp = 5.0f)
        : dim_(size), decay_(decay), clamp_(clamp) {
        buffer_A_ = new MassState(size);
        buffer_B_ = new MassState(size);
        active_.store(buffer_A_, std::memory_order_release);
    }

    ~HDQMemoryManager() {
        delete buffer_A_;
        delete buffer_B_;
    }

    // Forward pass: lock-free read of the current active mass.
    const MassState* get_read_state() const {
        return active_.load(std::memory_order_acquire);
    }

    // Condensation Loop: write accretion delta into the background buffer,
    // then atomically flip the active pointer. Bounded by decay + clamp.
    void apply_accretion(const std::vector<float>& delta) {
        if (delta.size() != dim_) {
            throw std::invalid_argument("accretion delta size mismatch");
        }
        MassState* cur = active_.load(std::memory_order_relaxed);
        MassState* bg = (cur == buffer_A_) ? buffer_B_ : buffer_A_;

        // Sync background from current active, then accrete + bound.
        std::copy(cur->weights.begin(), cur->weights.end(), bg->weights.begin());
        for (size_t i = 0; i < dim_; ++i) {
            float v = bg->weights[i] * decay_ + delta[i];
            v = std::max(-clamp_, std::min(clamp_, v));
            bg->weights[i] = v;
        }
        active_.store(bg, std::memory_order_release);
    }

    size_t dim() const { return dim_; }
    float decay() const { return decay_; }
    float clamp() const { return clamp_; }

    // Zero both buffers (used between sequences / on reset).
    void reset() {
        buffer_A_->zero();
        buffer_B_->zero();
        active_.store(buffer_A_, std::memory_order_release);
    }

    // Convenience: current active norm (used by Python tests / event horizon).
    float active_norm() const { return get_read_state()->norm(); }

private:
    size_t dim_;
    float decay_;
    float clamp_;
    MassState* buffer_A_;
    MassState* buffer_B_;
    std::atomic<MassState*> active_;
};

}  // namespace r2ie
