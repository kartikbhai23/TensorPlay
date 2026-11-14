#pragma once

// Shared helpers for the fused mean-reduced losses: a deterministic parallel
// fold, plus the operand tests that decide when a loss can take its joined
// single-pass form.

#include "Tensor.h"
#include "Parallel.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {

// Folds a body over [0, n) into one double answer.  work_per_unit is how many
// element operations one unit costs (1 for a plain elementwise walk, the
// reduced length for a row-wise walk), and it sets the chunk size so a walk
// with few but heavy units still spreads across the team.  Each range keeps
// its own partial sum, so the fold order is the same however the team splits
// the work; the per-range answers are added in range order afterwards.
template <typename Body>
double parallel_sum(int64_t n, int64_t work_per_unit, const Body& body) {
    if (n <= 0) return 0.0;
    const int64_t unit = std::max<int64_t>(1, work_per_unit);
    // Waking the team costs more than a short walk, so small surfaces stay on
    // the calling thread.
    if (n <= parallel::GRAIN_SIZE / unit) {
        double total = 0.0;
        for (int64_t i = 0; i < n; ++i) total += body(i);
        return total;
    }
    const int64_t team = std::max<int64_t>(1, parallel::get_num_threads());
    const int64_t target = std::max<int64_t>(1, (parallel::GRAIN_SIZE / 4) / unit);
    const int64_t chunks = std::min<int64_t>(
        std::min<int64_t>(team * 4, (n + target - 1) / target), n);
    const int64_t chunk = (n + chunks - 1) / chunks;
    const int64_t count = (n + chunk - 1) / chunk;
    std::vector<double> partial(static_cast<size_t>(count), 0.0);
    parallel::parallel_for(0, count, 1, [&](int64_t first, int64_t last) {
        for (int64_t k = first; k < last; ++k) {
            const int64_t begin = k * chunk;
            const int64_t end = std::min(n, begin + chunk);
            double local = 0.0;
            for (int64_t i = begin; i < end; ++i) local += body(i);
            partial[static_cast<size_t>(k)] = local;
        }
    });
    double total = 0.0;
    for (const double value : partial) total += value;
    return total;
}

// True when both operands span the same shape in the same memory order.
inline bool same_layout(const Tensor& a, const Tensor& b) {
    return a.is_contiguous() && b.is_contiguous() &&
           static_cast<std::vector<int64_t>>(a.sizes()) ==
               static_cast<std::vector<int64_t>>(b.sizes());
}

// True when a joined pass can read both operands as plain float values.
inline bool fused_float(const Tensor& a, const Tensor& b) {
    return a.dtype() == b.dtype() && same_layout(a, b) &&
           (a.dtype() == DType::Float32 || a.dtype() == DType::Float64);
}

} // namespace cpu
} // namespace tensorplay