#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512) folds for the mean-reduced losses.
// The implementations live in LossVecKernelsImpl.cpp, which is listed in
// TP_CPU_KERNEL_SRCS so each CPU capability tier compiles its own vectorized
// walk (see p10/CMakeLists.txt); the loss entry points in LossKernels.cpp and
// ExtraLossKernels.cpp stay in the base tier and hand the ready-to-run
// buffers to these stubs.
//
// Every stub folds contiguous same-dtype operands in the operand width and
// answers the accumulated value as a double.  dtype is the DType enumerator
// cast to int; only Float32 and Float64 are accepted.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// binary_cross_entropy: -[w (t log x + (1 - t) log (1 - x))] with both
// logarithms floored at -100.  The bounds [0, 1] are tested on the way; the
// flags answer whether a lane of input / target was outside them.
using bce_sum_fn = double (*)(const void* input, const void* target, int64_t n,
                              int* bad_input, int* bad_target, int dtype);

// The elementwise forms write the per-element answers into `out` (same length
// and dtype as the operands) and report the same bounds flags for input and
// target as the fold above.
using bce_into_fn = void (*)(const void* input, const void* target, void* out,
                             int64_t n, int* bad_input, int* bad_target,
                             int dtype);

// huber_loss: z = |x - t|; z <= delta ? 0.5 z^2 : delta (z - 0.5 delta).
using huber_sum_fn = double (*)(const void* input, const void* target,
                                int64_t n, double delta, int dtype);

using huber_into_fn = void (*)(const void* input, const void* target, void* out,
                               int64_t n, double delta, int dtype);

// multilabel_soft_margin_loss: sum over all elements of
// t log sigmoid(x) + (1 - t) log sigmoid(-x).
using multilabel_sum_fn = double (*)(const void* input, const void* target,
                                     int64_t n, int dtype);

// triplet_margin_loss: row r answers max(0, d(a, p) - d(a, n) + margin), where
// d(u, v) is the order-p norm of (u - v + eps) along the row.  With `swap` the
// anchor-to-negative distance is taken against the positive-to-negative one,
// whichever is shorter.
//
// The return value is the sum over rows.  When `out` is not null the walk also
// stores every row answer there (same dtype as the operands), so an unreduced
// call reads the rows back without a second walk over the operands.
using triplet_rows_fn = double (*)(const void* anchor, const void* positive,
                                   const void* negative, void* out, int64_t rows,
                                   int64_t width, double margin, double p,
                                   double eps, bool swap, int dtype);

DECLARE_DISPATCH(bce_sum_fn, bce_sum_stub)
DECLARE_DISPATCH(bce_into_fn, bce_into_stub)
DECLARE_DISPATCH(huber_into_fn, huber_into_stub)
DECLARE_DISPATCH(huber_sum_fn, huber_sum_stub)
DECLARE_DISPATCH(multilabel_sum_fn, multilabel_sum_stub)
DECLARE_DISPATCH(triplet_rows_fn, triplet_rows_stub)

} // namespace cpu
} // namespace tensorplay