#pragma once

// Tier-compiled (DEFAULT/AVX2/AVX512) walks for the losses.
// The implementations live in LossVecKernelsImpl.cpp, which is listed in
// TP_CPU_KERNEL_SRCS so each CPU capability tier compiles its own vectorized
// walk (see p10/CMakeLists.txt); the loss entry points in LossKernels.cpp and
// ExtraLossKernels.cpp stay in the base tier and hand the ready-to-run
// buffers to these stubs.
//
// Every stub walks contiguous same-dtype operands in the operand width and
// answers the accumulated value as a double; the walks that also have an
// unreduced form store the per-element or per-row answers into `out`.  dtype is
// the DType enumerator cast to int; only Float32 and Float64 are accepted.

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

// binary_cross_entropy_with_logits: w [pw t sp(-x) + (1 - t) sp(x)] with
// sp(y) = max(y, 0) + log1p(exp(-|y|)).  `weight` and `pos_weight` are null
// when the caller carries neither.
//
// The return value is the sum over all elements.  When `out` is not null the
// walk also stores every element answer there (same dtype as the operands), so
// an unreduced call reads the elements back without a second walk.
using bce_logits_rows_fn = double (*)(const void* input, const void* target,
                                      const void* weight, const void* pos_weight,
                                      void* out, int64_t n, int dtype);

// multilabel_soft_margin_loss: row r answers
// -[sum_c t log sigmoid(x) + (1 - t) log sigmoid(-x)] / width, where width is
// the row length.  The return value is the sum over rows; when `out` is not
// null the walk also stores every row answer there (same dtype as the operands).
using multilabel_rows_fn = double (*)(const void* input, const void* target,
                                      void* out, int64_t rows, int64_t width,
                                      int dtype);

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
DECLARE_DISPATCH(bce_logits_rows_fn, bce_logits_rows_stub)
DECLARE_DISPATCH(multilabel_rows_fn, multilabel_rows_stub)
DECLARE_DISPATCH(triplet_rows_fn, triplet_rows_stub)

} // namespace cpu
} // namespace tensorplay