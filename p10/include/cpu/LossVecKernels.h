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

// mse_loss: z = x - t, answered as z^2 per element.  Both walks read the two
// operands once and never build the deviation plane; the sum walk folds the
// squared deviations and the into walk stores them (same length and dtype as
// the operands).
using mse_sum_fn = double (*)(const void* input, const void* target, int64_t n,
                              int dtype);

using mse_into_fn = void (*)(const void* input, const void* target, void* out,
                             int64_t n, int dtype);

// l1_loss: the same deviation read without its sign.
using l1_sum_fn = double (*)(const void* input, const void* target, int64_t n,
                             int dtype);

using l1_into_fn = void (*)(const void* input, const void* target, void* out,
                            int64_t n, int dtype);

// The gradients the two squared and absolute losses answer: twice the
// deviation, and the side it sits on, each carried by the incoming gradient and
// by `scale` (the slope the reduction divides with, folded into the same walk).
// Both write one element plane.
using mse_grad_fn = void (*)(const void* grad_output, const void* input,
                             const void* target, void* out, int64_t n,
                             double scale, int dtype);

using l1_grad_fn = void (*)(const void* grad_output, const void* input,
                            const void* target, void* out, int64_t n,
                            double scale, int dtype);

// kl_div: t log t - t x with a zero lane contributing nothing, or, when the
// target already sits in log space, exp(t) (t - x).  The sum walk folds the
// lanes and the into walk stores them.
using kl_sum_fn = double (*)(const void* input, const void* target, int64_t n,
                             int log_target, int dtype);

using kl_into_fn = void (*)(const void* input, const void* target, void* out,
                            int64_t n, int log_target, int dtype);

// The gradient the divergence answers with respect to the input: the negated
// target when it sits in probability space (zero lanes contribute nothing), or
// the negated exponent in log space; carried by the incoming gradient and by
// the slope a mean divides with.
using kl_grad_fn = void (*)(const void* grad_output, const void* input,
                            const void* target, void* out, int64_t n,
                            int log_target, double scale, int dtype);

// huber_loss: z = |x - t|; z <= delta ? 0.5 z^2 : delta (z - 0.5 delta).
using huber_sum_fn = double (*)(const void* input, const void* target,
                                int64_t n, double delta, int dtype);

using huber_into_fn = void (*)(const void* input, const void* target, void* out,
                               int64_t n, double delta, int dtype);

// smooth_l1_loss: z = x - t; |z| < beta ? 0.5 z^2 / beta : |z| - 0.5 beta.
// The switch is strict, so a zero width (beta = 0) answers the deviation
// itself and the quadratic branch never divides.
using smooth_l1_sum_fn = double (*)(const void* input, const void* target,
                                    int64_t n, double beta, int dtype);

using smooth_l1_into_fn = void (*)(const void* input, const void* target,
                                   void* out, int64_t n, double beta,
                                   int dtype);

// The gradient's slope is z / beta inside the width and the two outer signs
// outside it, carried by the incoming gradient and by `scale`.
using smooth_l1_grad_fn = void (*)(const void* grad_output, const void* input,
                                   const void* target, void* out, int64_t n,
                                   double beta, double scale, int dtype);

// soft_margin_loss: log(1 + exp(-t x)) per element, read once per operand.
using softmargin_sum_fn = double (*)(const void* input, const void* target,
                                     int64_t n, int dtype);

using softmargin_into_fn = void (*)(const void* input, const void* target,
                                    void* out, int64_t n, int dtype);

// The gradient answers -t / (1 + exp(t x)), carried by the incoming gradient
// and by `scale`.
using softmargin_grad_fn = void (*)(const void* grad_output, const void* input,
                                    const void* target, void* out, int64_t n,
                                    double scale, int dtype);

// poisson_nll_loss: exp(x) - t x (log_input) or x - t log(x + eps), plus the
// Stirling term t log t - t + 0.5 log(2 pi t) when full and t > 1.
using poisson_sum_fn = double (*)(const void* input, const void* target,
                                  int64_t n, int log_input, int full,
                                  double eps, int dtype);

using poisson_into_fn = void (*)(const void* input, const void* target,
                                 void* out, int64_t n, int log_input,
                                 int full, double eps, int dtype);

// The gradient is exp(x) - t (log_input) or 1 - t / (x + eps), carried by the
// incoming gradient and by `scale`; the Stirling term does not depend on x.
using poisson_grad_fn = void (*)(const void* grad_output, const void* input,
                                 const void* target, void* out, int64_t n,
                                 int log_input, double eps, double scale,
                                 int dtype);

// hinge_embedding_loss: x when the target lane is 1, max(0, margin - x) when
// it is -1, and their sum for any other lane.
using hinge_sum_fn = double (*)(const void* input, const void* target, int64_t n,
                                double margin, int dtype);

using hinge_into_fn = void (*)(const void* input, const void* target, void* out,
                               int64_t n, double margin, int dtype);

// The gradient's slope is 1 for target 1, -(margin - x > 0) for target -1,
// and 1 - (margin - x > 0) otherwise, carried by the incoming gradient and by
// `scale`.
using hinge_grad_fn = void (*)(const void* grad_output, const void* input,
                               const void* target, void* out, int64_t n,
                               double margin, double scale, int dtype);

// margin_ranking_loss: max(0, margin - (x1 - x2) t) per lane.
using margin_rank_sum_fn = double (*)(const void* input1, const void* input2,
                                      const void* target, int64_t n,
                                      double margin, int dtype);

using margin_rank_into_fn = void (*)(const void* input1, const void* input2,
                                     const void* target, void* out, int64_t n,
                                     double margin, int dtype);

// The gradient answers -(active t) to the first operand and its negation to
// the second, where active is whether the margin expression stays positive.
using margin_rank_grad_fn = void (*)(const void* grad_output, const void* input1,
                                     const void* input2, const void* target,
                                     void* out1, void* out2, int64_t n,
                                     double margin, double scale, int dtype);

// cosine_embedding_loss: row r answers 1 - cos when its target is 1,
// max(0, cos - margin) when it is -1, and 0 otherwise, with cos the cosine of
// the two row vectors.  The return value is the sum over rows; when `out` is
// not null the walk also stores every row answer there.
using cosine_rows_fn = double (*)(const void* x1, const void* x2,
                                  const void* target, void* out, int64_t rows,
                                  int64_t width, double margin, int dtype);

// The gradient writes both input planes: row r carries
// (x2/d - cos x1/n1) and (x1/d - cos x2/n2) times the row's loss slope, the
// incoming gradient and `scale`.
using cosine_grad_fn = void (*)(const void* grad_output, const void* x1,
                                const void* x2, const void* target, void* out1,
                                void* out2, int64_t rows, int64_t width,
                                double margin, double scale, int dtype);

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

// nll_loss: row r answers -x[r, t] w[t], where t is the row's target class and
// rows whose target equals ignore_index contribute nothing.  The walk returns
// the (loss sum, contributing weight) pair; when `out` is not null it also
// stores every row answer there.  `weight` is null when the caller carries no
// per-class weights.
using nll_rows_fn = std::tuple<double, double> (*)(const void* input,
                                                   const void* target,
                                                   const void* weight,
                                                   void* out, int64_t rows,
                                                   int64_t cols,
                                                   int64_t ignore_index,
                                                   int has_weight, int dtype);

// The reduced mean divides by the total contributing weight; this walk sums
// just the weights (1.0 when no per-class weight is carried).
using nll_weight_total_fn = double (*)(const void* target, const void* weight,
                                       int64_t rows, int64_t ignore_index,
                                       int has_weight, int dtype);

// The backward scatters -w[t] times the incoming gradient into the target
// column of every row: per-row when reduction is none, otherwise the single
// value a reduced call leaves, divided by the total weight under mean.
using nll_grad_fn = void (*)(const void* grad_output, const void* target,
                             const void* weight, void* grad_input,
                             int64_t rows, int64_t cols, int64_t ignore_index,
                             int reduction, double total_weight, int dtype);

// multi_margin_loss: row r answers
// sum_{d != y} max(0, margin - x_y + x_d)^p * w[y] / C, with y the target
// class.  The return value is the sum over rows; when `out` is not null the
// walk also stores every row answer there.
using multi_margin_rows_fn = double (*)(const void* input, const void* target,
                                        const void* weight, void* out,
                                        int64_t rows, int64_t cols, int p,
                                        double margin, int dtype);

// The backward writes every row: column d receives the hinge derivative
// (p z^(p-1) / C, or 2 z / C when p is 2) times the row gradient and the
// target-class weight, and the target column receives the negated sum over the
// other columns.  `scale` already carries the reduction's divisor and `per_row`
// tells whether the incoming gradient has one value per row.
using multi_margin_grad_fn = void (*)(const void* grad_output, const void* input,
                                      const void* target, const void* weight,
                                      void* grad_input, int64_t rows,
                                      int64_t cols, int p, double margin,
                                      double scale, int per_row, int dtype);

// multilabel_margin_loss: row r answers
// sum_{y in targets} sum_{d not target} max(0, 1 - x_y + x_d) / C.  The walk
// writes the is_target mask (1.0 at every listed class, 0 elsewhere), returns
// the sum over rows, and stores the row answers when `out` is not null.
using multilabel_margin_rows_fn = double (*)(const void* input,
                                             const void* target, void* out,
                                             void* is_target, int64_t rows,
                                             int64_t cols, int dtype);

// The backward accumulates +g into every non-target column and -g into the
// target columns, for every (target, non-target) pair with positive margin.
// `scale` carries the reduction's divisor and `per_row` tells whether the
// incoming gradient has one value per row.
using multilabel_margin_grad_fn = void (*)(const void* grad_output,
                                           const void* input,
                                           const void* target,
                                           const void* is_target,
                                           void* grad_input, int64_t rows,
                                           int64_t cols, double scale,
                                           int per_row, int dtype);

DECLARE_DISPATCH(bce_sum_fn, bce_sum_stub)
DECLARE_DISPATCH(bce_into_fn, bce_into_stub)
DECLARE_DISPATCH(mse_sum_fn, mse_sum_stub)
DECLARE_DISPATCH(mse_into_fn, mse_into_stub)
DECLARE_DISPATCH(mse_grad_fn, mse_grad_stub)
DECLARE_DISPATCH(l1_sum_fn, l1_sum_stub)
DECLARE_DISPATCH(l1_into_fn, l1_into_stub)
DECLARE_DISPATCH(l1_grad_fn, l1_grad_stub)
DECLARE_DISPATCH(kl_sum_fn, kl_sum_stub)
DECLARE_DISPATCH(kl_into_fn, kl_into_stub)
DECLARE_DISPATCH(kl_grad_fn, kl_grad_stub)
DECLARE_DISPATCH(huber_into_fn, huber_into_stub)
DECLARE_DISPATCH(huber_sum_fn, huber_sum_stub)
DECLARE_DISPATCH(smooth_l1_sum_fn, smooth_l1_sum_stub)
DECLARE_DISPATCH(smooth_l1_into_fn, smooth_l1_into_stub)
DECLARE_DISPATCH(smooth_l1_grad_fn, smooth_l1_grad_stub)
DECLARE_DISPATCH(softmargin_sum_fn, softmargin_sum_stub)
DECLARE_DISPATCH(softmargin_into_fn, softmargin_into_stub)
DECLARE_DISPATCH(softmargin_grad_fn, softmargin_grad_stub)
DECLARE_DISPATCH(poisson_sum_fn, poisson_sum_stub)
DECLARE_DISPATCH(poisson_into_fn, poisson_into_stub)
DECLARE_DISPATCH(poisson_grad_fn, poisson_grad_stub)
DECLARE_DISPATCH(hinge_sum_fn, hinge_sum_stub)
DECLARE_DISPATCH(hinge_into_fn, hinge_into_stub)
DECLARE_DISPATCH(hinge_grad_fn, hinge_grad_stub)
DECLARE_DISPATCH(margin_rank_sum_fn, margin_rank_sum_stub)
DECLARE_DISPATCH(margin_rank_into_fn, margin_rank_into_stub)
DECLARE_DISPATCH(margin_rank_grad_fn, margin_rank_grad_stub)
DECLARE_DISPATCH(cosine_rows_fn, cosine_rows_stub)
DECLARE_DISPATCH(cosine_grad_fn, cosine_grad_stub)
DECLARE_DISPATCH(bce_logits_rows_fn, bce_logits_rows_stub)
DECLARE_DISPATCH(multilabel_rows_fn, multilabel_rows_stub)
DECLARE_DISPATCH(triplet_rows_fn, triplet_rows_stub)
DECLARE_DISPATCH(nll_rows_fn, nll_rows_stub)
DECLARE_DISPATCH(nll_weight_total_fn, nll_weight_total_stub)
DECLARE_DISPATCH(nll_grad_fn, nll_grad_stub)
DECLARE_DISPATCH(multi_margin_rows_fn, multi_margin_rows_stub)
DECLARE_DISPATCH(multi_margin_grad_fn, multi_margin_grad_stub)
DECLARE_DISPATCH(multilabel_margin_rows_fn, multilabel_margin_rows_stub)
DECLARE_DISPATCH(multilabel_margin_grad_fn, multilabel_margin_grad_stub)

} // namespace cpu
} // namespace tensorplay