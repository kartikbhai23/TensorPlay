// Loss operators use device-side elementwise loops and reduction kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "ErrorReporting.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

// Loss carry type: half/float accumulate in float, everything else in double.
inline DType loss_carry_dtype(DType t) {
    return (t == DType::Float32 || t == DType::Float16 || t == DType::BFloat16)
        ? DType::Float32
        : DType::Float64;
}

std::pair<Tensor, Tensor> pair_carry_dev(const Tensor& a, const Tensor& b,
                                         DType acc) {
    std::vector<int64_t> shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(acc),
            b.expand(shape).contiguous().to(acc)};
}

Tensor expand_carry_dev(const Tensor& a, const std::vector<int64_t>& shape,
                        DType acc) {
    return a.expand(shape).contiguous().to(acc);
}

void tp_validate_reduction(int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
}

Tensor tp_loss_reduce(const Tensor& loss, int64_t reduction) {
    tp_validate_reduction(reduction);
    if (reduction == 0) return loss;
    if (reduction == 1) return loss.mean();
    return loss.sum();
}

Tensor tp_scale_grad(const Tensor& grad, int64_t reduction, int64_t numel) {
    tp_validate_reduction(reduction);
    if (reduction == 1) return grad / static_cast<double>(numel);
    return grad;
}

// ---------------------------------------------------------------------------
// elementwise kernels (one thread per pair element)
// ---------------------------------------------------------------------------

__host__ __device__ inline double dsp(double y) {
    return ::fmax(y, 0.0) + ::log1p(::exp(-::fabs(y)));
}

__host__ __device__ inline float dspf(float y) {
    return ::fmaxf(y, 0.0f) + ::log1pf(::expf(-::fabsf(y)));
}

} // anonymous namespace

// ===========================================================================
// Public entry points
// ===========================================================================

Tensor binary_cross_entropy_with_logits_cuda(const Tensor& self, const Tensor& target,
                                             const std::optional<Tensor>& weight_opt,
                                             const std::optional<Tensor>& pos_weight_opt,
                                             int64_t reduction) {
    tp_validate_reduction(reduction);
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    const DType acc = loss_carry_dtype(self.dtype());
    auto pr = pair_carry_dev(self, target, acc);
    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    Tensor w = has_w ? expand_carry_dev(weight, shape_of(pr.first), acc)
                     : pr.first;  // dummy pointer when absent
    Tensor pw = has_pw ? expand_carry_dev(pos_weight, shape_of(pr.first), acc)
                       : pr.first;
    Tensor elems = Tensor::empty(shape_of(pr.first), acc, self.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .add_const_input(w)
            .add_const_input(pw)
            .build();
        if (acc == DType::Float64) {
            gpu_kernel(iter, [has_w, has_pw] __host__ __device__(
                double x, double t, double wv, double pwv) -> double {
                const double wi = has_w ? wv : 1.0;
                const double pi = has_pw ? pwv : 1.0;
                return wi * (pi * t * dsp(-x) + (1.0 - t) * dsp(x));
            });
        } else {
            gpu_kernel(iter, [has_w, has_pw] __host__ __device__(
                float x, float t, float wv, float pwv) -> float {
                const float wi = has_w ? wv : 1.0f;
                const float pi = has_pw ? pwv : 1.0f;
                return wi * (pi * t * dspf(-x) + (1.0f - t) * dspf(x));
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const DType out_dtype =
        self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    if (elems.dtype() != out_dtype) elems = elems.to(out_dtype);
    if (reduction == 0) return elems;
    // A mean over no elements carries the 0/0 it divides to.
    return reduction == 1 ? elems.mean() : elems.sum();
}

Tensor triplet_margin_loss_cuda(const Tensor& anchor, const Tensor& positive,
                                const Tensor& negative, double margin, double p,
                                double eps, bool swap, int64_t reduction) {
    tp_validate_reduction(reduction);
    const int64_t a_dim = anchor.dim();
    const int64_t p_dim = positive.dim();
    const int64_t n_dim = negative.dim();
    if (a_dim != p_dim || p_dim != n_dim) {
        TP_THROW(RuntimeError,
                 "The anchor, positive, and negative tensors are expected to "
                 "have the same number of dimensions, but got: anchor ", a_dim,
                 "D, positive ", p_dim, "D, and negative ", n_dim, "D inputs");
    }
    Tensor dist_pos = ops::pairwise_distance(anchor, positive, p, eps, false);
    Tensor dist_neg = ops::pairwise_distance(anchor, negative, p, eps, false);
    // The shorter positive-to-negative distance replaces the anchor-to-negative
    // one when the pair is asked to swap, so a positive sample that drifted
    // toward the negative one cannot shorten the margin for free.
    if (swap) {
        dist_neg = Tensor::minimum(
            dist_neg, ops::pairwise_distance(positive, negative, p, eps, false));
    }
    Tensor raw = dist_pos - dist_neg + margin;
    Tensor loss = Tensor::where(raw.lt(Scalar(0)), Scalar(0), raw);
    // The distances blend in floating literals, so a whole-number operand is
    // answered at single precision.
    const DType out_dtype =
        anchor.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return tp_loss_reduce(loss.to(out_dtype), reduction);
}

Tensor multilabel_soft_margin_loss_cuda(const Tensor& input, const Tensor& target,
                                         int64_t reduction) {
    tp_validate_reduction(reduction);
    Tensor positive = target * (-ops::log_sigmoid(input));
    Tensor negative = (Scalar(1) - target) * (-ops::log_sigmoid(-input));
    const int64_t class_dim = input.dim() - 1;
    Tensor row_loss = ops::mean(positive + negative, {class_dim}, false);
    const DType out_dtype =
        input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    if (row_loss.dtype() != out_dtype) row_loss = row_loss.to(out_dtype);
    if (reduction == 0) return row_loss;
    // The answer over an empty batch averages to nothing.
    return reduction == 1 ? row_loss.mean() : row_loss.sum();
}

// The gradient walks answer in the width the forward carried: half and single
// precision planes are read and written through single precision, and double
// planes stay double.
Tensor binary_cross_entropy_with_logits_backward_cuda(
    const Tensor& grad_output, const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  tp_validate_reduction(reduction);
  const bool has_w =
      weight_opt.has_value() && weight_opt->defined() && weight_opt->numel() > 0;
  const bool has_pw = pos_weight_opt.has_value() &&
                      pos_weight_opt->defined() &&
                      pos_weight_opt->numel() > 0;
  const DType carry = loss_carry_dtype(self.dtype());
  const Tensor x = ops::to(self, carry);
  const Tensor t = ops::to(target, carry);
  const Tensor sig = ops::sigmoid(x);
  Tensor grad_input;
  if (has_pw) {
    const Tensor pos_weight = ops::to(*pos_weight_opt, carry);
    const Tensor pos_t = ops::mul(pos_weight, t);
    grad_input = ops::sub(
        ops::mul(ops::add(ops::sub(pos_t, t), Scalar(1.0)), sig), pos_t);
  } else {
    grad_input = ops::sub(sig, t);
  }
  grad_input = ops::mul(grad_input, ops::to(grad_output, carry));
  if (has_w) grad_input = ops::mul(grad_input, ops::to(*weight_opt, carry));
  if (reduction == 1) {
    grad_input =
        ops::div(grad_input, Scalar(static_cast<double>(self.numel())));
  }
  return ops::to(grad_input, self.dtype());
}

Tensor binary_cross_entropy_with_logits_target_backward_cuda(
    const Tensor& grad_output, const Tensor& self, const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& pos_weight_opt, int64_t reduction) {
  tp_validate_reduction(reduction);
  const bool has_w =
      weight_opt.has_value() && weight_opt->defined() && weight_opt->numel() > 0;
  const bool has_pw = pos_weight_opt.has_value() &&
                      pos_weight_opt->defined() &&
                      pos_weight_opt->numel() > 0;
  const DType carry = loss_carry_dtype(target.dtype());
  const Tensor x = ops::to(self, carry);
  Tensor grad_target;
  if (has_pw) {
    const Tensor pos_weight = ops::to(*pos_weight_opt, carry);
    grad_target = ops::sub(ops::log_sigmoid(ops::neg(x)),
                           ops::mul(ops::log_sigmoid(x), pos_weight));
  } else {
    grad_target = ops::neg(x);
  }
  grad_target = ops::mul(grad_target, ops::to(grad_output, carry));
  if (has_w) grad_target = ops::mul(grad_target, ops::to(*weight_opt, carry));
  if (reduction == 1) {
    grad_target =
        ops::div(grad_target, Scalar(static_cast<double>(target.numel())));
  }
  return ops::to(grad_target, target.dtype());
}

Tensor multilabel_soft_margin_loss_backward_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    int64_t reduction) {
  tp_validate_reduction(reduction);
  const DType carry = loss_carry_dtype(input.dtype());
  const int64_t C = input.dim() > 0 ? input.size(-1) : 0;
  const int64_t rows = C > 0 ? input.numel() / C : 0;
  if (C <= 0) return ops::zeros_like(input);
  Tensor g = ops::div(ops::sub(ops::sigmoid(ops::to(input, carry)),
                               ops::to(target, carry)),
                      Scalar(static_cast<double>(C)));
  const Tensor grad = ops::to(grad_output, carry);
  if (reduction == 0) {
    std::vector<int64_t> shape = shape_of(grad);
    shape.push_back(1);
    g = ops::mul(g, grad.reshape(shape));
  } else {
    g = ops::mul(g, grad);
    if (reduction == 1 && rows > 0) {
      g = ops::div(g, Scalar(static_cast<double>(rows)));
    }
  }
  return ops::to(g, input.dtype());
}

Tensor multilabel_soft_margin_loss_target_backward_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& target,
    int64_t reduction) {
  tp_validate_reduction(reduction);
  const DType carry = loss_carry_dtype(target.dtype());
  const int64_t C = input.dim() > 0 ? input.size(-1) : 0;
  const int64_t rows = C > 0 ? input.numel() / C : 0;
  if (C <= 0) return ops::zeros_like(target);
  const Tensor x = ops::to(input, carry);
  Tensor g = ops::div(ops::sub(ops::log_sigmoid(ops::neg(x)),
                               ops::log_sigmoid(x)),
                      Scalar(static_cast<double>(C)));
  const Tensor grad = ops::to(grad_output, carry);
  if (reduction == 0) {
    std::vector<int64_t> shape = shape_of(grad);
    shape.push_back(1);
    g = ops::mul(g, grad.reshape(shape));
  } else {
    g = ops::mul(g, grad);
    if (reduction == 1 && rows > 0) {
      g = ops::div(g, Scalar(static_cast<double>(rows)));
    }
  }
  return ops::to(g, target.dtype());
}

Tensor tp_l1_loss_cuda(const Tensor& input, const Tensor& target,
                       int64_t reduction) {
    tp_validate_reduction(reduction);
    const DType acc = loss_carry_dtype(input.dtype());
    auto pr = pair_carry_dev(input, target, acc);
    Tensor elems = Tensor::empty(shape_of(pr.first), acc, input.device());
    const int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        if (acc == DType::Float64) {
            gpu_kernel(iter, [] __host__ __device__(double a, double b) -> double {
                return ::fabs(a - b);
            });
        } else {
            gpu_kernel(iter, [] __host__ __device__(float a, float b) -> float {
                return ::fabs(a - b);
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input.dtype());
    return tp_loss_reduce(elems, reduction);
}

Tensor tp_l1_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                const Tensor& target, int64_t reduction) {
    tp_validate_reduction(reduction);
    const DType acc = loss_carry_dtype(input.dtype());
    auto pr = pair_carry_dev(input, target, acc);
    // The kernels read grad linearly: materialize its broadcast.
    const Tensor g = (grad_output.shape() == pr.first.shape()
                          ? grad_output
                          : grad_output.expand(shape_of(pr.first)))
        .contiguous()
        .to(acc);
    Tensor out = Tensor::empty(shape_of(pr.first), acc, input.device());
    const int64_t n = out.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(out)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .add_const_input(g)
            .build();
        if (acc == DType::Float64) {
            const double norm =
                reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
            gpu_kernel(iter, [norm] __host__ __device__(
                double a, double b, double gv) -> double {
                const double d = a - b;
                const double s = d > 0.0 ? 1.0 : (d < 0.0 ? -1.0 : 0.0);
                return norm * s * gv;
            });
        } else {
            const float norm =
                reduction == 1 ? 1.0f / static_cast<float>(n) : 1.0f;
            gpu_kernel(iter, [norm] __host__ __device__(
                float a, float b, float gv) -> float {
                const float d = a - b;
                const float s = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
                return norm * s * gv;
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return out.to(input.dtype());
}

Tensor tp_kl_div_cuda(const Tensor& input, const Tensor& target,
                      int64_t reduction, bool log_target) {
    // One fused elementwise pass feeds the shared reduction helper.  The
    // non-log branch short-circuits on target == 0 so log(0) never enters
    // the arithmetic.
    Tensor loss = Tensor::empty(shape_of(input), input.dtype(), input.device());
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(loss)
        .add_const_input(input)
        .add_const_input(target)
        .build();
    if (log_target) {
        if (input.dtype() == DType::Float64) {
            gpu_kernel(iter, [] __host__ __device__(double i, double t) -> double {
                return ::exp(t) * (t - i);
            });
        } else {
            gpu_kernel(iter, [] __host__ __device__(float i, float t) -> float {
                return ::expf(t) * (t - i);
            });
        }
    } else {
        if (input.dtype() == DType::Float64) {
            gpu_kernel(iter, [] __host__ __device__(double i, double t) -> double {
                return t > 0 ? t * (::log(t) - i) : 0.0;
            });
        } else {
            gpu_kernel(iter, [] __host__ __device__(float i, float t) -> float {
                return t > 0 ? t * (::logf(t) - i) : 0.0f;
            });
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return tp_loss_reduce(loss, reduction);
}

Tensor tp_kl_div_backward_cuda(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction,
                               bool log_target) {
    Tensor grad;
    if (log_target) {
        grad = -target.exp() * grad_output;
    } else {
        Tensor nonzero = target.ne(Scalar(0)).to(input.dtype());
        grad = (-target * nonzero) * grad_output;
    }
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_margin_ranking_loss_cuda(const Tensor& input1, const Tensor& input2,
                                   const Tensor& target, double margin,
                                   int64_t reduction) {
    tp_validate_reduction(reduction);
    const DType acc = loss_carry_dtype(input1.dtype());
    // The difference, the target weight, the margin offset and the clamp all
    // land in one traversal: the staged form would write and re-read a full
    // elementwise tensor for every term.
    const std::vector<int64_t> shape = broadcast_shapes(
        broadcast_shapes(shape_of(input1), shape_of(input2)), shape_of(target));
    Tensor a = expand_carry_dev(input1, shape, acc);
    Tensor b = expand_carry_dev(input2, shape, acc);
    Tensor t = expand_carry_dev(target, shape, acc);
    Tensor elems = Tensor::empty(shape, acc, input1.device());
    const int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(a)
            .add_const_input(b)
            .add_const_input(t)
            .build();
        // The clamp keeps the raw value when it is not negative, so a NaN
        // passes through and a negative zero survives, as the minimum bound
        // of a value rejects neither.
        if (acc == DType::Float64) {
            gpu_kernel(iter, [margin] __host__ __device__(
                double x, double y, double tv) -> double {
                const double v = -tv * (x - y) + margin;
                return v < 0.0 ? 0.0 : v;
            });
        } else {
            const float m = static_cast<float>(margin);
            gpu_kernel(iter, [m] __host__ __device__(
                float x, float y, float tv) -> float {
                const float v = -tv * (x - y) + m;
                return v < 0.0f ? 0.0f : v;
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input1.dtype());
    return tp_loss_reduce(elems, reduction);
}

std::tuple<Tensor, Tensor> tp_margin_ranking_loss_backward_cuda(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    tp_validate_reduction(reduction);
    Tensor raw = -(input1 - input2) * target + margin;
    Tensor active = raw.gt(Scalar(0)).to(input1.dtype());
    Tensor grad = -active * target * grad_output;
    grad = tp_scale_grad(grad, reduction, input1.numel());
    return std::make_tuple(grad, -grad);
}

Tensor tp_hinge_embedding_loss_cuda(const Tensor& input, const Tensor& target,
                                    double margin, int64_t reduction) {
    Tensor zeros = Tensor::zeros_like(input);
    Tensor margin_diff = margin - input;
    Tensor margin_part = Tensor::where(
        target.ne(Scalar(1)),
        Tensor::where(margin_diff.lt(Scalar(0)), Scalar(0), margin_diff),
        zeros);
    Tensor self_part = Tensor::where(target.ne(Scalar(-1)), input, zeros);
    return tp_loss_reduce(margin_part + self_part, reduction);
}

Tensor tp_hinge_embedding_loss_backward_cuda(const Tensor& grad_output,
                                             const Tensor& input,
                                             const Tensor& target, double margin,
                                             int64_t reduction) {
    Tensor ones = Tensor::ones_like(input);
    Tensor active = (margin - input).gt(Scalar(0)).to(input.dtype());
    Tensor grad = Tensor::where(
        target.eq(Scalar(1)), ones,
        Tensor::where(target.eq(Scalar(-1)), -active, ones - active));
    grad = grad * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_soft_margin_loss_cuda(const Tensor& input, const Tensor& target,
                                int64_t reduction) {
    tp_validate_reduction(reduction);
    const DType acc = loss_carry_dtype(input.dtype());
    // One pass forms log1p(exp(-x * t)): the staged form allocates and
    // traverses a full elementwise tensor per term, and the log of one plus a
    // sum loses the small-argument digits that log1p keeps.
    auto pr = pair_carry_dev(input, target, acc);
    Tensor elems = Tensor::empty(shape_of(pr.first), acc, input.device());
    const int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        if (acc == DType::Float64) {
            gpu_kernel(iter, [] __host__ __device__(double x, double t) -> double {
                return ::log1p(::exp(-x * t));
            });
        } else {
            gpu_kernel(iter, [] __host__ __device__(float x, float t) -> float {
                return ::log1pf(::expf(-x * t));
            });
        }
        CUDA_CHECK(cudaGetLastError());
    }
    if (reduction == 0) return elems.to(input.dtype());
    return tp_loss_reduce(elems, reduction);
}

Tensor tp_soft_margin_loss_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         int64_t reduction) {
    // d/dx log(1 + exp(-t x)) = -t * sigmoid(-t x)
    Tensor s = ((input * target) * Scalar(-1)).sigmoid();
    Tensor grad = -target * s * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_cosine_embedding_loss_cuda(const Tensor& input1, const Tensor& input2,
                                     const Tensor& target, double margin,
                                     int64_t reduction) {
    tp_validate_reduction(reduction);
    const int64_t targ_dim = target.dim();
    if (targ_dim != 1 && targ_dim != 0) {
        TP_THROW(RuntimeError,
                 "0D or 1D target tensor expected, multi-target not supported");
    }
    if (targ_dim == 1 && (input1.dim() != 2 || input2.dim() != 2)) {
        TP_THROW(RuntimeError,
                 "1D target tensor expects 2D input tensors, but found inputs "
                 "with sizes ", format_sizes(shape_of(input1)), " and ",
                 format_sizes(shape_of(input2)), ".");
    }
    if (targ_dim == 0 && (input1.dim() != 1 || input2.dim() != 1)) {
        TP_THROW(RuntimeError,
                 "0D target tensor expects 1D input tensors, but found inputs "
                 "with sizes ", format_sizes(shape_of(input1)), " and ",
                 format_sizes(shape_of(input2)), ".");
    }
    const std::vector<int64_t> reduce_dims{targ_dim};
    Tensor prod_sum = (input1 * input2).sum(reduce_dims);
    Tensor mag1 = (input1 * input1).sum(reduce_dims) + 1e-12;
    Tensor mag2 = (input2 * input2).sum(reduce_dims) + 1e-12;
    // The product of the two magnitudes is a temporary nobody else holds, so
    // its root lands on it rather than on a fresh buffer.
    Tensor denom = (mag1 * mag2).sqrt_();
    Tensor cosine = prod_sum / denom;
    Tensor zeros = Tensor::zeros_like(cosine);
    // The two target cases are disjoint, so their contributions add: the
    // positive side carries the plain distance, the negative side the margin
    // one, and each is zero where its own target bit is clear.
    Tensor pos = Scalar(1) - cosine;
    Tensor neg = (cosine - margin).clamp_min_(Scalar(0));
    Tensor loss = Tensor::where(target.eq(Scalar(1)), pos, zeros) +
                  Tensor::where(target.eq(Scalar(-1)), neg, zeros);
    return tp_loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> tp_cosine_embedding_loss_backward_cuda(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    tp_validate_reduction(reduction);
    const int64_t targ_dim = target.dim();
    const std::vector<int64_t> reduce_dims{targ_dim};
    Tensor n1 = (input1 * input1).sum(reduce_dims) + 1e-12;
    Tensor n2 = (input2 * input2).sum(reduce_dims) + 1e-12;
    Tensor denom = (n1 * n2).sqrt_();
    Tensor cosine = (input1 * input2).sum(reduce_dims) / denom;

    const bool per_row = targ_dim == 1;
    Tensor ones = Tensor::ones(per_row ? std::vector<int64_t>{input1.size(0)}
                                       : std::vector<int64_t>{},
                               input1.dtype(), input1.device());
    Tensor dl_dcos = Tensor::where(
        target.eq(Scalar(1)), -ones,
        Tensor::where(target.eq(Scalar(-1)),
                      (cosine - margin).gt(Scalar(0)).to(input1.dtype()),
                      Tensor::zeros_like(ones)));
    if (reduction == 1) {
        dl_dcos = dl_dcos / static_cast<double>(ones.numel());
    }

    // With a per-row target the cosine and the magnitudes are columns, so they
    // stretch along the embedding axis; a single target leaves them scalars.
    auto as_column = [per_row](const Tensor& t) {
        return per_row ? t.unsqueeze(1) : t;
    };
    Tensor cosine_col = as_column(cosine);
    Tensor denom_col = as_column(denom);
    Tensor grad1 = (input2 / denom_col) -
                   cosine_col * (input1 / as_column(n1));
    Tensor grad2 = (input1 / denom_col) -
                   cosine_col * (input2 / as_column(n2));
    Tensor multiplier = as_column(dl_dcos * grad_output);
    return std::make_tuple(grad1 * multiplier, grad2 * multiplier);
}

Tensor tp_poisson_nll_loss_cuda(const Tensor& input, const Tensor& target,
                                bool log_input, bool full, double eps,
                                int64_t reduction) {
    Tensor loss = log_input ? input.exp() - target * input
                            : input - target * (input + eps).log();
    if (full) {
        Tensor active = target.gt(Scalar(1));
        Tensor safe_target = Tensor::where(active, target,
                                           Tensor::ones_like(target));
        Tensor stirling = target * safe_target.log() - target +
                          (safe_target * (2.0 * M_PI)).log() * 0.5;
        loss = loss + Tensor::where(active, stirling,
                                    Tensor::zeros_like(stirling));
    }
    return tp_loss_reduce(loss, reduction);
}

Tensor tp_poisson_nll_loss_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target, bool log_input,
                                         bool full, double eps,
                                         int64_t reduction) {
    static_cast<void>(full);
    Tensor grad = log_input ? input.exp() - target
                            : Scalar(1) - target / (input + eps);
    grad = grad * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LossKernels) {
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cuda);
    m.impl("binary_cross_entropy_with_logits_backward",
           binary_cross_entropy_with_logits_backward_cuda);
    m.impl("binary_cross_entropy_with_logits_target_backward",
           binary_cross_entropy_with_logits_target_backward_cuda);
    m.impl("triplet_margin_loss", triplet_margin_loss_cuda);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cuda);
    m.impl("multilabel_soft_margin_loss_backward",
           multilabel_soft_margin_loss_backward_cuda);
    m.impl("multilabel_soft_margin_loss_target_backward",
           multilabel_soft_margin_loss_target_backward_cuda);
    m.impl("tp_l1_loss", tp_l1_loss_cuda);
    m.impl("tp_l1_loss_backward", tp_l1_loss_backward_cuda);
    m.impl("tp_kl_div", tp_kl_div_cuda);
    m.impl("tp_kl_div_backward", tp_kl_div_backward_cuda);
    m.impl("tp_margin_ranking_loss", tp_margin_ranking_loss_cuda);
    m.impl("tp_margin_ranking_loss_backward", tp_margin_ranking_loss_backward_cuda);
    m.impl("tp_hinge_embedding_loss", tp_hinge_embedding_loss_cuda);
    m.impl("tp_hinge_embedding_loss_backward", tp_hinge_embedding_loss_backward_cuda);
    m.impl("tp_soft_margin_loss", tp_soft_margin_loss_cuda);
    m.impl("tp_soft_margin_loss_backward", tp_soft_margin_loss_backward_cuda);
    m.impl("tp_cosine_embedding_loss", tp_cosine_embedding_loss_cuda);
    m.impl("tp_cosine_embedding_loss_backward", tp_cosine_embedding_loss_backward_cuda);
    m.impl("tp_poisson_nll_loss", tp_poisson_nll_loss_cuda);
    m.impl("tp_poisson_nll_loss_backward", tp_poisson_nll_loss_backward_cuda);

    // The plain names answer the same walk with the same reduction argument.
    m.impl("l1_loss", tp_l1_loss_cuda);
    m.impl("kl_div", tp_kl_div_cuda);
    m.impl("margin_ranking_loss", tp_margin_ranking_loss_cuda);
    m.impl("hinge_embedding_loss", tp_hinge_embedding_loss_cuda);
    m.impl("soft_margin_loss", tp_soft_margin_loss_cuda);
    m.impl("cosine_embedding_loss", tp_cosine_embedding_loss_cuda);
    m.impl("poisson_nll_loss", tp_poisson_nll_loss_cuda);
}

} // namespace cuda
} // namespace tensorplay
