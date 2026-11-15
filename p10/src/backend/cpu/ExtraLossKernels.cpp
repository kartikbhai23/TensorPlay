// Losses kept outside the TensorIterator family:
// binary_cross_entropy_with_logits (stable softplus form),
// triplet_margin_loss, multilabel_soft_margin_loss.
//
// multi_margin_loss / multilabel_margin_loss live in MarginLossKernels.cpp;
// ctc_loss / gaussian_nll_loss intentionally omitted (heavy dispatch
// surface) — noted for a later pass.

#include "Tensor.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "LossReduce.h"
#include "cpu/LossVecKernels.h"

#include <vector>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace tensorplay {
namespace cpu {

namespace {

inline double softplus(double y) { return std::max(y, 0.0) + std::log1p(std::exp(-std::fabs(y))); }

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline std::pair<Tensor, Tensor> bcast2(const Tensor& a, const Tensor& b) {
    auto shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(DType::Float64),
            b.expand(shape).contiguous().to(DType::Float64)};
}

inline Tensor scalar_from(double v, DType dt, const Device& dev) {
    return Tensor::full({}, Scalar(v),
                        dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
}

} // anonymous namespace

Tensor binary_cross_entropy_with_logits_cpu(const Tensor& self, const Tensor& target,
                                            const std::optional<Tensor>& weight_opt,
                                            const std::optional<Tensor>& pos_weight_opt,
                                            int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
    // Stable form: l = w*(pw*t*sp(-x) + (1-t)*sp(x)) with
    // sp(y) = max(y, 0) + log1p(exp(-|y|)).
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    const bool has_w = weight.defined() && weight.numel() > 0;
    const bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    const DType out_dt =
        self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    const Device& dev = self.device();

    // The joined walk reads both operands where they lie, so only the two
    // optional planes are widened and laid out for it.
    if (fused_float(self, target)) {
        const std::vector<int64_t> shape = shape_of(self);
        const DType dt = self.dtype();
        Tensor w_plane;
        Tensor pw_plane;
        const void* wp = nullptr;
        const void* pwp = nullptr;
        if (has_w) {
            w_plane = weight.expand(shape).contiguous().to(dt);
            wp = w_plane.data_ptr();
        }
        if (has_pw) {
            pw_plane = pos_weight.expand(shape).contiguous().to(dt);
            pwp = pw_plane.data_ptr();
        }
        const int64_t n = self.numel();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(shape, dt, dev);
            bce_logits_rows_stub(DeviceType::CPU, self.data_ptr(),
                                 target.data_ptr(), wp, pwp, out.data_ptr(), n,
                                 dtype);
            return out;
        }
        const double total =
            bce_logits_rows_stub(DeviceType::CPU, self.data_ptr(),
                                 target.data_ptr(), wp, pwp, nullptr, n, dtype);
        // A mean over no elements carries the 0/0 it divides to.
        return scalar_from(reduction == 1 ? total / static_cast<double>(n)
                                          : total,
                           out_dt, dev);
    }

    // Everything else is widened to double first, so a whole-number or
    // mixed-precision pair still answers the same number.
    std::vector<int64_t> shape = broadcast_shapes(shape_of(self), shape_of(target));
    if (has_w) shape = broadcast_shapes(shape, shape_of(weight));
    if (has_pw) shape = broadcast_shapes(shape, shape_of(pos_weight));
    Tensor x = self.expand(shape).contiguous().to(DType::Float64);
    Tensor t = target.expand(shape).contiguous().to(DType::Float64);
    Tensor w_plane =
        has_w ? weight.expand(shape).contiguous().to(DType::Float64) : Tensor();
    Tensor pw_plane = has_pw
                          ? pos_weight.expand(shape).contiguous().to(DType::Float64)
                          : Tensor();
    const int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    const double* wp = has_w ? w_plane.data_ptr<double>() : nullptr;
    const double* pwp = has_pw ? pw_plane.data_ptr<double>() : nullptr;
    Tensor values;
    double* vp = nullptr;
    if (reduction == 0) {
        values = Tensor(shape, DType::Float64, dev);
        vp = values.data_ptr<double>();
    }
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        const double xv = xp[i];
        const double tv = tp[i];
        const double wi = wp ? wp[i] : 1.0;
        const double pi = pwp ? pwp[i] : 1.0;
        const double value =
            wi * (pi * tv * softplus(-xv) + (1.0 - tv) * softplus(xv));
        if (vp) vp[i] = value;
        total += value;
    }
    if (reduction == 0) return values.to(out_dt);
    return scalar_from(reduction == 1 ? total / static_cast<double>(n) : total,
                       out_dt, dev);
}

// Distances along the last axis for one row, read from the operand buffers.
template <typename scalar_t>
inline double row_distance(const scalar_t* u, const scalar_t* v, int64_t D,
                           double p, double eps) {
    if (p == 0.0) {
        double count = 0;
        for (int64_t j = 0; j < D; ++j) {
            if (static_cast<double>(u[j]) - static_cast<double>(v[j]) + eps != 0.0)
                count += 1.0;
        }
        return count;
    }
    if (std::isinf(p)) {
        double best = p > 0 ? 0.0 : std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < D; ++j) {
            const double d =
                static_cast<double>(u[j]) - static_cast<double>(v[j]) + eps;
            best = p > 0 ? std::max(best, std::fabs(d))
                         : std::min(best, std::fabs(d));
        }
        return best;
    }
    double s = 0;
    for (int64_t j = 0; j < D; ++j) {
        const double d =
            static_cast<double>(u[j]) - static_cast<double>(v[j]) + eps;
        s += p == 2.0 ? d * d : std::pow(std::fabs(d), p);
    }
    return p == 2.0 ? std::sqrt(s) : std::pow(s, 1.0 / p);
}

Tensor triplet_margin_loss_cpu(const Tensor& anchor, const Tensor& positive,
                               const Tensor& negative, double margin, double p,
                               double eps, bool swap, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
    const int64_t a_dim = anchor.dim();
    const int64_t p_dim = positive.dim();
    const int64_t n_dim = negative.dim();
    if (a_dim != p_dim || p_dim != n_dim) {
        TP_THROW(RuntimeError,
                 "The anchor, positive, and negative tensors are expected to "
                 "have the same number of dimensions, but got: anchor ", a_dim,
                 "D, positive ", p_dim, "D, and negative ", n_dim, "D inputs");
    }
    // A row is the last axis of the joined operands; the leading axes index
    // positions, and the unreduced answer keeps them.
    std::vector<int64_t> shape = broadcast_shapes(
        broadcast_shapes(shape_of(anchor), shape_of(positive)),
        shape_of(negative));
    const int64_t width = shape.empty() ? 0 : shape.back();
    int64_t rows = shape.empty() ? 0 : 1;
    for (size_t k = 0; k + 1 < shape.size(); ++k) rows *= shape[k];
    std::vector<int64_t> out_shape(shape);
    if (!out_shape.empty()) out_shape.pop_back();
    // The distances blend in floating literals, so a whole-number operand is
    // answered at single precision.  Only an unreduced call needs the
    // per-position answers materialized; the folded forms keep the running sum.
    const DType out_dtype =
        anchor.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor out;
    void* rows_out = nullptr;
    if (reduction == 0) {
        out = rows > 0 ? Tensor::empty(out_shape, out_dtype, anchor.device())
                       : Tensor::zeros(out_shape, out_dtype, anchor.device());
        if (rows > 0) rows_out = out.data_ptr();
    }

    double total = 0;
    if (rows > 0 && fused_float(anchor, positive) &&
        fused_float(anchor, negative)) {
        total = triplet_rows_stub(
            DeviceType::CPU, anchor.data_ptr(), positive.data_ptr(),
            negative.data_ptr(), rows_out, rows, width, margin, p, eps, swap,
            static_cast<int>(anchor.dtype()));
    } else if (rows > 0) {
        Tensor a = anchor.expand(shape).contiguous().to(DType::Float64);
        Tensor pos = positive.expand(shape).contiguous().to(DType::Float64);
        Tensor neg = negative.expand(shape).contiguous().to(DType::Float64);
        const double* ap = a.data_ptr<double>();
        const double* apos = pos.data_ptr<double>();
        const double* aneg = neg.data_ptr<double>();
        std::vector<double> row_values(static_cast<size_t>(rows));
        for (int64_t i = 0; i < rows; ++i) {
            const double* a_row = ap + i * width;
            double dn = row_distance(a_row, aneg + i * width, width, p, eps);
            if (swap) {
                dn = std::min(dn, row_distance(apos + i * width, aneg + i * width,
                                               width, p, eps));
            }
            const double value = std::max(
                0.0, row_distance(a_row, apos + i * width, width, p, eps) - dn +
                         margin);
            row_values[static_cast<size_t>(i)] = value;
            total += value;
        }
        if (reduction == 0) {
            Tensor rows_t = Tensor::tensor(row_values, DType::Float64,
                                           anchor.device())
                                .reshape(out_shape)
                                .to(out_dtype);
            out.copy_(rows_t);
        }
    }

    if (reduction == 0) return out;
    if (reduction == 1) {
        return scalar_from(rows > 0 ? total / static_cast<double>(rows)
                                    : std::numeric_limits<double>::quiet_NaN(),
                           anchor.dtype(), anchor.device());
    }
    return scalar_from(total, anchor.dtype(), anchor.device());
}

Tensor multilabel_soft_margin_loss_cpu(const Tensor& input,
                                       const Tensor& target,
                                       int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
    const DType out_dt =
        input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    const Device& dev = input.device();
    if (fused_float(input, target) && input.dim() > 0 && input.size(-1) > 0) {
        const int64_t C = input.size(-1);
        const int64_t rows = input.numel() / C;
        const std::vector<int64_t> full = shape_of(input);
        const std::vector<int64_t> row_shape(full.begin(), full.end() - 1);
        const int dtype = static_cast<int>(input.dtype());
        if (reduction == 0) {
            Tensor out = rows > 0 ? Tensor::empty(row_shape, out_dt, dev)
                                  : Tensor::zeros(row_shape, out_dt, dev);
            if (rows > 0) {
                multilabel_rows_stub(DeviceType::CPU, input.data_ptr(),
                                     target.data_ptr(), out.data_ptr(), rows, C,
                                     dtype);
            }
            return out;
        }
        const double total =
            rows > 0 ? multilabel_rows_stub(DeviceType::CPU, input.data_ptr(),
                                            target.data_ptr(), nullptr, rows, C,
                                            dtype)
                     : 0.0;
        // The row answers are averages, so an empty batch averages to nothing.
        if (reduction == 2) return scalar_from(total, out_dt, dev);
        return scalar_from(rows > 0 ? total / static_cast<double>(rows)
                                    : std::numeric_limits<double>::quiet_NaN(),
                           out_dt, dev);
    }
    auto pr = bcast2(input, target);
    Tensor x = pr.first;
    Tensor t = pr.second;
    const std::vector<int64_t> full = shape_of(x);
    const int64_t C = x.dim() > 0 ? x.size(-1) : 0;
    const int64_t rows = C > 0 ? x.numel() / C : 0;
    std::vector<int64_t> row_shape = full;
    if (!row_shape.empty()) row_shape.pop_back();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    std::vector<double> row_values(static_cast<size_t>(rows), 0.0);
    double total = 0;
    for (int64_t i = 0; i < rows; ++i) {
        double row = 0;
        for (int64_t c = 0; c < C; ++c) {
            const double xv = xp[i * C + c];
            const double tv = tp[i * C + c];
            row += tv * -softplus(-xv) + (1.0 - tv) * -softplus(xv);
        }
        row_values[static_cast<size_t>(i)] = -row / static_cast<double>(C);
        total += row_values[static_cast<size_t>(i)];
    }
    if (reduction == 0) {
        Tensor rows_t = Tensor::tensor(row_values, DType::Float64, dev)
                            .reshape(row_shape)
                            .to(out_dt);
        return rows_t;
    }
    if (reduction == 2) return scalar_from(total, out_dt, dev);
    return scalar_from(rows > 0 ? total / static_cast<double>(rows)
                                : std::numeric_limits<double>::quiet_NaN(),
                       out_dt, dev);
}

TENSORPLAY_LIBRARY_IMPL(CPU, ExtraLosses) {
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cpu);
    m.impl("triplet_margin_loss", triplet_margin_loss_cpu);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cpu);
}

}  // namespace cpu
}  // namespace tensorplay
