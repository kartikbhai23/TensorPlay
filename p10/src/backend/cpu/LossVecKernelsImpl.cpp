// Tier-compiled forward walks for the losses: binary cross entropy, binary
// cross entropy over logits, huber, multilabel soft margin and triplet margin.  Each CPU
// capability tier compiles its own copy of this file (see TP_CPU_KERNEL_SRCS
// in p10/CMakeLists.txt) and registers the slots declared in
// cpu/LossVecKernels.h; DispatchStub picks the widest copy at run time.
//
// The folds read the operand buffers as they are, so no widened copy of an
// input is materialized, and the per-lane arithmetic stays in the operand
// width (float for float inputs) while the range answers are folded in double.

#include "cpu/LossVecKernels.h"
#include "cpu/vec/vec.h"

#include "DType.h"
#include "Exception.h"

#include "LossReduce.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

// Work per unit of the elementwise walks and of the row-wise walk.
constexpr int64_t kElementwiseUnit = 1;

template <typename T>
double huber_sum_typed(const T* input, const T* target, int64_t n,
                       double delta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T d = static_cast<T>(delta);
    const Vec dv(d);
    const Vec half(static_cast<T>(0.5));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec z = Vec::loadu(input + i) - Vec::loadu(target + i);
            const Vec az = z.abs();
            const Vec quad = z * z * half;
            const Vec linear = dv * (az - dv * half);
            acc = acc + Vec::blendv(linear, quad, az <= dv);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const T z = input[i] - target[i];
            const T az = z < static_cast<T>(0) ? -z : z;
            total += static_cast<double>(az <= d
                                             ? z * z * static_cast<T>(0.5)
                                             : d * (az - d * static_cast<T>(0.5)));
        }
        return total;
    });
}


template <typename T>
void huber_into_typed(const T* input, const T* target, T* out, int64_t n,
                      double delta) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const T d = static_cast<T>(delta);
    const Vec dv(d);
    const Vec half(static_cast<T>(0.5));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z =
                                Vec::loadu(input + i) - Vec::loadu(target + i);
                            const Vec az = z.abs();
                            const Vec quad = z * z * half;
                            const Vec linear = dv * (az - dv * half);
                            Vec::blendv(linear, quad, az <= dv).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            const T az = z < static_cast<T>(0) ? -z : z;
                            out[i] = az <= d
                                         ? z * z * static_cast<T>(0.5)
                                         : d * (az - d * static_cast<T>(0.5));
                        }
                        return 0.0;
                    });
}

// z = x - t, squared and folded.  One load of each operand answers a whole
// lane of the sum, so the deviation plane the composed form would allocate
// never exists.
template <typename T>
double mse_sum_typed(const T* input, const T* target, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
                               Vec acc(static_cast<T>(0));
                               int64_t i = begin;
                               for (; i + W <= end; i += W) {
                                   const Vec z = Vec::loadu(input + i) -
                                                 Vec::loadu(target + i);
                                   acc = acc + z * z;
                               }
                               double total =
                                   static_cast<double>(acc.reduce_add());
                               for (; i < end; ++i) {
                                   const T z = input[i] - target[i];
                                   total += static_cast<double>(z * z);
                               }
                               return total;
                           });
}

template <typename T>
void mse_into_typed(const T* input, const T* target, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            (z * z).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = z * z;
                        }
                        return 0.0;
                    });
}

// The squared walk answers twice the deviation, moved by the incoming
// gradient lane by lane.
template <typename T>
void mse_grad_typed(const T* grad_output, const T* input, const T* target,
                    T* out, int64_t n, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec slope(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            (slope * z * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = static_cast<T>(scale) * z *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// The same deviation read without its sign.
template <typename T>
double l1_sum_typed(const T* input, const T* target, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
                               Vec acc(static_cast<T>(0));
                               int64_t i = begin;
                               for (; i + W <= end; i += W) {
                                   const Vec z = Vec::loadu(input + i) -
                                                 Vec::loadu(target + i);
                                   acc = acc + z.abs();
                               }
                               double total =
                                   static_cast<double>(acc.reduce_add());
                               for (; i < end; ++i) {
                                   const T z = input[i] - target[i];
                                   total += static_cast<double>(
                                       z < static_cast<T>(0) ? -z : z);
                               }
                               return total;
                           });
}

template <typename T>
void l1_into_typed(const T* input, const T* target, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            z.abs().store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            out[i] = z < static_cast<T>(0) ? -z : z;
                        }
                        return 0.0;
                    });
}

// The absolute walk answers the side the deviation sits on: both sides carry
// one, and the width itself answers nothing.
template <typename T>
void l1_grad_typed(const T* grad_output, const T* input, const T* target,
                   T* out, int64_t n, double scale) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(scale));
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec z = Vec::loadu(input + i) -
                                          Vec::loadu(target + i);
                            const Vec side = Vec::blendv(
                                zero, Vec::blendv(-one, one, z > zero),
                                z != zero);
                            (side * Vec::loadu(grad_output + i))
                                .store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T z = input[i] - target[i];
                            const T side = z > static_cast<T>(0)
                                               ? static_cast<T>(1)
                                               : (z < static_cast<T>(0)
                                                      ? static_cast<T>(-1)
                                                      : static_cast<T>(0));
                            out[i] = side * static_cast<T>(scale) *
                                     grad_output[i];
                        }
                        return 0.0;
                    });
}

// log x and log(1 - x), both floored at -100, in the operand width.
template <typename T>
double bce_sum_typed(const T* input, const T* target, int64_t n,
                     int* bad_input, int* bad_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec floor_value(static_cast<T>(-100));
    std::atomic<int> in_bad{0};
    std::atomic<int> tg_bad{0};
    const double total =
        parallel_ranges(n, kElementwiseUnit,
                        [&](int64_t begin, int64_t end) -> double {
                            Vec acc(static_cast<T>(0));
                            bool chunk_in_bad = false;
                            bool chunk_tg_bad = false;
                            int64_t i = begin;
                            for (; i + W <= end; i += W) {
                                const Vec iv = Vec::loadu(input + i);
                                const Vec tv = Vec::loadu(target + i);
                                const Vec in_ok = Vec::blendv(
                                    zero, one, (iv >= zero) & (iv <= one));
                                const Vec tg_ok = Vec::blendv(
                                    zero, one, (tv >= zero) & (tv <= one));
                                if (in_ok.reduce_min() == static_cast<T>(0)) chunk_in_bad = true;
                                if (tg_ok.reduce_min() == static_cast<T>(0)) chunk_tg_bad = true;
                                const Vec log_x =
                                    vec::maximum(iv.log(), floor_value);
                                const Vec log_1mx =
                                    vec::maximum((one - iv).log(), floor_value);
                                acc = acc + (tv - one) * log_1mx - tv * log_x;
                            }
                            double local = static_cast<double>(acc.reduce_add());
                            for (; i < end; ++i) {
                                const T iv = input[i];
                                const T tv = target[i];
                                if (!(iv >= static_cast<T>(0) &&
                                      iv <= static_cast<T>(1)))
                                    chunk_in_bad = true;
                                if (!(tv >= static_cast<T>(0) &&
                                      tv <= static_cast<T>(1)))
                                    chunk_tg_bad = true;
                                const T log_x = std::max(
                                    static_cast<T>(std::log(iv)),
                                    static_cast<T>(-100));
                                const T log_1mx = std::max(
                                    static_cast<T>(std::log1p(-iv)),
                                    static_cast<T>(-100));
                                local += static_cast<double>(
                                    (tv - static_cast<T>(1)) * log_1mx -
                                    tv * log_x);
                            }
                            if (chunk_in_bad)
                                in_bad.store(1, std::memory_order_relaxed);
                            if (chunk_tg_bad)
                                tg_bad.store(1, std::memory_order_relaxed);
                            return local;
                        });
    *bad_input = in_bad.load(std::memory_order_relaxed);
    *bad_target = tg_bad.load(std::memory_order_relaxed);
    return total;
}


template <typename T>
void bce_into_typed(const T* input, const T* target, T* out, int64_t n,
                    int* bad_input, int* bad_target) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    const Vec floor_value(static_cast<T>(-100));
    std::atomic<int> in_bad{0};
    std::atomic<int> tg_bad{0};
    parallel_ranges(n, kElementwiseUnit,
                    [&](int64_t begin, int64_t end) -> double {
                        bool chunk_in_bad = false;
                        bool chunk_tg_bad = false;
                        int64_t i = begin;
                        for (; i + W <= end; i += W) {
                            const Vec iv = Vec::loadu(input + i);
                            const Vec tv = Vec::loadu(target + i);
                            const Vec in_ok =
                                Vec::blendv(zero, one, (iv >= zero) & (iv <= one));
                            const Vec tg_ok =
                                Vec::blendv(zero, one, (tv >= zero) & (tv <= one));
                            if (in_ok.reduce_min() == static_cast<T>(0))
                                chunk_in_bad = true;
                            if (tg_ok.reduce_min() == static_cast<T>(0))
                                chunk_tg_bad = true;
                            const Vec log_x = vec::maximum(iv.log(), floor_value);
                            const Vec log_1mx =
                                vec::maximum((one - iv).log(), floor_value);
                            ((tv - one) * log_1mx - tv * log_x).store(out + i);
                        }
                        for (; i < end; ++i) {
                            const T iv = input[i];
                            const T tv = target[i];
                            if (!(iv >= static_cast<T>(0) && iv <= static_cast<T>(1)))
                                chunk_in_bad = true;
                            if (!(tv >= static_cast<T>(0) && tv <= static_cast<T>(1)))
                                chunk_tg_bad = true;
                            const T log_x = std::max(static_cast<T>(std::log(iv)),
                                                     static_cast<T>(-100));
                            const T log_1mx = std::max(
                                static_cast<T>(std::log1p(-iv)),
                                static_cast<T>(-100));
                            out[i] = (tv - static_cast<T>(1)) * log_1mx - tv * log_x;
                        }
                        if (chunk_in_bad)
                            in_bad.store(1, std::memory_order_relaxed);
                        if (chunk_tg_bad)
                            tg_bad.store(1, std::memory_order_relaxed);
                        return 0.0;
                    });
    *bad_input = in_bad.load(std::memory_order_relaxed);
    *bad_target = tg_bad.load(std::memory_order_relaxed);
}

// The loss one lane contributes, negated: t log sigmoid(x) + (1 - t)
// log sigmoid(-x) = t max(-x, 0) + (1 - t) max(x, 0) + log1p(exp(-|x|)), and
// the row answer is its negated sum divided by the row length.
template <typename T>
T multilabel_lane(const T x, const T t) {
    const double xv = static_cast<double>(x);
    const double tv = static_cast<double>(t);
    const double tail = std::log1p(std::exp(-std::fabs(xv)));
    return static_cast<T>(tv * std::max(-xv, 0.0) +
                          (1.0 - tv) * std::max(xv, 0.0) + tail);
}

// The row answer is the walk over the row divided by the row length, and the
// rows are cut off the flat operands, so a row walk needs no second pass.
template <typename T>
double multilabel_rows_typed(const T* input, const T* target, T* out,
                             int64_t rows, int64_t width) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    if (width <= 0) return 0.0;
    const T scale = static_cast<T>(1) / static_cast<T>(width);
    return parallel_ranges(rows, width,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* row_in = input + r * width;
            const T* row_tg = target + r * width;
            Vec acc(static_cast<T>(0));
            int64_t i = 0;
            for (; i + W <= width; i += W) {
                const Vec xv = Vec::loadu(row_in + i);
                const Vec tv = Vec::loadu(row_tg + i);
                const Vec tail = (-xv.abs()).exp().log1p();
                const Vec positive = vec::maximum(xv, zero);
                const Vec negative = vec::maximum(-xv, zero);
                acc = acc + (tv * negative + (one - tv) * positive + tail);
            }
            T sum = acc.reduce_add();
            for (; i < width; ++i) sum += multilabel_lane(row_in[i], row_tg[i]);
            const T answer = sum * scale;
            total += static_cast<double>(answer);
            if (out) out[r] = answer;
        }
        return total;
    });
}

// sp(y) = max(y, 0) + log1p(exp(-|y|)); the two softplus terms of one lane
// share their tail, so the walk builds it once and keeps the positive parts.
template <typename T>
double bce_logits_rows_typed(const T* input, const T* target, const T* weight,
                             const T* pos_weight, T* out, int64_t n) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec zero(static_cast<T>(0));
    const Vec one(static_cast<T>(1));
    return parallel_ranges(n, kElementwiseUnit,
                           [&](int64_t begin, int64_t end) -> double {
        Vec acc(static_cast<T>(0));
        int64_t i = begin;
        for (; i + W <= end; i += W) {
            const Vec xv = Vec::loadu(input + i);
            const Vec tv = Vec::loadu(target + i);
            const Vec tail = (-xv.abs()).exp().log1p();
            const Vec sp_x = vec::maximum(xv, zero) + tail;
            const Vec sp_neg_x = vec::maximum(-xv, zero) + tail;
            const Vec positive = pos_weight ? Vec::loadu(pos_weight + i) : one;
            Vec value = positive * tv * sp_neg_x + (one - tv) * sp_x;
            if (weight) value = value * Vec::loadu(weight + i);
            acc = acc + value;
            if (out) value.store(out + i);
        }
        double total = static_cast<double>(acc.reduce_add());
        for (; i < end; ++i) {
            const double xv = static_cast<double>(input[i]);
            const double tv = static_cast<double>(target[i]);
            const double tail = std::log1p(std::exp(-std::fabs(xv)));
            const double sp_x = std::max(xv, 0.0) + tail;
            const double sp_neg_x = std::max(-xv, 0.0) + tail;
            const double positive =
                pos_weight ? static_cast<double>(pos_weight[i]) : 1.0;
            double value = positive * tv * sp_neg_x + (1.0 - tv) * sp_x;
            if (weight) value *= static_cast<double>(weight[i]);
            total += value;
            if (out) out[i] = static_cast<T>(value);
        }
        return total;
    });
}

// Order-p norm of (u - v + eps) over one row: the row is shifted by eps
// elementwise before the magnitude is accumulated, so two rows that agree
// answer eps rather than zero.  Orders 1, 2 and the two infinite orders walk
// the row in vector width; the rest (including the zero-order count of unequal
// entries and negative finite orders) take the scalar recurrence.
template <typename T>
double row_norm(const T* u, const T* v, int64_t width, double p, double eps) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t W = Vec::size();
    const Vec shift(static_cast<T>(eps));
    int64_t i = 0;
    if (std::isinf(p)) {
        const bool largest = p > 0;
        Vec best(static_cast<T>(largest ? 0
                                        : std::numeric_limits<T>::max()));
        for (; i + W <= width; i += W) {
            const Vec d = ((Vec::loadu(u + i) - Vec::loadu(v + i)) + shift).abs();
            best = largest ? vec::maximum(best, d) : vec::minimum(best, d);
        }
        double answer = static_cast<double>(
            largest ? best.reduce_max() : best.reduce_min());
        for (; i < width; ++i) {
            const double d =
                std::fabs(static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps);
            answer = largest ? std::max(answer, d) : std::min(answer, d);
        }
        return answer;
    }
    if (p == 2.0) {
        Vec acc(static_cast<T>(0));
        for (; i + W <= width; i += W) {
            const Vec d = (Vec::loadu(u + i) - Vec::loadu(v + i)) + shift;
            acc = acc + d * d;
        }
        double answer = static_cast<double>(acc.reduce_add());
        for (; i < width; ++i) {
            const double d =
                static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps;
            answer += d * d;
        }
        return std::sqrt(answer);
    }
    if (p == 1.0) {
        Vec acc(static_cast<T>(0));
        for (; i + W <= width; i += W) {
            acc = acc + ((Vec::loadu(u + i) - Vec::loadu(v + i)) + shift).abs();
        }
        double answer = static_cast<double>(acc.reduce_add());
        for (; i < width; ++i)
            answer += std::fabs(static_cast<double>(u[i]) -
                                static_cast<double>(v[i]) + eps);
        return answer;
    }
    if (p == 0.0) {
        double count = 0.0;
        for (; i < width; ++i) {
            if (static_cast<double>(u[i]) - static_cast<double>(v[i]) + eps != 0.0)
                count += 1.0;
        }
        return count;
    }
    double answer = 0.0;
    for (; i < width; ++i) {
        answer += std::pow(std::fabs(static_cast<double>(u[i]) -
                                     static_cast<double>(v[i]) + eps), p);
    }
    return std::pow(answer, 1.0 / p);
}

template <typename T>
double triplet_rows_typed(const T* anchor, const T* positive, const T* negative,
                          T* out, int64_t rows, int64_t width, double margin,
                          double p, double eps, bool swap) {
    return parallel_ranges(rows, width,
                           [&](int64_t begin, int64_t end) -> double {
        double total = 0.0;
        for (int64_t r = begin; r < end; ++r) {
            const T* a = anchor + r * width;
            const T* pos = positive + r * width;
            const T* neg = negative + r * width;
            double dn = row_norm(a, neg, width, p, eps);
            if (swap) {
                dn = std::min(dn, row_norm(pos, neg, width, p, eps));
            }
            const double row =
                std::max(0.0, row_norm(a, pos, width, p, eps) - dn + margin);
            if (out != nullptr) out[r] = static_cast<T>(row);
            total += row;
        }
        return total;
    });
}


void bce_into_impl(const void* input, const void* target, void* out, int64_t n,
                   int* bad_input, int* bad_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            bce_into_typed(static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n, bad_input, bad_target);
            break;
        case DType::Float32:
            bce_into_typed(static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n, bad_input, bad_target);
            break;
        default:
            TP_THROW(TypeError, "binary_cross_entropy: unsupported dtype");
    }
}

void huber_into_impl(const void* input, const void* target, void* out, int64_t n,
                     double delta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            huber_into_typed(static_cast<const double*>(input),
                             static_cast<const double*>(target),
                             static_cast<double*>(out), n, delta);
            break;
        case DType::Float32:
            huber_into_typed(static_cast<const float*>(input),
                             static_cast<const float*>(target),
                             static_cast<float*>(out), n, delta);
            break;
        default:
            TP_THROW(TypeError, "huber_loss: unsupported dtype");
    }
}

double bce_sum_impl(const void* input, const void* target, int64_t n,
                    int* bad_input, int* bad_target, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return bce_sum_typed(static_cast<const double*>(input),
                                 static_cast<const double*>(target), n,
                                 bad_input, bad_target);
        case DType::Float32:
            return bce_sum_typed(static_cast<const float*>(input),
                                 static_cast<const float*>(target), n,
                                 bad_input, bad_target);
        default:
            TP_THROW(TypeError, "binary_cross_entropy: unsupported dtype");
    }
}

double huber_sum_impl(const void* input, const void* target, int64_t n,
                      double delta, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return huber_sum_typed(static_cast<const double*>(input),
                                   static_cast<const double*>(target), n, delta);
        case DType::Float32:
            return huber_sum_typed(static_cast<const float*>(input),
                                   static_cast<const float*>(target), n, delta);
        default:
            TP_THROW(TypeError, "huber_loss: unsupported dtype");
    }
}

double mse_sum_impl(const void* input, const void* target, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return mse_sum_typed(static_cast<const double*>(input),
                                 static_cast<const double*>(target), n);
        case DType::Float32:
            return mse_sum_typed(static_cast<const float*>(input),
                                 static_cast<const float*>(target), n);
        default:
            TP_THROW(TypeError, "mse_loss: unsupported dtype");
    }
}

void mse_into_impl(const void* input, const void* target, void* out, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            mse_into_typed(static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n);
            break;
        case DType::Float32:
            mse_into_typed(static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n);
            break;
        default:
            TP_THROW(TypeError, "mse_loss: unsupported dtype");
    }
}

void mse_grad_impl(const void* grad_output, const void* input,
                   const void* target, void* out, int64_t n, double scale,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            mse_grad_typed(static_cast<const double*>(grad_output),
                           static_cast<const double*>(input),
                           static_cast<const double*>(target),
                           static_cast<double*>(out), n, scale);
            break;
        case DType::Float32:
            mse_grad_typed(static_cast<const float*>(grad_output),
                           static_cast<const float*>(input),
                           static_cast<const float*>(target),
                           static_cast<float*>(out), n, scale);
            break;
        default:
            TP_THROW(TypeError, "mse_loss backward: unsupported dtype");
    }
}

double l1_sum_impl(const void* input, const void* target, int64_t n,
                   int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return l1_sum_typed(static_cast<const double*>(input),
                                static_cast<const double*>(target), n);
        case DType::Float32:
            return l1_sum_typed(static_cast<const float*>(input),
                                static_cast<const float*>(target), n);
        default:
            TP_THROW(TypeError, "l1_loss: unsupported dtype");
    }
}

void l1_into_impl(const void* input, const void* target, void* out, int64_t n,
                  int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            l1_into_typed(static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n);
            break;
        case DType::Float32:
            l1_into_typed(static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n);
            break;
        default:
            TP_THROW(TypeError, "l1_loss: unsupported dtype");
    }
}

void l1_grad_impl(const void* grad_output, const void* input,
                  const void* target, void* out, int64_t n, double scale,
                  int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            l1_grad_typed(static_cast<const double*>(grad_output),
                          static_cast<const double*>(input),
                          static_cast<const double*>(target),
                          static_cast<double*>(out), n, scale);
            break;
        case DType::Float32:
            l1_grad_typed(static_cast<const float*>(grad_output),
                          static_cast<const float*>(input),
                          static_cast<const float*>(target),
                          static_cast<float*>(out), n, scale);
            break;
        default:
            TP_THROW(TypeError, "l1_loss backward: unsupported dtype");
    }
}

double multilabel_rows_impl(const void* input, const void* target, void* out,
                            int64_t rows, int64_t width, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return multilabel_rows_typed(static_cast<const double*>(input),
                                         static_cast<const double*>(target),
                                         static_cast<double*>(out), rows, width);
        case DType::Float32:
            return multilabel_rows_typed(static_cast<const float*>(input),
                                         static_cast<const float*>(target),
                                         static_cast<float*>(out), rows, width);
        default:
            TP_THROW(TypeError, "multilabel_soft_margin_loss: unsupported dtype");
    }
}

double bce_logits_rows_impl(const void* input, const void* target,
                            const void* weight, const void* pos_weight,
                            void* out, int64_t n, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return bce_logits_rows_typed(
                static_cast<const double*>(input),
                static_cast<const double*>(target),
                static_cast<const double*>(weight),
                static_cast<const double*>(pos_weight),
                static_cast<double*>(out), n);
        case DType::Float32:
            return bce_logits_rows_typed(
                static_cast<const float*>(input),
                static_cast<const float*>(target),
                static_cast<const float*>(weight),
                static_cast<const float*>(pos_weight),
                static_cast<float*>(out), n);
        default:
            TP_THROW(TypeError,
                     "binary_cross_entropy_with_logits: unsupported dtype");
    }
}

double triplet_rows_impl(const void* anchor, const void* positive,
                         const void* negative, void* out, int64_t rows,
                         int64_t width, double margin, double p, double eps,
                         bool swap, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float64:
            return triplet_rows_typed(static_cast<const double*>(anchor),
                                      static_cast<const double*>(positive),
                                      static_cast<const double*>(negative),
                                      static_cast<double*>(out), rows, width,
                                      margin, p, eps, swap);
        case DType::Float32:
            return triplet_rows_typed(static_cast<const float*>(anchor),
                                      static_cast<const float*>(positive),
                                      static_cast<const float*>(negative),
                                      static_cast<float*>(out), rows, width,
                                      margin, p, eps, swap);
        default:
            TP_THROW(TypeError, "triplet_margin_loss: unsupported dtype");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(bce_sum_stub, &bce_sum_impl);
REGISTER_DISPATCH(bce_into_stub, &bce_into_impl);
REGISTER_DISPATCH(mse_sum_stub, &mse_sum_impl);
REGISTER_DISPATCH(mse_into_stub, &mse_into_impl);
REGISTER_DISPATCH(mse_grad_stub, &mse_grad_impl);
REGISTER_DISPATCH(l1_sum_stub, &l1_sum_impl);
REGISTER_DISPATCH(l1_into_stub, &l1_into_impl);
REGISTER_DISPATCH(l1_grad_stub, &l1_grad_impl);
REGISTER_DISPATCH(huber_into_stub, &huber_into_impl);
REGISTER_DISPATCH(huber_sum_stub, &huber_sum_impl);
REGISTER_DISPATCH(bce_logits_rows_stub, &bce_logits_rows_impl);
REGISTER_DISPATCH(multilabel_rows_stub, &multilabel_rows_impl);
REGISTER_DISPATCH(triplet_rows_stub, &triplet_rows_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(bce_sum_stub, &bce_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(bce_into_stub, &bce_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_sum_stub, &mse_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_into_stub, &mse_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(mse_grad_stub, &mse_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_sum_stub, &l1_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_into_stub, &l1_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(l1_grad_stub, &l1_grad_impl);
ALSO_REGISTER_AVX512_DISPATCH(huber_into_stub, &huber_into_impl);
ALSO_REGISTER_AVX512_DISPATCH(huber_sum_stub, &huber_sum_impl);
ALSO_REGISTER_AVX512_DISPATCH(bce_logits_rows_stub, &bce_logits_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(multilabel_rows_stub, &multilabel_rows_impl);
ALSO_REGISTER_AVX512_DISPATCH(triplet_rows_stub, &triplet_rows_impl);
#endif

} // namespace cpu
} // namespace tensorplay