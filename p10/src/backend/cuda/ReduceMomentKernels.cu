#include "ReduceEngine.cuh"

#include "QuantTensorOperators.h"
#include "Complex.h"

#include <limits>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

using namespace reduce_family;

std::tuple<Tensor, Tensor> std_mean_dim_kernel(
        const Tensor& self, const std::vector<int64_t>& dim,
        int64_t correction, bool keepdim);

namespace {

template <typename T, typename OutT = T>
Tensor mean_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim, DType dtype) {
    using AccT = same_dtype_acc_t<T>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), dtype, input.device());
    }
    using FactorT = typename tensorplay::scalar_value_type<AccT>::type;
    const FactorT factor = FactorT(1) / static_cast<FactorT>(spec.reduced_numel);
    return run_reduction_typed<T, AccT, OutT>(
        input, spec, keepdim, dtype, MeanOps<T, AccT, OutT>{factor}, AccT(0));
}

template <typename T, typename IndexT>
Tensor welford_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    using AccT = same_dtype_acc_t<T>;
    using StateT = WelfordData<AccT, IndexT>;
    using Ops = WelfordOps<AccT, T, IndexT>;
    if (input.numel() == 0) {
        return Tensor::full(
            reduction_output_shape(input, spec, keepdim),
            Scalar(std::numeric_limits<float>::quiet_NaN()), input.dtype(), input.device());
    }
    return run_reduction_typed<T, StateT, T, Ops, 4>(
        input, spec, keepdim, input.dtype(),
        Ops{static_cast<AccT>(correction), take_sqrt}, StateT{AccT(0), AccT(0), 0, AccT(0)});
}

// var (or std) and mean out of one Welford pass: the accumulator already
// carries both, so the mean rides along instead of re-reading the whole input
// and the square root lands in the same projection.
template <typename T, typename IndexT>
std::tuple<Tensor, Tensor> welford_var_mean_same_dtype(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    using AccT = same_dtype_acc_t<T>;
    using StateT = WelfordData<AccT, IndexT>;
    using Ops = WelfordOps<AccT, T, IndexT>;
    const auto shape = reduction_output_shape(input, spec, keepdim);
    if (input.numel() == 0) {
        const Scalar nan_value{std::numeric_limits<float>::quiet_NaN()};
        return {Tensor::full(shape, nan_value, input.dtype(), input.device()),
                Tensor::full(shape, nan_value, input.dtype(), input.device())};
    }
    Tensor mean = Tensor::empty(shape, input.dtype(), input.device());
    // Four independent accumulators per thread: the per-element update is a
    // serial chain through the running mean, so the group supplies the ILP.
    Tensor var = run_reduction_typed<T, StateT, T, Ops, 4>(
        input, spec, keepdim, input.dtype(),
        Ops{static_cast<AccT>(correction), take_sqrt},
        StateT{AccT(0), AccT(0), 0, AccT(0)}, &mean);
    return {var, mean};
}

// The dtype dispatch macro fixes the first template argument, so the count
// width is chosen here.
template <typename T>
std::tuple<Tensor, Tensor> welford_var_mean_dispatch(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    if (spec.reduced_numel <=
        static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return welford_var_mean_same_dtype<T, int32_t>(
            input, spec, keepdim, correction, take_sqrt);
    }
    return welford_var_mean_same_dtype<T, int64_t>(
        input, spec, keepdim, correction, take_sqrt);
}


std::tuple<Tensor, Tensor> var_mean_cuda(const Tensor& self, const std::vector<int64_t>& dim,
                                         bool unbiased, bool keepdim) {
    // One Welford pass yields both; a separate mean pass would re-read the
    // whole input for a value the accumulator already holds.
    return var_mean_dim_kernel(self, dim, unbiased ? 1 : 0, keepdim);
}

std::tuple<Tensor, Tensor> std_mean_cuda(const Tensor& self, const std::vector<int64_t>& dim,
                                         bool unbiased, bool keepdim) {
    return std_mean_dim_kernel(self, dim, unbiased ? 1 : 0, keepdim);
}




// NaN-aware mean: counts the finite elements per slice and divides.
__global__ void nanmean_zero_mask_kernel(int64_t n, const float* count, float* data) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (count[i] == 0.0f) data[i] = std::numeric_limits<float>::quiet_NaN();
    }
}


Tensor nanmean_cuda(const Tensor& self, std::optional<int64_t> dim_opt, bool keepdim,
                    std::optional<DType> dtype) {
    DType acc_dt = dtype.value_or(DType::Undefined);
    if (!isFloatingType(self.dtype()) && !isComplexType(self.dtype())) {
        TP_THROW(TypeError,
                 "nanmean(): expected input to have floating point or complex dtype but got ",
                 toString(self.dtype()));
    }
    if (acc_dt != DType::Undefined && !isFloatingType(acc_dt) &&
        !isComplexType(acc_dt)) {
        TP_THROW(TypeError,
                 "nanmean(): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: ",
                 toString(acc_dt));
    }
    Tensor x = self;
    if (acc_dt != DType::Undefined && x.dtype() != acc_dt) {
        x = x.to(acc_dt);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) dims.push_back(*dim_opt);
    else {
        // global reduction over every dimension
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    // Two reductions and one divide: a finite-element count and a NaN-
    // skipping sum.  An all-NaN slice yields 0/0 = NaN, so no explicit
    // zero-count mask is needed.
    Tensor factor =
        ops::isnan(x).logical_not().sum(dims, keepdim, DType::Float32);
    Tensor result =
        nansum_cuda2(x, dims, keepdim).div(factor);
    return result.to(acc_dt != DType::Undefined ? acc_dt : result.dtype());
}

} // namespace

// Mean
Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, DType dtype) {
    if (isQuantizedType(self.dtype())) return quantized::moment(self, dim, keepdim, false, 1., dtype);

    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
        out_dtype = isFloatingOrComplexType(self.dtype()) ? self.dtype() : DType::Float32;
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    if (self.dtype() == DType::Float16 && out_dtype == DType::Float32) {
        return mean_same_dtype<Half, float>(self, spec, keepdim, out_dtype);
    }
    if (self.dtype() == DType::BFloat16 && out_dtype == DType::Float32) {
        return mean_same_dtype<BFloat16, float>(self, spec, keepdim, out_dtype);
    }
    if (isComplexType(self.dtype())) {
        // Single-pass complex mean: MeanOps accumulates the complex sum and
        // scales it by a real factor (1/count) on projection.
        if (self.dtype() == DType::ComplexHalf || self.dtype() == DType::BComplex32) {
            if (out_dtype == DType::ComplexFloat) {
                return mean_same_dtype<tensorplay::complex<Half>, tensorplay::complex<float>>(
                    self, spec, keepdim, out_dtype);
            }
            return mean_same_dtype<tensorplay::complex<Half>, tensorplay::complex<Half>>(
                self, spec, keepdim, out_dtype);
        }
        if (self.dtype() == DType::ComplexFloat) {
            return mean_same_dtype<tensorplay::complex<float>>(self, spec, keepdim, out_dtype);
        }
        return mean_same_dtype<tensorplay::complex<double>>(self, spec, keepdim, out_dtype);
    }
    Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    const ReductionSpec input_spec = make_reduction_spec(input, dim);
    TP_DISPATCH_FLOAT_REDUCTION(mean_same_dtype, input.dtype(), input, input_spec, keepdim, out_dtype);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    if (isQuantizedType(self.dtype())) return quantized::moment(self, {}, false, false, 1., dtype);

    return mean_dim_kernel(self, {}, false, dtype);
}

// Var / Std
// 32-bit counts whenever the reduction extent allows: the count rides along
// every shuffle and staging step, and a narrower count shrinks both.
template <typename T>
Tensor welford_dispatch(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        int64_t correction, bool take_sqrt) {
    if (spec.reduced_numel <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        return welford_same_dtype<T, int32_t>(input, spec, keepdim, correction, take_sqrt);
    }
    return welford_same_dtype<T, int64_t>(input, spec, keepdim, correction, take_sqrt);
}

// Two-pass variance for full reductions: each pass runs at sum speed, so
// the pair beats the serial Welford chain, and no host sync is needed
// because the second pass reads the mean straight from device memory.
// The correction division is linear in the sum, so each block applies it
// before the atomic accumulation.
__global__ void var_two_pass_kernel(int64_t n, double divisor,
                                    const float* __restrict__ mean_ptr,
                                    const float* __restrict__ input,
                                    float* __restrict__ accumulator) {
    const double mean = mean_ptr[0];
    double value = 0.0;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const double d = static_cast<double>(input[i]) - mean;
        value += d * d;
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
    }
    __shared__ double warp_values[32];
    if (lane == 0) warp_values[warp] = value;
    __syncthreads();
    if (warp == 0) {
        const int warp_count = (blockDim.x + 31) / 32;
        value = lane < warp_count ? warp_values[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffffffffffull, value, offset);
        }
        if (lane == 0) {
            atomicAdd(accumulator, static_cast<float>(value / divisor));
        }
    }
}

Tensor var_two_pass_from_mean(const Tensor& self, Tensor& mean_t,
                              int64_t correction) {
    Tensor out = Tensor::empty({}, DType::Float32, self.device());
    out.zero_();
    const int64_t n = self.numel();
    const int threads = 64;
    const int64_t blocks =
        std::min<int64_t>((n + threads * 4 - 1) / (threads * 4), 256);
    var_two_pass_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                           getCurrentCUDAStream().stream()>>>(
        n, static_cast<double>(n) - static_cast<double>(correction), mean_t.data_ptr<float>(),
        self.data_ptr<float>(), out.data_ptr<float>());
    return out;
}

Tensor var_two_pass(const Tensor& self, int64_t correction) {
    Tensor mean_t = mean_dim_kernel(self, {}, false, DType::Float32);
    return var_two_pass_from_mean(self, mean_t, correction);
}

static bool can_use_two_pass_variance(
        const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    // Empty inputs and a zero divisor need the Welford path's NaN and Inf handling.
    return dim.empty() && !keepdim && self.dtype() == DType::Float32 &&
           self.is_contiguous() && self.numel() > std::max<int64_t>(0, correction);
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        Tensor real = tensorplay::tpx::ops::real(self);
        Tensor imag = tensorplay::tpx::ops::imag(self);
        return var_dim_kernel(real, dim, correction, keepdim) +
               var_dim_kernel(imag, dim, correction, keepdim);
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    // Full-reduction float fast path: the two-pass pair runs at sum speed,
    // well ahead of the serial Welford chain on ordinary sizes.
    if (can_use_two_pass_variance(self, dim, correction, keepdim)) {
        return var_two_pass(self, correction);
    }
    TP_DISPATCH_FLOAT_REDUCTION(welford_dispatch, self.dtype(), self, spec,
                                keepdim, correction, false);
}

std::tuple<Tensor, Tensor> var_mean_dim_kernel(
        const Tensor& self, const std::vector<int64_t>& dim,
        int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        // Complex keeps the two-pass form: the variance adds over the
        // components while the mean has to be rebuilt as a complex value.
        return {var_dim_kernel(self, dim, correction, keepdim),
                mean_dim_kernel(self, dim, keepdim, DType::Undefined)};
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    // Full-reduction float fast path: the two-pass pair, with the mean the
    // first pass already produced returned as the second output.
    if (can_use_two_pass_variance(self, dim, correction, keepdim)) {
        Tensor mean_t = mean_dim_kernel(self, {}, false, DType::Float32);
        Tensor var = var_two_pass_from_mean(self, mean_t, correction);
        return {var, mean_t};
    }
    TP_DISPATCH_FLOAT_REDUCTION(welford_var_mean_dispatch, self.dtype(), self, spec,
                                keepdim, correction, false);
}

std::tuple<Tensor, Tensor> std_mean_dim_kernel(
        const Tensor& self, const std::vector<int64_t>& dim,
        int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        return {var_dim_kernel(self, dim, correction, keepdim).sqrt(),
                mean_dim_kernel(self, dim, keepdim, DType::Undefined)};
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_var_mean_dispatch, self.dtype(), self, spec,
                                keepdim, correction, true);
}

Tensor var_kernel(const Tensor& self, int64_t correction) {
    return var_dim_kernel(self, {}, correction, false);
}

Tensor std_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (isQuantizedType(self.dtype())) return quantized::moment(self, dim, keepdim, true, correction);

    if (isComplexType(self.dtype())) {
        return var_dim_kernel(self, dim, correction, keepdim).sqrt();
    }
    // Same two-pass fast path as var, with the sqrt folded into the caller.
    if (can_use_two_pass_variance(self, dim, correction, keepdim)) {
        return var_two_pass(self, correction).sqrt();
    }
    const ReductionSpec spec = make_reduction_spec(self, dim);
    TP_DISPATCH_FLOAT_REDUCTION(welford_dispatch, self.dtype(), self, spec,
                                keepdim, correction, true);
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    if (isQuantizedType(self.dtype())) return quantized::moment(self, {}, false, true, correction);

    if (isComplexType(self.dtype())) {
        return var_kernel(self, correction).sqrt();
    }
    return std_dim_kernel(self, {}, correction, false);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, ReduceMomentKernels) {
    m.impl("var_mean", var_mean_cuda);
    m.impl("std_mean", std_mean_cuda);
    m.impl("nanmean", nanmean_cuda);
    m.impl("mean", mean_kernel);
    m.impl("mean.dim", mean_dim_kernel);
    m.impl("var", var_kernel);
    m.impl("var.dim", var_dim_kernel);
    m.impl("std", std_kernel);
    m.impl("std.dim", std_dim_kernel);
}

} // namespace cuda
} // namespace tensorplay
