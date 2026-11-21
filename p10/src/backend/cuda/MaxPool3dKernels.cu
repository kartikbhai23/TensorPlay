#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

template <typename T> struct PoolMath { using type = T; };
template <> struct PoolMath<tensorplay::Half> { using type = float; };
template <> struct PoolMath<tensorplay::BFloat16> { using type = float; };

int64_t pool_grid_blocks(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

// The per-output window walk is a long serial chain, so a small output pool
// needs the grid spread over several waves of multiprocessors rather than one
// wide block per 256 outputs.  Halving the block keeps one output per thread
// while multiplying the block count, down to a single warp per block.
int pool_threads(int64_t total) {
    constexpr int widest = 256;
    constexpr int narrowest = 32;
    static thread_local int sm_count = 0;
    if (sm_count == 0) {
        checkCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                         currentDevice()),
                  "max_pool multiprocessor count");
    }
    int threads = widest;
    while (threads > narrowest &&
           (total + threads - 1) / threads < 4 * sm_count) {
        threads >>= 1;
    }
    return threads;
}

int64_t pool_div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t pool_output_shape(int64_t in, int64_t k, int64_t pad,
                          int64_t stride, int64_t dilation, bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    int64_t out = pool_div_rtn(
        in + 2 * pad - dilation * (k - 1) - 1 +
            (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= in + pad) --out;
    return out;
}

std::vector<int64_t> pool_expand_param(const std::vector<int64_t>& list,
                                       const char* name, int64_t n,
                                       int64_t default_val) {
    if (list.empty()) return std::vector<int64_t>(n, default_val);
    if (list.size() == 1) return std::vector<int64_t>(n, list[0]);
    if (static_cast<int64_t>(list.size()) != n) {
        TP_THROW(ValueError,
                 std::string(name) + ": expected " + std::to_string(n) + " values");
    }
    return list;
}

Tensor& write_pooling_out(const char* op, Tensor value, Tensor& out) {
    if (!out.defined()) {
        out = std::move(value);
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

// The index width is the launch's choice: the four divisions that decode an
// output position are the expensive part of this kernel, and a 32-bit divide
// costs a fraction of the 64-bit one.  The wide form keeps inputs whose
// position arithmetic does not fit in 32 bits.
template <typename T, typename M, typename DimT>
__global__ void max_pool3d_wi_fwd_kernel(
    DimT total, DimT D_in, DimT H_in, DimT W_in,
    DimT D_out, DimT H_out, DimT W_out,
    DimT kD, DimT kH, DimT kW, DimT sD, DimT sH, DimT sW,
    DimT pD, DimT pH, DimT pW, DimT dD, DimT dH, DimT dW,
    const T* __restrict__ input, T* __restrict__ output,
    int64_t* __restrict__ indices) {
    DimT i = blockIdx.x * DimT(blockDim.x) + threadIdx.x;
    const DimT span = DimT(blockDim.x) * DimT(gridDim.x);
    const DimT out_spatial = D_out * H_out * W_out;
    for (; i < total; i += span) {
        const DimT w = i % W_out;
        const DimT h = (i / W_out) % H_out;
        const DimT d = (i / (W_out * H_out)) % D_out;
        const DimT nc = i / out_spatial;
        const T* vol = input + static_cast<int64_t>(nc) * D_in * H_in * W_in;
        // The window is contiguous per axis once walked with the dilation as
        // the step: clamping its ends and skipping the padded prefix removes
        // the per-tap bound checks from the hot loop entirely.
        const DimT d_begin_raw = d * sD - pD;
        const DimT h_begin_raw = h * sH - pH;
        const DimT w_begin_raw = w * sW - pW;
        const DimT d_end = std::min(d_begin_raw + (kD - 1) * dD + 1, D_in);
        const DimT h_end = std::min(h_begin_raw + (kH - 1) * dH + 1, H_in);
        const DimT w_end = std::min(w_begin_raw + (kW - 1) * dW + 1, W_in);
        DimT d_begin = d_begin_raw;
        DimT h_begin = h_begin_raw;
        DimT w_begin = w_begin_raw;
        while (d_begin < 0) d_begin += dD;
        while (h_begin < 0) h_begin += dH;
        while (w_begin < 0) w_begin += dW;

        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (DimT di = d_begin; di < d_end; di += dD) {
            const DimT plane = di * H_in;
            for (DimT hi = h_begin; hi < h_end; hi += dH) {
                const DimT row = (plane + hi) * W_in;
                for (DimT wi = w_begin; wi < w_end; wi += dW) {
                    const DimT idx = row + wi;
                    const M val = static_cast<M>(vol[idx]);
                    if ((val > max_val) || ::isnan(val)) {
                        max_val = val;
                        max_idx = static_cast<int64_t>(idx);
                    }
                }
            }
        }
        output[i] = static_cast<T>(max_val);
        indices[i] = max_idx;
    }
}

// Dispatches one launch of the forward kernel at the narrowest index width
// that holds every position this input can address.
template <typename DimT>
void launch_max_pool3d_wi_fwd(
    DType dtype, int64_t blocks, int threads, cudaStream_t stream,
    int64_t total, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW, int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW,
    const void* input, void* output, int64_t* indices) {
#define TP_POOL3D_FWD_CASE(ctype, name)                                       \
    case DType::name: {                                                       \
        using M = typename PoolMath<ctype>::type;                             \
        max_pool3d_wi_fwd_kernel<ctype, M, DimT>                              \
            <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(          \
                static_cast<DimT>(total), static_cast<DimT>(D_in),            \
                static_cast<DimT>(H_in), static_cast<DimT>(W_in),             \
                static_cast<DimT>(D_out), static_cast<DimT>(H_out),           \
                static_cast<DimT>(W_out), static_cast<DimT>(kD),              \
                static_cast<DimT>(kH), static_cast<DimT>(kW),                 \
                static_cast<DimT>(sD), static_cast<DimT>(sH),                 \
                static_cast<DimT>(sW), static_cast<DimT>(pD),                 \
                static_cast<DimT>(pH), static_cast<DimT>(pW),                 \
                static_cast<DimT>(dD), static_cast<DimT>(dH),                 \
                static_cast<DimT>(dW),                                        \
                static_cast<const ctype*>(input),                             \
                static_cast<ctype*>(output), indices);                        \
        break;                                                                \
    }
    switch (dtype) {
        TP_POOL3D_FWD_CASE(float, Float32)
        TP_POOL3D_FWD_CASE(double, Float64)
        TP_POOL3D_FWD_CASE(tensorplay::Half, Float16)
        TP_POOL3D_FWD_CASE(tensorplay::BFloat16, BFloat16)
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d_with_indices CUDA supports "
                     "Float32/Float64/Float16/BFloat16 only");
    }
#undef TP_POOL3D_FWD_CASE
}

template <typename T>
__global__ void max_pool_wi_bwd_kernel(
    int64_t total, int64_t out_spatial, int64_t in_plane,
    const T* __restrict__ grad_output, const int64_t* __restrict__ indices,
    T* __restrict__ grad_input) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        const int64_t idx = indices[i];
        if (idx < 0) continue;
        const int64_t nc = i / out_spatial;
        gpuAtomicAdd(grad_input + nc * in_plane + idx, grad_output[i]);
    }
}

}

#define POOL_CUDA_DISPATCH(ctype, name, ...)                                \
    case DType::name: {                                                     \
        using M = typename PoolMath<ctype>::type;                           \
        __VA_ARGS__;                                                        \
        break;                                                              \
    }

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    // A batch-less input has the same memory and the same per-channel walk as
    // its one-batch form, so it is answered directly: keeping the leading
    // dimension out of the shape saves building, wrapping and dropping it.
    if (input.dim() != 4 && input.dim() != 5) {
        TP_THROW(RuntimeError,
                 "max_pool3d_with_indices: Expected 4D or 5D input");
    }
    const bool batchless = input.dim() == 4;
    const Tensor input_c = input.contiguous();
    const int64_t N = batchless ? 1 : input_c.size(0);
    const int64_t C = input_c.size(batchless ? 0 : 1);
    const int64_t D_in = input_c.size(batchless ? 1 : 2);
    const int64_t H_in = input_c.size(batchless ? 2 : 3);
    const int64_t W_in = input_c.size(batchless ? 3 : 4);

    const auto ks = pool_expand_param(kernel_size, "max_pool3d kernel_size", 3, 1);
    const auto st = pool_expand_param(
        stride.empty() ? ks : stride, "max_pool3d stride", 3, ks[0]);
    const auto pd = pool_expand_param(padding, "max_pool3d padding", 3, 0);
    const auto dl = pool_expand_param(dilation, "max_pool3d dilation", 3, 1);
    const int64_t kD = ks[0], kH = ks[1], kW = ks[2];
    const int64_t sD = st[0], sH = st[1], sW = st[2];
    const int64_t pD = pd[0], pH = pd[1], pW = pd[2];
    const int64_t dD = dl[0], dH = dl[1], dW = dl[2];

    const int64_t D_out = pool_output_shape(D_in, kD, pD, sD, dD, ceil_mode);
    const int64_t H_out = pool_output_shape(H_in, kH, pH, sH, dH, ceil_mode);
    const int64_t W_out = pool_output_shape(W_in, kW, pW, sW, dW, ceil_mode);
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
        TP_THROW(RuntimeError, "max_pool3d: Calculated output size is too small");
    }

    const std::vector<int64_t> out_shape = batchless
        ? std::vector<int64_t>{C, D_out, H_out, W_out}
        : std::vector<int64_t>{N, C, D_out, H_out, W_out};
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t total = N * C * D_out * H_out * W_out;
    const int threads = pool_threads(total);
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();

    // Every position addressable by this launch fits in 32 bits unless one of
    // the extents or the output count says otherwise.
    const int64_t widest = std::max(
        {total, D_in, H_in, W_in, D_out, H_out, W_out,
         D_in * H_in * W_in, N * C * D_in * H_in * W_in});
    if (widest <= std::numeric_limits<int32_t>::max()) {
        launch_max_pool3d_wi_fwd<int32_t>(
            input.dtype(), blocks, threads, stream, total,
            D_in, H_in, W_in, D_out, H_out, W_out,
            kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
            input_c.const_data_ptr(), out.data_ptr(),
            indices.data_ptr<int64_t>());
    } else {
        launch_max_pool3d_wi_fwd<int64_t>(
            input.dtype(), blocks, threads, stream, total,
            D_in, H_in, W_in, D_out, H_out, W_out,
            kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW,
            input_c.const_data_ptr(), out.data_ptr(),
            indices.data_ptr<int64_t>());
    }
    return std::make_tuple(out, indices);
}

Tensor max_pool3d_cuda(const Tensor& input, const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (input.dim() == 4) {
        return max_pool3d_cuda(input.unsqueeze(0), kernel_size, stride, padding,
                               dilation, ceil_mode).squeeze(0);
    }
    return std::get<0>(max_pool3d_with_indices_cuda(
        input, kernel_size, stride, padding, dilation, ceil_mode));
}

Tensor max_pool3d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt);

Tensor max_pool3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& kernel_size,
                                const std::vector<int64_t>& stride,
                                const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation,
                                bool ceil_mode) {
    if (grad_output.dim() == 4 && input.dim() == 4) {
        return max_pool3d_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode).squeeze(0);
    }
    auto fw = max_pool3d_with_indices_cuda(
        input, kernel_size, stride, padding, dilation, ceil_mode);
    return max_pool3d_with_indices_backward_cuda(
        grad_output, input, kernel_size, stride, padding, dilation, ceil_mode,
        std::get<1>(fw));
}

Tensor max_pool3d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const std::optional<Tensor>& indices_opt) {
    (void)kernel_size;
    (void)stride;
    (void)padding;
    (void)dilation;
    (void)ceil_mode;
    if (!indices_opt.has_value() || !indices_opt->defined()) {
        TP_THROW(RuntimeError, "max_pool3d_with_indices_backward: indices is required");
    }
    if (input.dim() == 4) {
        return max_pool3d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0), kernel_size,
                   stride, padding, dilation, ceil_mode,
                   indices_opt->unsqueeze(0)).squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5) {
        TP_THROW(RuntimeError,
                 "max_pool3d_with_indices_backward: Expected 5D input and grad_output");
    }
    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices_opt->contiguous();
    Tensor grad_input = Tensor::zeros_like(input);
    const int64_t total = go.numel();
    const int64_t out_spatial = go.size(2) * go.size(3) * go.size(4);
    const int64_t in_plane = input.size(2) * input.size(3) * input.size(4);
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    switch (input.dtype()) {
        POOL_CUDA_DISPATCH(float, Float32,
            max_pool_wi_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<float>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<float>()))
        POOL_CUDA_DISPATCH(double, Float64,
            max_pool_wi_bwd_kernel<double><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<double>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<double>()))
        POOL_CUDA_DISPATCH(tensorplay::Half, Float16,
            (max_pool_wi_bwd_kernel<tensorplay::Half><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::Half>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::Half>())))
        POOL_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            (max_pool_wi_bwd_kernel<tensorplay::BFloat16><<<blocks, threads, 0, stream>>>(
                total, out_spatial, in_plane, go.data_ptr<tensorplay::BFloat16>(),
                idx.data_ptr<int64_t>(), grad_input.data_ptr<tensorplay::BFloat16>())))
        default:
            TP_THROW(NotImplementedError,
                     "max_pool3d_with_indices_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

std::tuple<Tensor, Tensor> interop_max_pool3d_with_indices_out_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode, Tensor& out,
    Tensor& indices) {
    auto result = max_pool3d_with_indices_cuda(
        self, kernel_size, stride, padding, dilation, ceil_mode);
    write_pooling_out("max_pool3d_with_indices", std::get<0>(result), out);
    write_pooling_out("max_pool3d_with_indices", std::get<1>(result), indices);
    return {out, indices};
}

Tensor& interop_max_pool3d_with_indices_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& self,
    const std::vector<int64_t>& kernel_size, const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation,
    bool ceil_mode, const Tensor& indices, Tensor& grad_input) {
    return write_pooling_out(
        "max_pool3d_with_indices_backward",
        max_pool3d_with_indices_backward_cuda(
            grad_output, self, kernel_size, stride, padding, dilation,
            ceil_mode, indices),
        grad_input);
}

#undef POOL_CUDA_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CUDA, MaxPool3dKernels) {
    m.impl("max_pool3d_with_indices", max_pool3d_with_indices_cuda);
    m.impl("max_pool3d_with_indices_backward", max_pool3d_with_indices_backward_cuda);
    m.impl("max_pool3d_backward", max_pool3d_backward_cuda);
    m.impl("max_pool3d_with_indices.out", interop_max_pool3d_with_indices_out_cuda);
    m.impl("max_pool3d_with_indices_backward.grad_input",
           interop_max_pool3d_with_indices_backward_grad_input_cuda);
}

}
}
