#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
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

bool is_adaptive_pool_cuda_dtype(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float64 ||
           dtype == DType::Float16 || dtype == DType::BFloat16;
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

// The index width is the launch's choice: the four divisions that decode an
// output position are the expensive part of this kernel, and a 32-bit divide
// costs a fraction of the 64-bit one.  The wide form keeps inputs whose
// position arithmetic does not fit in 32 bits.
template <typename T, typename M, typename DimT>
__global__ void adaptive_max_pool3d_fwd_kernel(
    DimT total, DimT D, DimT H, DimT W,
    DimT oD, DimT oH, DimT oW,
    const T* __restrict__ input, T* __restrict__ output,
    int64_t* __restrict__ indices) {
    DimT i = blockIdx.x * DimT(blockDim.x) + threadIdx.x;
    const DimT span = DimT(blockDim.x) * DimT(gridDim.x);
    const DimT out_spatial = oD * oH * oW;
    for (; i < total; i += span) {
        const DimT w = i % oW;
        const DimT h = (i / oW) % oH;
        const DimT d = (i / (oW * oH)) % oD;
        const DimT nc = i / out_spatial;
        const T* vol = input + static_cast<int64_t>(nc) * D * H * W;
        const DimT ds = d * D / oD;
        const DimT de = 1 + (((d + 1) * D) - 1) / oD;
        const DimT hs = h * H / oH;
        const DimT he = 1 + (((h + 1) * H) - 1) / oH;
        const DimT ws = w * W / oW;
        const DimT we = 1 + (((w + 1) * W) - 1) / oW;
        M max_val = -std::numeric_limits<M>::infinity();
        int64_t max_idx = -1;
        for (DimT z = ds; z < de; ++z) {
            for (DimT y = hs; y < he; ++y) {
                for (DimT x = ws; x < we; ++x) {
                    const int64_t idx =
                        (static_cast<int64_t>(z) * H + y) * W + x;
                    const M val = static_cast<M>(vol[idx]);
                    if ((val > max_val) || ::isnan(val)) {
                        max_val = val;
                        max_idx = idx;
                    }
                }
            }
        }
        output[i] = static_cast<T>(max_val);
        indices[i] = max_idx;
    }
}

}

#define POOL_CUDA_DISPATCH(ctype, name, ...)                                \
    case DType::name: {                                                     \
        using M = typename PoolMath<ctype>::type;                           \
        __VA_ARGS__;                                                        \
        break;                                                              \
    }

std::tuple<Tensor, Tensor> adaptive_max_pool3d_with_indices_cuda(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    // A batch-less input has the same memory and the same per-channel walk as
    // its one-batch form, so it is answered directly: keeping the leading
    // dimension out of the shape saves building, wrapping and dropping it.
    if (input.dim() != 4 && input.dim() != 5) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool3d: Expected 4D or 5D input");
    }
    const bool batchless = input.dim() == 4;
    const Tensor input_c = input.contiguous();
    const int64_t N = batchless ? 1 : input_c.size(0);
    const int64_t C = input_c.size(batchless ? 0 : 1);
    const int64_t D = input_c.size(batchless ? 1 : 2);
    const int64_t H = input_c.size(batchless ? 2 : 3);
    const int64_t W = input_c.size(batchless ? 3 : 4);
    if (output_size.size() != 3) {
        TP_THROW(ValueError,
                 "adaptive_max_pool3d: output_size must have three elements");
    }
    const int64_t oD = output_size[0], oH = output_size[1], oW = output_size[2];
    if (oD <= 0 || oH <= 0 || oW <= 0) {
        TP_THROW(RuntimeError, "adaptive_max_pool3d: Invalid output size");
    }
    const std::vector<int64_t> out_shape = batchless
        ? std::vector<int64_t>{C, oD, oH, oW}
        : std::vector<int64_t>{N, C, oD, oH, oW};
    Tensor out = Tensor::empty(out_shape, input.dtype(), input.device());
    Tensor indices = Tensor::empty(out_shape, DType::Int64, input.device());
    const int64_t total = N * C * oD * oH * oW;
    const int threads = 256;
    const int64_t blocks = pool_grid_blocks(total, threads);
    const auto stream = getCurrentCUDAStream().stream();
    // Every position addressable by this launch fits in 32 bits unless one of
    // the extents or the output count says otherwise.
    const int64_t extents[] = {total, D, H, W, oD, oH, oW,
                               D * H * W, N * C * D * H * W};
    const int64_t widest =
        *std::max_element(std::begin(extents), std::end(extents));
    if (widest <= std::numeric_limits<int32_t>::max()) {
#define TP_AMP3D_FWD_CASE(ctype, name)                                       \
        case DType::name: {                                                   \
            using M = typename PoolMath<ctype>::type;                         \
            adaptive_max_pool3d_fwd_kernel<ctype, M, int32_t>                  \
                <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(      \
                    static_cast<int32_t>(total), static_cast<int32_t>(D),     \
                    static_cast<int32_t>(H), static_cast<int32_t>(W),         \
                    static_cast<int32_t>(oD), static_cast<int32_t>(oH),       \
                    static_cast<int32_t>(oW),                                 \
                    static_cast<const ctype*>(input_c.const_data_ptr()),      \
                    static_cast<ctype*>(out.data_ptr()),                             \
                    indices.data_ptr<int64_t>());            \
            break;                                                            \
        }
        switch (input.dtype()) {
            TP_AMP3D_FWD_CASE(float, Float32)
            TP_AMP3D_FWD_CASE(double, Float64)
            TP_AMP3D_FWD_CASE(tensorplay::Half, Float16)
            TP_AMP3D_FWD_CASE(tensorplay::BFloat16, BFloat16)
            default:
                TP_THROW(NotImplementedError,
                         "adaptive_max_pool3d CUDA supports "
                         "Float32/Float64/Float16/BFloat16 only");
        }
#undef TP_AMP3D_FWD_CASE
    } else {
#define TP_AMP3D_FWD_CASE(ctype, name)                                       \
        case DType::name: {                                                   \
            using M = typename PoolMath<ctype>::type;                         \
            adaptive_max_pool3d_fwd_kernel<ctype, M, int64_t>                  \
                <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(      \
                    total, D, H, W, oD, oH, oW,                               \
                    static_cast<const ctype*>(input_c.const_data_ptr()),      \
                    static_cast<ctype*>(out.data_ptr()),                             \
                    indices.data_ptr<int64_t>());            \
            break;                                                            \
        }
        switch (input.dtype()) {
            TP_AMP3D_FWD_CASE(float, Float32)
            TP_AMP3D_FWD_CASE(double, Float64)
            TP_AMP3D_FWD_CASE(tensorplay::Half, Float16)
            TP_AMP3D_FWD_CASE(tensorplay::BFloat16, BFloat16)
            default:
                TP_THROW(NotImplementedError,
                         "adaptive_max_pool3d CUDA supports "
                         "Float32/Float64/Float16/BFloat16 only");
        }
#undef TP_AMP3D_FWD_CASE
    }
    return std::make_tuple(out, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool3d_cuda(
    const Tensor& input, const std::vector<int64_t>& output_size) {
    return adaptive_max_pool3d_with_indices_cuda(input, output_size);
}

Tensor adaptive_max_pool3d_with_indices_backward_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& indices) {
    if (input.dim() == 4 && grad_output.dim() == 4 && indices.dim() == 4) {
        return adaptive_max_pool3d_with_indices_backward_cuda(
                   grad_output.unsqueeze(0), input.unsqueeze(0),
                   indices.unsqueeze(0)).squeeze(0);
    }
    if (grad_output.dim() != 5 || input.dim() != 5 || indices.dim() != 5) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool3d_backward: Expected 4D or 5D tensors");
    }
    if (!is_adaptive_pool_cuda_dtype(input.dtype()) ||
        input.dtype() != grad_output.dtype()) {
        TP_THROW(NotImplementedError,
                 "adaptive_max_pool3d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    if (indices.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "adaptive_max_pool3d_backward: indices must be Int64");
    }
    if (input.device() != grad_output.device() ||
        input.device() != indices.device()) {
        TP_THROW(DeviceMismatchError,
                 "adaptive_max_pool3d_backward: all tensors must be on the same device");
    }
    if (indices.shape() != grad_output.shape()) {
        TP_THROW(RuntimeError,
                 "adaptive_max_pool3d_backward: indices shape must match grad_output shape");
    }

    const Tensor go = grad_output.contiguous();
    const Tensor idx = indices.contiguous();
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
                     "adaptive_max_pool3d_backward CUDA supports Float32/Float64/Float16/BFloat16 only");
    }
    return grad_input;
}

std::tuple<Tensor, Tensor> interop_adaptive_max_pool3d_out_cuda(
    const Tensor& self, const std::vector<int64_t>& output_size,
    Tensor& out, Tensor& indices) {
    auto result = adaptive_max_pool3d_with_indices_cuda(self, output_size);
    write_pooling_out("adaptive_max_pool3d", std::get<0>(result), out);
    write_pooling_out("adaptive_max_pool3d", std::get<1>(result), indices);
    return {out, indices};
}

// The scatter is driven by the indices the forward saved: one linear offset
// into each (D, H, W) volume per output element.
Tensor adaptive_max_pool3d_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& indices) {
    return adaptive_max_pool3d_with_indices_backward_cuda(grad_output, input, indices);
}

Tensor& interop_adaptive_max_pool3d_backward_grad_input_cuda(
    const Tensor& grad_output, const Tensor& input, const Tensor& indices,
    Tensor& grad_input) {
    return write_pooling_out(
        "adaptive_max_pool3d_backward",
        adaptive_max_pool3d_with_indices_backward_cuda(grad_output, input, indices),
        grad_input);
}

#undef POOL_CUDA_DISPATCH

TENSORPLAY_LIBRARY_IMPL(CUDA, AdaptiveMaxPool3dKernels) {
    m.impl("adaptive_max_pool3d", adaptive_max_pool3d_cuda);
    m.impl("adaptive_max_pool3d_backward", adaptive_max_pool3d_backward_cuda);
    m.impl("adaptive_max_pool3d.out", interop_adaptive_max_pool3d_out_cuda);
    m.impl("adaptive_max_pool3d_backward.grad_input",
           interop_adaptive_max_pool3d_backward_grad_input_cuda);
}

}
}
