#pragma once

#include "ReduceEngine.cuh"

namespace tensorplay::cuda::reduce_family {

constexpr int kSmallArgThreads = 256;

// The block folds through the shuffle tree and stages one key per warp: the
// generic block-reduce body drags a large template into a launch that only
// reads a short contiguous run.
template <typename T, typename Ops>
__global__ void __launch_bounds__(kSmallArgThreads)
small_arg_reduce_kernel(
        const T* __restrict__ input, int count, unsigned long long identity,
        int64_t* __restrict__ output) {
    constexpr int kWarps = kSmallArgThreads / reduction::kWarpSize;
    using Key = unsigned long long;
    const Ops ops{};
    Key key = identity;
    for (int i = threadIdx.x; i < count; i += kSmallArgThreads) {
        key = ops.reduce(key, input[i], i);
    }
    for (int offset = reduction::kWarpSize / 2; offset > 0; offset >>= 1) {
        key = ops.combine(key, reduction::reduce_warp_shuffle_down(key, 0xffffffffu, offset));
    }
    __shared__ Key warp_key[kWarps];
    if ((threadIdx.x & (reduction::kWarpSize - 1)) == 0) {
        warp_key[threadIdx.x / reduction::kWarpSize] = key;
    }
    __syncthreads();
    if (threadIdx.x < reduction::kWarpSize) {
        Key value = threadIdx.x < kWarps ? warp_key[threadIdx.x] : identity;
        for (int offset = reduction::kWarpSize / 2; offset > 0; offset >>= 1) {
            value = ops.combine(value,
                reduction::reduce_warp_shuffle_down(value, 0xffffffffu, offset));
        }
        if (threadIdx.x == 0) *output = ops.project(value);
    }
}

inline bool use_small_arg_reduce(const Tensor& input, const ReductionSpec& spec) {
    // A single block can reduce a short contiguous input without staging.
    const auto count = input.numel();
    return count > 0 && count <= 16 * kSmallArgThreads &&
           spec.reduced_numel == count && input.is_contiguous();
}

template <typename T, typename Ops>
Tensor small_arg_reduce(
        const Tensor& input, const ReductionSpec& spec, bool keepdim,
        unsigned long long identity) {
    Tensor result = Tensor::empty(
        reduction_output_shape(input, spec, keepdim), DType::Int64, input.device());
    small_arg_reduce_kernel<T, Ops><<<1, kSmallArgThreads, 0, getCurrentCUDAStream().stream()>>>(
        input.data_ptr<T>(), static_cast<int>(input.numel()), identity, result.data_ptr<int64_t>());
    checkCuda(cudaGetLastError(), "CUDA argument reduction launch");
    return result;
}

} // namespace tensorplay::cuda::reduce_family
