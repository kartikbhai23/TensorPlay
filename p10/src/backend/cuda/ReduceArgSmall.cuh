#pragma once

#include "ReduceEngine.cuh"

#include <cub/block/block_reduce.cuh>

namespace tensorplay::cuda::reduce_family {

constexpr int kSmallArgThreads = 256;

template <typename T, typename Ops>
__global__ void small_arg_reduce_kernel(
        const T* input, int count, unsigned long long identity, int64_t* output) {
    using Key = unsigned long long;
    using BlockReduce = cub::BlockReduce<Key, kSmallArgThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    const Ops ops{};
    Key key = identity;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        key = ops.reduce(key, input[i], i);
    }
    key = BlockReduce(storage).Reduce(
        key, [ops] __device__(Key a, Key b) { return ops.combine(a, b); });
    if (threadIdx.x == 0) *output = ops.project(key);
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
