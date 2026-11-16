#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "GPUPrimitives.cuh"
#include "Atomic.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

__device__ __forceinline__ void atomic_add_rel(int64_t* addr, int64_t v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(float* addr, float v) {
    gpuAtomicAdd(addr, v);
}
__device__ __forceinline__ void atomic_add_rel(double* addr, double v) {
    gpuAtomicAdd(addr, v);
}

template <typename T>
__global__ void minmax_reduce_kernel(int64_t n, const T* x, T* min_out,
                                     T* max_out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    for (; i < n; i += stride) {
        local_min = ::min(local_min, x[i]);
        local_max = ::max(local_max, x[i]);
    }
    __shared__ T min_shm[kThreads];
    __shared__ T max_shm[kThreads];
    int tid = threadIdx.x;
    min_shm[tid] = local_min;
    max_shm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            min_shm[tid] = ::min(min_shm[tid], min_shm[tid + s]);
            max_shm[tid] = ::max(max_shm[tid], max_shm[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_out[blockIdx.x] = min_shm[0];
        max_out[blockIdx.x] = max_shm[0];
    }
}

// Vectorized form of the minmax reduction for 8-byte values: each thread
// folds four 16-byte loads per grid stride into min/max updates, so the
// input is read with a quarter of the load instructions of the scalar loop
// and several loads stay in flight per thread.  Requires the base pointer to
// be 16-byte aligned; the caller picks this kernel only when that holds, and
// the scalar kernel remains the fallback.
template <typename T>
__global__ void minmax_reduce_vec_kernel(int64_t n, const T* x, T* min_out,
                                         T* max_out) {
    const int64_t nthreads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    if constexpr (sizeof(T) == 8) {
        const long4* x4 = reinterpret_cast<const long4*>(x);
        const int64_t n4 = n / 4;
        for (int64_t base = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
             base < n4; base += nthreads * 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t at = base + static_cast<int64_t>(j) * blockDim.x;
                if (at < n4) {
                    const long4 v = x4[at];
                    local_min = ::min(local_min, static_cast<T>(v.x));
                    local_min = ::min(local_min, static_cast<T>(v.y));
                    local_min = ::min(local_min, static_cast<T>(v.z));
                    local_min = ::min(local_min, static_cast<T>(v.w));
                    local_max = ::max(local_max, static_cast<T>(v.x));
                    local_max = ::max(local_max, static_cast<T>(v.y));
                    local_max = ::max(local_max, static_cast<T>(v.z));
                    local_max = ::max(local_max, static_cast<T>(v.w));
                }
            }
        }
        const int64_t tail = n4 * 4;
        for (int64_t i = tail + static_cast<int64_t>(threadIdx.x); i < n;
             i += static_cast<int64_t>(blockDim.x)) {
            local_min = ::min(local_min, x[i]);
            local_max = ::max(local_max, x[i]);
        }
    } else {
        for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             i < n; i += nthreads) {
            local_min = ::min(local_min, x[i]);
            local_max = ::max(local_max, x[i]);
        }
    }
    __shared__ T min_shm[kThreads];
    __shared__ T max_shm[kThreads];
    const int tid = threadIdx.x;
    min_shm[tid] = local_min;
    max_shm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            min_shm[tid] = ::min(min_shm[tid], min_shm[tid + s]);
            max_shm[tid] = ::max(max_shm[tid], max_shm[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_out[blockIdx.x] = min_shm[0];
        max_out[blockIdx.x] = max_shm[0];
    }
}

// Folds the per-block partials of minmax_reduce_kernel into the final pair.
template <typename T>
__global__ void minmax_combine_kernel(int64_t nblocks, const T* min_in,
                                      const T* max_in, T* min_out, T* max_out) {
    __shared__ T min_shm[kThreads];
    __shared__ T max_shm[kThreads];
    T local_min = std::numeric_limits<T>::max();
    T local_max = std::numeric_limits<T>::lowest();
    for (int64_t i = threadIdx.x; i < nblocks; i += blockDim.x) {
        local_min = ::min(local_min, min_in[i]);
        local_max = ::max(local_max, max_in[i]);
    }
    int tid = threadIdx.x;
    min_shm[tid] = local_min;
    max_shm[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            min_shm[tid] = ::min(min_shm[tid], min_shm[tid + s]);
            max_shm[tid] = ::max(max_shm[tid], max_shm[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        min_out[0] = min_shm[0];
        max_out[0] = max_shm[0];
    }
}

template <typename T>
__global__ void bincount_count_kernel(int64_t n, const T* x, int64_t* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], static_cast<int64_t>(1));
}

template <typename W>
__global__ void bincount_weighted_kernel(int64_t n, const int64_t* x, const W* wp, W* bins) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) atomic_add_rel(&bins[x[i]], wp[i]);
}

// A bounded bin count fits one histogram per block in shared memory: the
// block folds its elements there and adds each bin to the output once, so
// the per-element atomics stay on chip and only one pass per block reaches
// global memory.  A wider histogram keeps the flat global pass.
constexpr int64_t kSharedBinBudgetBytes = 16 * 1024;

template <typename T, typename B>
__global__ void bincount_shared_kernel(int64_t n, const T* x, const B* wp,
                                       B* bins, int nbins) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    B* hist = reinterpret_cast<B*>(smem_raw);
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) hist[b] = B(0);
    __syncthreads();
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const B v = wp != nullptr ? wp[i] : B(1);
        atomic_add_rel(&hist[x[i]], v);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) {
        if (hist[b] != B(0)) atomic_add_rel(&bins[b], hist[b]);
    }
}

inline int bincount_shared_blocks(int64_t n) {
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    return static_cast<int>(std::min<int64_t>(std::max<int64_t>(blocks, 1), 512));
}

template <typename B>
inline bool bincount_shared_ok(int64_t nbins) {
    return nbins <= std::numeric_limits<int>::max() &&
           nbins * static_cast<int64_t>(sizeof(B)) <= kSharedBinBudgetBytes;
}

}

Tensor bincount_cuda(const Tensor& self, const std::optional<Tensor>& weights_opt, int64_t minlength) {
    Tensor weights = weights_opt.value_or(Tensor());
    globalContext().alertNotDeterministic("bincount_cuda");
    if (minlength < 0) TP_THROW(RuntimeError, "minlength should be >= 0");
    if (isFloatingType(self.dtype())) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    Tensor inp = self.to(DType::Int64).contiguous();
    int64_t n = inp.numel();
    if (self.dim() == 1 && n == 0) {
        return Tensor::zeros({minlength}, DType::Int64, self.device());
    }
    if (self.dim() != 1) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    bool has_weights = weights.defined() && weights.numel() > 0;
    if (has_weights && (weights.dim() != 1 || weights.size(0) != self.size(0))) {
        TP_THROW(RuntimeError, "weights should be 1-d and have the same length as input");
    }
    auto stream = getCurrentCUDAStream().stream();
    // The value scan is spread over many blocks; a tiny second kernel folds
    // the partials so only the final pair crosses to the host.
    int64_t partial_blocks = (n + kThreads - 1) / kThreads;
    partial_blocks = std::min<int64_t>(std::max<int64_t>(partial_blocks, 1), 4096);
    Tensor partial = Tensor::empty({partial_blocks * 2}, DType::Int64, self.device());
    Tensor bounds_d = Tensor::empty({2}, DType::Int64, self.device());
    const int64_t* xp = inp.data_ptr<int64_t>();
    const bool aligned16 = (reinterpret_cast<uintptr_t>(xp) & 0xf) == 0;
    if (aligned16) {
        const int64_t n4 = n / 4;
        const int64_t vec_blocks =
            (n4 + kThreads * 4 - 1) / (kThreads * 4);
        partial_blocks = std::min<int64_t>(
            std::max<int64_t>(vec_blocks, 1), 4096);
        minmax_reduce_vec_kernel<int64_t>
            <<<static_cast<unsigned>(partial_blocks), kThreads, 0, stream>>>(
                n, xp, partial.data_ptr<int64_t>(),
                partial.data_ptr<int64_t>() + partial_blocks);
    } else {
        minmax_reduce_kernel<int64_t>
            <<<static_cast<unsigned>(partial_blocks), kThreads, 0, stream>>>(
                n, xp, partial.data_ptr<int64_t>(),
                partial.data_ptr<int64_t>() + partial_blocks);
    }
    minmax_combine_kernel<int64_t><<<1, kThreads, 0, stream>>>(
        partial_blocks, partial.data_ptr<int64_t>(),
        partial.data_ptr<int64_t>() + partial_blocks, bounds_d.data_ptr<int64_t>(),
        bounds_d.data_ptr<int64_t>() + 1);
    CUDA_CHECK(cudaGetLastError());
    int64_t bounds[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(bounds, bounds_d.data_ptr<int64_t>(),
                          sizeof(bounds), cudaMemcpyDeviceToHost));
    const int64_t min_v = bounds[0];
    const int64_t max_v = bounds[1];
    if (min_v < 0) {
        TP_THROW(RuntimeError, "bincount only supports 1-d non-negative integral inputs.");
    }
    if (max_v >= std::numeric_limits<int64_t>::max()) {
        TP_THROW(RuntimeError, "maximum value of input overflowed");
    }
    int64_t nbins = std::max(max_v + 1, minlength);
    if (has_weights) {
        if (weights.dtype() == DType::Float32) {
            Tensor rf = Tensor::zeros({nbins}, DType::Float32, self.device());
            Tensor w = weights.contiguous();
            if (bincount_shared_ok<float>(nbins)) {
                bincount_shared_kernel<int64_t, float>
                    <<<bincount_shared_blocks(n), kThreads,
                       static_cast<size_t>(nbins) * sizeof(float), stream>>>(
                        n, inp.data_ptr<int64_t>(), w.data_ptr<float>(),
                        rf.data_ptr<float>(), static_cast<int>(nbins));
            } else {
                bincount_weighted_kernel<float><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    n, inp.data_ptr<int64_t>(), w.data_ptr<float>(), rf.data_ptr<float>());
            }
            CUDA_CHECK(cudaGetLastError());
            return rf;
        }
        Tensor rf = Tensor::zeros({nbins}, DType::Float64, self.device());
        Tensor w = weights.to(DType::Float64).contiguous();
        if (bincount_shared_ok<double>(nbins)) {
            bincount_shared_kernel<int64_t, double>
                <<<bincount_shared_blocks(n), kThreads,
                   static_cast<size_t>(nbins) * sizeof(double), stream>>>(
                    n, inp.data_ptr<int64_t>(), w.data_ptr<double>(),
                    rf.data_ptr<double>(), static_cast<int>(nbins));
        } else {
            bincount_weighted_kernel<double><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                n, inp.data_ptr<int64_t>(), w.data_ptr<double>(), rf.data_ptr<double>());
        }
        CUDA_CHECK(cudaGetLastError());
        return rf;
    }
    Tensor result = Tensor::zeros({nbins}, DType::Int64, self.device());
    if (bincount_shared_ok<int64_t>(nbins)) {
        bincount_shared_kernel<int64_t, int64_t>
            <<<bincount_shared_blocks(n), kThreads,
               static_cast<size_t>(nbins) * sizeof(int64_t), stream>>>(
                n, inp.data_ptr<int64_t>(), nullptr, result.data_ptr<int64_t>(),
                static_cast<int>(nbins));
    } else {
        bincount_count_kernel<int64_t><<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            n, inp.data_ptr<int64_t>(), result.data_ptr<int64_t>());
    }
    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, BincountKernels) {
    m.impl("bincount", bincount_cuda);
}

}
}
