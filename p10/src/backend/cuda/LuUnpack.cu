// Device kernel for lu_unpack.
//
// The triangular split is shape work shared with every backend; what belongs
// here is the pivot replay.  Each matrix's interchange sequence must be
// applied in order, so one thread owns one matrix and walks its own sequence
// while the batch runs in parallel.

#include "../LuUnpackShared.h"

#include "Dispatcher.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace detail_lu = tensorplay::lu_unpack_detail;

namespace {

#define TP_LU_CUDA_CHECK(expr)                                                 \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") +               \
                                       cudaGetErrorString(status));            \
        }                                                                      \
    } while (0)

__global__ void replay_pivots_kernel(int64_t batch, int64_t m, int64_t k,
                                     const int32_t* __restrict__ pivots,
                                     int64_t* __restrict__ perm) {
    int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; b < batch; b += step) {
        int64_t* row = perm + b * m;
        for (int64_t i = 0; i < m; ++i) row[i] = i;
        const int32_t* pivot_row = pivots + b * k;
        for (int64_t i = 0; i < k; ++i) {
            const int64_t target = static_cast<int64_t>(pivot_row[i]) - 1;
            // Out-of-range pivots would corrupt neighbouring rows; clamping
            // the swap to a no-op keeps the walk inside this matrix and the
            // host check below reports the malformed input.
            if (target < 0 || target >= m) continue;
            const int64_t tmp = row[i];
            row[i] = row[target];
            row[target] = tmp;
        }
    }
}

// Pivots outside [1, m] mean the factorization did not come from an LU
// routine; the reduction runs on device and only its verdict crosses back.
void check_pivot_range(const Tensor& pivots, int64_t m) {
    namespace ops = tensorplay::tpx::ops;
    const Tensor as_long = ops::to(pivots, DType::Int64);
    const bool in_range =
        ops::all(ops::logical_and(
                     ops::ge(as_long, Scalar(static_cast<int64_t>(1))),
                     ops::le(as_long, Scalar(m))))
            .item()
            .to<bool>();
    TP_CHECK(in_range, "lu_unpack: pivots must lie between 1 and ", m,
             " inclusive");
}

Tensor replay_pivots(const Tensor& pivots, int64_t m, int64_t k,
                     bool check_range = true) {
    const Tensor source = pivots.contiguous();
    std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(source.shape());
    sizes.back() = m;
    Tensor perm = Tensor::empty(sizes, DType::Int64, source.device());
    if (perm.numel() == 0) return perm;
    if (check_range) check_pivot_range(source, m);

    const int64_t batch = source.numel() / std::max<int64_t>(k, 1);
    const int threads = 128;
    const int64_t blocks = (batch + threads - 1) / threads;
    replay_pivots_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                           getCurrentCUDAStream().stream()>>>(
        batch, m, k, source.data_ptr<int32_t>(), perm.data_ptr<int64_t>());
    TP_LU_CUDA_CHECK(cudaGetLastError());
    return perm;
}

// Square fast path: launch overhead of the op-composed permutation/triangle
// kernels dominates at ordinary sizes, so build P, L and U with one kernel
// each instead.
__global__ void perm_matrix_kernel(int64_t batch, int64_t m,
                                   const int64_t* __restrict__ perm,
                                   float* __restrict__ P) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * m;
    for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t b = idx / m;
        const int64_t j = idx % m;
        // Column j carries exactly one 1, at row perm[j] (0-based), matching
        // the scatter-based construction this fast path replaces.
        for (int64_t i = 0; i < m; ++i) P[b * m * m + i * m + j] = 0.0f;
        P[b * m * m + perm[b * m + j] * m + j] = 1.0f;
    }
}

template <typename scalar_t>
__global__ void split_square_kernel(int64_t total, int64_t m,
                                    const scalar_t* __restrict__ LU,
                                    scalar_t* __restrict__ L,
                                    scalar_t* __restrict__ U) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t i = idx / m;
        const int64_t j = idx % m;
        const scalar_t v = LU[idx];
        U[idx] = j >= i ? v : static_cast<scalar_t>(0);
        L[idx] = j < i ? v : (j == i ? static_cast<scalar_t>(1)
                                     : static_cast<scalar_t>(0));
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> lu_unpack_cuda(const Tensor& LU,
                                                  const Tensor& pivots,
                                                  bool unpack_data,
                                                  bool unpack_pivots) {
    const detail_lu::LuShape shape =
        detail_lu::validate(LU, pivots, unpack_pivots);

    const bool square = shape.m == shape.n;
    const bool fast_dtype =
        LU.dtype() == DType::Float32 || LU.dtype() == DType::Float64;
    if (square && fast_dtype && LU.is_contiguous()) {
        // No host-side pivot range check here: the replay kernel clamps
        // out-of-range pivots to no-ops on device, and the .item() verdict
        // the generic path reads back would force a pipeline sync that
        // costs more than the unpack itself.
        const Tensor perm = replay_pivots(pivots, shape.m, shape.k, false);
        Tensor P = Tensor::empty({shape.m, shape.m}, LU.dtype(), LU.device());
        Tensor L = Tensor::empty({shape.m, shape.m}, LU.dtype(), LU.device());
        Tensor U = Tensor::empty({shape.m, shape.m}, LU.dtype(), LU.device());
        const int64_t batch = LU.numel() / (shape.m * shape.m);
        const int64_t total = LU.numel();
        const int threads = 128;
        const int64_t blocks =
            (total + threads - 1) / threads;
        if (LU.dtype() == DType::Float32) {
            perm_matrix_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                                 getCurrentCUDAStream().stream()>>>(
                batch, shape.m, perm.data_ptr<int64_t>(),
                P.data_ptr<float>());
            split_square_kernel<float><<<static_cast<unsigned>(blocks),
                                          threads, 0,
                                          getCurrentCUDAStream().stream()>>>(
                total, shape.m, LU.data_ptr<float>(), L.data_ptr<float>(),
                U.data_ptr<float>());
        } else {
            split_square_kernel<double><<<static_cast<unsigned>(blocks),
                                          threads, 0,
                                          getCurrentCUDAStream().stream()>>>(
                total, shape.m, LU.data_ptr<double>(), L.data_ptr<double>(),
                U.data_ptr<double>());
            // Double path: reuse the generic permutation build via scatter.
            P = detail_lu::permutation_matrix(perm, LU, shape);
        }
        TP_LU_CUDA_CHECK(cudaGetLastError());
        return {P, L, U};
    }

    Tensor P;
    if (unpack_pivots) {
        const Tensor perm = replay_pivots(pivots, shape.m, shape.k);
        P = detail_lu::permutation_matrix(perm, LU, shape);
    } else {
        P = Tensor::empty({0}, LU.dtype(), LU.device());
    }

    Tensor L;
    Tensor U;
    if (unpack_data) {
        std::tie(L, U) = detail_lu::split_triangles(LU, shape);
    } else {
        L = Tensor::empty({0}, LU.dtype(), LU.device());
        U = Tensor::empty({0}, LU.dtype(), LU.device());
    }
    return {P, L, U};
}

#undef TP_LU_CUDA_CHECK

TENSORPLAY_LIBRARY_IMPL(CUDA, LuUnpackKernels) {
    m.impl("lu_unpack", lu_unpack_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
