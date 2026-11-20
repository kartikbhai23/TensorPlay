// Device kernel for lu_unpack.
//
// Each matrix's interchange sequence must be applied in order. A block can
// keep the permutation of a small matrix in shared memory while it writes
// the requested factors.

#include "../LuUnpackShared.h"

#include "Dispatcher.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace detail_lu = tensorplay::lu_unpack_detail;

namespace {

constexpr int kSmallLuExtent = 64;
constexpr int kLuThreads = 128;

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
        if (k == 0) continue;
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

Tensor replay_pivots(const Tensor& pivots, int64_t m, int64_t k) {
    const Tensor source = pivots.contiguous();
    std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(source.shape());
    sizes.back() = m;
    Tensor perm = Tensor::empty(sizes, DType::Int64, source.device());
    if (perm.numel() == 0) return perm;
    if (k != 0) check_pivot_range(source, m);

    const int64_t batch = perm.numel() / m;
    const int threads = 128;
    const int64_t blocks = (batch + threads - 1) / threads;
    replay_pivots_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                           getCurrentCUDAStream().stream()>>>(
        batch, m, k, source.data_ptr<int32_t>(), perm.data_ptr<int64_t>());
    TP_LU_CUDA_CHECK(cudaGetLastError());
    return perm;
}

template <typename T>
__global__ void unpack_small_lu(
        const T* lu, const int32_t* pivots, T* p, T* l, T* u,
        int m, int n, int k, int64_t row_stride, int64_t column_stride) {
    const int64_t batch = blockIdx.x;
    const int lane = threadIdx.x;
    if (p != nullptr) {
        __shared__ int permutation[kSmallLuExtent];
        if (lane == 0) {
            for (int i = 0; i < m; ++i) permutation[i] = i;
            for (int i = 0; i < k; ++i) {
                const int target = pivots[batch * k + i];
                assert(target >= 1 && target <= m);
                const int row = permutation[i];
                permutation[i] = permutation[target - 1];
                permutation[target - 1] = row;
            }
        }
        __syncthreads();
        p += batch * m * m;
        for (int i = lane; i < m * m; i += blockDim.x) {
            p[i] = T(i / m == permutation[i % m]);
        }
    }
    if (l != nullptr) {
        lu += batch * m * n;
        l += batch * m * k;
        u += batch * k * n;
        for (int i = lane; i < m * k; i += blockDim.x) {
            const int row = i / k;
            const int col = i % k;
            l[i] = row > col ? lu[row * row_stride + col * column_stride] : T(row == col);
        }
        for (int i = lane; i < k * n; i += blockDim.x) {
            const int row = i / n;
            const int col = i % n;
            u[i] = row <= col ? lu[row * row_stride + col * column_stride] : T(0);
        }
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> lu_unpack_cuda(const Tensor& LU,
                                                  const Tensor& pivots,
                                                  bool unpack_data,
                                                  bool unpack_pivots) {
    const detail_lu::LuShape shape =
        detail_lu::validate(LU, pivots, unpack_pivots);

    OptionalCUDAGuard guard(LU.device());
    const bool fast_dtype =
        LU.dtype() == DType::Float32 || LU.dtype() == DType::Float64;
    if (fast_dtype && shape.m <= kSmallLuExtent && shape.n <= kSmallLuExtent &&
        LU.numel() != 0 && LU.numel() / (shape.m * shape.n) <= INT32_MAX) {
        auto sizes = static_cast<std::vector<int64_t>>(LU.shape());
        auto factor = [&](bool needed, int64_t rows, int64_t cols) {
            if (!needed) return Tensor::empty({0}, LU.dtype(), LU.device());
            sizes[sizes.size() - 2] = rows;
            sizes.back() = cols;
            return Tensor::empty(sizes, LU.dtype(), LU.device());
        };
        Tensor P = factor(unpack_pivots, shape.m, shape.m);
        Tensor L = factor(unpack_data, shape.m, shape.k);
        Tensor U = factor(unpack_data, shape.k, shape.n);
        if (unpack_data || unpack_pivots) {
            const Tensor input = unpack_data && !LU.is_contiguous() &&
                    !LU.transpose(-2, -1).is_contiguous() ? LU.contiguous() : LU;
            const Tensor source = unpack_pivots ? pivots.contiguous() : Tensor();
            const auto blocks = static_cast<unsigned>(LU.numel() / (shape.m * shape.n));
            const auto stream = getCurrentCUDAStream().stream();
            auto launch = [&]<typename T>() {
                unpack_small_lu<T><<<blocks, kLuThreads, 0, stream>>>(
                    unpack_data ? input.const_data_ptr<T>() : nullptr,
                    unpack_pivots ? source.const_data_ptr<int32_t>() : nullptr,
                    unpack_pivots ? P.data_ptr<T>() : nullptr,
                    unpack_data ? L.data_ptr<T>() : nullptr,
                    unpack_data ? U.data_ptr<T>() : nullptr,
                    static_cast<int>(shape.m), static_cast<int>(shape.n),
                    static_cast<int>(shape.k), input.stride(-2), input.stride(-1));
            };
            if (LU.dtype() == DType::Float32) launch.template operator()<float>();
            else launch.template operator()<double>();
            TP_LU_CUDA_CHECK(cudaGetLastError());
        }
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
