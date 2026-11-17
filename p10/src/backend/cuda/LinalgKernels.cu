// CUDA linear algebra kernels using column-major batched buffers and device
// error-code tensors.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "DTypeNames.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include "CudaDispatchHelpers.cuh"

#define CUDA_CHECK(condition)                                                 \
  do {                                                                        \
    cudaError_t error = (condition);                                          \
    if (error != cudaSuccess) {                                               \
      TP_THROW(RuntimeError,                                                  \
               std::string("CUDA Error: ") + cudaGetErrorString(error));      \
    }                                                                         \
  } while (0)

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

namespace {

#define CUSOLVER_CHECK(condition)                                              \
    do {                                                                       \
        const cusolverStatus_t _tp_cs = (condition);                           \
        if (_tp_cs != CUSOLVER_STATUS_SUCCESS) {                               \
            TP_THROW(RuntimeError, "cuSOLVER error ",                          \
                     std::to_string(static_cast<int>(_tp_cs)));                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(condition)                                                \
    do {                                                                       \
        const cublasStatus_t _tp_cb = (condition);                             \
        if (_tp_cb != CUBLAS_STATUS_SUCCESS) {                                 \
            TP_THROW(RuntimeError, "cuBLAS error ",                            \
                     std::to_string(static_cast<int>(_tp_cb)));                \
        }                                                                      \
    } while (0)

constexpr cublasFillMode_t fill_mode(char uplo) {
    return uplo == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}

// ------------------------------------------------------------- helpers -----

template <class Kernel>
decltype(auto) run_real(DType dt, Kernel&& k) {
    switch (dt) {
        case DType::Float32:
            return k(static_cast<float*>(nullptr));
        case DType::Float64:
            return k(static_cast<double*>(nullptr));
        default:
            TP_THROW(NotImplementedError,
                     "linalg: unsupported dtype ", scalarTypeName(dt),
                     " on CUDA (only float32/float64 are implemented)");
    }
}

void write_linalg_output(const char* op, const Tensor& value, Tensor& out) {
    if (!out.defined()) {
        out = value;
        return;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError, op, ": output dtype must match result dtype");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 op, ": output device must match input device");
    }
    out.resize_(static_cast<std::vector<int64_t>>(value.shape()));
    out.copy_(value);
}

void check_is_matrix(const Tensor& A, const char* fn, const char* arg = "A") {
    if (A.dim() < 2) {
        TP_THROW(RuntimeError, fn, ": The input tensor ", arg,
                 " must have at least 2 dimensions.");
    }
}

void square_check_inputs(const Tensor& A, const char* fn, const char* arg = "A") {
    check_is_matrix(A, fn, arg);
    if (A.size(-1) != A.size(-2)) {
        TP_THROW(RuntimeError, fn, ": ", arg,
                 " must be batches of square matrices, but they are ",
                 A.size(-2), " by ", A.size(-1), " matrices");
    }
}

std::vector<int64_t> batch_shape_of(const Tensor& t) {
    const Size t_shape = t.shape();
    return std::vector<int64_t>(t_shape.begin(), t_shape.end() - 2);
}

int64_t matrix_stride_of(const Tensor& t) { return t.size(-1) * t.size(-2); }

int64_t batch_count_of(const Tensor& t) {
    const int64_t plane = matrix_stride_of(t);
    return plane == 0 ? 0 : t.numel() / plane;
}

Tensor clone_batched_column_major(const Tensor& src) {
    // The clone must physically transpose into a contiguous buffer of the
    // transposed shape: clone() with Preserve would keep the transposed view's
    // strides (non-overlapping-and-dense), leaving the data row-major.
    auto result = src.transpose(-2, -1).clone(static_cast<int64_t>(MemoryFormat::Contiguous));
    return result.transpose(-2, -1);
}

Tensor empty_column_major(std::vector<int64_t> shape, DType dt, Device dev) {
    std::swap(shape[shape.size() - 2], shape[shape.size() - 1]);
    return Tensor::empty(shape, dt, dev).transpose(-2, -1);
}

std::vector<int64_t> cat_batch(const std::vector<int64_t>& batch,
                               std::vector<int64_t> tail) {
    std::vector<int64_t> out = batch;
    for (int64_t v : tail) out.push_back(v);
    return out;
}

int64_t linear_batch_size(const std::vector<int64_t>& batch) {
    return static_cast<int64_t>(std::accumulate(
        batch.begin(), batch.end(), int64_t{1}, std::multiplies<int64_t>()));
}

std::vector<int64_t> broadcast_batch(const Tensor& a, const Tensor& b) {
    const auto as = batch_shape_of(a);
    const auto bs = batch_shape_of(b);
    const size_t rank = std::max(as.size(), bs.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = i < rank - as.size() ? 1 : as[i - (rank - as.size())];
        const int64_t db = i < rank - bs.size() ? 1 : bs[i - (rank - bs.size())];
        if (da != db && da != 1 && db != 1) {
            TP_THROW(RuntimeError, "The size of tensor a (", da,
                     ") must match the size of tensor b (", db, ")");
        }
        out[i] = std::max(da, db);
    }
    return out;
}

Tensor expand_to_batch(const Tensor& t, const std::vector<int64_t>& batch) {
    return t.expand(cat_batch(batch, {t.size(-2), t.size(-1)}));
}

int64_t first_nonzero_info(const Tensor& infos_host) {
    const auto* ptr = infos_host.data_ptr<int32_t>();
    for (int64_t i = 0; i < infos_host.numel(); ++i)
        if (ptr[i] != 0) return i;
    return -1;
}

void check_infos(const Tensor& infos_dev, std::string_view api_name, bool is_matrix) {
    Tensor infos = infos_dev.to(Device(DeviceType::CPU), DType::Int32).contiguous();
    const int64_t first = first_nonzero_info(infos);
    if (first < 0) return;
    const auto* ptr = infos.data_ptr<int32_t>();
    const int64_t info = ptr[first];
    const std::string b =
        is_matrix ? "" : ": (Batch element " + std::to_string(first) + ")";
    if (info < 0) {
        TP_THROW(RuntimeError, api_name, b, ": Argument ", -info,
                 " has illegal value.");
    }
    if (api_name.find("inv") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The diagonal element ", info, " is zero, the inversion could not be completed because the input matrix is singular.");
    } else if (api_name.find("lu_factor") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": U[", info, ",", info, "] is zero and using it on lu_solve would result in a division by zero. "
                 "If you still want to perform the factorization, consider calling linalg.lu(A, pivot) or "
                 "linalg.lu_factor_ex(A, pivot)");
    } else if (api_name.find("cholesky") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The factorization could not be completed because the input is not positive-definite (the leading minor of order ", info, " is not positive-definite).");
    } else if (api_name.find("svd") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated singular values (error code: ", info, ").");
    } else if (api_name.find("eig") != std::string_view::npos ||
               api_name.find("syevd") != std::string_view::npos) {
        TP_THROW(RuntimeError, api_name, b,
                 ": The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: ", info, ").");
    } else {
        TP_THROW(RuntimeError, api_name, b,
                 ": The solver failed because the input matrix is singular.");
    }
}

void linalg_check_errors_kernel(const Tensor& infos, const std::string& api_name,
                                bool is_matrix) {
    check_infos(infos, api_name, is_matrix);
}

// ------------------------------------------------- cusolver entry traits ---

template <typename scalar_t>
struct CusolverTraits;

template <>
struct CusolverTraits<float> {
    static constexpr bool is_float = true;
    using T = float;
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, char uplo, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSpotrf_bufferSize(h, fill_mode(uplo), n, a, lda, lw);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t h, char uplo, int n, float* a,
                                  int lda, float* work, int lwork, int* info) {
        return cusolverDnSpotrf(h, fill_mode(uplo), n, a, lda, work, lwork, info);
    }
    static cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t uplo,
                                  int n, int nrhs, const float* a, int lda,
                                  float* b, int ldb, int* info) {
        return cusolverDnSpotrs(h, uplo, n, nrhs, a, lda, b, ldb, info);
    }
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSgetrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, float* a,
                                  int lda, int* ipiv, float* work, int lwork,
                                  int* info) {
        return cusolverDnSgetrf(h, m, n, a, lda, work, ipiv, info); (void)lwork;
    }
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             float* a, int lda, int* lw) {
        return cusolverDnSgeqrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, float* a,
                                  int lda, float* tau, float* work, int lwork,
                                  int* info) {
        return cusolverDnSgeqrf(h, m, n, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             int k, const float* a, int lda,
                                             const float* tau, int* lw) {
        return cusolverDnSorgqr_bufferSize(h, m, n, k, a, lda, tau, lw);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k,
                                  float* a, int lda, const float* tau, float* work,
                                  int lwork, int* info) {
        return cusolverDnSorgqr(h, m, n, k, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t ormqr_bufferSize(cusolverDnHandle_t h,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans, int m,
                                             int n, int k, const float* a,
                                             int lda, const float* tau,
                                             const float* c, int ldc, int* lw) {
        return cusolverDnSormqr_bufferSize(h, side, trans, m, n, k, a, lda, tau,
                                           c, ldc, lw);
    }
    static cusolverStatus_t ormqr(cusolverDnHandle_t h, cublasSideMode_t side,
                                  cublasOperation_t trans, int m, int n, int k,
                                  const float* a, int lda, const float* tau,
                                  float* c, int ldc, float* work, int lwork,
                                  int* info) {
        return cusolverDnSormqr(h, side, trans, m, n, k, a, lda, tau, c, ldc,
                                work, lwork, info);
    }
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                             cusolverEigMode_t jobz, char uplo,
                                             int n, const float* a, int lda,
                                             const float* w, int* lw) {
        return cusolverDnSsyevd_bufferSize(h, jobz, fill_mode(uplo), n, a, lda, w, lw);
    }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz,
                                  char uplo, int n, float* a, int lda, float* w,
                                  float* work, int lwork, int* info) {
        return cusolverDnSsyevd(h, jobz, fill_mode(uplo), n, a, lda, w, work,
                                lwork, info);
    }
    static cusolverStatus_t syevj_bufferSize(
            cusolverDnHandle_t h, cusolverEigMode_t jobz,
            cublasFillMode_t uplo, int n, const float* a, int lda,
            const float* w, int* lwork, syevjInfo_t params, int batch) {
        return cusolverDnSsyevjBatched_bufferSize(
                h, jobz, uplo, n, a, lda, w, lwork, params, batch);
    }
    static cusolverStatus_t syevj(
            cusolverDnHandle_t h, cusolverEigMode_t jobz,
            cublasFillMode_t uplo, int n, float* a, int lda, float* w,
            float* work, int lwork, int* info, syevjInfo_t params,
            int batch) {
        return cusolverDnSsyevjBatched(
                h, jobz, uplo, n, a, lda, w, work, lwork, info, params, batch);
    }
    static cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char jobu,
                                  signed char jobvt, int m, int n, float* a,
                                  int lda, float* s, float* u, int ldu, float* vt,
                                  int ldvt, float* work, int lwork, float* rwork,
                                  int* info) {
        return cusolverDnSgesvd(h, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                work, lwork, rwork, info);
    }
};

template <>
struct CusolverTraits<double> {
    static constexpr bool is_float = false;
    using T = double;
    static cusolverStatus_t potrf_bufferSize(cusolverDnHandle_t h, char uplo, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDpotrf_bufferSize(h, fill_mode(uplo), n, a, lda, lw);
    }
    static cusolverStatus_t potrf(cusolverDnHandle_t h, char uplo, int n, double* a,
                                  int lda, double* work, int lwork, int* info) {
        return cusolverDnDpotrf(h, fill_mode(uplo), n, a, lda, work, lwork, info);
    }
    static cusolverStatus_t potrs(cusolverDnHandle_t h, cublasFillMode_t uplo,
                                  int n, int nrhs, const double* a, int lda,
                                  double* b, int ldb, int* info) {
        return cusolverDnDpotrs(h, uplo, n, nrhs, a, lda, b, ldb, info);
    }
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDgetrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, double* a,
                                  int lda, int* ipiv, double* work, int lwork,
                                  int* info) {
        return cusolverDnDgetrf(h, m, n, a, lda, work, ipiv, info); (void)lwork;
    }
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             double* a, int lda, int* lw) {
        return cusolverDnDgeqrf_bufferSize(h, m, n, a, lda, lw);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, double* a,
                                  int lda, double* tau, double* work, int lwork,
                                  int* info) {
        return cusolverDnDgeqrf(h, m, n, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n,
                                             int k, const double* a, int lda,
                                             const double* tau, int* lw) {
        return cusolverDnDorgqr_bufferSize(h, m, n, k, a, lda, tau, lw);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k,
                                  double* a, int lda, const double* tau,
                                  double* work, int lwork, int* info) {
        return cusolverDnDorgqr(h, m, n, k, a, lda, tau, work, lwork, info);
    }
    static cusolverStatus_t ormqr_bufferSize(cusolverDnHandle_t h,
                                             cublasSideMode_t side,
                                             cublasOperation_t trans, int m,
                                             int n, int k, const double* a,
                                             int lda, const double* tau,
                                             const double* c, int ldc,
                                             int* lw) {
        return cusolverDnDormqr_bufferSize(h, side, trans, m, n, k, a, lda, tau,
                                           c, ldc, lw);
    }
    static cusolverStatus_t ormqr(cusolverDnHandle_t h, cublasSideMode_t side,
                                  cublasOperation_t trans, int m, int n, int k,
                                  const double* a, int lda, const double* tau,
                                  double* c, int ldc, double* work, int lwork,
                                  int* info) {
        return cusolverDnDormqr(h, side, trans, m, n, k, a, lda, tau, c, ldc,
                                work, lwork, info);
    }
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h,
                                             cusolverEigMode_t jobz, char uplo,
                                             int n, const double* a, int lda,
                                             const double* w, int* lw) {
        return cusolverDnDsyevd_bufferSize(h, jobz, fill_mode(uplo), n, a, lda, w, lw);
    }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz,
                                  char uplo, int n, double* a, int lda, double* w,
                                  double* work, int lwork, int* info) {
        return cusolverDnDsyevd(h, jobz, fill_mode(uplo), n, a, lda, w, work,
                                lwork, info);
    }
    static cusolverStatus_t syevj_bufferSize(
            cusolverDnHandle_t h, cusolverEigMode_t jobz,
            cublasFillMode_t uplo, int n, const double* a, int lda,
            const double* w, int* lwork, syevjInfo_t params, int batch) {
        return cusolverDnDsyevjBatched_bufferSize(
                h, jobz, uplo, n, a, lda, w, lwork, params, batch);
    }
    static cusolverStatus_t syevj(
            cusolverDnHandle_t h, cusolverEigMode_t jobz,
            cublasFillMode_t uplo, int n, double* a, int lda, double* w,
            double* work, int lwork, int* info, syevjInfo_t params,
            int batch) {
        return cusolverDnDsyevjBatched(
                h, jobz, uplo, n, a, lda, w, work, lwork, info, params, batch);
    }
    static cusolverStatus_t gesvd(cusolverDnHandle_t h, signed char jobu,
                                  signed char jobvt, int m, int n, double* a,
                                  int lda, double* s, double* u, int ldu,
                                  double* vt, int ldvt, double* work, int lwork,
                                  double* rwork, int* info) {
        return cusolverDnDgesvd(h, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,
                                work, lwork, rwork, info);
    }
};

// ------------------------------------------------------------- getrf -------

template <typename scalar_t>
void apply_getrf(const Tensor& input, const Tensor& pivots, const Tensor& info) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = input.data_ptr<scalar_t>();
    auto* ipiv = pivots.data_ptr<int32_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t ms = matrix_stride_of(input);
    const int64_t piv_stride = pivots.dim() > 1 ? pivots.size(-1) : pivots.numel();
    const int64_t batch = batch_count_of(input);
    const int m = static_cast<int>(input.size(-2));
    const int n = static_cast<int>(input.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();

    int lwork = 0;
    CUSOLVER_CHECK(Tr::getrf_bufferSize(handle, m, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, input.dtype(),
                                input.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::getrf(handle, m, n, &a[i * ms], lda, &ipiv[i * piv_stride],
                                 work_ptr, lwork, &info_data[i]));
    }
}

std::tuple<Tensor, Tensor, Tensor> lu_factor_ex_cuda_impl(const Tensor& A,
                                                          bool check_errors) {
    check_is_matrix(A, "linalg.lu_factor_ex");
    Tensor LU = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    const int64_t k = std::min(A.size(-2), A.size(-1));
    Tensor pivots = Tensor::zeros(cat_batch(batch, {k}), DType::Int32, A.device());
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_getrf<T>(LU, pivots, info);
    });
    if (check_errors) check_infos(info, "linalg.lu_factor_ex", A.dim() == 2);
    return {LU.contiguous(), pivots.contiguous(), info.contiguous()};
}

std::tuple<Tensor, Tensor, Tensor> linalg_lu_factor_ex_kernel_cuda(const Tensor& A,
                                                                   bool pivot,
                                                                   bool check_errors) {
    (void)pivot;
    return lu_factor_ex_cuda_impl(A, check_errors);
}

// det and slogdet both read one diagonal and the sign of the permutation.
// pass answers them per matrix: a warp walks the diagonal in strides and folds
// its counts together, so nothing of the matrix travels through the host.  An
// entry without a sign -- a zero or a missing value -- makes the sign zero,
// while the logarithm follows the magnitudes and goes to -inf or undefined.
template <typename T>
__global__ void det_slogdet_kernel(int64_t n, const T* lu,
                                   const int32_t* pivots, T* det_out,
                                   T* sign_out, T* logabsdet_out) {
    constexpr unsigned kFullWarp = 0xffffffffu;
    const int64_t b = blockIdx.x;
    const T* diag = lu + b * n * n;
    const int32_t* piv = pivots + b * n;
    T product = T(1);
    T log_sum = T(0);
    int negative = 0;
    int zero = 0;
    unsigned long long sign_changes = 0;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const T v = diag[i * n + i];
        product *= v;
        if constexpr (std::is_same_v<T, float>) {
            log_sum += logf(fabsf(v));
        } else {
            log_sum += ::log(::fabs(v));
        }
        negative ^= (v < T(0)) ? 1 : 0;
        // A value that is neither above nor below zero carries no sign: a zero
        // diagonal makes the whole sign zero, and a missing entry does too.
        zero |= (v > T(0) || v < T(0)) ? 0 : 1;
        sign_changes += (piv[i] - 1 != static_cast<int32_t>(i)) ? 1ULL : 0ULL;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        product *= __shfl_down_sync(kFullWarp, product, offset);
        log_sum += __shfl_down_sync(kFullWarp, log_sum, offset);
        negative ^= __shfl_down_sync(kFullWarp, negative, offset);
        zero |= __shfl_down_sync(kFullWarp, zero, offset);
        sign_changes += __shfl_down_sync(kFullWarp, sign_changes, offset);
    }
    if (threadIdx.x != 0) return;
    const T permutation = (sign_changes % 2 == 0) ? T(1) : T(-1);
    if (det_out != nullptr) {
        det_out[b] = permutation * product;
    }
    if (sign_out != nullptr) {
        sign_out[b] = zero ? T(0)
                           : permutation * (negative ? T(-1) : T(1));
        logabsdet_out[b] = log_sum;
    }
}

template <typename T>
void device_det_slogdet(const Tensor& LU, const Tensor& pivots, Tensor* det_out,
                        Tensor* sign_out, Tensor* logabsdet_out) {
    // The matrix count is the product of the leading extents; an empty matrix
    // still counts as one, whose answer is the empty product the kernel's
    // untouched accumulators carry.
    int64_t bs = 1;
    for (int64_t dim = 0; dim + 2 < LU.dim(); ++dim) bs *= LU.size(dim);
    if (bs == 0) return;
    const int64_t n = LU.size(-1);
    det_slogdet_kernel<T><<<static_cast<unsigned>(bs), 32, 0,
                            getCurrentCUDAStream().stream()>>>(
        n, LU.data_ptr<T>(), pivots.data_ptr<int32_t>(),
        det_out == nullptr ? nullptr : det_out->data_ptr<T>(),
        sign_out == nullptr ? nullptr : sign_out->data_ptr<T>(),
        logabsdet_out == nullptr ? nullptr : logabsdet_out->data_ptr<T>());
    CUDA_CHECK(cudaGetLastError());
}

std::tuple<Tensor, Tensor, Tensor> linalg_det_internal_kernel_cuda(
        const Tensor& A) {
    square_check_inputs(A, "linalg.det");
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(src, false);
    (void)info;
    Tensor out = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        device_det_slogdet<T>(LU, pivots, &out, nullptr, nullptr);
    });
    return {out, LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor> linalg_det_internal_out_kernel_cuda(
        const Tensor& A, Tensor& result, Tensor& LU, Tensor& pivots) {
    auto values = linalg_det_internal_kernel_cuda(A);
    write_linalg_output("linalg.det", std::get<0>(values), result);
    write_linalg_output("linalg.det", std::get<1>(values), LU);
    write_linalg_output("linalg.det", std::get<2>(values), pivots);
    return {result, LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_slogdet_internal_kernel_cuda(
        const Tensor& A) {
    square_check_inputs(A, "linalg.slogdet");
    const Tensor src = A.is_contiguous() ? A.transpose(-2, -1) : A;
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(src, false);
    (void)info;
    auto sign = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    auto logabsdet = Tensor::empty(batch_shape_of(A), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        device_det_slogdet<T>(LU, pivots, nullptr, &sign, &logabsdet);
    });
    return {sign, logabsdet, LU, pivots};
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
linalg_slogdet_internal_out_kernel_cuda(
        const Tensor& A, Tensor& sign, Tensor& logabsdet, Tensor& LU,
        Tensor& pivots) {
    auto values = linalg_slogdet_internal_kernel_cuda(A);
    write_linalg_output("linalg.slogdet", std::get<0>(values), sign);
    write_linalg_output("linalg.slogdet", std::get<1>(values), logabsdet);
    write_linalg_output("linalg.slogdet", std::get<2>(values), LU);
    write_linalg_output("linalg.slogdet", std::get<3>(values), pivots);
    return {sign, logabsdet, LU, pivots};
}

// ------------------------------------------- solving against the factors --

// The row permutation the factorisation applied, resolved once on the device
// so that every row of the right-hand side can be placed at the same time:
// the swaps themselves are one dependent chain over the pivots, so a single
// thread walks them and the gathered copy below does the real work.
__global__ void row_permutation_kernel(int64_t n, const int32_t* pivots,
                                       int32_t* rows, int32_t* inverse) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t i = 0; i < n; ++i) rows[i] = static_cast<int32_t>(i);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t j = static_cast<int64_t>(pivots[i]) - 1;
        const int32_t held = rows[i];
        rows[i] = rows[j];
        rows[j] = held;
    }
    if (inverse == nullptr) return;
    for (int64_t i = 0; i < n; ++i) {
        inverse[rows[i]] = static_cast<int32_t>(i);
    }
}

// out[row, col] = src[rows[row], col] over a block of columns, both buffers
// column-major.  One block per column keeps every access inside that column's
// own contiguous run, which the rows only reorder.
template <typename scalar_t>
__global__ void gather_rows_kernel(const scalar_t* src, scalar_t* out,
                                   const int32_t* rows, int64_t ldb,
                                   int64_t n, int64_t nrhs) {
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t col = static_cast<int64_t>(blockIdx.y);
    if (row >= n || col >= nrhs) return;
    out[row + col * ldb] = src[rows[row] + col * ldb];
}

// One triangular solve over the whole right-hand side: op(A) X = alpha B,
// where A is the upper or the unit lower factor of the factorisation.
template <typename scalar_t>
void apply_one_trsm(cublasHandle_t handle, const scalar_t* a, int64_t lda,
                    scalar_t* b, int64_t ldb, int64_t n, int64_t nrhs,
                    cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag) {
    const scalar_t alpha = scalar_t(1);
    if constexpr (std::is_same_v<scalar_t, float>) {
        CUBLAS_CHECK(cublasStrsm(handle, CUBLAS_SIDE_LEFT, uplo, trans, diag,
                                 static_cast<int>(n), static_cast<int>(nrhs),
                                 &alpha, a, static_cast<int>(lda), b,
                                 static_cast<int>(ldb)));
    } else {
        CUBLAS_CHECK(cublasDtrsm(handle, CUBLAS_SIDE_LEFT, uplo, trans, diag,
                                 static_cast<int>(n), static_cast<int>(nrhs),
                                 &alpha, a, static_cast<int>(lda), b,
                                 static_cast<int>(ldb)));
    }
}

// Solves op(A) X = B from the factorisation, both sides column-major, and
// answers in a column-major buffer of its own.  The permutation travels
// through one gathered copy and the two factors go through the tuned
// triangular solve, which answers this shape in a third of the time the
// general solver takes for it.
template <typename scalar_t>
Tensor solve_from_factors(const Tensor& LU_cm, const Tensor& pivots,
                   const Tensor& B_cm, bool transpose) {
    constexpr int64_t kColumnChunk = 65535;
    constexpr int64_t kRowsPerBlock = 256;
    const int64_t n = LU_cm.size(-2);
    const int64_t nrhs = B_cm.size(-1);
    const int64_t bs = linear_batch_size(batch_shape_of(B_cm));
    if (bs == 0 || n == 0 || nrhs == 0) return B_cm;
    const int64_t lu_ms = matrix_stride_of(LU_cm);
    const int64_t b_ms = matrix_stride_of(B_cm);
    const int64_t ldb = B_cm.size(-2);
    auto stream = getCurrentCUDAStream().stream();
    Tensor routes = Tensor::empty({2 * n}, DType::Int32, B_cm.device());
    auto* route_data = routes.data_ptr<int32_t>();
    auto* rows_data = route_data;
    auto* inverse_data = route_data + n;
    const auto* piv = pivots.data_ptr<int32_t>();
    for (int64_t i = 0; i < bs; ++i) {
        row_permutation_kernel<<<1, 1, 0, stream>>>(
            n, &piv[i * n], rows_data, transpose ? inverse_data : nullptr);
    }
    Tensor solved = empty_column_major(
        static_cast<std::vector<int64_t>>(B_cm.shape()), B_cm.dtype(),
        B_cm.device());
    auto* lu = LU_cm.data_ptr<scalar_t>();
    auto* src = B_cm.data_ptr<scalar_t>();
    auto* dst = solved.data_ptr<scalar_t>();
    const auto handle = CUDAContext::getCublasHandle();
    const unsigned row_blocks =
        static_cast<unsigned>((n + kRowsPerBlock - 1) / kRowsPerBlock);
    const int32_t* routes_used = transpose ? inverse_data : rows_data;
    for (int64_t i = 0; i < bs; ++i) {
        scalar_t* moved = dst + i * b_ms;
        if (!transpose) {
            // op(A) = A: place the rows first, then solve the two factors.
            for (int64_t c0 = 0; c0 < nrhs; c0 += kColumnChunk) {
                const int64_t cols = std::min<int64_t>(kColumnChunk, nrhs - c0);
                gather_rows_kernel<scalar_t><<<
                    dim3(row_blocks, static_cast<unsigned>(cols)),
                    kRowsPerBlock, 0, stream>>>(
                    src + i * b_ms + c0 * ldb, moved + c0 * ldb, rows_data,
                    ldb, n, cols);
            }
            apply_one_trsm<scalar_t>(handle, lu + i * lu_ms, n, moved, ldb, n,
                                     nrhs, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_UNIT);
            apply_one_trsm<scalar_t>(handle, lu + i * lu_ms, n, moved, ldb, n,
                                     nrhs, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT);
        } else {
            // op(A) = A^T: the transposed factors are solved straight onto the
            // right-hand side, and the permutation lands on the answer.
            scalar_t* work = src + i * b_ms;
            apply_one_trsm<scalar_t>(handle, lu + i * lu_ms, n, work, ldb, n,
                                     nrhs, CUBLAS_FILL_MODE_UPPER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT);
            apply_one_trsm<scalar_t>(handle, lu + i * lu_ms, n, work, ldb, n,
                                     nrhs, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_UNIT);
            for (int64_t c0 = 0; c0 < nrhs; c0 += kColumnChunk) {
                const int64_t cols = std::min<int64_t>(kColumnChunk, nrhs - c0);
                gather_rows_kernel<scalar_t><<<
                    dim3(row_blocks, static_cast<unsigned>(cols)),
                    kRowsPerBlock, 0, stream>>>(
                    work + c0 * ldb, moved + c0 * ldb, routes_used, ldb, n,
                    cols);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    return solved;
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
linalg_solve_ex_impl(const Tensor& A, const Tensor& B, bool left,
                     bool check_errors, bool keep_factors) {
    const char* api = "linalg.solve_ex";
    check_is_matrix(A, api, "A");
    bool vector_case = B.dim() == 1;
    if (!vector_case && A.dim() - 1 == B.dim()) {
        vector_case = true;
        for (int64_t i = 0; i < A.dim() - 1; ++i) {
            if (A.size(i) != B.size(i)) {
                vector_case = false;
                break;
            }
        }
    }
    Tensor B_2d = vector_case ? B.unsqueeze(-1) : B;
    check_is_matrix(B_2d, api, "B");
    if (!left && vector_case) {
        TP_THROW(RuntimeError,
                 "linalg.solve: Vector right-hand sides are not supported when left is false");
    }
    if (!(left ? A.size(-2) == B_2d.size(-2) : A.size(-1) == B_2d.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of A and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", A.size(-2), "x", A.size(-1), " and ",
                 B_2d.size(-2), "x", B_2d.size(-1), ")");
    }
    const auto batch = broadcast_batch(A, B_2d);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(A, batch));
    const int64_t n = A.size(-1);
    Tensor pivots = Tensor::zeros(cat_batch(batch_shape_of(LU_work), {n}),
                                  DType::Int32, A.device());
    Tensor info = Tensor::zeros(batch_shape_of(LU_work), DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_getrf<T>(LU_work, pivots, info);
    });

    Tensor result;
    if (left) {
        // op(A) = A or conj-transposed A; real dtypes: adjoint handled via 'T'.
        Tensor B_cm = clone_batched_column_major(expand_to_batch(B_2d, batch));
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            B_cm = solve_from_factors<T>(LU_work, pivots, B_cm, false);
        });
        result = B_cm.contiguous();
    } else {
        // X A = B <=> A^T X^T = B^T: solve against the transposed RHS.
        Tensor BT_cm =
            clone_batched_column_major(expand_to_batch(B_2d, batch).transpose(-2, -1));
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            BT_cm = solve_from_factors<T>(LU_work, pivots, BT_cm, true);
        });
        result = BT_cm.contiguous().transpose(-2, -1).contiguous();
    }
    if (vector_case) result = result.squeeze(-1);
    if (check_errors) check_infos(info, api, A.dim() == 2 && B.dim() == 2);
    if (!keep_factors) {
        // The caller throws the factors away; copying the column-major working
        // buffer back to the caller's layout would be one full pass for nothing.
        return {result, LU_work, pivots, info};
    }
    return {result, LU_work.contiguous(), pivots.contiguous(), info.contiguous()};
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
linalg_solve_ex_internal_kernel_cuda(const Tensor& A, const Tensor& B,
                                     bool left, bool check_errors) {
    return linalg_solve_ex_impl(A, B, left, check_errors,
                                /*keep_factors=*/true);
}

std::tuple<Tensor, Tensor> linalg_solve_ex_kernel_cuda(const Tensor& A,
                                                       const Tensor& B,
                                                       bool left, bool check_errors) {
    auto values = linalg_solve_ex_impl(A, B, left, check_errors,
                                       /*keep_factors=*/false);
    return {std::get<0>(values), std::get<3>(values)};
}

std::tuple<Tensor, Tensor, Tensor, Tensor>
linalg_solve_ex_internal_out_kernel_cuda(
        const Tensor& A, const Tensor& B, bool left, bool check_errors,
        Tensor& result, Tensor& LU, Tensor& pivots, Tensor& info) {
    auto values = linalg_solve_ex_internal_kernel_cuda(A, B, left, check_errors);
    write_linalg_output("linalg.solve_ex", std::get<0>(values), result);
    write_linalg_output("linalg.solve_ex", std::get<1>(values), LU);
    write_linalg_output("linalg.solve_ex", std::get<2>(values), pivots);
    write_linalg_output("linalg.solve_ex", std::get<3>(values), info);
    return {result, LU, pivots, info};
}

// The right-hand side of an inverse is a stack of identities.  Writing it on
// the device keeps the host out of the loop: staging it through a host buffer
// costs a full-size upload plus a stream synchronization, and the solve that
// follows is only tens of microseconds.
template <typename T>
__global__ void eye_batched_kernel(int64_t n, int64_t matrix_stride,
                                   int64_t batch, T* out) {
    int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * matrix_stride;
    const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; index < total; index += step) {
        const int64_t i = index % matrix_stride;
        out[index] = (i / n == i % n) ? T(1) : T(0);
    }
}

std::tuple<Tensor, Tensor> linalg_inv_ex_kernel_cuda(const Tensor& A,
                                                     bool check_errors) {
    square_check_inputs(A, "linalg.inv_ex");
    const int64_t n = A.size(-1);
    const int64_t ms = n * n;
    const int64_t bs = linear_batch_size(batch_shape_of(A));
    if (n == 0) {
        run_real(A.dtype(), [](auto) {});
        return {Tensor::empty(A.shape(), A.dtype(), A.device()),
                Tensor::zeros(batch_shape_of(A), DType::Int32, A.device())};
    }
    Tensor identity = Tensor::empty(A.shape(), A.dtype(), A.device());
    const auto stream = getCurrentCUDAStream().stream();
    if (bs * ms > 0) {
        run_real(A.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            const auto blocks = static_cast<unsigned>(std::min<int64_t>((bs * ms + 255) / 256, 65535));
            eye_batched_kernel<<<blocks, 256, 0, stream>>>(n, ms, bs, identity.data_ptr<T>());
        });
        checkCuda(cudaGetLastError(), "linalg.inv identity");
    }
    auto [inv, info] = linalg_solve_ex_kernel_cuda(A, identity, true, false);
    if (check_errors) check_infos(info, "linalg.inv_ex", A.dim() == 2);
    return {inv, info};
}

// ------------------------------------------------------------- potrf -------

template <typename scalar_t>
__global__ void zero_potrf_opposite_triangle_kernel(
    scalar_t* data, int64_t total, int64_t n, int64_t matrix_stride,
    char uplo) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int64_t local = index % matrix_stride;
    const int64_t row = local % n;
    const int64_t col = local / n;
    if ((uplo == 'U' && row > col) || (uplo == 'L' && row < col)) {
        data[index] = scalar_t(0);
    }
}

template <typename scalar_t>
void apply_potrf(const Tensor& L, const Tensor& info, char uplo) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = L.data_ptr<scalar_t>();
    auto* info_data = info.data_ptr<int32_t>();
    const int64_t ms = matrix_stride_of(L);
    const int64_t batch = batch_count_of(L);
    const int n = static_cast<int>(L.size(-1));
    const int lda = std::max(1, n);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::potrf_bufferSize(handle, uplo, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, L.dtype(), L.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::potrf(handle, uplo, n, &a[i * ms], lda, work_ptr,
                                 lwork, &info_data[i]));
    }
    const int64_t total = batch * ms;
    if (total > 0) {
        zero_potrf_opposite_triangle_kernel<scalar_t>
            <<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                                      getCurrentCUDAStream().stream()>>>(
                a, total, n, ms, uplo);
    }
}

std::tuple<Tensor, Tensor> linalg_cholesky_ex_kernel_cuda(const Tensor& A, bool upper,
                                                          bool check_errors) {
    square_check_inputs(A, "linalg.cholesky_ex");
    Tensor L = clone_batched_column_major(A);
    const auto batch = batch_shape_of(A);
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_potrf<T>(L, info, upper ? 'U' : 'L');
    });
    if (check_errors) check_infos(info, "linalg.cholesky_ex", A.dim() == 2);
    return {L.contiguous(), info.contiguous()};
}

// -------------------------------------------------------------- lu_solve ---

Tensor linalg_lu_solve_kernel_cuda(const Tensor& LU, const Tensor& pivots,
                                   const Tensor& B, bool left, bool adjoint) {
    const char* api = "linalg.lu_solve";
    square_check_inputs(LU, api, "LU");
    check_is_matrix(B, api, "B");
    if (!(left ? LU.size(-2) == B.size(-2) : LU.size(-1) == B.size(-1))) {
        TP_THROW(RuntimeError, api, ": Incompatible shapes of LU and B for the equation ",
                 left ? "AX = B" : "XA = B",
                 " (", LU.size(-2), "x", LU.size(-1), " and ",
                 B.size(-2), "x", B.size(-1), ")");
    }
    const auto batch = broadcast_batch(LU, B);
    Tensor LU_work = clone_batched_column_major(expand_to_batch(LU, batch));
    std::vector<int64_t> piv_shape = cat_batch(batch, {pivots.size(-1)});
    Tensor piv_exp = pivots.expand(piv_shape).contiguous();
    if (left) {
        Tensor B_cm = clone_batched_column_major(expand_to_batch(B, batch));
        run_real(LU.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            B_cm = solve_from_factors<T>(LU_work, piv_exp, B_cm, adjoint);
        });
        return B_cm.contiguous();
    }
    Tensor BT_cm =
        clone_batched_column_major(expand_to_batch(B, batch).transpose(-2, -1));
    run_real(LU.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        BT_cm = solve_from_factors<T>(LU_work, piv_exp, BT_cm, !adjoint);
    });
    return BT_cm.contiguous().transpose(-2, -1).contiguous();
}

// --------------------------------------------------------- geqrf / orgqr ---

template <typename scalar_t>
void apply_geqrf(const Tensor& QR_cm, const Tensor& tau) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = QR_cm.data_ptr<scalar_t>();
    auto* tau_ptr = tau.data_ptr<scalar_t>();
    const int64_t ms = matrix_stride_of(QR_cm);
    const int64_t tau_stride = tau.dim() > 1 ? tau.size(-1) : tau.numel();
    const int64_t batch = batch_count_of(QR_cm);
    const int m = static_cast<int>(QR_cm.size(-2));
    const int n = static_cast<int>(QR_cm.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::geqrf_bufferSize(handle, m, n, a, lda, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, QR_cm.dtype(),
                                QR_cm.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    Tensor dev_info = Tensor::zeros({batch}, DType::Int32, QR_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::geqrf(handle, m, n, &a[i * ms], lda,
                                 &tau_ptr[i * tau_stride], work_ptr, lwork,
                                 &info_data[i]));
    }
}

template <typename scalar_t>
void apply_orgqr(Tensor& Q_cm, const Tensor& tau) {
    using Tr = CusolverTraits<scalar_t>;
    if (Q_cm.numel() == 0) return;
    auto* a = Q_cm.data_ptr<scalar_t>();
    const auto* tau_ptr = tau.data_ptr<scalar_t>();
    const int64_t ms = matrix_stride_of(Q_cm);
    const int64_t tau_stride = tau.dim() > 1 ? tau.size(-1) : tau.numel();
    const int64_t batch = batch_count_of(Q_cm);
    const int m = static_cast<int>(Q_cm.size(-2));
    const int n = static_cast<int>(Q_cm.size(-1));
    const int k = static_cast<int>(tau.size(-1));
    const int lda = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();
    int lwork = 0;
    CUSOLVER_CHECK(Tr::orgqr_bufferSize(handle, m, n, k, a, lda, tau_ptr, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, Q_cm.dtype(),
                                Q_cm.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    Tensor dev_info = Tensor::zeros({batch}, DType::Int32, Q_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::orgqr(handle, m, n, k, &a[i * ms], lda,
                                 &tau_ptr[i * tau_stride], work_ptr, lwork,
                                 &info_data[i]));
    }
}

// C <- op(Q) C from the left, or C <- C op(Q) from the right, with Q held
// implicitly by the geqrf output (QR_cm, tau).  Every batch plane of the
// column-major C buffer takes the same reflectors.
template <typename scalar_t>
void apply_ormqr(const Tensor& QR_cm, const Tensor& tau, Tensor& C_cm,
                 bool transpose,
                 cublasSideMode_t side = CUBLAS_SIDE_LEFT) {
    using Tr = CusolverTraits<scalar_t>;
    auto* a = QR_cm.data_ptr<scalar_t>();
    auto* tau_ptr = tau.data_ptr<scalar_t>();
    auto* c = C_cm.data_ptr<scalar_t>();
    const int64_t a_ms = matrix_stride_of(QR_cm);
    const int64_t tau_stride = tau.dim() > 1 ? tau.size(-1) : tau.numel();
    const int64_t c_ms = matrix_stride_of(C_cm);
    const int64_t batch = batch_count_of(C_cm);
    const int m = static_cast<int>(C_cm.size(-2));
    const int n = static_cast<int>(C_cm.size(-1));
    const int k = static_cast<int>(tau.size(-1));
    const int lda = std::max(1, static_cast<int>(QR_cm.size(-2)));
    const int ldc = std::max(1, m);
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cublasOperation_t trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lwork = 0;
    CUSOLVER_CHECK(Tr::ormqr_bufferSize(handle, side, trans, m, n, k, a, lda,
                                        tau_ptr, c, ldc, &lwork));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, C_cm.dtype(),
                                C_cm.device());
    scalar_t* work_ptr =
        Tr::is_float ? reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<float>()))
                     : reinterpret_cast<scalar_t*>(static_cast<void*>(work.data_ptr<double>()));
    // The solver only flags an invalid argument in info, and every argument
    // was validated before dispatch, so the flag is never read back; a single
    // device slot receives each call's (ignored) status.
    Tensor dev_info = Tensor::empty({1}, DType::Int32, C_cm.device());
    auto* info_data = dev_info.data_ptr<int32_t>();
    for (int64_t i = 0; i < batch; ++i) {
        CUSOLVER_CHECK(Tr::ormqr(handle, side, trans, m, n, k, &a[i * a_ms], lda,
                                 &tau_ptr[i * tau_stride], &c[i * c_ms], ldc,
                                 work_ptr, lwork, info_data));
    }
}

template <typename scalar_t>
__global__ void triu_extract_kernel(const scalar_t* src, scalar_t* dst,
                                    int64_t ld_src, int64_t rows, int64_t cols,
                                    int64_t total) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // dst is column-major: a plane element (row, col) sits at linear offset
    // col * rows + row, so the row is the low-order residue.
    const int64_t b = idx / (rows * cols);
    const int64_t local = idx % (rows * cols);
    const int64_t row = local % rows;
    const int64_t col = local / rows;
    dst[idx] = col >= row ? src[b * ld_src * cols + col * ld_src + row]
                          : scalar_t(0);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> linalg_qr_kernel_cuda_impl(const Tensor& A,
                                                      const std::string& mode) {
    Tensor QR = clone_batched_column_major(A);
    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const int64_t k = std::min(m, n);
    const auto batch = batch_shape_of(A);
    const bool reduced = mode != "complete";
    Tensor tau = Tensor::zeros(cat_batch(batch, {k}), A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geqrf<T>(QR, tau);
    });

    const int64_t qcols = reduced ? k : m;
    const int64_t rrows = reduced ? k : m;
    const int64_t bs = linear_batch_size(batch);

    // Pack the first qcols reflector columns of each matrix into orgqr's
    // input.  Both buffers are column-major with leading dimension m, so a
    // column is one contiguous run of m elements and the packed region of a
    // matrix is one contiguous run of m*qcols elements; the loop walks the
    // batch because the packed regions of consecutive matrices do not
    // concatenate when qcols < n.
    Tensor Q_in = empty_column_major(cat_batch(batch, {m, qcols}),
                                     A.dtype(), A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        if (m > 0 && qcols > 0) {
            const int64_t qr_ms = matrix_stride_of(QR);
            for (int64_t b = 0; b < bs; ++b) {
                CUDA_CHECK(cudaMemcpyAsync(
                    Q_in.data_ptr<T>() + b * m * qcols,
                    QR.data_ptr<T>() + b * qr_ms, sizeof(T) * m * qcols,
                    cudaMemcpyDeviceToDevice,
                    getCurrentCUDAStream().stream()));
            }
        }
        apply_orgqr<T>(Q_in, tau);
    });

    // R: upper triangle of the geqrf buffer.
    Tensor R = empty_column_major(cat_batch(batch, {rrows, n}), A.dtype(),
                                  A.device());
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const int64_t total = bs * rrows * n;
        if (total > 0) {
            triu_extract_kernel<T><<<static_cast<unsigned>((total + 255) / 256),
                                     256, 0,
                                     getCurrentCUDAStream().stream()>>>(
                QR.data_ptr<T>(), R.data_ptr<T>(), m, rrows, n, total);
        }
    });
    return {Q_in.contiguous(), R.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_qr_kernel_cuda(const Tensor& A,
                                                 const std::string& mode) {
    check_is_matrix(A, "linalg.qr");
    if (mode != "reduced" && mode != "complete" && mode != "r" && mode != "R") {
        TP_THROW(RuntimeError, "linalg.qr: mode '", mode, "' not recognized.");
    }
    return run_real(A.dtype(), [&](auto tag) -> std::tuple<Tensor, Tensor> {
        using T = std::remove_pointer_t<decltype(tag)>;
        return linalg_qr_kernel_cuda_impl<T>(A, mode);
    });
}

std::tuple<Tensor, Tensor> linalg_qr_out_kernel_cuda(const Tensor& A,
                                                     const std::string& mode,
                                                     Tensor& Q, Tensor& R) {
    auto result = linalg_qr_kernel_cuda(A, mode);
    write_linalg_output("linalg.qr", std::get<0>(result), Q);
    write_linalg_output("linalg.qr", std::get<1>(result), R);
    return {Q, R};
}

// ------------------------------------------------------- geqrf / ormqr -----
// The packed Householder form of one factorization is shared by both ops: the
// reflectors live below the diagonal of the buffer geqrf writes, the scales
// in tau.  Both run on column-major planes through the device solver, and the
// result is handed back in the plain (row-major) layout the copy contract
// implies.

std::tuple<Tensor, Tensor> geqrf_cuda(const Tensor& self) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError, "geqrf(): input must have at least 2 dimensions");
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError,
                 "geqrf(): only float32 and float64 matrices are supported on "
                 "the CUDA backend");
    }
    const int64_t m = self.size(-2);
    const int64_t n = self.size(-1);
    const int64_t k = std::min(m, n);
    Tensor tau = Tensor::zeros(cat_batch(batch_shape_of(self), {k}),
                               self.dtype(), self.device());
    Tensor QR = clone_batched_column_major(self);
    if (k == 0) return {QR.contiguous(), tau};
    run_real(self.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_geqrf<T>(QR, tau);
    });
    return {QR.contiguous(), tau};
}

Tensor ormqr_cuda(const Tensor& self, const Tensor& tau, const Tensor& other,
                  bool left, bool transpose) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError, "ormqr(): input must have at least 2 dimensions");
    }
    if (self.dtype() != tau.dtype() || self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError,
                 "ormqr(): self, tau and other must share the same dtype");
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError,
                 "ormqr(): only float32 and float64 reflectors are supported "
                 "on the CUDA backend");
    }
    const int64_t m = self.size(-2);
    const int64_t cols = self.size(-1);
    const int64_t q = left ? other.size(-2) : other.size(-1);
    if (q != m) {
        TP_THROW(RuntimeError,
                 "ormqr(): reflector matrix and other disagree on the matched "
                 "dimension: got ", m, " vs ", q);
    }
    // The packed buffer always carries min(rows, cols) reflectors, however many
    // columns the factorization kept.
    const int64_t k = tau.size(-1);
    if (k != std::min(m, cols)) {
        TP_THROW(RuntimeError,
                 "ormqr(): tau must hold one scale per reflector, expected ",
                 std::min(m, cols), " but got ", k);
    }
    if (self.dim() - tau.dim() != 1) {
        TP_THROW(RuntimeError,
                 "ormqr(): tau must have exactly one dimension less than the "
                 "reflector matrix, got ", tau.dim(), " and ", self.dim());
    }
    if (self.dim() != other.dim()) {
        TP_THROW(RuntimeError,
                 "ormqr(): reflector matrix and other must have the same rank, "
                 "got ", self.dim(), " and ", other.dim());
    }
    const auto batch = batch_shape_of(self);
    if (self.dim() > 2) {
        const Size t_shape = tau.shape();
        const std::vector<int64_t> tau_batch(t_shape.begin(), t_shape.end() - 1);
        const auto other_batch = batch_shape_of(other);
        if (tau_batch != batch) {
            TP_THROW(RuntimeError,
                     "ormqr(): tau must carry the same batch extents as the "
                     "reflector matrix");
        }
        if (other_batch != batch) {
            TP_THROW(RuntimeError,
                     "ormqr(): other must carry the same batch extents as the "
                     "reflector matrix");
        }
    }
    if (k == 0 || q == 0) return other.clone();
    Tensor tau_c = tau.contiguous();
    const Tensor QR = clone_batched_column_major(self);
    // The solver writes into a column-major plane; that plane is already the
    // batched-column-major view the result is returned in, so no row-major
    // conversion pass is needed on the way out.
    if (left) {
        Tensor C = clone_batched_column_major(other);
        run_real(self.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            apply_ormqr<T>(QR, tau_c, C, transpose);
        });
        return C;
    }
    // From the right the reflectors act on the columns of `other`, which is
    // exactly the side the solver applies them to when told so; no transposed
    // staging copy is needed.
    Tensor C = clone_batched_column_major(other);
    run_real(self.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_ormqr<T>(QR, tau_c, C, transpose, CUBLAS_SIDE_RIGHT);
    });
    return C;
}

std::tuple<Tensor, Tensor> linalg_polar_kernel_cuda(const Tensor& A) {
    check_is_matrix(A, "linalg.polar");
    if (A.size(-2) < A.size(-1)) {
        TP_THROW(RuntimeError,
                 "linalg.polar: input must have at least as many rows as columns, but got ",
                 A.size(-2), " by ", A.size(-1), " matrices");
    }

    auto [Up, S, Vh] = ops::linalg_svd(A, false, std::nullopt);
    Tensor scaled_vh = ops::mul(ops::unsqueeze(S, -1), Vh);
    Tensor H = ops::matmul(ops::mH(Vh), scaled_vh);
    H = ops::mul(ops::add(H, ops::mH(H)), Scalar(0.5));
    Tensor U = ops::matmul(Up, Vh);
    return {U.contiguous(), H.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_polar_out_kernel_cuda(const Tensor& A,
                                                        Tensor& U, Tensor& H) {
    auto result = linalg_polar_kernel_cuda(A);
    write_linalg_output("linalg.polar", std::get<0>(result), U);
    write_linalg_output("linalg.polar", std::get<1>(result), H);
    return {U, H};
}

Tensor linalg_householder_product_kernel_cuda(const Tensor& input, const Tensor& tau) {
    const char* api = "linalg.householder_product";
    check_is_matrix(input, api);
    if (input.size(-2) < input.size(-1)) {
        TP_THROW(RuntimeError, api, ": If input has size (..., m, n), "
                                    "n must be less than or equal to m, but got n = ",
                 input.size(-1), " and m = ", input.size(-2));
    }
    if (tau.dim() < 1 || tau.size(-1) != std::min(input.size(-2), input.size(-1))) {
        TP_THROW(RuntimeError, api,
                 ": If tau has size (..., k), then when input has size (..., m, n) "
                 "we require k == min(m, n)");
    }
    if (tau.dtype() != input.dtype()) {
        TP_THROW(RuntimeError, api, ": input and tau must have the same dtype");
    }
    Tensor result = clone_batched_column_major(input);
    run_real(input.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_orgqr<T>(result, tau);
    });
    return result.contiguous();
}

Tensor& linalg_householder_product_out_kernel_cuda(const Tensor& input,
                                                   const Tensor& tau,
                                                   Tensor& out) {
    write_linalg_output("linalg.householder_product",
                        linalg_householder_product_kernel_cuda(input, tau), out);
    return out;
}

// --------------------------------------------------------------- syevj ----

template <typename scalar_t>
void apply_syevj_batched(const Tensor& values, const Tensor& vectors,
                         const Tensor& infos, bool upper,
                         bool compute_eigenvectors) {
    using Tr = CusolverTraits<scalar_t>;
    const int n = static_cast<int>(vectors.size(-1));
    const int lda = std::max(1, n);
    const int batch = static_cast<int>(batch_count_of(vectors));
    auto* vectors_data = vectors.data_ptr<scalar_t>();
    auto* values_data = values.data_ptr<scalar_t>();
    auto* infos_data = infos.data_ptr<int32_t>();
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER
                                        : CUBLAS_FILL_MODE_LOWER;
    const cusolverEigMode_t jobz =
            compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR
                                  : CUSOLVER_EIG_MODE_NOVECTOR;

    syevjInfo_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));
    CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(params, 1));

    int lwork = 0;
    CUSOLVER_CHECK(Tr::syevj_bufferSize(
            handle, jobz, uplo, n, vectors_data, lda, values_data, &lwork,
            params, batch));
    Tensor work = Tensor::empty({std::max(lwork, 1)}, vectors.dtype(),
                                 vectors.device());
    auto* work_data = work.data_ptr<scalar_t>();
    CUSOLVER_CHECK(Tr::syevj(
            handle, jobz, uplo, n, vectors_data, lda, values_data, work_data,
            lwork, infos_data, params, batch));
    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(params));
}

// 64-bit batched driver: same Jacobi semantics as the legacy syevjBatched,
// but computes through the typed-params entry point that the current
// cuSOLVER optimizes for large batches.
template <typename scalar_t>
void apply_xsyev_batched(const Tensor& values, const Tensor& vectors,
                         const Tensor& infos, bool upper,
                         bool compute_eigenvectors) {
    const int n = static_cast<int>(vectors.size(-1));
    const int lda = std::max(1, n);
    const int64_t batch = static_cast<int64_t>(batch_count_of(vectors));
    auto* vectors_data = vectors.data_ptr<scalar_t>();
    auto* values_data = values.data_ptr<scalar_t>();
    auto* infos_data = infos.data_ptr<int32_t>();
    const auto handle = CUDAContext::getCusolverDnHandle();
    const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER
                                        : CUBLAS_FILL_MODE_LOWER;
    const cusolverEigMode_t jobz =
            compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR
                                  : CUSOLVER_EIG_MODE_NOVECTOR;
    const cudaDataType dtype = std::is_same_v<scalar_t, float>
            ? CUDA_R_32F : CUDA_R_64F;

    cusolverDnParams_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    size_t ws_dev = 0, ws_host = 0;
    CUSOLVER_CHECK(cusolverDnXsyevBatched_bufferSize(
            handle, params, jobz, uplo, n, dtype, vectors_data, lda, dtype,
            values_data, dtype, &ws_dev, &ws_host, batch));
    Tensor work = Tensor::empty(
            {static_cast<int64_t>((ws_dev + sizeof(scalar_t) - 1) / sizeof(scalar_t) + 1)},
            vectors.dtype(), vectors.device());
    std::vector<char> work_host(ws_host);
    CUSOLVER_CHECK(cusolverDnXsyevBatched(
            handle, params, jobz, uplo, n, dtype, vectors_data, lda, dtype,
            values_data, dtype, work.data_ptr<scalar_t>(), ws_dev,
            work_host.data(), ws_host, infos_data, batch));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
}

std::tuple<Tensor, Tensor> eigh_impl_cuda(const Tensor& A, bool upper,
                                          bool compute_eigenvectors) {
    square_check_inputs(A, "linalg.eigh");
    const auto batch = batch_shape_of(A);
    const int64_t n = A.size(-1);
    if (A.numel() == 0) {
        Tensor values = Tensor::empty(cat_batch(batch, {n}), A.dtype(),
                                      A.device());
        Tensor vectors = compute_eigenvectors
                ? Tensor::empty(cat_batch(batch, {n, n}), A.dtype(), A.device())
                : Tensor::empty({0}, A.dtype(), A.device());
        return {values, vectors};
    }
    Tensor vectors = clone_batched_column_major(A);
    Tensor values = Tensor::empty(cat_batch(batch, {n}), A.dtype(), A.device());
    Tensor info = Tensor::zeros(batch, DType::Int32, A.device());
    // The 64-bit typed-params batched driver is the entry point the current
    // cuSOLVER optimizes for batched eigensolves; it replaces both the
    // per-matrix syevd loop and the legacy syevjBatched pass.
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        apply_xsyev_batched<T>(values, vectors, info, upper,
                               compute_eigenvectors);
    });
    check_infos(info, "linalg.eigh", A.dim() == 2);
    return {values.contiguous(), vectors.contiguous()};
}

std::tuple<Tensor, Tensor> linalg_eigh_internal_kernel_cuda(
        const Tensor& A, const std::string& UPLO, bool compute_v) {
    if (UPLO != "U" && UPLO != "L") {
        TP_THROW(RuntimeError, "linalg.eigh: UPLO argument must be 'U' or 'L', got ", UPLO);
    }
    return eigh_impl_cuda(A, UPLO == "U", compute_v);
}

std::tuple<Tensor, Tensor> linalg_eigh_internal_out_kernel_cuda(
        const Tensor& A, const std::string& UPLO, bool compute_v, Tensor& values,
        Tensor& vectors) {
    auto result = linalg_eigh_internal_kernel_cuda(A, UPLO, compute_v);
    write_linalg_output("linalg.eigh", std::get<0>(result), values);
    write_linalg_output("linalg.eigh", std::get<1>(result), vectors);
    return {values, vectors};
}

std::tuple<Tensor, Tensor> linalg_eigh_eigvals_out_kernel_cuda(
        const Tensor& A, const std::string& UPLO, Tensor& values, Tensor& vectors) {
    return linalg_eigh_internal_out_kernel_cuda(A, UPLO, true, values, vectors);
}

Tensor& linalg_eigvalsh_out_kernel_cuda(const Tensor& A, const std::string& UPLO,
                                        Tensor& out) {
    auto result = linalg_eigh_internal_kernel_cuda(A, UPLO, false);
    write_linalg_output("linalg.eigvalsh", std::get<0>(result), out);
    return out;
}

// ---------------------------------------------------- triangular solve -----

Tensor linalg_solve_triangular_kernel_cuda(const Tensor& A, const Tensor& B,
                                           bool upper, bool left, bool unitriangular) {
    const char* api = "linalg.solve_triangular";
    check_is_matrix(A, api, "A");
    check_is_matrix(B, api, "B");
    const auto batch = broadcast_batch(A, B);
    Tensor B_cm = clone_batched_column_major(expand_to_batch(B, batch));
    Tensor A_cm = clone_batched_column_major(expand_to_batch(A, batch));
    const int64_t bs = linear_batch_size(batch);
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        auto* a = A_cm.data_ptr<T>();
        auto* b = B_cm.data_ptr<T>();
        const cublasSideMode_t side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
        const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        const cublasDiagType_t diag =
            unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
        const int m = static_cast<int>(B.size(-2));
        const int n = static_cast<int>(B.size(-1));
        const int lda = std::max<int>(1, static_cast<int>(A.size(-2)));
        const int ldb = std::max(1, m);
        const T alpha = T(1);
        const int64_t a_ms = matrix_stride_of(A_cm);
        const int64_t b_ms = matrix_stride_of(B_cm);
        const auto handle = CUDAContext::getCublasHandle();
        for (int64_t i = 0; i < bs; ++i) {
            if constexpr (std::is_same_v<T, float>) {
                CUBLAS_CHECK(cublasStrsm(handle, side, uplo, CUBLAS_OP_N, diag,
                                         m, n, &alpha, &a[i * a_ms], lda,
                                         &b[i * b_ms], ldb));
            } else {
                CUBLAS_CHECK(cublasDtrsm(handle, side, uplo, CUBLAS_OP_N, diag,
                                         m, n, &alpha, &a[i * a_ms], lda,
                                         &b[i * b_ms], ldb));
            }
        }
    });
    return B_cm.contiguous();
}

// ----------------------------------------------------- cholesky solve -----

Tensor cholesky_solve_kernel_cuda(const Tensor& self, const Tensor& input2,
                                  bool upper) {
    const char* api = "cholesky_solve";
    check_is_matrix(self, api, "self");
    square_check_inputs(input2, api, "input2");
    if (self.dtype() != input2.dtype()) {
        TP_THROW(RuntimeError, api, ": Expected self and input2 to have the same "
                 "dtype, but found self of type ", scalarTypeName(self.dtype()),
                 " and input2 of type ", scalarTypeName(input2.dtype()),
                 " instead.");
    }
    if (self.size(-2) != input2.size(-1)) {
        TP_THROW(RuntimeError, api, ": input2.size(-1) must equal self.size(-2), "
                 "got ", input2.size(-1), " and ", self.size(-2));
    }
    // Both operands broadcast to one batch; the solver consumes the
    // column-major planes and writes the solution over the right-hand side.
    const auto batch = broadcast_batch(self, input2);
    Tensor B_cm = clone_batched_column_major(expand_to_batch(self, batch));
    Tensor A_cm = clone_batched_column_major(expand_to_batch(input2, batch));
    const int64_t bs = linear_batch_size(batch);
    const int n = static_cast<int>(input2.size(-1));
    const int nrhs = static_cast<int>(self.size(-1));
    const int lda = std::max(1, static_cast<int>(input2.size(-2)));
    const int ldb = std::max(1, static_cast<int>(self.size(-2)));
    const cublasFillMode_t uplo =
        upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    run_real(self.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        using Tr = CusolverTraits<T>;
        auto* a = A_cm.data_ptr<T>();
        auto* b = B_cm.data_ptr<T>();
        const int64_t a_ms = matrix_stride_of(A_cm);
        const int64_t b_ms = matrix_stride_of(B_cm);
        const auto handle = CUDAContext::getCusolverDnHandle();
        // info only flags an invalid argument, which the validation above has
        // already ruled out, so the status is never read back.
        Tensor info = Tensor::empty({1}, DType::Int32, self.device());
        int32_t* info_data = info.data_ptr<int32_t>();
        for (int64_t i = 0; i < bs; ++i) {
            CUSOLVER_CHECK(Tr::potrs(handle, uplo, n, nrhs, &a[i * a_ms], lda,
                                     &b[i * b_ms], ldb, info_data));
        }
    });
    return B_cm.contiguous();
}

// ------------------------------------------------------------------ lstsq --

// Squares and sums the trailing rows of the column-major working RHS. After
// the QR solve those rows hold Q^T B below the solution block, i.e. the
// residual vector of every right-hand side.
template <typename scalar_t>
__global__ void lstsq_residual_kernel(const scalar_t* b_cm, scalar_t* res,
                                      int64_t ldb, int64_t nrhs,
                                      int64_t row_begin, int64_t row_end,
                                      int64_t count) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const int64_t i = idx / nrhs;
    const int64_t col = idx % nrhs;
    const scalar_t* column = b_cm + i * ldb * nrhs + col * ldb;
    scalar_t acc = scalar_t(0);
    for (int64_t r = row_begin; r < row_end; ++r) acc += column[r] * column[r];
    res[idx] = acc;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> linalg_lstsq_kernel_cuda(
        const Tensor& A, const Tensor& B, std::optional<double> rcond,
        const std::optional<std::string>& driver_opt) {
    (void)rcond;  // the QR driver assumes full column rank, so no cutoff
    const char* api = "linalg.lstsq";
    const std::string driver = driver_opt.value_or("gels");
    if (driver != "gels") {
        TP_THROW(NotImplementedError, api, ": driver '", driver,
                 "' is not supported on CUDA; only 'gels' is implemented");
    }
    check_is_matrix(A, api);
    if (B.dim() < 1) {
        TP_THROW(RuntimeError, api, ": B must have at least 1 dimension");
    }
    const int64_t dim_diff = A.dim() - B.dim();
    if (dim_diff < 0 || dim_diff > 1) {
        TP_THROW(RuntimeError, api,
                 ": A and B must have compatible numbers of dimensions");
    }
    if (A.device() != B.device()) {
        TP_THROW(DeviceMismatchError, api, ": A and B must be on the same device");
    }
    if (A.dtype() != B.dtype()) {
        TP_THROW(RuntimeError, api, ": A and B must have the same dtype, but got ",
                 scalarTypeName(A.dtype()), " and ", scalarTypeName(B.dtype()));
    }
    bool vector_case = B.dim() == 1;
    if (!vector_case && A.dim() - 1 == B.dim()) {
        vector_case = true;
        for (int64_t i = 0; i < A.dim() - 1; ++i) {
            if (A.size(i) != B.size(i)) {
                vector_case = false;
                break;
            }
        }
    }
    Tensor B_2d = vector_case ? B.unsqueeze(-1) : B;
    if (A.size(-2) != B_2d.size(-2)) {
        TP_THROW(RuntimeError, api, ": A and B have incompatible row dimensions: ",
                 A.size(-2), " and ", B_2d.size(-2));
    }

    const int64_t m = A.size(-2);
    const int64_t n = A.size(-1);
    const auto batch = broadcast_batch(A, B_2d);
    const int64_t nrhs = B_2d.size(-1);
    const int64_t ldb = std::max<int64_t>({int64_t{1}, m, n});
    const int64_t bs = linear_batch_size(batch);

    // The RHS lives in a column-major working buffer with max(m, n) rows: the
    // strided copy fills the first m rows, a tall system later overwrites rows
    // n..m-1 with the residual vectors, and a wide one zeroes its tail so the
    // free components of Z vanish before Q multiplies back onto it.
    Tensor B_cm = empty_column_major(cat_batch(batch, {ldb, nrhs}),
                                     B.dtype(), B.device());
    B_cm.slice(-2, 0, m).copy_(expand_to_batch(B_2d, batch));

    if (m > 0 && n > 0 && nrhs > 0) {
        run_real(B.dtype(), [&](auto tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            const T alpha = T(1);
            const auto handle = CUDAContext::getCublasHandle();
            if (m >= n) {
                Tensor A_cm =
                    clone_batched_column_major(expand_to_batch(A, batch));
                Tensor tau = Tensor::zeros(cat_batch(batch, {n}), A.dtype(),
                                           A.device());
                apply_geqrf<T>(A_cm, tau);
                // B <- Q^T B over all m rows, then R X = Y on the leading n
                // rows leaves the residual vectors in rows n..m-1.
                apply_ormqr<T>(A_cm, tau, B_cm, true);
                for (int64_t i = 0; i < bs; ++i) {
                    if constexpr (std::is_same_v<T, float>) {
                        CUBLAS_CHECK(cublasStrsm(
                            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            static_cast<int>(n), static_cast<int>(nrhs), &alpha,
                            A_cm.data_ptr<float>() + i * matrix_stride_of(A_cm),
                            static_cast<int>(m),
                            B_cm.data_ptr<float>() + i * ldb * nrhs,
                            static_cast<int>(ldb)));
                    } else {
                        CUBLAS_CHECK(cublasDtrsm(
                            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            static_cast<int>(n), static_cast<int>(nrhs), &alpha,
                            A_cm.data_ptr<double>() + i * matrix_stride_of(A_cm),
                            static_cast<int>(m),
                            B_cm.data_ptr<double>() + i * ldb * nrhs,
                            static_cast<int>(ldb)));
                    }
                }
            } else {
                // Wide system: factor A^H = Q R, solve R^H Z = B into the
                // first m rows (the zero tail completes Z), then X = Q Z.
                Tensor Ah_cm =
                    clone_batched_column_major(expand_to_batch(A, batch).mH());
                Tensor tau = Tensor::zeros(cat_batch(batch, {m}), A.dtype(),
                                           A.device());
                apply_geqrf<T>(Ah_cm, tau);
                for (int64_t i = 0; i < bs; ++i) {
                    if constexpr (std::is_same_v<T, float>) {
                        CUBLAS_CHECK(cublasStrsm(
                            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                            static_cast<int>(m), static_cast<int>(nrhs), &alpha,
                            Ah_cm.data_ptr<float>() + i * matrix_stride_of(Ah_cm),
                            static_cast<int>(n),
                            B_cm.data_ptr<float>() + i * ldb * nrhs,
                            static_cast<int>(ldb)));
                    } else {
                        CUBLAS_CHECK(cublasDtrsm(
                            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                            static_cast<int>(m), static_cast<int>(nrhs), &alpha,
                            Ah_cm.data_ptr<double>() + i * matrix_stride_of(Ah_cm),
                            static_cast<int>(n),
                            B_cm.data_ptr<double>() + i * ldb * nrhs,
                            static_cast<int>(ldb)));
                    }
                }
                // The untouched tail rows hold the free components of Z;
                // zero them so Q multiplies onto the exact minimizer.
                B_cm.slice(-2, m, n).zero_();
                apply_ormqr<T>(Ah_cm, tau, B_cm, false);
            }
        });
    }

    Tensor solution = B_cm.contiguous().slice(-2, 0, n).contiguous();
    if (vector_case) solution = solution.squeeze(-1);

    Tensor residuals;
    if (m > n) {
        residuals = Tensor::empty(cat_batch(batch, {nrhs}), B.dtype(), B.device());
        const int64_t count = bs * nrhs;
        if (count > 0) {
            run_real(B.dtype(), [&](auto tag) {
                using T = std::remove_pointer_t<decltype(tag)>;
                const int block = 256;
                const int grid =
                    static_cast<int>((count + block - 1) / block);
                lstsq_residual_kernel<T><<<grid, block, 0,
                                           getCurrentCUDAStream().stream()>>>(
                    B_cm.data_ptr<T>(), residuals.data_ptr<T>(), ldb, nrhs, n,
                    m, count);
            });
        }
    } else {
        residuals = Tensor::empty({0}, B.dtype(), B.device());
    }
    // The QR driver assumes full rank: no per-batch rank and no singular
    // values are reported.
    Tensor rank = Tensor::empty({0}, DType::Int64, B.device());
    Tensor singular_values =
        Tensor::empty({0}, B.dtype(), B.device());
    return {solution, residuals, rank, singular_values};
}

// -------------------------------------------------------- lu with unpack ---

// A = P L U for any m x n matrix: the packed factorization unpacked on the
// device into its permutation and its two triangles.
std::tuple<Tensor, Tensor, Tensor> linalg_lu_kernel_cuda(const Tensor& A, bool pivot) {
    (void)pivot;
    check_is_matrix(A, "linalg.lu");
    auto [LU, pivots, info] = lu_factor_ex_cuda_impl(A, false);
    (void)info;
    return ops::lu_unpack(LU, pivots, true, true);
}

// ------------------------------------------------------- eig (no MAGMA) ----

[[noreturn]] void throw_no_magma(const char* api) {
    TP_THROW(RuntimeError, "Calling ", api,
             " on a CUDA tensor requires MAGMA support. "
             "Please move the tensor to the CPU.");
}

std::tuple<Tensor, Tensor> linalg_eig_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.eig");
    throw_no_magma("linalg.eig");
}

Tensor linalg_eigvals_kernel_cuda(const Tensor& A) {
    square_check_inputs(A, "linalg.eigvals");
    throw_no_magma("linalg.eigvals");
}

std::tuple<Tensor, Tensor> linalg_eig_out_kernel_cuda(const Tensor& A,
                                                      Tensor& values,
                                                      Tensor& vectors) {
    auto result = linalg_eig_kernel_cuda(A);
    write_linalg_output("linalg.eig", std::get<0>(result), values);
    write_linalg_output("linalg.eig", std::get<1>(result), vectors);
    return {values, vectors};
}

Tensor& linalg_eigvals_out_kernel_cuda(const Tensor& A, Tensor& values) {
    write_linalg_output("linalg.eigvals", linalg_eigvals_kernel_cuda(A), values);
    return values;
}

// --------------------------------------------------------------- diagonal --

template <typename scalar_t>
__global__ void diagonal_gather_kernel(
    const scalar_t* src, scalar_t* dst, int64_t total, int64_t diag_len,
    int64_t outer, int64_t s_outer, int64_t s1, int64_t s2, int64_t base_off1,
    int64_t base_off2) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t t = idx % diag_len;
    const int64_t o = idx / diag_len;
    dst[idx] = src[o * s_outer + base_off1 + t * s1 + base_off2 + t * s2];
}

Tensor linalg_diagonal_kernel_cuda(const Tensor& A, int64_t offset, int64_t dim1,
                                   int64_t dim2) {
    const char* api = "linalg.diagonal";
    check_is_matrix(A, api);
    const int64_t ndim = A.dim();
    const auto norm_dim = [&](int64_t d) {
        const int64_t r = d < 0 ? d + ndim : d;
        if (r < 0 || r >= ndim) {
            TP_THROW(RuntimeError, "Dimension out of range");
        }
        return r;
    };
    dim1 = norm_dim(dim1);
    dim2 = norm_dim(dim2);
    if (dim1 == dim2) {
        TP_THROW(RuntimeError, api, ": dimension 1 and dimension 2 cannot be equal");
    }
    // Move (dim1, dim2) to trailing positions via permute for a flat gather.
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < ndim; ++i)
        if (i != dim1 && i != dim2) perm.push_back(i);
    perm.push_back(dim1);
    perm.push_back(dim2);
    Tensor work = A.permute(perm).contiguous();
    const int64_t d1 = work.size(-2);
    const int64_t d2 = work.size(-1);
    const int64_t outer = work.numel() / (d1 * d2 == 0 ? 1 : d1 * d2);
    const int64_t diag_len =
        offset >= 0 ? std::max<int64_t>(std::min(d1, d2 - offset), 0)
                    : std::max<int64_t>(std::min(d1 + offset, d2), 0);
    Tensor out = Tensor::empty(cat_batch(std::vector<int64_t>{outer}, {diag_len}),
                               A.dtype(), A.device());
    if (out.numel() == 0) return out.reshape([&] {
        std::vector<int64_t> shape(perm.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i)
            shape[i] = i == shape.size() - 1 ? diag_len
                                             : A.size(static_cast<int64_t>(perm[i]));
        return shape;
    }());

    std::vector<int64_t> strides(ndim);
    {
        std::vector<int64_t> st(ndim, 1);
        for (int64_t i = ndim - 2; i >= 0; --i) st[i] = st[i + 1] * A.size(i + 1);
        strides = st;
    }
    const int64_t s_outer = [&] {
        int64_t acc = 1;
        bool first = true;
        for (int64_t i = 0; i < ndim; ++i) {
            if (i == dim1 || i == dim2) continue;
            if (first) { acc = strides[i]; first = false; }
        }
        return first ? 0 : acc;
    }();
    // Base offsets fold the fixed index of every non-gathered axis.
    (void)s_outer;
    // Simpler correct path: iterate with explicit strides over the two dims.
    const int64_t total = outer * diag_len;
    run_real(A.dtype(), [&](auto tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const int64_t inner_block = d1 * d2;
        diagonal_gather_kernel<T><<<static_cast<unsigned>((total + 255) / 256),
                                    256, 0,
                                    getCurrentCUDAStream().stream()>>>(
            work.data_ptr<T>(), out.data_ptr<T>(), total, diag_len, outer,
            inner_block,
            /*s1=*/strides[ndim - 2], /*s2=*/strides[ndim - 1],
            offset >= 0 ? 0 : -offset * strides[ndim - 2],
            offset >= 0 ? offset * strides[ndim - 1] : 0);
    });
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim1 && i != dim2) out_shape.push_back(A.size(i));
    }
    out_shape.push_back(diag_len);
    return out.reshape(out_shape);
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, LinalgKernels) {
    m.impl("_linalg_check_errors", linalg_check_errors_kernel);
    m.impl("linalg_cholesky_ex", linalg_cholesky_ex_kernel_cuda);
    m.impl("linalg_inv_ex", linalg_inv_ex_kernel_cuda);
    m.impl("_linalg_det", linalg_det_internal_kernel_cuda);
    m.impl("_linalg_det.result", linalg_det_internal_out_kernel_cuda);
    m.impl("_linalg_slogdet", linalg_slogdet_internal_kernel_cuda);
    m.impl("_linalg_slogdet.sign", linalg_slogdet_internal_out_kernel_cuda);
    m.impl("_linalg_solve_ex", linalg_solve_ex_internal_kernel_cuda);
    m.impl("_linalg_solve_ex.result", linalg_solve_ex_internal_out_kernel_cuda);
    m.impl("linalg_lu_factor_ex", linalg_lu_factor_ex_kernel_cuda);
    m.impl("linalg_lu", linalg_lu_kernel_cuda);
    m.impl("linalg_lu_solve", linalg_lu_solve_kernel_cuda);
    m.impl("linalg_solve_triangular", linalg_solve_triangular_kernel_cuda);
    m.impl("cholesky_solve", cholesky_solve_kernel_cuda);
    m.impl("_linalg_eigh", linalg_eigh_internal_kernel_cuda);
    m.impl("_linalg_eigh.eigenvalues", linalg_eigh_internal_out_kernel_cuda);
    m.impl("linalg_eigh.eigvals", linalg_eigh_eigvals_out_kernel_cuda);
    m.impl("linalg_eigvalsh.out", linalg_eigvalsh_out_kernel_cuda);
    m.impl("linalg_eig", linalg_eig_kernel_cuda);
    m.impl("linalg_eig.out", linalg_eig_out_kernel_cuda);
    m.impl("_linalg_eigvals", linalg_eigvals_kernel_cuda);
    m.impl("linalg_eigvals.out", linalg_eigvals_out_kernel_cuda);
    m.impl("linalg_polar", linalg_polar_kernel_cuda);
    m.impl("linalg_polar.out", linalg_polar_out_kernel_cuda);
    m.impl("linalg_lstsq", linalg_lstsq_kernel_cuda);
    m.impl("linalg_qr", linalg_qr_kernel_cuda);
    m.impl("linalg_qr.out", linalg_qr_out_kernel_cuda);
    m.impl("geqrf", geqrf_cuda);
    m.impl("ormqr", ormqr_cuda);
    m.impl("linalg_householder_product", linalg_householder_product_kernel_cuda);
    m.impl("linalg_householder_product.out", linalg_householder_product_out_kernel_cuda);
    m.impl("linalg_diagonal", linalg_diagonal_kernel_cuda);
}

namespace {


// ---------------------------------------------------------------------------
// _cholesky_solve_helper: the dispatch-level spelling of cholesky_solve.
// ---------------------------------------------------------------------------

Tensor interop__cholesky_solve_helper_cuda(const Tensor& self, const Tensor& A,
                                           bool upper) {
    return dispatch_cuda<Tensor>("cholesky_solve", self, A, upper);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, LinalgInterop) {
    // linear algebra helper spellings
    m.impl("_cholesky_solve_helper", interop__cholesky_solve_helper_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
