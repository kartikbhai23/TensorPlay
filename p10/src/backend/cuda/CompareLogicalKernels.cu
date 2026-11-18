// Core operators - CUDA kernels: comparison, logical-connective and floating-point predicate
// operators.
//
// Pointwise kernels over a grid-stride loop; the shared plumbing comes
// from the pointwise header and the per-operator bodies live here.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {
namespace {

Tensor greater_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn10{}, "greater");
}

Tensor greater_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn11{}, "greater_equal");
}

Tensor less_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn12{}, "less");
}

Tensor less_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn13{}, "less_equal");
}

Tensor not_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn14{}, "not_equal");
}

Tensor signbit_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn15{}, "signbit");
}

Tensor logical_not_cuda(const Tensor& self) {
    return logical_unary_cuda(self, HFn16{},
                              "logical_not");
}

Tensor& logical_not_out_cuda(const Tensor& self, Tensor& out) {
    return logical_unary_out(self, out, HFn16{},
                             "logical_not");
}

// Dtype-preserving in-place negation functor: nvcc rejects generic
// __host__ __device__ lambdas, so the operation is a callable struct.
struct LogicalNotDtypeFn {
    template <typename T>
    __host__ __device__ T operator()(T value) const {
        return value == T(0) ? T(1) : T(0);
    }
};

Tensor& logical_not__cuda(Tensor& self) {
    // Dtype-preserving in-place negation: one kernel writing `self`
    // directly.  Routing through the out= form would build a Bool temporary
    // and cast it back, doubling the kernel count on every call.
    dtype_unary_into(self, self, LogicalNotDtypeFn{}, "logical_not_");
    return self;
}

// Dtype-preserving in-place binary connective functor: nvcc rejects generic
// __host__ __device__ lambdas, so the operation is a callable struct.
struct LogicalAndDtypeFn {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        const bool truth = (a != T(0)) && (b != T(0));
        return truth ? T(1) : T(0);
    }
};

struct LogicalOrDtypeFn {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        const bool truth = (a != T(0)) || (b != T(0));
        return truth ? T(1) : T(0);
    }
};

struct LogicalXorDtypeFn {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        const bool truth = (a != T(0)) != (b != T(0));
        return truth ? T(1) : T(0);
    }
};

template <typename F>
Tensor& logical_binary_dtype_into(Tensor& self, const Tensor& other, F f,
                                  const char* name) {
    if (self.numel() == 0) return self;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(self)
        .add_const_input(self)
        .add_const_input(other)
        .build();
#define TP_LOGICAL_BIN_INPLACE(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [f] __host__ __device__(ctype a, ctype b) -> ctype { \
            return f(a, b); \
        }); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_LOGICAL_BIN_INPLACE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_LOGICAL_BIN_INPLACE
    CUDA_CHECK(cudaGetLastError());
    return self;
}

Tensor& logical_and__cuda(Tensor& self, const Tensor& other) {
    if (other.dtype() != self.dtype()) {
        logical_binary_dtype_into(self, other.to(self.dtype()),
                                  LogicalAndDtypeFn{}, "logical_and_");
        return self;
    }
    return logical_binary_dtype_into(self, other, LogicalAndDtypeFn{},
                                     "logical_and_");
}

Tensor& logical_or__cuda(Tensor& self, const Tensor& other) {
    if (other.dtype() != self.dtype()) {
        logical_binary_dtype_into(self, other.to(self.dtype()),
                                  LogicalOrDtypeFn{}, "logical_or_");
        return self;
    }
    return logical_binary_dtype_into(self, other, LogicalOrDtypeFn{},
                                     "logical_or_");
}

Tensor& logical_xor__cuda(Tensor& self, const Tensor& other) {
    if (other.dtype() != self.dtype()) {
        logical_binary_dtype_into(self, other.to(self.dtype()),
                                  LogicalXorDtypeFn{}, "logical_xor_");
        return self;
    }
    return logical_binary_dtype_into(self, other, LogicalXorDtypeFn{},
                                     "logical_xor_");
}

Tensor logical_and_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn17{}, "logical_and");
}

Tensor logical_or_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn18{}, "logical_or");
}

Tensor logical_xor_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn19{}, "logical_xor");
}

Tensor isfinite_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn20{}, "isfinite");
}

Tensor isinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn21{}, "isinf");
}

Tensor isnan_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn22{}, "isnan");
}

Tensor isneginf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn23{}, "isneginf");
}

Tensor isposinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn24{}, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor isnan_cuda(const Tensor& self);

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CompareLogicalKernels) {
    m.impl("greater", greater_cuda);
    m.impl("greater_equal", greater_equal_cuda);
    m.impl("less", less_cuda);
    m.impl("less_equal", less_equal_cuda);
    m.impl("not_equal", not_equal_cuda);
    m.impl("signbit", signbit_cuda);
    m.impl("logical_not", logical_not_cuda);
    m.impl("logical_not.out", logical_not_out_cuda);
    m.impl("logical_not_", logical_not__cuda);
    m.impl("logical_and", logical_and_cuda);
    m.impl("logical_or", logical_or_cuda);
    m.impl("logical_xor", logical_xor_cuda);
    m.impl("logical_and_", logical_and__cuda);
    m.impl("logical_or_", logical_or__cuda);
    m.impl("logical_xor_", logical_xor__cuda);
    m.impl("isfinite", isfinite_cuda);
    m.impl("isinf", isinf_cuda);
    m.impl("isnan", isnan_cuda);
    m.impl("isneginf", isneginf_cuda);
    m.impl("isposinf", isposinf_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
