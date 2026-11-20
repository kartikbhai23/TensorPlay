#include "CudaDispatchHelpers.cuh"
#include "Complex.h"

#include <cstring>
#include <type_traits>

namespace tensorplay {
namespace cuda {


namespace {


template <typename T>
Scalar scalar_from_bytes(const void* bytes) {
    if constexpr (std::is_same_v<T, bool>) {
        return Scalar(*static_cast<const unsigned char*>(bytes) != 0);
    } else {
        T value;
        std::memcpy(&value, bytes, sizeof(T));
        if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
            return Scalar(static_cast<uint64_t>(value));
        } else if constexpr (std::is_integral_v<T>) {
            return Scalar(static_cast<int64_t>(value));
        } else if constexpr (std::is_floating_point_v<T>) {
            return Scalar(static_cast<double>(value));
        } else if constexpr (std::is_same_v<T, complex<double>> ||
                             std::is_same_v<T, complex<float>>) {
            return Scalar(value);
        } else if constexpr (!std::is_same_v<typename scalar_value_type<T>::type, T>) {
            return Scalar(complex<float>(static_cast<float>(value.real()),
                                         static_cast<float>(value.imag())));
        } else {
            return Scalar(static_cast<float>(value));
        }
    }
}

Scalar interop__local_scalar_dense_cuda(const Tensor& self) {
    TP_CHECK(self.numel() == 1,
             "_local_scalar_dense only supports 1-element tensors, got ",
             self.numel());
    OptionalCUDAGuard guard(self.device());
    const auto stream = getCurrentCUDAStream();
    auto host = getPinnedMemoryAllocator()->allocate(self.itemsize());
    checkCuda(cudaMemcpyAsync(host.get(), self.const_data_ptr(), self.itemsize(),
                              cudaMemcpyDeviceToHost, stream.stream()),
              "_local_scalar_dense copy");
    stream.synchronize();

#define TP_READ_SCALAR(ctype, name) \
    case DType::name: return scalar_from_bytes<ctype>(host.get());
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(TP_READ_SCALAR)
        default:
            TP_THROW(TypeError, "_local_scalar_dense: unsupported dtype ",
                     toString(self.dtype()));
    }
#undef TP_READ_SCALAR
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CUDAScalarKernels) {
    m.impl("_local_scalar_dense", interop__local_scalar_dense_cuda);
}

} // namespace cuda
} // namespace tensorplay
