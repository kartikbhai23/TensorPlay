#include "CudaDispatchHelpers.cuh"
#include "CUDAScalar.h"
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

void copy_scalar_to_host(const Tensor& input, void* out, const CUDAStream& stream) {
    TP_CHECK(input.numel() == 1,
             "_local_scalar_dense only supports 1-element tensors, got ", input.numel());
    checkCuda(cudaMemcpyAsync(out, input.const_data_ptr(), input.itemsize(),
                              cudaMemcpyDeviceToHost, stream.stream()),
              "_local_scalar_dense copy");
}

Scalar decode_scalar(DType dtype, const void* bytes) {
#define TP_READ_SCALAR(ctype, name) \
    case DType::name: return scalar_from_bytes<ctype>(bytes);
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_FP8(TP_READ_SCALAR)
        default:
            TP_THROW(TypeError, "_local_scalar_dense: unsupported dtype ",
                     toString(dtype));
    }
#undef TP_READ_SCALAR
}

Scalar interop__local_scalar_dense_cuda(const Tensor& self) {
    OptionalCUDAGuard guard(self.device());
    const auto stream = getCurrentCUDAStream();
    auto host = getPinnedMemoryAllocator()->allocate(self.itemsize());
    copy_scalar_to_host(self, host.get(), stream);
    stream.synchronize();
    return decode_scalar(self.dtype(), host.get());
}

} // namespace

std::pair<Scalar, Scalar> read_cuda_scalar_pair(const Tensor& first, const Tensor& second) {
    OptionalCUDAGuard guard(first.device());
    const auto stream = getCurrentCUDAStream();
    auto host = getPinnedMemoryAllocator()->allocate(first.itemsize() + second.itemsize());
    auto* first_bytes = static_cast<char*>(host.get());
    auto* second_bytes = first_bytes + first.itemsize();
    copy_scalar_to_host(first, first_bytes, stream);
    try {
        copy_scalar_to_host(second, second_bytes, stream);
        stream.synchronize();
    } catch (...) {
        recordPinnedStream(host.get(), stream);
        throw;
    }
    return {decode_scalar(first.dtype(), first_bytes), decode_scalar(second.dtype(), second_bytes)};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, CUDAScalarKernels) {
    m.impl("_local_scalar_dense", interop__local_scalar_dense_cuda);
}

} // namespace cuda
} // namespace tensorplay
