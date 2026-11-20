#pragma once

// Launch helpers for the packed pointwise case: the destination already
// carries the extent the result needs and every operand is tightly packed.
// Those calls hand the buffers straight to the vectorized driver, which is the
// same body the iterator would run; the iterator stays for every layout it
// exists to reconcile -- broadcasting, strides, a destination that has to be
// resized or written at another width.
//
// The helpers only answer whether a launch happened.  Callers keep their
// iterator construction behind a false return so unsupported widths and
// non-packed layouts reach it unchanged.

#include "Tensor.h"
#include "OutWrite.h"
#include "CUDALoops.cuh"

#include <array>
#include <limits>

namespace tensorplay {
namespace cuda {

// A single-input pointwise pass may run flat when the destination matches the
// input element for element and both buffers are packed.  The width may differ
// (a boolean result, for example) because the element body does the conversion
// itself; what it may not do is read and write one window at two widths.
inline bool packed_pointwise_destination(const Tensor& in, const Tensor& out) {
    if (!in.defined() || !out.defined()) return false;
    if (!in.is_contiguous() || !out.is_contiguous()) return false;
    if (in.numel() != out.numel() || in.sizes() != out.sizes()) return false;
    if (in.numel() <= 0 ||
        in.numel() > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    return read_while_write_is_safe(in, out);
}

// Same question for the destination of a two-input pass.
inline bool packed_pointwise_destination(const Tensor& first,
                                         const Tensor& second,
                                         const Tensor& out) {
    if (!packed_pointwise_destination(first, out)) return false;
    if (!second.is_contiguous() || second.numel() != out.numel() ||
        second.sizes() != out.sizes()) {
        return false;
    }
    return read_while_write_is_safe(second, out);
}

template <typename body_t>
inline void launch_packed_unary(const Tensor& in, const Tensor& out,
                                const body_t& body) {
    std::array<char*, 2> data{
        static_cast<char*>(out.data_ptr()),
        static_cast<char*>(const_cast<void*>(in.const_data_ptr()))};
    launch_vectorized_kernel(in.numel(), body, data);
}

template <typename body_t>
inline void launch_packed_binary(const Tensor& first, const Tensor& second,
                                 const Tensor& out, const body_t& body) {
    std::array<char*, 3> data{
        static_cast<char*>(out.data_ptr()),
        static_cast<char*>(const_cast<void*>(first.const_data_ptr())),
        static_cast<char*>(const_cast<void*>(second.const_data_ptr()))};
    launch_vectorized_kernel(out.numel(), body, data);
}

} // namespace cuda
} // namespace tensorplay