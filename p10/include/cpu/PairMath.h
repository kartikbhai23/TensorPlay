#pragma once

// The pairwise maps that answer several lanes at a time: one chunk kernel
// call per block of the destination, the width taken from the operands
// themselves.
//
// The flat loop answers the same maps one element at a time, and a
// reduced-width pair would cross between the two widths on a pass of its own
// before the answer is computed.  The path below is taken when both operands
// carry the same floating width, have the same shape and are packed, so the
// walk can read and write the buffers the caller already holds.
//
// Everything else -- a broadcast pair, a mixed pair, an integral pair -- is
// left to the flat map the caller hands in.  The chunk kernels are
// translation-unit-local templates built with per-function target
// attributes, so they exist in every same-width floating pair immediately.

#include <cstdint>
#include <type_traits>
#include <vector>

#include "DType.h"
#include "OutWrite.h"
#include "Parallel.h"
#include "Tensor.h"
#include "cpu/FloatMathLoops.h"
#include "cpu/VecBinaryMath.h"

namespace tensorplay {
namespace cpu {

// Whether a pair of operands is one the vector walk answers: both carry the
// same floating width, share their shape, and are packed.
inline bool pair_vector_takes(const Tensor& a, const Tensor& b) {
    if (!vecbinary::avx2_available() || a.dtype() != b.dtype()) return false;
    const DType dt = a.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        return false;
    }
    return a.shape() == b.shape() && a.is_contiguous() && b.is_contiguous();
}

// The same question for a destination the caller holds: the answer is laid on
// it directly when it is packed, holds the width the map answers in, and is
// not a second window over the bytes an operand reads.
inline bool pair_math_takes(const Tensor& a, const Tensor& b, const Tensor& out) {
    if (!pair_vector_takes(a, b) || out.dtype() != a.dtype() ||
        a.shape() != out.shape() || !out.is_contiguous()) {
        return false;
    }
    return destination_takes_flat(
        out, out.device(), a.dtype(), static_cast<std::vector<int64_t>>(a.shape()),
        a, b);
}

// The walk is spread over the pool: a chunk of the answer is handed to a
// thread, and the thread answers the elements of that chunk.
template <vecbinary::BOp op, typename T>
inline void pair_math_walk(const T* a, const T* b, T* d, int64_t n) {
    // The chunk kernels exist only where SLEEF's AVX2 bodies are built;
    // elsewhere avx2_available() is false and no caller reaches the walk.
#if defined(TP_VECBINARY_SLEEF)
    parallel::parallel_for_balanced(0, n, parallel::GRAIN_SIZE,
                                    [&](int64_t begin, int64_t end) {
        if constexpr (std::is_same_v<T, float>) {
            vecbinary::f32_chunk_avx2<op>(
                a + begin, b + begin, d + begin, 0, end - begin);
        } else if constexpr (std::is_same_v<T, double>) {
            vecbinary::f64_chunk_avx2<op>(a + begin, b + begin, d + begin, 0,
                                          end - begin);
        } else if constexpr (std::is_same_v<T, tensorplay::BFloat16>) {
            vecbinary::bf16_chunk_avx2<op>(
                reinterpret_cast<const uint16_t*>(a + begin),
                reinterpret_cast<const uint16_t*>(b + begin),
                reinterpret_cast<uint16_t*>(d + begin), 0, end - begin);
        } else {
            vecbinary::f16_chunk_avx2<op>(
                reinterpret_cast<const uint16_t*>(a + begin),
                reinterpret_cast<const uint16_t*>(b + begin),
                reinterpret_cast<uint16_t*>(d + begin), 0, end - begin);
        }
    });
#endif
}

// The answer of one pair over the whole tensor, allocating it here: the walk
// reads the buffers the operands already hold, at their own width.
template <vecbinary::BOp op>
inline Tensor pair_math_alloc(const Tensor& a, const Tensor& b) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(a.shape()), a.dtype(), a.device());
    const int64_t n = result.numel();
    if (n > 0) {
        if (a.dtype() == DType::Float32) {
            pair_math_walk<op>(a.data_ptr<float>(), b.data_ptr<float>(),
                               result.data_ptr<float>(), n);
        } else if (a.dtype() == DType::Float64) {
            pair_math_walk<op>(a.data_ptr<double>(), b.data_ptr<double>(),
                               result.data_ptr<double>(), n);
        } else if (a.dtype() == DType::BFloat16) {
            pair_math_walk<op>(a.data_ptr<tensorplay::BFloat16>(),
                               b.data_ptr<tensorplay::BFloat16>(),
                               result.data_ptr<tensorplay::BFloat16>(), n);
        } else {
            pair_math_walk<op>(a.data_ptr<tensorplay::Half>(),
                               b.data_ptr<tensorplay::Half>(),
                               result.data_ptr<tensorplay::Half>(), n);
        }
    }
    return result;
}

// One map over a pair the caller holds, into a result this makes: the vector
// walk where the pair is one, the flat map otherwise.
template <vecbinary::BOp op, typename F>
inline Tensor pair_math_kernel(const Tensor& a, const Tensor& b, F f,
                               const char* name) {
    if (pair_vector_takes(a, b)) return pair_math_alloc<op>(a, b);
    return binary_float_kernel(a, b, f, name);
}

// The same map into the destination the caller holds, wherever it takes the
// pass; everywhere else the value comes from the allocating form.
template <vecbinary::BOp op, typename F>
inline Tensor& pair_math_out_kernel(const Tensor& a, const Tensor& b,
                                    Tensor& out, F f, const char* name) {
    if (pair_math_takes(a, b, out)) {
        const int64_t n = out.numel();
        if (n > 0) {
            if (out.dtype() == DType::Float32) {
                pair_math_walk<op>(a.data_ptr<float>(), b.data_ptr<float>(),
                                   out.data_ptr<float>(), n);
            } else if (out.dtype() == DType::Float64) {
                pair_math_walk<op>(a.data_ptr<double>(), b.data_ptr<double>(),
                                   out.data_ptr<double>(), n);
            } else if (out.dtype() == DType::BFloat16) {
                pair_math_walk<op>(a.data_ptr<tensorplay::BFloat16>(),
                                   b.data_ptr<tensorplay::BFloat16>(),
                                   out.data_ptr<tensorplay::BFloat16>(), n);
            } else {
                pair_math_walk<op>(a.data_ptr<tensorplay::Half>(),
                                   b.data_ptr<tensorplay::Half>(),
                                   out.data_ptr<tensorplay::Half>(), n);
            }
        }
        return out;
    }
    return binary_float_out_kernel(
        a, b, out,
        [=](const Tensor&, const Tensor&) {
            return pair_math_kernel<op>(a, b, f, name);
        },
        f, name);
}

}  // namespace cpu
}  // namespace tensorplay