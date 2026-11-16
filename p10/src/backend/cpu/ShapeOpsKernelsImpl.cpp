// Tier-compiled pixel shuffle core (see cpu/ShapeOpsKernels.h).  Listed in
// TP_CPU_KERNEL_SRCS, so each capability tier compiles the interleaving with
// that tier's vector width.

#include "cpu/ShapeOpsKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

// One output row is the interleave of two input rows.  A vector pair of each
// input row becomes two vectors of the output row, so the row is written by
// full-width contiguous stores; a tail narrower than a vector pair is copied
// one element pair at a time.
template <typename T>
void pixel_shuffle_r2_typed(const T* in, T* out, int64_t groups, int64_t H,
                            int64_t W) {
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int64_t S = Vec::size();
    const int64_t plane = H * W;
    const int64_t half_out = 2 * W;
    parallel_for(0, groups * H, 1, [&](int64_t b, int64_t e) {
        for (int64_t task = b; task < e; ++task) {
            const int64_t h = task % H;
            const int64_t g = task / H;
            const T* src = in + g * 4 * plane + h * W;
            T* dst = out + g * (H * 2 * half_out) + (h * 2) * half_out;
            for (int64_t ih = 0; ih < 2; ++ih) {
                const T* row0 = src + (ih * 2) * plane;
                const T* row1 = row0 + plane;
                T* out_row = dst + ih * half_out;
                int64_t k = 0;
                for (; k + S <= W; k += S) {
                    const Vec a = Vec::loadu(row0 + k);
                    const Vec bb = Vec::loadu(row1 + k);
                    auto [v0, v1] = tensorplay::vec::interleave2(a, bb);
                    v0.store(out_row + 2 * k);
                    v1.store(out_row + 2 * k + S);
                }
                for (; k < W; ++k) {
                    out_row[2 * k] = row0[k];
                    out_row[2 * k + 1] = row1[k];
                }
            }
        }
    });
}

void pixel_shuffle_r2_impl(const void* in, void* out, int64_t groups, int64_t H,
                           int64_t W, int dtype) {
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            pixel_shuffle_r2_typed<float>(static_cast<const float*>(in),
                                          static_cast<float*>(out), groups, H, W);
            return;
        case DType::Float64:
            pixel_shuffle_r2_typed<double>(static_cast<const double*>(in),
                                           static_cast<double*>(out), groups, H, W);
            return;
        default:
            TP_THROW(NotImplementedError,
                     "pixel_shuffle core: unsupported dtype");
    }
}

} // namespace

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(pixel_shuffle_r2_stub, &pixel_shuffle_r2_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(pixel_shuffle_r2_stub, &pixel_shuffle_r2_impl);
#endif

} // namespace cpu
} // namespace tensorplay