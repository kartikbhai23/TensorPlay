#pragma once

// Tier-compiled pixel shuffle core.  ShapeOpsKernelsImpl.cpp is listed in
// TP_CPU_KERNEL_SRCS, so each CPU capability tier gets its own copy of the
// interleaving loop and picks the vector width of that tier.
//
// The upscale factor 2 case, which is the one models use, writes an output row
// as the interleave of the two input rows that share its height: a vector of
// each row becomes two vectors of the output with the tier's interleave, so a
// row is written as contiguous, full-width stores instead of one element at a
// time.  Other factors keep the row walk in the base tier.

#include "DispatchStub.h"
#include <cstdint>

namespace tensorplay {
namespace cpu {

// `in` holds `groups` channel groups of 4 planes of H*W elements laid out
// channel-major; `out` holds `groups` groups of 2H rows of 2W elements.  The
// buffers are contiguous and of `dtype` (a DType value).
using pixel_shuffle_r2_fn = void (*)(const void* in, void* out, int64_t groups,
                                     int64_t H, int64_t W, int dtype);

DECLARE_DISPATCH(pixel_shuffle_r2_fn, pixel_shuffle_r2_stub)

} // namespace cpu
} // namespace tensorplay