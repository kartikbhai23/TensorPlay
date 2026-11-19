#pragma once

#include "SizesAndStrides.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {

struct DropoutBuffers {
    Tensor source;
    Tensor output;
    Tensor mask;
};

inline DropoutBuffers allocate_dropout_buffers(const Tensor& input) {
    const auto shape = static_cast<std::vector<int64_t>>(input.shape());
    Tensor source = input;
    if (!input.is_contiguous()) {
        const auto strides = static_cast<std::vector<int64_t>>(input.strides());
        // The fused loop can walk any dense layout if all three buffers use it.
        if (SizesAndStrides::is_non_overlapping_and_dense(shape, strides)) {
            return {
                input,
                tpx::ops::empty_strided(shape, strides, input.dtype(), input.device()),
                tpx::ops::empty_strided(shape, strides, DType::Bool, input.device()),
            };
        }
        source = input.contiguous();
    }
    return {
        std::move(source),
        Tensor(shape, input.dtype(), input.device()),
        Tensor(shape, DType::Bool, input.device()),
    };
}

} // namespace tensorplay
