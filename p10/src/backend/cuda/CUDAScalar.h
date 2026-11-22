#pragma once

#include <utility>

#include "Scalar.h"

namespace tensorplay {

class Tensor;

namespace cuda {

std::pair<Scalar, Scalar> read_cuda_scalar_pair(const Tensor& first, const Tensor& second);

} // namespace cuda
} // namespace tensorplay
