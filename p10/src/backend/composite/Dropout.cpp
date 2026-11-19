// Composite kernels: the dropout family.
// the no-op/zero fast paths wrap the fused native_* primitives (which reject
// p == 1 themselves).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_dropout_p(double p) {
    if (p < 0 || p > 1) {
        TP_THROW(RuntimeError,
                 "dropout probability has to be between 0 and 1, but got ", p);
    }
}

Tensor zero_like_mul(const Tensor& input) {
    return ops::mul(input, ops::zeros({}, input.dtype(), input.device()));
}

Tensor make_feature_noise(const Tensor& input) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError,
                 "Feature dropout requires at least 2 dimensions in the input");
    }
    std::vector<int64_t> shape(input.dim(), 1);
    shape[0] = input.size(0);
    shape[1] = input.size(1);
    return ops::new_empty(input, shape);
}

} // anonymous namespace

Tensor dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_dropout(input, p, train));
}

Tensor alpha_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_alpha_dropout(input, p));
}

Tensor feature_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_feature_dropout(input, p));
}

// No fused native kernel exists for the feature+alpha combination, so it is
// bernoulli noise, alpha affine rescaling.
Tensor feature_alpha_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    constexpr double alpha = 1.7580993408473766;
    const double a = 1.0 / std::sqrt((alpha * alpha * p + 1.0) * (1.0 - p));
    Tensor noise = make_feature_noise(input);
    ops::bernoulli_(noise, 1.0 - p, std::nullopt);
    // out = input * (noise * a) + (noise - 1) * (alpha * a) + alpha * a * p
    const Tensor b = ops::add(ops::mul(ops::sub(noise, Scalar(1)),
                                       Scalar(alpha * a)),
                              Scalar(alpha * a * p));
    return ops::add(ops::mul(input, ops::mul(noise, Scalar(a))), b);
}

namespace {

Tensor& dropout_inplace(Tensor& self, double p, bool train, bool feature, bool alpha_dropout) {
    check_dropout_p(p);
    if (p == 0 || !train || self.numel() == 0) return self;
    if (p == 1) return ops::mul_(self, ops::zeros({}, self.dtype(), self.device()));

    Tensor noise = feature ? make_feature_noise(self) : ops::empty_like(self);
    ops::bernoulli_(noise, 1.0 - p, std::nullopt);
    if (!alpha_dropout) {
        ops::div_(noise, Scalar(1.0 - p));
        return ops::mul_(self, noise);
    }

    constexpr double alpha = 1.7580993408473766;
    const double scale = 1.0 / std::sqrt((alpha * alpha * p + 1.0) * (1.0 - p));
    Tensor bias = ops::sub(noise, Scalar(1));
    ops::mul_(bias, Scalar(alpha * scale));
    ops::add_(bias, Scalar(alpha * scale * p));
    ops::mul_(noise, Scalar(scale));
    ops::mul_(self, noise);
    return ops::add_(self, bias);
}

} // anonymous namespace

Tensor& dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, false, false);
}

Tensor& alpha_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, false, true);
}

Tensor& feature_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, true, false);
}

Tensor& feature_alpha_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, true, true);
}

TENSORPLAY_LIBRARY_IMPL(Composite, DropoutComposite) {
    m.impl("dropout", dropout_native);
    m.impl("dropout_", dropout__native);
    m.impl("alpha_dropout", alpha_dropout_native);
    m.impl("alpha_dropout_", alpha_dropout__native);
    m.impl("feature_dropout", feature_dropout_native);
    m.impl("feature_dropout_", feature_dropout__native);
    m.impl("feature_alpha_dropout", feature_alpha_dropout_native);
    m.impl("feature_alpha_dropout_", feature_alpha_dropout__native);
}

} // namespace composite
} // namespace tensorplay
