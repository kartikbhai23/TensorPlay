// Composite kernels: resolve_conj / resolve_neg.
// unchanged).

#include "Tensor.h"
#include "Dispatcher.h"

namespace tensorplay {
namespace composite {

Tensor resolve_conj_native(const Tensor& self) { return self; }

// Clearing the deferred sign means making the bytes hold the values they
// already stand for: the tensor then carries no sign and no longer needs its
// own copy, so the values are unchanged and the tensor itself answers.
Tensor resolve_neg_native(const Tensor& self) {
    const TensorImpl* impl = self.unsafeGetTensorImpl().get();
    if (impl != nullptr && impl->storage().is_negated()) impl->cow_materialize();
    return self;
}

TENSORPLAY_LIBRARY_IMPL(Composite, UnaryOpsComposite) {
    m.impl("resolve_conj", resolve_conj_native);
    m.impl("resolve_neg", resolve_neg_native);
}

} // namespace composite
} // namespace tensorplay
