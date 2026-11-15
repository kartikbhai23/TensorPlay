"""Orthogonal polynomial families evaluated by the native pointwise kernels.

Each takes ``(x, n)`` and evaluates the degree-``n`` member of the family at
``x``; ``n`` broadcasts against ``x``.
"""
from tensorplay import (
    chebyshev_polynomial_t,
    chebyshev_polynomial_u,
    chebyshev_polynomial_v,
    chebyshev_polynomial_w,
    hermite_polynomial_h,
    hermite_polynomial_he,
    laguerre_polynomial_l,
    legendre_polynomial_p,
    shifted_chebyshev_polynomial_t,
    shifted_chebyshev_polynomial_u,
    shifted_chebyshev_polynomial_v,
    shifted_chebyshev_polynomial_w,
)
from functools import wraps
from numbers import Number

__all__ = [
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
]


def _accept_scalar_operands(op):
    @wraps(op)
    def polynomial(x, n):
        import tensorplay as tp

        if isinstance(x, tp.Tensor) and isinstance(n, Number):
            n = tp.tensor(n, device=x.device)
        elif isinstance(n, tp.Tensor) and isinstance(x, Number):
            x = tp.tensor(x, device=n.device)
        return op(x, n)

    return polynomial


for _name in __all__:
    globals()[_name] = _accept_scalar_operands(globals()[_name])
