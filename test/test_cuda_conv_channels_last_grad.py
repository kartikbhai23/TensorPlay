"""A channels-last convolution answers both gradients in the row-major order.

The engines that read channels-last operands write a channels-last filter
gradient, but the parameter a weight gradient belongs to is row-major, so the
call repacks it back once instead of letting the engine repack each operand
around every read.  Each case checks that layout and compares the values with
a float64 reference computed on the same numbers.
"""

import numpy as np
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")


def _channels_last(x):
    """The same values in the channels-last order, as a fresh tensor."""
    n, c, h, w = x.shape
    return tp.as_strided(
        x.detach().contiguous(memory_format=tp.channels_last).cuda(),
        (n, c, h, w), (c * h * w, 1, w * c, c),
    )


@pytest.mark.parametrize("shape, weight", [
    ((8, 64, 56, 56), (64, 64, 3, 3)),
    ((8, 64, 56, 56), (128, 64, 1, 1)),
    ((8, 256, 14, 14), (256, 256, 3, 3)),
    ((2, 17, 23, 23), (5, 17, 3, 3)),
])
def test_channels_last_gradients_are_row_major(shape, weight):
    tp.manual_seed(0)
    x = _channels_last(tp.randn(*shape, device="cuda")).requires_grad_()
    w = _channels_last(tp.randn(*weight, device="cuda")).requires_grad_()
    pad = weight[-1] // 2
    y = F.conv2d(x, w, padding=pad)
    assert not y.is_contiguous()
    g = tp.randn(*y.shape, device="cuda").contiguous(memory_format=tp.channels_last)
    got = (y,) + tuple(tp.autograd.grad(y, (x, w), g))

    x64 = x.detach().double().contiguous().requires_grad_(True)
    w64 = w.detach().double().contiguous().requires_grad_(True)
    y64 = F.conv2d(x64, w64, padding=pad)
    want = (y64,) + tuple(tp.autograd.grad(y64, (x64, w64), g.double().contiguous()))
    for name, a, b in zip(("y", "dx", "dw"), got, want):
        # A channels-last input gradient keeps the order its input carries; a
        # weight gradient does not, and is row-major whatever the operands are.
        if name in ("y", "dx"):
            assert not a.is_contiguous(), name
            assert a.is_contiguous(memory_format=tp.channels_last), name
        else:
            assert a.is_contiguous(), name
        # TF32 is allowed here, so the comparison is looser than a full
        # precision run would need.
        assert ((a.double() - b).abs().max() / b.abs().max()).item() < 1e-3, name