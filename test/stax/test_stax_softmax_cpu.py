"""CPU softmax retains values across axes, layouts, and special inputs."""

import numpy as np
import pytest
import tensorplay as tp


@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("transpose", [False, True])
def test_softmax_keeps_values_for_each_axis_and_layout(dtype, dim, transpose):
    x = tp.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=dtype)
    if transpose:
        x = x.transpose(0, 1)

    def fn(value):
        return tp.softmax(value, dim)

    actual = tp.compile(fn, strict_native=True)(x)
    np.testing.assert_allclose(actual.numpy(), fn(x).numpy(), rtol=2e-5, atol=2e-6)


def test_softmax_preserves_nonfinite_results():
    x = tp.tensor([[float("inf"), 0.0, 1.0], [float("-inf")] * 3, [0.0, 2.0, -4.0]])

    def fn(value):
        return tp.softmax(value, -1)

    actual = tp.compile(fn, strict_native=True)(x)
    np.testing.assert_allclose(actual.numpy(), fn(x).numpy(), rtol=2e-5, atol=2e-6, equal_nan=True)
