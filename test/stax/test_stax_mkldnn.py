"""MKLDNN ops compiled through the stax backend.

The three operations that move a value into oneDNN layout and compute in it --
``to_mkldnn``, ``mkldnn_linear`` and ``mkldnn_convolution`` -- are lowered as
fallbacks, so the compiled graph hands each call to its registered kernel.
These tests check that the compiled path runs, that the layout metadata the
kernels need reaches them, and that the results match a plain reference.
"""

import numpy as np
import pytest

import tensorplay as tp

pytestmark = [
    pytest.mark.skipif(not tp.has_mkldnn(), reason="oneDNN not available"),
    pytest.mark.skipif(
        not tp.is_mkldnn_enabled(), reason="oneDNN not enabled at import"
    ),
]


def _compiled(fn):
    compiled = tp.compile(fn, backend="stax")

    def run(*args):
        out = compiled(*args)
        lowering = next(iter(compiled._tensorplay_cache.values()))
        assert lowering._tensorplay_codegen is not None
        return out

    return run


def test_to_mkldnn_compiled():
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))

    def program(x):
        return tp.functional.to_mkldnn(x)

    got = _compiled(program)(x)
    assert got.shape == x.shape
    assert got._has_onednn_md()
    assert np.allclose(got.numpy(), x.numpy(), atol=1e-5)


def test_mkldnn_linear_compiled():
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))

    def program(x, w):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.mkldnn_linear(xm, w)

    got = _compiled(program)(x, w)
    assert got.shape == (4, 16)
    # The result is still in oneDNN layout, so it carries the metadata.
    assert got._has_onednn_md()
    ref = x.numpy() @ w.numpy().T
    assert np.allclose(got.numpy(), ref, atol=1e-4)


def test_mkldnn_linear_with_bias_compiled():
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    b = tp.tensor(np.random.randn(16).astype(np.float32))

    def program(x, w, b):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.mkldnn_linear(xm, w, b)

    got = _compiled(program)(x, w, b)
    assert got.shape == (4, 16)
    ref = x.numpy() @ w.numpy().T + b.numpy()
    assert np.allclose(got.numpy(), ref, atol=1e-4)


def test_mkldnn_convolution_2d_compiled():
    x = tp.tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
    b = tp.tensor(np.random.randn(4).astype(np.float32))

    def program(x, w, b):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.mkldnn_convolution(
            xm, w, b, padding=1, stride=1, dilation=1, groups=1
        )

    got = _compiled(program)(x, w, b)
    assert got.shape == (2, 4, 8, 8)
    # The result is still in oneDNN layout, so it carries the metadata.
    assert got._has_onednn_md()


def test_mkldnn_convolution_3d_compiled():
    x = tp.tensor(np.random.randn(2, 3, 4, 6, 6).astype(np.float32))
    w = tp.tensor(np.random.randn(4, 3, 3, 3, 3).astype(np.float32))
    b = tp.tensor(np.random.randn(4).astype(np.float32))

    def program(x, w, b):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.mkldnn_convolution(
            xm, w, b, padding=1, stride=1, dilation=1, groups=1
        )

    got = _compiled(program)(x, w, b)
    assert got.shape == (2, 4, 4, 6, 6)


def test_mkldnn_convolution_with_stride_padding():
    x = tp.tensor(np.random.randn(2, 3, 10, 10).astype(np.float32))
    w = tp.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
    b = tp.tensor(np.random.randn(4).astype(np.float32))

    def program(x, w, b):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.mkldnn_convolution(
            xm, w, b, padding=2, stride=2, dilation=1, groups=1
        )

    got = _compiled(program)(x, w, b)
    # (10 + 2*2 - 3) / 2 + 1 = 6
    assert got.shape == (2, 4, 6, 6)


def test_mkldnn_convolution_without_to_mkldnn():
    """The convolution kernel accepts both plain and blocked inputs."""
    x = tp.tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
    b = tp.tensor(np.random.randn(4).astype(np.float32))

    def program(x, w, b):
        return tp.functional.mkldnn_convolution(
            x, w, b, padding=1, stride=1, dilation=1, groups=1
        )

    got = _compiled(program)(x, w, b)
    assert got.shape == (2, 4, 8, 8)


def test_mkldnn_linear_without_to_mkldnn_raises():
    """A linear on a plain tensor has no layout to read, so the kernel refuses."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))

    def program(x, w):
        return tp.functional.mkldnn_linear(x, w)

    compiled = tp.compile(program, backend="stax")
    with pytest.raises(RuntimeError, match="mkldnn layout"):
        compiled(x, w)


def test_mkldnn_chained_linear_convolution():
    """to_mkldnn -> linear -> convolution in one compiled graph."""
    x = tp.tensor(np.random.randn(2, 8).astype(np.float32))
    w1 = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    w2 = tp.tensor(np.random.randn(4, 16, 3, 3).astype(np.float32))
    b2 = tp.tensor(np.random.randn(4).astype(np.float32))

    def program(x, w1, w2, b2):
        xm = tp.functional.to_mkldnn(x)
        h = tp.functional.mkldnn_linear(xm, w1)
        h2 = h.reshape(2, 16, 1, 1)
        h2m = tp.functional.to_mkldnn(h2)
        return tp.functional.mkldnn_convolution(
            h2m, w2, b2, padding=1, stride=1, dilation=1, groups=1
        )

    got = _compiled(program)(x, w1, w2, b2)
    assert got.shape == (2, 4, 1, 1)


def test_mkldnn_to_mkldnn_is_noop():
    """Calling to_mkldnn on a value already in oneDNN layout keeps the layout."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))

    def program(x):
        xm = tp.functional.to_mkldnn(x)
        return tp.functional.to_mkldnn(xm)

    got = _compiled(program)(x)
    assert got.shape == x.shape
    assert got._has_onednn_md()
    assert np.allclose(got.numpy(), x.numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# Backward ops
# ---------------------------------------------------------------------------


def test_to_mkldnn_backward_compiled():
    """to_mkldnn_backward converts a blocked gradient back to dense."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    grad = tp.tensor(np.random.randn(4, 8).astype(np.float32))

    def program(x, grad):
        grad_m = tp.functional.to_mkldnn(grad)
        return tp.functional.to_mkldnn_backward(grad_m, x)

    got = _compiled(program)(x, grad)
    assert got.shape == grad.shape
    # The result is dense, so it carries no layout metadata.
    assert not got._has_onednn_md()
    assert np.allclose(got.numpy(), grad.numpy(), atol=1e-5)


def test_mkldnn_linear_backward_input_compiled():
    """mkldnn_linear_backward_input computes the input gradient."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward_input([4, 8], grad_y_m, w)

    got = _compiled(program)(x, w, grad_y)
    assert got.shape == (4, 8)
    # y = x @ w.T, so grad_x = grad_y @ w
    ref = grad_y.numpy() @ w.numpy()
    assert np.allclose(got.numpy(), ref, atol=1e-4)


def test_mkldnn_linear_backward_weights_compiled():
    """mkldnn_linear_backward_weights computes weight and bias gradients."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward_weights(
            grad_y_m, xm, w, True
        )

    grad_w, grad_b = _compiled(program)(x, w, grad_y)
    assert grad_w.shape == (16, 8)
    assert grad_b.shape == (16,)
    # grad_w = grad_y.T @ x, grad_b = grad_y.sum(axis=0)
    ref_w = grad_y.numpy().T @ x.numpy()
    ref_b = grad_y.numpy().sum(axis=0)
    assert np.allclose(grad_w.numpy(), ref_w, atol=1e-4)
    assert np.allclose(grad_b.numpy(), ref_b, atol=1e-4)


def test_mkldnn_linear_backward_weights_no_bias_compiled():
    """mkldnn_linear_backward_weights with bias_defined=False omits grad_bias."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward_weights(
            grad_y_m, xm, w, False
        )

    grad_w, grad_b = _compiled(program)(x, w, grad_y)
    assert grad_w.shape == (16, 8)
    # With no bias the bias gradient is a defined zero-dimensional tensor.
    assert grad_b.shape == ()
    assert grad_b.numel() == 1
    ref_w = grad_y.numpy().T @ x.numpy()
    assert np.allclose(grad_w.numpy(), ref_w, atol=1e-4)


def test_mkldnn_linear_backward_compiled():
    """mkldnn_linear_backward combines input and weights backward."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward(
            xm, grad_y_m, w, [True, True, True]
        )

    grad_x, grad_w, grad_b = _compiled(program)(x, w, grad_y)
    assert grad_x.shape == (4, 8)
    assert grad_w.shape == (16, 8)
    assert grad_b.shape == (16,)
    ref_x = grad_y.numpy() @ w.numpy()
    ref_w = grad_y.numpy().T @ x.numpy()
    ref_b = grad_y.numpy().sum(axis=0)
    assert np.allclose(grad_x.numpy(), ref_x, atol=1e-4)
    assert np.allclose(grad_w.numpy(), ref_w, atol=1e-4)
    assert np.allclose(grad_b.numpy(), ref_b, atol=1e-4)


def test_mkldnn_linear_backward_partial_mask_compiled():
    """mkldnn_linear_backward with only the input mask set skips weights."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward(
            xm, grad_y_m, w, [True, False, False]
        )

    grad_x, grad_w, grad_b = _compiled(program)(x, w, grad_y)
    assert grad_x.shape == (4, 8)
    assert not grad_w.defined()
    assert not grad_b.defined()
    ref_x = grad_y.numpy() @ w.numpy()
    assert np.allclose(grad_x.numpy(), ref_x, atol=1e-4)


def test_mkldnn_linear_backward_weights_mask_without_bias_compiled():
    """A weights mask without the bias bit still yields a bias gradient."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward(
            xm, grad_y_m, w, [True, True, False]
        )

    grad_x, grad_w, grad_b = _compiled(program)(x, w, grad_y)
    assert grad_x.shape == (4, 8)
    assert grad_w.shape == (16, 8)
    # The bias gradient is the defined zero-dimensional empty result.
    assert grad_b.shape == ()
    ref_w = grad_y.numpy().T @ x.numpy()
    assert np.allclose(grad_w.numpy(), ref_w, atol=1e-4)


def test_mkldnn_linear_backward_input_without_mkldnn_raises():
    """grad_output without mkldnn layout is rejected by the kernel."""
    x = tp.tensor(np.random.randn(4, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(4, 16).astype(np.float32))

    def program(x, w, grad_y):
        return tp.functional.mkldnn_linear_backward_input([4, 8], grad_y, w)

    compiled = tp.compile(program, backend="stax")
    with pytest.raises(RuntimeError, match="mkldnn layout"):
        compiled(x, w, grad_y)


def test_mkldnn_linear_backward_flattens_rank3_compiled():
    """Backward flattens a rank-above-2 batch to one inner product."""
    x = tp.tensor(np.random.randn(2, 3, 8).astype(np.float32))
    w = tp.tensor(np.random.randn(16, 8).astype(np.float32))
    grad_y = tp.tensor(np.random.randn(2, 3, 16).astype(np.float32))

    def program_input(x, w, grad_y):
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward_input(
            [2, 3, 8], grad_y_m, w
        )

    got = _compiled(program_input)(x, w, grad_y)
    assert got.shape == (2, 3, 8)
    # The reshaped result stays in oneDNN layout.
    assert got._has_onednn_md()
    ref = (grad_y.numpy().reshape(6, 16) @ w.numpy()).reshape(2, 3, 8)
    assert np.allclose(got.numpy(), ref, atol=1e-4)

    def program_weights(x, w, grad_y):
        xm = tp.functional.to_mkldnn(x)
        grad_y_m = tp.functional.to_mkldnn(grad_y)
        return tp.functional.mkldnn_linear_backward_weights(
            grad_y_m, xm, w, True
        )

    grad_w, grad_b = _compiled(program_weights)(x, w, grad_y)
    ref_w = grad_y.numpy().reshape(6, 16).T @ x.numpy().reshape(6, 8)
    ref_b = grad_y.numpy().reshape(6, 16).sum(axis=0)
    assert np.allclose(grad_w.numpy(), ref_w, atol=1e-4)
    assert np.allclose(grad_b.numpy(), ref_b, atol=1e-4)


def test_mkldnn_linear_backward_input_rejects_non_float32_weight():
    """A weight that is not dense CPU float32 is refused."""
    x = tp.randn(4, 8)
    grad_y = tp.randn(4, 16)
    w64 = tp.tensor(np.random.randn(16, 8).astype(np.float64))
    grad_y_m = tp.functional.to_mkldnn(grad_y)
    with pytest.raises(RuntimeError, match="float32"):
        tp.functional.mkldnn_linear_backward_input([4, 8], grad_y_m, w64)


def test_to_mkldnn_backward_rejects_layout_input():
    """The reference input to to_mkldnn_backward must be dense."""
    grad = tp.randn(4, 8)
    grad_m = tp.functional.to_mkldnn(grad)
    with pytest.raises(RuntimeError, match="dense tensor"):
        tp.functional.to_mkldnn_backward(grad_m, grad_m)
