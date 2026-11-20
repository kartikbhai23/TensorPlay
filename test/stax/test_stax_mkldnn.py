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
