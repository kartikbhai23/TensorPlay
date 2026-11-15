import numpy as np
import pytest
import torch
import tensorplay as tp
import tensorplay._C as C

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")


def _tensor(values, dtype):
    return tp.tensor(values, device="cuda").to(getattr(tp, dtype))


def _close(actual, expected, dtype):
    actual = np.asarray(actual.double().cpu().tolist())
    expected = expected.detach().double().cpu().numpy()
    error = np.max(np.abs(actual - expected)) / max(np.max(np.abs(expected)), 1e-10)
    tolerance = 1e-2 if dtype in ("float16", "bfloat16") else 1e-11 if dtype == "float64" else 1e-4
    assert error < tolerance


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "bfloat16"])
@pytest.mark.parametrize("groups,stride,dilation", [(1, 1, 1), (1, 2, 1), (1, 1, 2), (2, 1, 1)])
def test_conv3d_gradients(dtype, groups, stride, dilation):
    rng = np.random.default_rng(10)
    x = rng.normal(size=(2, 4, 6, 7, 8)).astype(np.float32)
    w = rng.normal(size=(6, 4 // groups, 3, 3, 3)).astype(np.float32)
    tx = torch.tensor(x, device="cuda", dtype=getattr(torch, dtype), requires_grad=True)
    tw = torch.tensor(w, device="cuda", dtype=getattr(torch, dtype), requires_grad=True)
    a, b = _tensor(x, dtype), _tensor(w, dtype)
    st, pad, dil, zero = [stride] * 3, [dilation] * 3, [dilation] * 3, [0] * 3
    old_tp, old_ref = tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        tp.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        output = C.convolution(a, b, None, st, pad, dil, False, zero, groups)
        ref = torch.nn.functional.conv3d(tx, tw, None, st, pad, dil, groups)
        g = rng.normal(size=tuple(output.shape)).astype(np.float32)
        got = C.convolution_backward(_tensor(g, dtype), a, b, None, st, pad, dil,
                                     False, zero, groups, [True, True, False])
        expected = torch.autograd.grad(ref, (tx, tw), torch.tensor(g, device="cuda", dtype=getattr(torch, dtype)))
        for actual, wanted in zip(got[:2], expected):
            _close(actual, wanted, dtype)
    finally:
        tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tp, old_ref


@pytest.mark.parametrize("dtype", ["float32", "float64", "float16", "bfloat16"])
@pytest.mark.parametrize("groups", [1, 2])
def test_transpose3d_output_padding(dtype, groups):
    rng = np.random.default_rng(20)
    x = rng.normal(size=(1, 4, 4, 5, 6)).astype(np.float32)
    w = rng.normal(size=(4, 3, 3, 3, 3)).astype(np.float32)
    a, b = _tensor(x, dtype), _tensor(w, dtype)
    tx = torch.tensor(x, device="cuda", dtype=getattr(torch, dtype))
    tw = torch.tensor(w, device="cuda", dtype=getattr(torch, dtype))
    old_tp, old_ref = tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        tp.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for outpad in ([0, 0, 0], [1, 0, 1], [0, 0, 0]):
            got = C.convolution(a, b, None, [2] * 3, [1] * 3, [1] * 3, True, outpad, groups)
            expected = torch.nn.functional.conv_transpose3d(tx, tw, None, 2, 1, outpad, groups)
            assert tuple(got.shape) == tuple(expected.shape)
            _close(got, expected, dtype)
    finally:
        tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tp, old_ref


def test_transpose3d_bias_and_gradients():
    rng = np.random.default_rng(33)
    x = rng.normal(size=(1, 2, 4, 4, 4)).astype(np.float32)
    w = rng.normal(size=(2, 3, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(size=(3,)).astype(np.float32)
    a, b, c = (_tensor(v, "float32") for v in (x, w, bias))
    tx, tw, tb = (torch.tensor(v, device="cuda", requires_grad=True) for v in (x, w, bias))
    st, pad, dil, outpad = [2] * 3, [1] * 3, [1] * 3, [1, 0, 1]
    old_tp, old_ref = tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        tp.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        got = C.convolution(a, b, c, st, pad, dil, True, outpad, 1)
        expected = torch.nn.functional.conv_transpose3d(tx, tw, tb, st, pad, outpad)
        _close(got, expected, "float32")
        g = rng.normal(size=tuple(got.shape)).astype(np.float32)
        grads = C.convolution_backward(_tensor(g, "float32"), a, b, [3], st, pad, dil,
                                       True, outpad, 1, [True, True, True])
        expected_grads = torch.autograd.grad(expected, (tx, tw, tb), torch.tensor(g, device="cuda"))
        for actual, wanted in zip(grads, expected_grads):
            _close(actual, wanted, "float32")
    finally:
        tp.backends.cudnn.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tp, old_ref
