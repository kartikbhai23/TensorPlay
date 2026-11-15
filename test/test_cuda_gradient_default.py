import numpy as np
import pytest
import torch
import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(13,), (5, 7), (3, 4, 5)])
@pytest.mark.parametrize('edge_order', [1, 2])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_cuda_gradient_default_spacing(shape, edge_order, dtype):
    data = np.random.default_rng(12).normal(size=shape).astype(getattr(np, dtype))
    actual = tp.tensor(data, device='cuda').requires_grad_(True)
    expected = torch.tensor(data, device='cuda', requires_grad=True)
    got = tp.gradient(actual, edge_order=edge_order)
    want = torch.gradient(expected, edge_order=edge_order)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)
    ga = tp.autograd.grad(sum(a.sum() for a in got), actual)[0]
    gb = torch.autograd.grad(sum(b.sum() for b in want), expected)[0]
    np.testing.assert_allclose(ga.cpu().numpy(), gb.cpu().numpy(), rtol=1e-6, atol=1e-6)
