import numpy as np
import pytest
import torch
import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(32, 7), (64, 128), (512, 513), (32, 4096)])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('limit', [0., 1., 1e6])
def test_row_renorm_matches_reference(shape, dtype, limit):
    rng = np.random.default_rng(13)
    data = rng.normal(size=shape).astype(getattr(np, dtype))
    actual = tp.renorm(tp.tensor(data, device='cuda'), 2, 0, limit)
    expected = torch.renorm(torch.tensor(data, device='cuda'), 2, 0, limit)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-6, atol=1e-7)


def test_row_renorm_backward():
    data = np.random.default_rng(9).normal(size=(64, 33)).astype(np.float64)
    x = tp.tensor(data, device='cuda').requires_grad_(True)
    tx = torch.tensor(data, device='cuda', requires_grad=True)
    gx = tp.autograd.grad(tp.renorm(x, 2, 0, 2.).sum(), x)[0]
    gt = torch.autograd.grad(torch.renorm(tx, 2, 0, 2.).sum(), tx)[0]
    np.testing.assert_allclose(gx.cpu().numpy(), gt.cpu().numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('dim', [0, 1])
def test_renorm_nonfinite_and_column(dtype, dim):
    data = np.random.default_rng(4).normal(size=(64, 128)).astype(getattr(np, dtype))
    data[2, 0] = np.inf
    data[3, 1] = np.nan
    x = tp.tensor(data, device='cuda')
    tx = torch.tensor(data, device='cuda')
    actual = tp.renorm(x, 2, dim, 1.).cpu().numpy()
    expected = torch.renorm(tx, 2, dim, 1.).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7, equal_nan=True)


def test_row_renorm_aliased_destination():
    data = np.random.default_rng(7).normal(size=(64, 65)).astype(np.float32)
    actual = tp.tensor(data, device='cuda')
    expected = torch.tensor(data, device='cuda')
    tp.renorm(actual, 2, 0, 1., out=actual)
    torch.renorm(expected, 2, 0, 1., out=expected)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.cpu().numpy(), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('p,limit', [(0., 1.), (-1., 1.), (float('nan'), 1.), (2., -1.), (2., float('nan'))])
def test_renorm_rejects_invalid_parameters(p, limit):
    with pytest.raises((ValueError, RuntimeError)):
        tp.renorm(tp.ones(32, 8, device='cuda'), p, 0, limit)
