import numpy as np
import pytest
import torch
import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_concatenated_ctc_values_and_gradients(reduction):
    data = np.random.default_rng(3).normal(size=(6, 2, 5)).astype(np.float64)
    x = tp.tensor(data, device='cuda').requires_grad_(True)
    tx = torch.tensor(data, device='cuda', requires_grad=True)
    target = [1, 2, 2, 3, 4]
    actual = F.ctc_loss(x.log_softmax(2), tp.tensor(target, device='cuda'), [6, 5], [2, 3], reduction=reduction)
    expected = torch.nn.functional.ctc_loss(tx.log_softmax(2), torch.tensor(target, device='cuda'), [6, 5], [2, 3], reduction=reduction)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected.detach().cpu().numpy(), rtol=1e-9, atol=1e-9)
    ag = tp.autograd.grad(actual.sum(), x)[0]
    eg = torch.autograd.grad(expected.sum(), tx)[0]
    np.testing.assert_allclose(ag.cpu().numpy(), eg.cpu().numpy(), rtol=1e-8, atol=1e-9)


def test_concatenated_ctc_rejects_extra_target():
    x = tp.randn(6, 2, 5, device='cuda').log_softmax(2)
    with pytest.raises((ValueError, RuntimeError)):
        F.ctc_loss(x, tp.tensor([1, 2, 3, 4, 1], device='cuda'), [6, 6], [2, 2])


def test_concatenated_ctc_second_derivative():
    x = tp.randn(4, 1, 3, device='cuda', dtype=tp.float64).requires_grad_(True)
    target = tp.tensor([1, 2], device='cuda')
    assert gradgradcheck(lambda y: F.ctc_loss(y.log_softmax(2), target, [4], [2]),
                         (x,), atol=1e-6, rtol=1e-4)
