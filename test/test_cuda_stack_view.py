import numpy as np
import pytest
import torch
import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(32, 32), (2, 0, 3), (3, 5, 7)])
@pytest.mark.parametrize('dim', [0, 1, -1])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_cuda_stack_matches_reference(shape, dim, noncontiguous):
    data = [np.random.default_rng(i).normal(size=shape).astype(np.float32) for i in range(3)]
    a = [tp.tensor(v, device='cuda') for v in data]
    b = [torch.tensor(v, device='cuda') for v in data]
    if noncontiguous:
        a = [v.transpose(0, 1) for v in a]
        b = [v.transpose(0, 1) for v in b]
    actual, expected = tp.stack(a, dim), torch.stack(b, dim)
    assert tuple(actual.shape) == tuple(expected.shape)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected.cpu().numpy())


def test_cuda_stack_mixed_dtype_and_backward():
    a = tp.tensor([[1., 2.], [3., 4.]], device='cuda').requires_grad_(True)
    b = tp.tensor([[1, 2], [3, 4]], device='cuda')
    result = tp.stack([a, b], 0)
    assert result.dtype == tp.float32
    np.testing.assert_array_equal(result.detach().cpu().numpy(), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    grad = tp.autograd.grad(result.sum(), a)[0]
    np.testing.assert_array_equal(grad.cpu().numpy(), np.ones((2, 2)))


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('count', [9, 16, 32])
def test_cuda_stack_many_last_dim(dtype, count):
    data = [np.arange(63, dtype=getattr(np, dtype)).reshape(7, 9) + i for i in range(count)]
    actual = tp.stack([tp.tensor(x, device='cuda') for x in data], -1).cpu().numpy()
    expected = torch.stack([torch.tensor(x, device='cuda') for x in data], -1).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_cuda_stack_many_last_dim_backward():
    values = [tp.tensor(np.arange(21, dtype=np.float32).reshape(3, 7) + i,
                        device='cuda').requires_grad_(True) for i in range(16)]
    result = tp.stack(values, -1)
    grads = tp.autograd.grad((result * result).sum(), values)
    for value, grad in zip(values, grads):
        np.testing.assert_array_equal(grad.cpu().numpy(), 2 * value.detach().cpu().numpy())
