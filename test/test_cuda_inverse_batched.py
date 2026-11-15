import numpy as np
import pytest
import torch
import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('shape', [(3, 3), (33, 33), (2, 17, 17), (2, 3, 4, 4)])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_inverse_cuda_batch(shape, dtype):
    data = np.random.default_rng(14).normal(size=shape).astype(getattr(np, dtype))
    data += 2 * shape[-1] * np.eye(shape[-1], dtype=getattr(np, dtype))
    actual = tp.linalg.inv(tp.tensor(data, device='cuda')).cpu().numpy()
    expected = torch.linalg.inv(torch.tensor(data, device='cuda')).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('shape', [(0, 0), (2, 0, 0), (0, 4, 4)])
def test_inverse_empty_cuda(shape):
    actual, info = tp.linalg.inv_ex(tp.empty(shape, device='cuda'))
    expected, expected_info = torch.linalg.inv_ex(torch.empty(shape, device='cuda'))
    assert tuple(actual.shape) == tuple(expected.shape)
    np.testing.assert_array_equal(info.cpu().numpy(), expected_info.cpu().numpy())
