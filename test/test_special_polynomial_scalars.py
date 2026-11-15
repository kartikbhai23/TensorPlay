import numpy as np
import pytest
import torch
import tensorplay as tp

NAMES = [
    'chebyshev_polynomial_t', 'chebyshev_polynomial_u',
    'chebyshev_polynomial_v', 'chebyshev_polynomial_w',
    'shifted_chebyshev_polynomial_t', 'shifted_chebyshev_polynomial_u',
    'shifted_chebyshev_polynomial_v', 'shifted_chebyshev_polynomial_w',
    'hermite_polynomial_h', 'hermite_polynomial_he', 'laguerre_polynomial_l',
    'legendre_polynomial_p',
]


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not tp.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('scalar_position', ['x', 'n'])
def test_polynomial_scalar_operands(device, name, scalar_position):
    if scalar_position == 'n':
        values = np.array([-.7, 0., .4, .9], dtype=np.float64)
        a, b = (tp.tensor(values, device=device), 3), (torch.tensor(values, device=device), 3)
    else:
        values = np.array([0, 1, 2, 3], dtype=np.int64)
        a, b = (.4, tp.tensor(values, device=device)), (.4, torch.tensor(values, device=device))
    actual = getattr(tp.special, name)(*a).cpu().numpy()
    expected = getattr(torch.special, name)(*b).cpu().numpy()
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
