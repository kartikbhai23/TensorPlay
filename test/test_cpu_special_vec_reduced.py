import pytest
import torch

import tensorplay as tp


@pytest.mark.parametrize("name", ["i0e", "i1e"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_scaled_bessel_reduced_cpu(name, dtype, strided):
    values = torch.cat((
        torch.linspace(-100, 100, 1027),
        torch.tensor([0.0, -0.0, 1e-7, float("inf"), -float("inf"), float("nan")]),
    )).to(dtype)
    if strided:
        values = torch.stack((values, values), dim=1)[:, 0]
    source = tp.from_dlpack(values)
    result = getattr(tp.special, name)(source)
    actual = torch.from_dlpack(result)
    expected = getattr(torch.special, name)(values.float()).to(dtype)
    assert result.dtype == (tp.float16 if dtype == torch.float16 else tp.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.0005, equal_nan=True)
