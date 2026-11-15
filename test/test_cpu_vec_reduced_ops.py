import pytest
import torch
import torch.nn.functional as torch_nn

import tensorplay as tp
import tensorplay.nn.functional as tp_nn


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["reciprocal", "exp2", "sinc", "hardshrink", "softshrink"])
@pytest.mark.parametrize("strided", [False, True])
def test_reduced_unary_vector_values(dtype, name, strided):
    values = torch.cat((torch.linspace(-4, 4, 1031), torch.tensor([
        0.0, -0.0, float("inf"), -float("inf"), float("nan"),
        0.1, -0.1, 0.1001, -0.1001,
    ]))).to(dtype)
    if strided:
        values = torch.stack((values, values), dim=1)[:, 0]
    source = tp.from_dlpack(values)
    if name in ("hardshrink", "softshrink"):
        actual = getattr(tp_nn, name)(source, lambd=0.1001)
        expected = getattr(torch_nn, name)(values, lambd=0.1001)
    else:
        actual = getattr(tp, name)(source)
        expected = getattr(torch, name)(values)
    assert actual.dtype == (tp.float16 if dtype == torch.float16 else tp.bfloat16)
    torch.testing.assert_close(torch.from_dlpack(actual), expected,
                               rtol=0.01, atol=0.0005, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 8, 9, 31, 32, 33])
def test_reduced_sinc_vector_tail(dtype, length):
    values = torch.linspace(-0.75, 0.75, length).to(dtype)
    actual = tp.sinc(tp.from_dlpack(values))
    torch.testing.assert_close(torch.from_dlpack(actual), torch.sinc(values),
                               rtol=0.01, atol=0.0005)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 31, 32, 33])
def test_bfloat16_sinc_nan_vector_and_tail(length):
    values = torch.full((length,), float("-inf"), dtype=torch.bfloat16)
    actual = torch.from_dlpack(tp.sinc(tp.from_dlpack(values)))
    assert torch.isnan(actual).all()


@pytest.mark.parametrize("lambd", [0.0, 0.1001, 0.5, 1.0, float("inf"), -1.0, float("nan")])
def test_bfloat16_hardshrink_all_encodings(lambd):
    values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    actual = torch.from_dlpack(tp.hardshrink(tp.from_dlpack(values), lambd))
    expected = torch_nn.hardshrink(values, lambd)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16, tp.float32, tp.float64])
@pytest.mark.parametrize("alias", [False, True])
def test_hardshrink_out_and_alias(dtype, alias):
    values = tp.tensor([-2.0, -0.5, 0.0, 0.5, 2.0] * 7, dtype=dtype)
    output = values if alias else tp.empty(values.shape, dtype=dtype)
    pointer = output.data_ptr()
    result = tp.hardshrink(values, 0.5, out=output)
    assert result.data_ptr() == pointer
    assert output.tolist() == [-2.0, 0.0, 0.0, 0.0, 2.0] * 7


def test_bfloat16_hardshrink_strided_output():
    source = tp.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=tp.bfloat16)
    backing = tp.ones(10, dtype=tp.bfloat16)
    output = backing[::2]
    pointer = output.data_ptr()
    result = tp.hardshrink(source, 0.5, out=output)
    assert result.data_ptr() == pointer
    assert backing.tolist() == [-1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
