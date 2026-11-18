"""Device argument handling of the placement methods."""

import pytest

import tensorplay as tp


def test_invalid_device_argument_is_rejected():
    x = tp.tensor([1.0])
    with pytest.raises(TypeError, match="Invalid CUDA device"):
        x.cuda(object())
    with pytest.raises(TypeError, match="Invalid MPS device"):
        x.mps(object())


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("device", [0, "cuda", "cuda:0", tp.device("cuda")])
def test_cuda_accepts_every_device_form(device):
    placed = tp.tensor([1.0]).cuda(device)
    assert placed.device.type == "cuda"
    assert placed.device.index == 0