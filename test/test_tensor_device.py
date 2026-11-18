"""Device argument handling of the placement methods."""

import pytest

import tensorplay as tp


def test_invalid_device_argument_is_rejected():
    x = tp.tensor([1.0])
    with pytest.raises(TypeError, match="Invalid CUDA device"):
        x.cuda(object())
    with pytest.raises(TypeError, match="Invalid MPS device"):
        x.mps(object())
    with pytest.raises(TypeError, match="Invalid XPU device"):
        x.xpu(object())
    with pytest.raises(TypeError, match="Invalid XPU device"):
        x.xpu(1.5)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("device", [0, "cuda", "cuda:0", tp.device("cuda")])
def test_cuda_accepts_every_device_form(device):
    placed = tp.tensor([1.0]).cuda(device)
    assert placed.device.type == "cuda"
    assert placed.device.index == 0


@pytest.mark.parametrize("device", [None, 0, 2, "xpu", "xpu:3", tp.device("xpu:1")])
def test_xpu_accepts_every_device_form(device):
    # The argument handling decides what this test can observe: without an XPU
    # backend a placement is served by the CPU fallback and succeeds, and a
    # rejection while the argument is read is the only failure this can see.
    tensor = tp.tensor([1.0])
    tensor.xpu() if device is None else tensor.xpu(device)