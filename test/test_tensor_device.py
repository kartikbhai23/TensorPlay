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


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("device", ["cpu", tp.device("cpu")])
def test_cuda_rejects_another_device_type(device):
    # A spelling or device naming another type is a mistake, not an index on
    # the type the placement lands on.
    with pytest.raises(ValueError, match="Expected a CUDA device"):
        tp.tensor([1.0]).cuda(device)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is not available")
def test_placement_on_the_same_device_returns_the_same_object():
    cpu = tp.tensor([1.0])
    assert cpu.cpu() is cpu
    cuda = cpu.cuda()
    assert cuda.cuda() is cuda
    assert cuda.cuda(tp.device("cuda")) is cuda


@pytest.mark.parametrize("device", [None, 0, 2, "xpu", "xpu:3", tp.device("xpu:1")])
def test_xpu_accepts_every_device_form(device):
    # The argument handling decides what this test can observe: without an XPU
    # backend a placement is served by the CPU fallback and succeeds, and a
    # rejection while the argument is read is the only failure this can see.
    tensor = tp.tensor([1.0])
    tensor.xpu() if device is None else tensor.xpu(device)