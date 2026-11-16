"""Device guards avoid redundant switches and restore the prior device."""

import pytest
import tensorplay as tp
from tensorplay.compiler.backends.stax.codegen.common import CudaDeviceOpOverrides


def test_device_exchange_only_switches_when_needed(monkeypatch):
    state = {"device": 0, "calls": []}
    monkeypatch.setattr(tp.cuda, "current_device", lambda: state["device"])

    def switch(device):
        state["calls"].append(device)
        state["device"] = device

    monkeypatch.setattr(tp.cuda._lcuda, "set_device", switch)
    assert tp.cuda._exchange_device(-1) == -1
    assert tp.cuda._exchange_device(0) == 0
    assert state["calls"] == []
    assert tp.cuda._exchange_device(1) == 0
    assert tp.cuda._maybe_exchange_device(0) == 1
    assert state["calls"] == [1, 0]


def test_generated_guard_uses_an_integer_device():
    assert CudaDeviceOpOverrides().device_guard(0) == "tp.cuda._DeviceGuard(0)"


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA is unavailable")
def test_nested_device_guards_restore_after_an_exception():
    original = tp.cuda.current_device()
    other = (original + 1) % tp.cuda.device_count()
    with pytest.raises(RuntimeError, match="inside"):
        with tp.cuda._DeviceGuard(other):
            assert tp.cuda.current_device() == other
            with tp.cuda._DeviceGuard(original):
                assert tp.cuda.current_device() == original
            assert tp.cuda.current_device() == other
            raise RuntimeError("inside")
    assert tp.cuda.current_device() == original
