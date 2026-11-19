"""Deployment packages for regions that run on the GPU.

A GPU region has no host kernel to call: its schedule is a set of compiled
device kernels, and a package carries their binaries beside its entry point.
Building one needs the compiler; running one must not.  The tests below
therefore load every package through the package entry point alone -- once
through the Python loader, once through a plain ``ctypes`` call in a fresh
interpreter that never imports this project at all.
"""

import json
import os
import subprocess
import sys

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.gpu_package import (
    GpuPackageError,
    build_gpu_package,
    load_gpu_package,
)

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)

_PKG = "gpu_package"
_DEVICE = tp.device("cuda")


def _build(path, fn, args):
    return build_gpu_package(fn, args, str(path))


def _close(got, expected, atol=1e-6):
    if isinstance(got, (tuple, list)):
        assert isinstance(expected, (tuple, list))
        assert len(got) == len(expected)
        assert all(tp.allclose(a, b, atol=atol) for a, b in zip(got, expected))
    else:
        assert tp.allclose(got, expected, atol=atol)


@requires_cuda
def test_package_layout_and_metadata(tmp_path):
    values = tp.randn(1024, device=_DEVICE)
    path = _build(tmp_path / _PKG, lambda a: tp.relu(a), [values])

    assert {
        "model.so",
        "metadata.json",
        "aoti_metadata.h",
        "aoti_entry.cpp",
        "kernels",
    } <= set(os.listdir(path))

    with open(os.path.join(path, "metadata.json")) as fp:
        payload = json.load(fp)
    assert [slot["role"] for slot in payload["slots"]] == ["input", "output"]
    assert payload["outputs"] == [1]

    kernels = payload["kernels"]
    assert len(kernels) == 1
    kernel = kernels[0]
    assert kernel["symbol"]
    assert kernel["num_warps"] >= 1
    assert kernel["grid"][0] >= 1
    assert kernel["grid"][1:] == [1, 1]
    # one pointer for the input, one for the output, and the extent
    assert kernel["params"][:3] == [0, 0, 1]
    assert kernel["ptr_slots"] == [0, 1]
    assert kernel["scalars"] == [1024]
    # the binary the table names is the binary the package carries
    assert os.path.exists(os.path.join(path, "kernels", kernel["cubin"]))


@requires_cuda
def test_loaded_package_matches_eager(tmp_path):
    for shape in ((1024,), (777,), (32, 64)):
        values = tp.randn(*shape, device=_DEVICE)
        path = _build(
            tmp_path / f"{_PKG}_{len(shape)}_{shape[-1]}",
            lambda a: tp.relu(a) * 2,
            [values],
        )
        module = load_gpu_package(path)
        assert module.kernels
        _close(module(values), tp.relu(values) * 2)


@requires_cuda
def test_package_holds_a_region_that_reads_two_arguments(tmp_path):
    left = tp.randn(512, device=_DEVICE)
    right = tp.randn(512, device=_DEVICE)
    path = _build(tmp_path / _PKG, lambda a, b: tp.abs(a + b), [left, right])
    module = load_gpu_package(path)
    assert len([slot for slot in module.slots if slot["role"] == "input"]) == 2
    _close(module(left, right), tp.abs(left + right))


@requires_cuda
def test_package_holds_a_region_with_several_results(tmp_path):
    values = tp.randn(512, device=_DEVICE)
    path = _build(tmp_path / _PKG, lambda a: (tp.relu(a), tp.abs(a)), [values])
    module = load_gpu_package(path)
    got = module(values)
    assert isinstance(got, tuple) and len(got) == 2
    _close(got, (tp.relu(values), tp.abs(values)))


@requires_cuda
def test_package_holds_a_reduction(tmp_path):
    values = tp.randn(256, 512, device=_DEVICE)
    path = _build(
        tmp_path / _PKG,
        lambda a: tp.nn.functional.layer_norm(a, (512,)),
        [values],
    )
    module = load_gpu_package(path)
    # a normalised value near zero divides two nearly equal numbers, so the
    # last digits differ from the eager result; the shape is the same either way
    _close(module(values), tp.nn.functional.layer_norm(values, (512,)), atol=1e-5)


@requires_cuda
def test_a_region_without_device_kernels_is_refused(tmp_path):
    values = tp.randn(256, 512, device=_DEVICE)
    with pytest.raises(GpuPackageError, match="no GPU kernel"):
        _build(tmp_path / _PKG, lambda a: tp.softmax(a, -1), [values])


@requires_cuda
def test_a_cpu_region_is_refused(tmp_path):
    values = tp.randn(8)
    with pytest.raises(GpuPackageError, match="lives on cpu"):
        _build(tmp_path / _PKG, lambda a: tp.relu(a), [values])


@requires_cuda
def test_package_runs_without_this_project_or_its_compiler(tmp_path):
    values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    path = _build(tmp_path / _PKG, lambda a: tp.relu(a), [tp.tensor(values, device=_DEVICE)])

    script = tmp_path / "runner.py"
    script.write_text(
        _RUNNER.format(package=str(path), values=", ".join(repr(v) for v in values))
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, result.stderr
    assert [float(v) for v in result.stdout.split()] == [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


_RUNNER = """\
import ctypes
import json

package = {package!r}
with open(package + "/metadata.json") as fp:
    spec = json.load(fp)

driver = ctypes.CDLL("libcuda.so.1")
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
driver.cuDevicePrimaryCtxRetain.argtypes = [
    ctypes.POINTER(ctypes.c_void_p), ctypes.c_int
]
driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
driver.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
driver.cuMemcpyHtoD_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
driver.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

assert driver.cuInit(0) == 0
device = ctypes.c_int()
assert driver.cuDeviceGet(ctypes.byref(device), 0) == 0
context = ctypes.c_void_p()
assert driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
assert driver.cuCtxSetCurrent(context) == 0

values = [{values}]
count = len(values)
host = (ctypes.c_float * count)(*values)
inputs_slot = [s for s in spec["slots"] if s["role"] == "input"]
buffers_slot = [s for s in spec["slots"] if s["role"] != "input"]

def allocate(slot):
    elements = 1
    for size in slot["shape"]:
        elements *= size
    memory = ctypes.c_void_p()
    assert driver.cuMemAlloc_v2(ctypes.byref(memory), elements * slot["itemsize"]) == 0
    return memory

inputs = (ctypes.c_void_p * max(len(inputs_slot), 1))()
inputs[0] = allocate(inputs_slot[0])
assert driver.cuMemcpyHtoD_v2(inputs[0], host, count * 4) == 0
buffers = (ctypes.c_void_p * max(len(buffers_slot), 1))()
for index, slot in enumerate(buffers_slot):
    buffers[index] = allocate(slot)

library = ctypes.CDLL(package + "/model.so")
library.aoti_set_kernels_dir.argtypes = [ctypes.c_char_p]
assert library.aoti_set_kernels_dir((package + "/kernels").encode()) == 0
entry = library.aoti_entry
entry.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
                  ctypes.POINTER(ctypes.c_long)]
entry.restype = ctypes.c_int
assert entry(inputs, buffers, None) == 0

back = (ctypes.c_float * count)()
assert driver.cuMemcpyDtoH_v2(back, buffers[spec["outputs"][0] - len(inputs_slot)], count * 4) == 0
print(" ".join(repr(float(v)) for v in back))
"""