"""Deployment packages: what a built package holds, and how it runs.

Building a package needs the compiler; running one must not.  The tests below
therefore load every package through the package entry point alone -- once
through the Python loader, once through a plain ``ctypes`` call in a fresh
interpreter that never imports the compiler, and (when a C++ compiler is
available) once through a compiled program that only includes the generated
header.
"""

import ctypes
import json
import os
import shutil
import subprocess
import sys

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.aot_package import (
    PackageError,
    PackageSpec,
    build_package,
    load_package,
)

_PKG = "package"


def _build(path, fn, args):
    return build_package(fn, args, str(path))


def _w():
    """The constant a package bakes: built here, read from the program."""

    return tp.arange(6, dtype=tp.float32).reshape(2, 3)


def _tensor_bytes(value):
    """A CPU tensor's bytes, in the order its elements are read."""

    return ctypes.string_at(value.data_ptr(), value.numel() * value.dtype.itemsize)


def test_package_layout_and_metadata(tmp_path):
    weight = _w()
    path = _build(tmp_path / _PKG, lambda a: (a * weight).relu().sum(dim=-1), [tp.ones(2, 3)])
    names = set(os.listdir(path))
    assert {
        "model.so",
        "weights.bin",
        "metadata.json",
        "aoti_metadata.h",
        "aoti_entry.cpp",
    } <= names

    with open(os.path.join(path, "metadata.json")) as fp:
        payload = json.load(fp)
    spec = PackageSpec.from_metadata(payload)
    assert [slot.role for slot in spec.slots].count("constant") == 1
    assert [slot.role for slot in spec.slots].count("input") == 1
    assert [slot.role for slot in spec.slots].count("output") == 1
    constant = spec.constants[0]
    assert tuple(constant.shape) == (2, 3)
    assert constant.itemsize == 4
    assert constant.nbytes == 24
    # the constant's bytes are the weights file, at the recorded offset
    with open(os.path.join(path, "weights.bin"), "rb") as fp:
        blob = fp.read()
    assert len(blob) == 24
    assert blob[constant.offset : constant.offset + constant.nbytes] == _tensor_bytes(_w())
    # the header describes the same table for C++ callers
    header = open(os.path.join(path, "aoti_metadata.h")).read()
    assert "static constexpr Slot kSlots[]" in header
    assert "GraphLowering_free_w" in header


def test_loaded_package_matches_eager(tmp_path):
    weight = _w()
    fn = lambda a: (a * weight).relu().sum(dim=-1)
    path = _build(tmp_path / _PKG, fn, [tp.ones(2, 3)])
    module = load_package(path)
    assert (module.num_inputs, module.num_outputs) == (1, 1)

    for values in (
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]],
    ):
        x = tp.tensor(values)
        got = module(x)
        assert tp.allclose(got, fn(x))


def test_constants_are_baked_into_the_package(tmp_path):
    weight = tp.ones(2, 3)
    fn = lambda a: a * weight
    path = _build(tmp_path / _PKG, fn, [tp.zeros(2, 3)])
    # the tensor the program read is gone; the package still has its values
    weight.copy_(tp.full((2, 3), 5.0))
    module = load_package(path)
    assert module(tp.ones(2, 3)).tolist() == [[1.0] * 3] * 2


def test_sliced_constant_is_stored_contiguous(tmp_path):
    base = tp.arange(12, dtype=tp.float32).reshape(4, 3)
    view = base[::2]
    assert not view.is_contiguous()
    fn = lambda a: a * view
    path = _build(tmp_path / _PKG, fn, [tp.zeros(2, 3)])
    module = load_package(path)
    x = tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert tp.allclose(module(x), fn(x))


def test_results_of_one_call_survive_the_next(tmp_path):
    weight = _w()
    path = _build(tmp_path / _PKG, lambda a: (a * weight).relu().sum(dim=-1), [tp.ones(2, 3)])
    module = load_package(path)
    first = module(tp.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    second = module(tp.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]))
    assert first is not second
    assert first.tolist() == [3.0, 12.0]
    assert second.tolist() == [6.0, 24.0]


def test_region_with_several_results(tmp_path):
    fn = lambda a: (a * 2, a + 1)
    path = _build(tmp_path / _PKG, fn, [tp.tensor([1.0, 2.0])])
    module = load_package(path)
    assert module.num_outputs == 2
    doubled, incremented = module(tp.tensor([3.0, 4.0]))
    assert doubled.tolist() == [6.0, 8.0]
    assert incremented.tolist() == [4.0, 5.0]


def test_region_with_intermediate_buffers(tmp_path):
    # built for one row: softmax's own kernels materialise partial results,
    # so the package has scratch buffers to describe and hand out
    fn = lambda a: a.softmax(dim=-1)
    path = _build(tmp_path / _PKG, fn, [tp.tensor([[1.0, 2.0, 3.0]])])
    with open(os.path.join(path, "metadata.json")) as fp:
        spec = PackageSpec.from_metadata(json.load(fp))
    roles = [slot.role for slot in spec.slots]
    assert roles.count("input") == 1
    assert roles.count("output") == 1
    assert roles.count("intermediate") >= 1
    module = load_package(path)
    x = tp.tensor([[1.0, 2.0, 3.0]])
    assert tp.allclose(module(x), fn(x))
    # the region was built for one row; another number of rows is another
    # region, not a shape the package can serve
    with pytest.raises(TypeError, match="recorded as"):
        module(tp.ones(2, 3))


def test_region_without_a_cpp_wrapper_is_refused(tmp_path):
    weight = tp.randn(3, 4)
    bias = tp.randn(3)
    fn = lambda a: tp.nn.functional.linear(a, weight, bias)
    with pytest.raises(PackageError, match=r"no C\+\+ wrapper"):
        _build(tmp_path / _PKG, fn, [tp.randn(2, 4)])


def test_non_tensor_argument_is_refused(tmp_path):
    with pytest.raises(PackageError, match="tensor inputs"):
        _build(tmp_path / _PKG, lambda a, k: a * k, [tp.ones(2), 2.0])


def test_existing_package_is_not_overwritten(tmp_path):
    path = tmp_path / _PKG
    _build(path, lambda a: a * 2, [tp.ones(2)])
    with pytest.raises(PackageError, match="already holds a package"):
        _build(path, lambda a: a * 3, [tp.ones(2)])
    # ...unless the caller says so
    build_package(lambda a: a * 3, [tp.ones(2)], str(path), overwrite=True)
    assert load_package(path)(tp.ones(2)).tolist() == [3.0, 3.0]


def test_bad_metadata_is_refused(tmp_path):
    path = _build(tmp_path / _PKG, lambda a: a * 2, [tp.ones(2)])
    metadata_path = os.path.join(path, "metadata.json")
    with open(metadata_path) as fp:
        payload = json.load(fp)
    payload["version"] = 99
    with open(metadata_path, "w") as fp:
        json.dump(payload, fp)
    with pytest.raises(PackageError, match="version"):
        load_package(path)


# --------------------------------------------------------------------------
# running a package without the compiler
# --------------------------------------------------------------------------


def test_package_runs_from_a_compiler_free_interpreter(tmp_path):
    """The entry point takes raw buffers: no tensor type, no compiler."""

    weight = _w()
    path = _build(tmp_path / _PKG, lambda a: (a * weight).relu().sum(dim=-1), [tp.ones(2, 3)])
    program = _COMPILER_FREE_RUNNER.format(
        package=str(path),
        values=", ".join(repr(float(v)) for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
    )
    script = tmp_path / "runner.py"
    script.write_text(program)
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
    # (a * w).relu().sum(-1) over w = arange(6).reshape(2, 3)
    assert [float(v) for v in result.stdout.split()] == [8.0, 62.0]


_COMPILER_FREE_RUNNER = """\
import ctypes
import json

with open({package!r} + "/metadata.json") as fp:
    spec = json.load(fp)

slots = spec["slots"]
library = ctypes.CDLL({package!r} + "/model.so")
entry = library.aoti_entry
entry.restype = ctypes.c_int
entry.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_long),
]
set_constants = library.aoti_set_constants
set_constants.restype = ctypes.c_int
set_constants.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

with open({package!r} + "/weights.bin", "rb") as fp:
    blob = fp.read()
constants = (ctypes.c_char * max(1, len(blob))).from_buffer_copy(blob)
constant_ptrs = (ctypes.c_void_p * 1)()
for position, slot in enumerate(s for s in slots if s["role"] == "constant"):
    constant_ptrs[position] = ctypes.addressof(constants) + slot["offset"]
set_constants(constant_ptrs)

inputs = (ctypes.c_void_p * 1)()
source = (ctypes.c_float * 6)({values})
inputs[0] = ctypes.addressof(source)

result_slots = [s for s in slots if s["role"] == "output"]
outputs = (ctypes.c_void_p * max(1, len(result_slots)))()
results = []
for position, slot in enumerate(result_slots):
    elements = 1
    for extent in slot["shape"]:
        elements *= extent
    buffer = (ctypes.c_float * elements)()
    results.append(buffer)
    outputs[position] = ctypes.addressof(buffer)

sizevars = (ctypes.c_long * 1)()
assert entry(inputs, outputs, sizevars) == 0
for buffer in results:
    print(*buffer)
"""


_CPP_SMOKE_MAIN = """\
// Compiled without Python headers: the package is deployed as a plain
// shared object.
#include "aoti_metadata.h"

#include <dlfcn.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: smoke <model.so> <weights.bin>\\n");
        return 2;
    }
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::fprintf(stderr, "dlopen failed: %s\\n", dlerror());
        return 3;
    }
    using Entry = int (*)(const void* const*, void* const*, const long*);
    using SetConstants = int (*)(const void* const*);
    auto entry = reinterpret_cast<Entry>(dlsym(library, "aoti_entry"));
    auto set_constants = reinterpret_cast<SetConstants>(dlsym(library, "aoti_set_constants"));
    if (entry == nullptr || set_constants == nullptr) {
        std::fprintf(stderr, "entry point missing\\n");
        return 4;
    }

    std::ifstream stream(argv[2], std::ios::binary);
    std::vector<char> weights((std::istreambuf_iterator<char>(stream)),
                              std::istreambuf_iterator<char>());

    std::vector<const void*> constants;
    std::vector<void*> inputs;
    std::vector<void*> outputs;
    std::vector<std::vector<float>> input_storage;
    std::vector<std::vector<float>> output_storage;
    for (int i = 0; i < aoti::kNumSlots; ++i) {
        const aoti::Slot& slot = aoti::kSlots[i];
        std::size_t elements = 1;
        for (int d = 0; d < slot.dim; ++d) {
            elements *= static_cast<std::size_t>(slot.sizes[d]);
        }
        if (slot.role == aoti::kConstant) {
            constants.push_back(weights.data() + slot.offset);
        } else if (slot.role == aoti::kInput) {
            input_storage.emplace_back(elements, 1.0f);
            inputs.push_back(input_storage.back().data());
        } else if (slot.role == aoti::kOutput) {
            output_storage.emplace_back(elements, 0.0f);
            outputs.push_back(output_storage.back().data());
        }
    }
    if (set_constants(constants.data()) != 0) {
        std::fprintf(stderr, "set_constants failed\\n");
        return 5;
    }
    const long sizevars[aoti::kNumSizevars > 0 ? aoti::kNumSizevars : 1] = {0};
    if (entry(inputs.data(), outputs.data(), sizevars) != 0) {
        std::fprintf(stderr, "entry failed\\n");
        return 6;
    }
    for (std::size_t i = 0; i < output_storage.size(); ++i) {
        for (std::size_t j = 0; j < output_storage[i].size(); ++j) {
            std::printf("%g ", static_cast<double>(output_storage[i][j]));
        }
    }
    std::printf("\\n");
    return 0;
}
"""


def _compiler_available():
    from tensorplay.compiler.backends.stax.cpp_builder import get_cpp_compiler

    return bool(get_cpp_compiler())


@pytest.mark.skipif(not _compiler_available(), reason="no C++ compiler available")
def test_package_runs_from_a_compiled_program(tmp_path):
    weight = _w()
    path = _build(tmp_path / _PKG, lambda a: (a * weight).relu().sum(dim=-1), [tp.ones(2, 3)])
    main_cpp = tmp_path / "smoke.cpp"
    main_cpp.write_text(_CPP_SMOKE_MAIN)
    binary = tmp_path / "smoke"
    from tensorplay.compiler.backends.stax.cpp_builder import get_cpp_compiler

    build = subprocess.run(
        [
            get_cpp_compiler(),
            "-std=c++20",
            "-O2",
            "-I",
            str(path),
            str(main_cpp),
            "-o",
            str(binary),
            "-ldl",
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run(
        [str(binary), os.path.join(path, "model.so"), os.path.join(path, "weights.bin")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    # the compiled program feeds every input with 1.0
    assert [float(v) for v in run.stdout.split()] == [3.0, 12.0]


def test_package_can_be_rebuilt_in_place(tmp_path):
    """A second build of the same region reuses the library it wrote."""

    path = tmp_path / _PKG
    fn = lambda a: a * 2
    _build(path, fn, [tp.ones(2)])
    library = os.path.join(path, "model.so")
    stamp = os.path.getmtime(library)
    build_package(fn, [tp.ones(2)], str(path), overwrite=True)
    assert os.path.getmtime(library) == stamp


def test_compiler_exposes_the_package_api(tmp_path):
    """The package entry points are reachable from ``tensorplay.compiler``."""

    from tensorplay import compiler

    path = compiler.build_aot_package(
        lambda a: a * 2, [tp.ones(2)], str(tmp_path / _PKG)
    )
    module = compiler.load_aot_package(path)
    assert module(tp.ones(2)).tolist() == [2.0, 2.0]


def test_package_directory_is_what_the_caller_asked_for(tmp_path):
    target = tmp_path / "nested" / "pkg"
    written = build_package(lambda a: a * 2, [tp.ones(2)], str(target))
    assert written == str(target)
    assert os.path.exists(os.path.join(written, "model.so"))
    shutil.rmtree(target)
    assert not os.path.exists(target)