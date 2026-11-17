"""Deployment packages for compiled regions.

A package is a directory holding everything a compiled region needs to run
without this compiler:

===========  ==========================================================
``model.so`` the generated kernels plus the entry point
``weights.bin`` the constants the region reads, packed in slot order
``metadata.json`` the slot table: which buffer is an input, an output, a
             constant or scratch space, with shapes, strides and dtypes
``aoti_metadata.h`` the same table as C++ ``constexpr`` values
``aoti_entry.cpp`` the C entry point, generated from the table
===========  ==========================================================

The entry point is

.. code-block:: c

    int aoti_entry(const void* const* inputs, void* const* outputs,
                   const long* sizevars);

``inputs`` are the region's own arguments in order; ``outputs`` are buffers
the caller provides, one per result; ``sizevars`` carries the values of the
symbols the region was built with (none for a package built from concrete
inputs).  Constants the region reads from its program are baked into
``weights.bin`` and handed over once:

.. code-block:: c

    int aoti_set_constants(const void* const* constants);

A package is built for the CPU schedule the C++ wrapper writes: regions whose
schedule has no C++ form (Triton kernels, graph-free fallbacks) cannot be
packaged and are refused.  The result links against the runtime library the
kernels were generated for, so a package runs wherever that library runs;
it does not need Python.

The Python side (:func:`load_package`) loads a package and calls it with
tensors.  ``aoti_metadata.h`` is what a C++ caller includes to learn the
shapes and dtypes of the arguments without parsing the JSON.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import tensorplay as tp

from .codegen.cpp_wrapper import CppWrapperModule
from .cpp_builder import CppBuilder, CppOptions, cpp_wrapper_build_options

ROLE_INPUT = "input"
ROLE_OUTPUT = "output"
ROLE_CONSTANT = "constant"
ROLE_INTERMEDIATE = "intermediate"

#: Written into every package so a reader can refuse a format it does not
#: know rather than misread it.
PACKAGE_FORMAT = "tensorplay-aot-package"
PACKAGE_VERSION = 1

#: The files a package is made of.
MODEL_LIBRARY = "model.so"
WEIGHTS_FILE = "weights.bin"
METADATA_FILE = "metadata.json"
METADATA_HEADER = "aoti_metadata.h"
ENTRY_SOURCE = "aoti_entry.cpp"

_ENTRY_SYMBOL = "aoti_entry"
_CONSTANTS_SYMBOL = "aoti_set_constants"

#: Device types as the entry point's table records them; the numbers are the
#: ones the generated wrapper's ``TpTensorBuffer`` carries.
_DEVICE_CODES = {
    "cpu": 0,
    "cuda": 1,
    "hip": 2,
    "mps": 3,
    "mtia": 4,
    "xpu": 5,
    "hpu": 6,
    "privateuse1": 7,
    "vulkan": 8,
    "meta": 9,
    "unknown": 10,
}

_ROLE_CODES = {
    ROLE_INPUT: 0,
    ROLE_OUTPUT: 1,
    ROLE_CONSTANT: 2,
    ROLE_INTERMEDIATE: 3,
}


class PackageError(RuntimeError):
    """A region cannot be written as a package, or a package cannot be read."""


@dataclass(frozen=True)
class PackageSlot:
    """One buffer the entry point's table describes.

    ``offset`` and ``nbytes`` are meaningful for a constant: they say where
    its bytes sit in ``weights.bin``.
    """

    index: int
    name: str
    role: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    dtype_code: int
    itemsize: int
    device: str
    offset: int = 0
    nbytes: int = 0

    @property
    def numel(self) -> int:
        result = 1
        for extent in self.shape:
            result *= int(extent)
        return result


@dataclass(frozen=True)
class PackageSpec:
    """What a package contains, as data."""

    slots: tuple[PackageSlot, ...]
    sizevar_names: tuple[str, ...]
    #: Digest of the generated sources and the flags they were built with.
    source_digest: str

    @property
    def inputs(self) -> tuple[PackageSlot, ...]:
        return tuple(slot for slot in self.slots if slot.role == ROLE_INPUT)

    @property
    def outputs(self) -> tuple[PackageSlot, ...]:
        return tuple(slot for slot in self.slots if slot.role == ROLE_OUTPUT)

    @property
    def constants(self) -> tuple[PackageSlot, ...]:
        return tuple(slot for slot in self.slots if slot.role == ROLE_CONSTANT)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "format": PACKAGE_FORMAT,
            "version": PACKAGE_VERSION,
            "entry": _ENTRY_SYMBOL,
            "constants_entry": _CONSTANTS_SYMBOL,
            "source_digest": self.source_digest,
            "sizevars": list(self.sizevar_names),
            "slots": [
                {
                    "index": slot.index,
                    "name": slot.name,
                    "role": slot.role,
                    "role_code": _ROLE_CODES[slot.role],
                    "shape": list(slot.shape),
                    "stride": list(slot.stride),
                    "dtype": slot.dtype,
                    "dtype_code": slot.dtype_code,
                    "itemsize": slot.itemsize,
                    "device": slot.device,
                    "offset": slot.offset,
                    "nbytes": slot.nbytes,
                }
                for slot in self.slots
            ],
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "PackageSpec":
        if payload.get("format") != PACKAGE_FORMAT:
            raise PackageError(
                f"{METADATA_FILE} declares format {payload.get('format')!r}, "
                f"expected {PACKAGE_FORMAT!r}"
            )
        version = payload.get("version")
        if version != PACKAGE_VERSION:
            raise PackageError(
                f"{METADATA_FILE} declares version {version!r}, this build "
                f"reads {PACKAGE_VERSION}"
            )
        slots = []
        for entry in payload.get("slots", []):
            role = entry.get("role")
            if role not in _ROLE_CODES:
                raise PackageError(f"unknown slot role {role!r} in {METADATA_FILE}")
            slots.append(
                PackageSlot(
                    index=int(entry["index"]),
                    name=str(entry["name"]),
                    role=role,
                    shape=tuple(int(s) for s in entry.get("shape", ())),
                    stride=tuple(int(s) for s in entry.get("stride", ())),
                    dtype=str(entry["dtype"]),
                    dtype_code=int(entry["dtype_code"]),
                    itemsize=int(entry["itemsize"]),
                    device=str(entry["device"]),
                    offset=int(entry.get("offset", 0)),
                    nbytes=int(entry.get("nbytes", 0)),
                )
            )
        slots.sort(key=lambda slot: slot.index)
        return cls(
            slots=tuple(slots),
            sizevar_names=tuple(str(name) for name in payload.get("sizevars", ())),
            source_digest=str(payload.get("source_digest", "")),
        )


# ---------------------------------------------------------------------------
# C++ code generation
# ---------------------------------------------------------------------------

_HEADER_TEMPLATE = """\
// Generated by the TensorPlay AOT package builder.  Do not edit.
#pragma once

#include <cstdint>

namespace aoti {{

enum Role : int {{
    kInput = {input},
    kOutput = {output},
    kConstant = {constant},
    kIntermediate = {intermediate},
}};

struct Slot {{
    const char* name;
    int role;
    int dim;
    int dtype;
    int itemsize;
    int device;
    const int64_t* sizes;
    const int64_t* strides;
    // Where this slot's bytes sit in weights.bin, for a constant.
    int64_t offset;
    int64_t nbytes;
}};

{shape_arrays}

static constexpr Slot kSlots[] = {{
{slot_rows}
}};

static constexpr int kNumSlots = {num_slots};
static constexpr int kNumInputs = {num_inputs};
static constexpr int kNumOutputs = {num_outputs};
static constexpr int kNumConstants = {num_constants};
static constexpr int kNumIntermediates = {num_intermediates};
static constexpr int kNumSizevars = {num_sizevars};
{sizevar_names}
}}  // namespace aoti
"""


def _cpp_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _emit_metadata_header(spec: PackageSpec) -> str:
    shape_arrays = []
    slot_rows = []
    for slot in spec.slots:
        # A zero-length C array is not valid C++; a dim-0 buffer is described
        # by a single unused element instead.
        shape_name = f"kSizes{slot.index}"
        stride_name = f"kStrides{slot.index}"
        shape = tuple(slot.shape) or (0,)
        stride = tuple(slot.stride) or (0,)
        shape_arrays.append(
            f"static constexpr int64_t {shape_name}[] = "
            f"{{{', '.join(str(int(s)) for s in shape)}}};"
        )
        shape_arrays.append(
            f"static constexpr int64_t {stride_name}[] = "
            f"{{{', '.join(str(int(s)) for s in stride)}}};"
        )
        slot_rows.append(
            f"    {{{_cpp_string(slot.name)}, {_ROLE_CODES[slot.role]}, "
            f"{len(slot.shape)}, {slot.dtype_code}, {slot.itemsize}, "
            f"{_DEVICE_CODES.get(slot.device, 10)}, {shape_name}, {stride_name}, "
            f"{int(slot.offset)}, {int(slot.nbytes)}}}"
        )
    sizevar_names = ""
    if spec.sizevar_names:
        rendered = ", ".join(_cpp_string(name) for name in spec.sizevar_names)
        sizevar_names = f"static constexpr const char* kSizevarNames[] = {{{rendered}}};"
    counts = {role: 0 for role in _ROLE_CODES}
    for slot in spec.slots:
        counts[slot.role] += 1
    return _HEADER_TEMPLATE.format(
        input=_ROLE_CODES[ROLE_INPUT],
        output=_ROLE_CODES[ROLE_OUTPUT],
        constant=_ROLE_CODES[ROLE_CONSTANT],
        intermediate=_ROLE_CODES[ROLE_INTERMEDIATE],
        shape_arrays="\n".join(shape_arrays),
        slot_rows=",\n".join(slot_rows),
        num_slots=len(spec.slots),
        num_inputs=counts[ROLE_INPUT],
        num_outputs=counts[ROLE_OUTPUT],
        num_constants=counts[ROLE_CONSTANT],
        num_intermediates=counts[ROLE_INTERMEDIATE],
        num_sizevars=len(spec.sizevar_names),
        sizevar_names=sizevar_names,
    )


_ENTRY_TEMPLATE = """\
// Generated by the TensorPlay AOT package builder.  Do not edit.
#include "aoti_metadata.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>

// The buffer layout the generated wrapper reads.  It is the same plain
// structure the wrapper itself declares; keeping the two in step is the
// package builder's job.
struct TpTensorBuffer {
    void* data;
    const int64_t* sizes;
    const int64_t* strides;
    int64_t dim;
    int32_t dtype;
    int32_t device_type;
};

extern "C" void call(TpTensorBuffer* tensor_data, long* sizevars);

using aoti::kSlots;

namespace {

TpTensorBuffer g_table[aoti::kNumSlots];
void* g_constants[aoti::kNumConstants > 0 ? aoti::kNumConstants : 1] = {nullptr};
void* g_scratch[aoti::kNumIntermediates > 0 ? aoti::kNumIntermediates : 1] = {nullptr};
bool g_ready = false;

std::size_t slot_bytes(const aoti::Slot& slot) {
    std::size_t elements = 1;
    for (int d = 0; d < slot.dim; ++d) {
        elements *= static_cast<std::size_t>(slot.sizes[d]);
    }
    return elements * static_cast<std::size_t>(slot.itemsize);
}

void* allocate_scratch(std::size_t bytes) {
    // The generated kernels read with the widest vector the host ISA offers,
    // so scratch buffers are aligned to a full vector register.
    constexpr std::size_t kAlignment = 64;
    const std::size_t rounded = ((bytes + kAlignment - 1) / kAlignment) * kAlignment;
    void* memory = nullptr;
    if (posix_memalign(&memory, kAlignment, rounded == 0 ? kAlignment : rounded) != 0) {
        return nullptr;
    }
    return memory;
}

bool prepare() {
    if (g_ready) {
        return true;
    }
    int index = 0;
    for (int i = 0; i < aoti::kNumSlots; ++i) {
        if (kSlots[i].role != aoti::kIntermediate) {
            continue;
        }
        g_scratch[index] = allocate_scratch(slot_bytes(kSlots[i]));
        if (g_scratch[index] == nullptr) {
            std::fprintf(stderr, "aoti: scratch buffer %s could not be allocated\\n",
                         kSlots[i].name);
            return false;
        }
        ++index;
    }
    g_ready = true;
    return true;
}

}  // namespace

extern "C" int aoti_set_constants(const void* const* constants) {
    for (int i = 0; i < aoti::kNumConstants; ++i) {
        g_constants[i] = const_cast<void*>(constants[i]);
    }
    return 0;
}

extern "C" int aoti_entry(const void* const* inputs, void* const* outputs,
                          const long* sizevars) {
    if (!prepare()) {
        return 1;
    }
    if (aoti::kNumSizevars > 0 && sizevars == nullptr) {
        std::fprintf(stderr, "aoti: the package was built with %d sizevars "
                             "and none were supplied\\n",
                     aoti::kNumSizevars);
        return 2;
    }
    for (int i = 0; i < aoti::kNumSlots; ++i) {
        g_table[i].sizes = kSlots[i].sizes;
        g_table[i].strides = kSlots[i].strides;
        g_table[i].dim = kSlots[i].dim;
        g_table[i].dtype = kSlots[i].dtype;
        g_table[i].device_type = kSlots[i].device;
        g_table[i].data = nullptr;
    }
    int input_index = 0;
    int output_index = 0;
    int constant_index = 0;
    int scratch_index = 0;
    for (int i = 0; i < aoti::kNumSlots; ++i) {
        void* data = nullptr;
        switch (kSlots[i].role) {
            case aoti::kInput:
                data = const_cast<void*>(inputs[input_index++]);
                break;
            case aoti::kOutput:
                data = outputs[output_index++];
                break;
            case aoti::kConstant:
                data = g_constants[constant_index++];
                break;
            default:
                data = g_scratch[scratch_index++];
                break;
        }
        if (data == nullptr) {
            std::fprintf(stderr, "aoti: slot %s was not given a buffer\\n",
                         kSlots[i].name);
            return 3;
        }
        g_table[i].data = data;
    }
    call(g_table, const_cast<long*>(sizevars));
    return 0;
}
"""


def _emit_entry_source() -> str:
    return _ENTRY_TEMPLATE


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------


def _slot_table(
    module: CppWrapperModule, constants: Sequence[Any]
) -> tuple[PackageSlot, ...]:
    """Describe every buffer the generated wrapper addresses.

    The wrapper reads its buffers through one flat array, so the table has to
    cover every index it may touch: the region's arguments first, then
    whatever it allocates for itself.
    """

    input_names = list(module.graph_input_names)
    constant_names = input_names[: len(constants)]
    names_by_index = {index: name for name, index in module.tensor_index.items()}
    slots: list[PackageSlot] = []
    for index in range(module.num_tensors):
        name = names_by_index.get(index)
        if name is None:
            raise PackageError(
                f"buffer {index} has no name; the generated wrapper would "
                "read a slot nothing describes"
            )
        meta = module.buffer_meta.get(name)
        if meta is None:
            if name in module.view_names:
                raise PackageError(
                    f"buffer {name!r} is a view of another buffer; a "
                    "package lays out storage of its own and cannot "
                    "describe one"
                )
            raise PackageError(
                f"buffer {name!r} has no recorded shape; the package could "
                "not describe it"
            )
        shape, stride, dtype, device = meta
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(stride, int):
            stride = (stride,)
        shape = tuple(int(s) for s in shape)
        stride = tuple(int(s) for s in stride)
        if name in constant_names:
            role = ROLE_CONSTANT
        elif name in input_names:
            role = ROLE_INPUT
        elif name in module.output_names:
            role = ROLE_OUTPUT
        else:
            role = ROLE_INTERMEDIATE
        slots.append(
            PackageSlot(
                index=index,
                name=str(name),
                role=role,
                shape=shape,
                stride=stride,
                dtype=str(dtype),
                dtype_code=int(dtype),
                itemsize=int(tp.dtype(int(dtype)).itemsize),
                device=str(device),
            )
        )
    return tuple(slots)


def _slots_and_weights(
    module: CppWrapperModule, constants: Sequence[Any]
) -> tuple[tuple[PackageSlot, ...], bytes]:
    """The slot table and the constant bytes it points into.

    The constants arrive contiguous (the builder makes them so), and the
    kernels were generated against exactly those strides, so the bytes written
    here are the ones the table describes.
    """

    slots = _slot_table(module, constants)
    # A constant is named by the argument the region was captured with, and the
    # names are read in slot order while the values were handed over in the
    # region's own order: the two are matched by name, not by position.
    values = dict(zip(module.graph_input_names, constants))
    constant_names = [slot.name for slot in slots if slot.role == ROLE_CONSTANT]
    blob = bytearray()
    offsets: dict[str, tuple[int, int]] = {}
    for name in constant_names:
        value = values.get(name)
        if value is None:
            raise PackageError(
                f"constant {name!r} has no value; the region read more "
                "buffers than it was given"
            )
        if not value.is_contiguous():
            raise PackageError(
                f"constant {name!r} is a view; a package can only bake a "
                "buffer of its own"
            )
        nbytes = int(value.numel()) * int(value.dtype.itemsize)
        offsets[name] = (len(blob), nbytes)
        blob.extend(ctypes.string_at(value.data_ptr(), nbytes))

    described: list[PackageSlot] = []
    for slot in slots:
        if slot.role != ROLE_CONSTANT:
            described.append(slot)
            continue
        offset, nbytes = offsets[slot.name]
        described.append(
            PackageSlot(
                index=slot.index,
                name=slot.name,
                role=slot.role,
                shape=slot.shape,
                stride=slot.stride,
                dtype=slot.dtype,
                dtype_code=slot.dtype_code,
                itemsize=slot.itemsize,
                device=slot.device,
                offset=offset,
                nbytes=nbytes,
            )
        )
    return tuple(described), bytes(blob)


def _package_paths(path: str) -> dict[str, str]:
    return {
        "model": os.path.join(path, MODEL_LIBRARY),
        "weights": os.path.join(path, WEIGHTS_FILE),
        "metadata": os.path.join(path, METADATA_FILE),
        "header": os.path.join(path, METADATA_HEADER),
        "entry": os.path.join(path, ENTRY_SOURCE),
    }


def _cpp_build_options() -> CppOptions:
    """The compiler invocation a package's shared object is built with.

    A package binds its own symbols: two packages loaded into one process must
    not resolve each other's entry points or tables through the global scope.
    """

    options = cpp_wrapper_build_options()
    options.ldflags.append("-Wl,-Bsymbolic")
    return options


def build_package(
    fn: Any,
    example_inputs: Sequence[Any],
    path: str,
    *,
    options: Optional[dict[str, bool]] = None,
    overwrite: bool = False,
) -> str:
    """Write ``fn`` for ``example_inputs`` as a package under ``path``.

    The region is lowered to the C++ wrapper the ``stax.cuda_codegen`` option
    selects, so a region whose schedule has no C++ form raises
    :class:`PackageError` rather than producing a package that computes
    nothing.  Tensors the program reads from its surroundings are baked into
    ``weights.bin``; the region's own arguments become the package's inputs.

    Returns the package directory.
    """

    if not callable(fn):
        raise TypeError(f"build_package() expected a callable, got {type(fn)!r}")
    inputs = list(example_inputs)
    for value in inputs:
        if not isinstance(value, tp.Tensor):
            raise PackageError(
                "a package takes tensor inputs; "
                f"{type(value).__name__} cannot be described"
            )
        if value.device.type != "cpu":
            raise PackageError(
                "a package is built for the CPU schedule; "
                f"input lives on {value.device.type}"
            )
    if os.path.exists(path) and os.listdir(path) and not overwrite:
        raise PackageError(f"{path} already holds a package; pass overwrite=True")

    from tensorplay.graph import Tracer
    from tensorplay.graph.passes import ShapeProp

    from . import backend as stax_backend
    from ..._core.api import _lift_free_tensors

    lifted = _lift_free_tensors(fn)
    if lifted is not None:
        program, read_free = lifted
    else:
        program, read_free = fn, lambda: ()
    constants = list(read_free())
    for value in constants:
        if not isinstance(value, tp.Tensor):
            raise PackageError(
                "a package can only bake tensor constants; the program reads "
                f"a {type(value).__name__} from its surroundings"
            )
        if value.device.type != "cpu":
            raise PackageError(
                f"constant lives on {value.device.type}; packages are CPU only"
            )
    # A constant the kernels read through the layout it was recorded with:
    # a view would be read through strides into storage the package does not
    # hold, so it is copied into a buffer of its own first.  The lowering
    # below then sees (and generates against) the layout the package stores.
    constants = [
        value if value.is_contiguous() else value.contiguous() for value in constants
    ]

    sample = [*constants, *inputs]
    graph_module = Tracer().trace(
        program, sample_inputs={f"p{i}": value for i, value in enumerate(sample)}
    )
    ShapeProp(sample)(graph_module)
    stax_options = {"stax.cuda_codegen": True, "stax.cudagraphs": False}
    stax_options.update(options or {})
    with tp.no_grad():
        compiled = stax_backend.stax(graph_module, sample, options=stax_options)

    module_call = getattr(compiled, "_tensorplay_module_call", None)
    module = getattr(module_call, "__self__", None)
    if not isinstance(module, CppWrapperModule):
        raise PackageError(
            "the region has no C++ wrapper to deploy: its schedule is not "
            "made of host kernels (Triton kernels and regions that fell back "
            "to the graph executor cannot be packaged)"
        )
    if len(module.graph_input_names) != len(sample):
        raise PackageError(
            f"the region reads {len(module.graph_input_names)} buffers but "
            f"{len(sample)} were handed over; a package can only bake what it "
            "was given"
        )
    for name, value in zip(module.graph_input_names, sample):
        meta = module.buffer_meta.get(name)
        if meta is None or tuple(int(s) for s in meta[0]) != tuple(value.shape):
            raise PackageError(
                f"buffer {name!r} was recorded with a different shape than "
                "the value it stands for"
            )

    slots, blob = _slots_and_weights(module, constants)

    # The symbols the generated code reads through ``sizevars``, in the order
    # the entry point's array is indexed.  A package built from concrete
    # inputs has none; one that does has the caller supply them.
    sizevar_names = tuple(
        str(symbol)
        for symbol, _ in sorted(
            module.wrapper_code.sizevar_index.items(), key=lambda item: item[1]
        )
    )
    header = _emit_metadata_header(PackageSpec(slots, sizevar_names, ""))
    entry = _emit_entry_source()
    wrapper_source = module.wrapper_code.value
    digest = hashlib.sha256(
        "\n".join([wrapper_source, header, entry]).encode()
    ).hexdigest()
    spec = PackageSpec(slots, sizevar_names, digest)

    files = _package_paths(path)
    os.makedirs(path, exist_ok=True)
    _write_text(files["header"], header)
    _write_text(files["entry"], entry)
    # The wrapper source is written beside the entry so the package can be
    # rebuilt or inspected without this installation.
    _write_text(os.path.join(path, "model.cpp"), wrapper_source)
    _write_bytes(files["weights"], blob)
    _write_text(files["metadata"], json.dumps(spec.to_metadata(), indent=2) + "\n")

    if not (os.path.exists(files["model"]) and _library_is_current(files, spec)):
        options_cpp = _cpp_build_options()
        builder = CppBuilder(
            name=MODEL_LIBRARY,
            sources=[os.path.join(path, "model.cpp"), files["entry"]],
            options=options_cpp,
            output_dir=path,
        )
        builder.build()
    return path


def _write_text(path: str, text: str) -> None:
    """Write ``text`` unless the file already holds it.

    A package is rebuilt for the same region whenever the compiler is asked
    again; rewriting files whose content did not change keeps the library
    it already built current, which is what makes a rebuild cheap.
    """

    try:
        with open(path) as fp:
            if fp.read() == text:
                return
    except (OSError, ValueError):
        pass
    with open(path, "w") as fp:
        fp.write(text)


def _write_bytes(path: str, blob: bytes) -> None:
    try:
        with open(path, "rb") as fp:
            if fp.read() == blob:
                return
    except OSError:
        pass
    with open(path, "wb") as fp:
        fp.write(blob)


def _library_is_current(files: dict[str, str], spec: PackageSpec) -> bool:
    """Whether an already built library matches this package's sources."""

    try:
        with open(files["metadata"]) as fp:
            stored = json.load(fp)
    except (OSError, ValueError):
        return False
    if stored.get("source_digest") != spec.source_digest:
        return False
    built = os.path.getmtime(files["model"])
    return built >= max(
        os.path.getmtime(files[entry])
        for entry in ("metadata", "header", "entry")
        if os.path.exists(files[entry])
    ) and built >= os.path.getmtime(os.path.join(os.path.dirname(files["model"]), "model.cpp"))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


class AOTIModule:
    """A package, loaded and callable with tensors.

    The constants are mapped from ``weights.bin`` once; every call hands the
    region's arguments to the entry point and returns buffers of its own, so
    results stay valid across calls.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        files = _package_paths(self.path)
        for key in ("model", "weights", "metadata"):
            if not os.path.exists(files[key]):
                raise PackageError(f"{files[key]} is missing; not a package")
        with open(files["metadata"]) as fp:
            self.spec = PackageSpec.from_metadata(json.load(fp))
        self._files = files
        self._library: Any = None
        self._weights: Any = None
        self._constant_ptrs: Any = None
        self._load()

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        import mmap

        # A package with no constants has an empty weights file, which cannot
        # be mapped; the constant array stays empty and is never read.
        with open(self._files["weights"], "rb") as fp:
            if os.path.getsize(self._files["weights"]) == 0:
                self._weights = b""
            else:
                self._weights = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_COPY)
        library = ctypes.CDLL(self._files["model"])
        entry = getattr(library, _ENTRY_SYMBOL)
        entry.restype = ctypes.c_int
        entry.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_long),
        ]
        set_constants = getattr(library, _CONSTANTS_SYMBOL)
        set_constants.restype = ctypes.c_int
        set_constants.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._library = library
        self._entry = entry
        self._set_constants = set_constants
        # The mapping must outlive the pointers handed to the library, and
        # every pointer must be taken from one mapping that stays put.
        constants = self.spec.constants
        self._constant_ptrs = (ctypes.c_void_p * max(1, len(constants)))()
        for position, slot in enumerate(constants):
            address = ctypes.addressof(
                ctypes.c_char.from_buffer(self._weights, slot.offset)
            )
            self._constant_ptrs[position] = address
        if constants:
            self._set_constants(self._constant_ptrs)

    # -- description --------------------------------------------------------

    @property
    def num_inputs(self) -> int:
        return len(self.spec.inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.spec.outputs)

    def __repr__(self) -> str:
        return (
            f"AOTIModule({self.path!r}, inputs={self.num_inputs}, "
            f"outputs={self.num_outputs})"
        )

    # -- calling ------------------------------------------------------------

    def __call__(self, *args: Any, sizevars: Optional[dict[str, int]] = None) -> Any:
        return self.run(list(args), sizevars=sizevars)

    def run(
        self,
        inputs: Sequence[Any],
        *,
        outputs: Optional[Sequence[Any]] = None,
        sizevars: Optional[dict[str, int]] = None,
    ) -> Any:
        """Run the region, returning its results.

        ``outputs`` lets a caller provide the result buffers (and keep them
        across calls); otherwise buffers of the recorded shapes are made for
        each call.
        """

        expected = self.spec.inputs
        if len(inputs) != len(expected):
            raise TypeError(
                f"package expects {len(expected)} inputs, got {len(inputs)}"
            )
        for value, slot in zip(inputs, expected):
            if not isinstance(value, tp.Tensor):
                raise TypeError(
                    f"package input {slot.name!r} needs a tensor, "
                    f"got {type(value).__name__}"
                )
            if tuple(int(s) for s in value.shape) != tuple(slot.shape):
                raise TypeError(
                    f"package input {slot.name!r} was recorded as "
                    f"{tuple(slot.shape)}, got {tuple(value.shape)}"
                )

        if outputs is None:
            allocated = [
                tp.empty_strided(
                    tuple(int(s) for s in slot.shape),
                    tuple(int(s) for s in slot.stride),
                    dtype=tp.dtype(slot.dtype_code),
                    device=slot.device,
                )
                for slot in self.spec.outputs
            ]
        else:
            allocated = list(outputs)
            if len(allocated) != len(self.spec.outputs):
                raise TypeError(
                    f"package expects {len(self.spec.outputs)} output buffers, "
                    f"got {len(allocated)}"
                )

        input_ptrs = (ctypes.c_void_p * max(1, len(inputs)))()
        for index, value in enumerate(inputs):
            input_ptrs[index] = value.data_ptr()
        output_ptrs = (ctypes.c_void_p * max(1, len(allocated)))()
        for index, value in enumerate(allocated):
            output_ptrs[index] = value.data_ptr()

        sizevar_names = self.spec.sizevar_names
        if sizevars is None:
            values = [0] * len(sizevar_names)
        else:
            missing = [name for name in sizevar_names if name not in sizevars]
            if missing:
                raise TypeError(
                    f"package was built with sizevars {list(sizevar_names)}; "
                    f"no value for {missing}"
                )
            values = [int(sizevars[name]) for name in sizevar_names]
        sizevar_array = (ctypes.c_long * max(1, len(values)))(*values)

        status = self._entry(input_ptrs, output_ptrs, sizevar_array)
        if status != 0:
            raise PackageError(f"the package entry returned {status}")

        if len(allocated) == 1:
            return allocated[0]
        return tuple(allocated)


def load_package(path: str) -> AOTIModule:
    """Load the package written under ``path``."""

    return AOTIModule(path)


__all__ = [
    "AOTIModule",
    "PackageError",
    "PackageSlot",
    "PackageSpec",
    "build_package",
    "load_package",
]