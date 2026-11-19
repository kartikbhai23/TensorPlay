"""C++ wrapper generation for ahead-of-time compiled graphs.

The Python wrapper calls kernels through Python objects; the C++ wrapper
compiles the same schedule into a single translation unit that exposes one
entry point:

    extern "C" void call(TpTensorBuffer* tensor_data, long* sizevars);

``tensor_data`` holds one descriptor per buffer the graph touches -- graph
inputs, outputs, intermediates and views -- addressed by an index assigned
during codegen, and each descriptor says where the bytes are and how they are
read.  ``sizevars`` holds the value of every symbolic size in the graph.  The
Python runtime allocates the buffers, fills ``tensor_data`` and ``sizevars``,
and invokes ``call`` through the loaded shared object.
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass, field
from typing import Any

import sympy

import tensorplay as tp

from .. import config
from ..loops import V

from .. import ir
from ..utils import LineContext
from .common import IndentedBuffer
from .wrapper import (
    AllocateLine,
    FreeLine,
    MemoryPlanningLine,
    PythonWrapperCodegen,
    WrapperLine,
)


@dataclass
class CppWrapperCode:
    """The printed C++ wrapper plus everything needed to build and run it."""

    value: str
    line_map: list[tuple[int, LineContext]]
    buffer_meta: dict[str, tuple[Any, Any, Any, Any]]
    output_names: list[str]
    tensor_index: dict[str, int]
    sizevar_index: dict[Any, int]
    num_tensors: int
    num_sizevars: int
    graph_input_names: list[str]
    #: Views the program reads or returns, keyed by the name the generated
    #: code calls them: the buffer they sit on, their shape, stride, offset
    #: and type.  A view owns no storage, so the Python side that fills
    #: ``tensor_data`` has to work these out from the buffer underneath.
    view_meta: dict[str, tuple[str, tuple[Any, ...], tuple[Any, ...], Any, Any]] = (
        field(default_factory=dict)
    )


class _TpTensorBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("sizes", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("dim", ctypes.c_int64),
        ("dtype", ctypes.c_int32),
        ("device_type", ctypes.c_int32),
    ]


_DEVICE_TYPE_TO_INT = {
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


class CppWrapperModule:
    """A built C++ wrapper, loaded and ready to be called.

    The shared object exposes ``call(TpTensorBuffer* tensor_data,
    long* sizevars)``.  This class owns the Python-side buffers: graph
    inputs are aliased from the caller's tensors, while intermediate and
    output buffers are allocated here according to the metadata recorded
    during codegen.
    """

    def __init__(
        self, call_fn: Any, wrapper_code: CppWrapperCode, so_path: str | None = None
    ) -> None:
        self._call_fn = call_fn
        #: The printed program this module was built from.  A deployment
        #: package is written from it, so it stays with the module rather
        #: than only in the code cache entry it was written to.
        self.wrapper_code = wrapper_code
        #: Where the built shared object lives, when the builder recorded it.
        self.so_path = so_path
        self._tensor_index = wrapper_code.tensor_index
        self._num_tensors = wrapper_code.num_tensors
        self._num_sizevars = wrapper_code.num_sizevars
        self._output_names = wrapper_code.output_names
        self._buffer_meta = wrapper_code.buffer_meta
        self._graph_input_names = wrapper_code.graph_input_names
        #: Views the program reads or returns: the name they are called by,
        #: the buffer they sit on, their shape, stride, offset and type.
        self._view_meta = wrapper_code.view_meta

    # -- what the module was built from --------------------------------------
    #
    # A deployment package is written from these: which buffer each index
    # stands for, what shape it has, and which of them the caller supplies.

    @property
    def graph_input_names(self) -> list[str]:
        """The names the region's arguments are recorded under, in order."""

        return list(self._graph_input_names)

    @property
    def output_names(self) -> list[str]:
        """The names of the region's results, in order."""

        return list(self._output_names)

    @property
    def tensor_index(self) -> dict[str, int]:
        """Which slot of the entry point's table each buffer occupies."""

        return dict(self._tensor_index)

    @property
    def buffer_meta(self) -> dict[str, tuple[Any, Any, Any, Any]]:
        """Shape, stride, dtype and device recorded for every buffer."""

        return dict(self._buffer_meta)

    @property
    def num_tensors(self) -> int:
        """How many slots the entry point's table has."""

        return self._num_tensors

    @property
    def view_names(self) -> list[str]:
        """The names of the views the program reads or returns."""

        return list(self._view_meta)

    @staticmethod
    def _fill_buffer(
        buf: _TpTensorBuffer,
        data_ptr: int,
        sizes: tuple[int, ...],
        strides: tuple[int, ...],
        dtype: Any,
        device_type: str,
        keep_alive: list,
    ) -> None:
        dim = len(sizes)
        size_arr = (ctypes.c_int64 * max(dim, 1))(*sizes)
        stride_arr = (ctypes.c_int64 * max(dim, 1))(*strides)
        buf.data = ctypes.c_void_p(data_ptr)
        buf.sizes = ctypes.cast(size_arr, ctypes.POINTER(ctypes.c_int64))
        buf.strides = ctypes.cast(stride_arr, ctypes.POINTER(ctypes.c_int64))
        buf.dim = dim
        buf.dtype = int(dtype)
        buf.device_type = _DEVICE_TYPE_TO_INT.get(device_type, 10)
        keep_alive.extend([size_arr, stride_arr])

    def call(self, args: list) -> tuple[Any, ...]:
        import ctypes

        if len(args) != len(self._graph_input_names):
            raise RuntimeError(
                f"expected {len(self._graph_input_names)} arguments, "
                f"got {len(args)}"
            )
        num_tensors = self._num_tensors
        tensor_data = (_TpTensorBuffer * num_tensors)()
        keep_alive: list[Any] = []
        # Every slot that holds storage, by name: what the caller supplied,
        # what was allocated here, and the views that share one of those.
        tensors: dict[str, Any] = {}
        for name, tensor in zip(self._graph_input_names, args):
            tensors[name] = tensor
            idx = self._tensor_index.get(name)
            if idx is None:
                continue
            buf = _TpTensorBuffer()
            self._fill_buffer(
                buf,
                tensor.data_ptr(),
                tuple(int(s) for s in tensor.shape),
                tuple(int(s) for s in tensor.stride()),
                tensor.dtype,
                tensor.device.type,
                keep_alive,
            )
            tensor_data[idx] = buf

        allocated: dict[str, Any] = {}
        for name, meta in self._buffer_meta.items():
            idx = self._tensor_index.get(name)
            if idx is None or name in self._graph_input_names:
                continue
            if name not in allocated:
                shape, stride, dtype, device = meta
                if shape is None:
                    continue
                try:
                    shape_i = tuple(int(s) for s in shape)
                    stride_i = tuple(int(s) for s in stride)
                except (TypeError, ValueError):
                    raise RuntimeError(
                        "C++ wrapper does not support dynamic shapes for "
                        f"buffer {name!r}"
                    ) from None
                allocated[name] = tp.empty_strided(
                    shape_i, stride_i, dtype=dtype, device=device
                )
            tensors[name] = allocated[name]
            buf = _TpTensorBuffer()
            self._fill_buffer(
                buf,
                allocated[name].data_ptr(),
                tuple(int(s) for s in allocated[name].shape),
                tuple(int(s) for s in allocated[name].stride()),
                allocated[name].dtype,
                allocated[name].device.type,
                keep_alive,
            )
            tensor_data[idx] = buf

        # A view owns no storage: it is the same bytes read as another shape.
        # The kernel was written against the view's own layout, so what it
        # needs is the address the view's first element sits at, and the layout
        # it was given.  A view may itself sit on a view, so these are settled
        # in passes until every one has found storage to sit on.
        pending = dict(self._view_meta)
        while pending:
            settled = False
            for name, view in list(pending.items()):
                base, size, stride, offset, dtype = view
                base_tensor = tensors.get(base)
                if base_tensor is None:
                    continue
                idx = self._tensor_index.get(name)
                if idx is not None:
                    buf = _TpTensorBuffer()
                    self._fill_buffer(
                        buf,
                        base_tensor.data_ptr()
                        + int(offset) * int(dtype.itemsize),
                        size,
                        stride,
                        dtype,
                        base_tensor.device.type,
                        keep_alive,
                    )
                    tensor_data[idx] = buf
                seen = tp.as_strided(base_tensor, size, stride, int(offset))
                if base_tensor.dtype != dtype:
                    seen = seen.view(dtype)
                tensors[name] = seen
                del pending[name]
                settled = True
            if not settled:
                raise RuntimeError(
                    f"view {min(pending)!r} sits on storage nothing provides"
                )

        sizevars = (ctypes.c_long * self._num_sizevars)()
        self._call_fn(tensor_data, sizevars)

        outputs = []
        for name in self._output_names:
            value = tensors.get(name)
            if value is None:
                raise RuntimeError(
                    f"output buffer {name!r} was not allocated"
                )
            outputs.append(value)
        return tuple(outputs)


_DISPATCH_BINARY_TEMPLATE = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&, const tensorplay::Tensor&>::call("
    'tensorplay::Dispatcher::singleton().findHandle("{op}"), '
    "tensorplay::DispatchKey::CPU, {0}, {1})"
)

_DISPATCH_UNARY_TEMPLATE = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&>::call("
    'tensorplay::Dispatcher::singleton().findHandle("{op}"), '
    "tensorplay::DispatchKey::CPU, {0})"
)

_DISPATCH_LINEAR_TEMPLATE = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&, const tensorplay::Tensor&, "
    "const std::optional<tensorplay::Tensor>&>::call("
    'tensorplay::Dispatcher::singleton().findHandle("linear"), '
    "tensorplay::DispatchKey::CPU, {0}, {1}, {2})"
)

#: The same call without the optional argument: what goes in its place is
#: written as one of the literals, so the form is the same shape of call with a
#: different argument where the absent one would be.
_DISPATCH_LINEAR_NO_BIAS_TEMPLATE = _DISPATCH_LINEAR_TEMPLATE.replace(
    "{2})", "{s0})"
)

_CONVOLUTION_DISPATCH = (
    "tensorplay::DispatchStub<tensorplay::Tensor, "
    "const tensorplay::Tensor&, const tensorplay::Tensor&, "
    "const std::optional<tensorplay::Tensor>&, "
    "const std::vector<int64_t>&, const std::vector<int64_t>&, "
    "const std::vector<int64_t>&, bool, const std::vector<int64_t>&, "
    "int64_t>::call("
    'tensorplay::Dispatcher::singleton().findHandle("convolution"), '
    "tensorplay::DispatchKey::CPU, "
)

#: A convolution, and the same call with the bias written where it stands
#: instead of handed over as a buffer.
_DISPATCH_CONVOLUTION_TEMPLATE = _CONVOLUTION_DISPATCH + (
    "{0}, {1}, {2}, {s0}, {s1}, {s2}, {s3}, {s4}, {s5})"
)

_DISPATCH_CONVOLUTION_NO_BIAS_TEMPLATE = _CONVOLUTION_DISPATCH + (
    "{0}, {1}, {s0}, {s1}, {s2}, {s3}, {s4}, {s5}, {s6})"
)

_EXTERN_KERNEL_SHIMS: dict[str, tuple[str, str]] = {
    "matmul": (
        "tp_cpu_matmul",
        _DISPATCH_BINARY_TEMPLATE.replace("{op}", "matmul"),
    ),
    "mm": ("tp_cpu_mm", _DISPATCH_BINARY_TEMPLATE.replace("{op}", "mm")),
    "bmm": ("tp_cpu_bmm", _DISPATCH_BINARY_TEMPLATE.replace("{op}", "bmm")),
    "mul": ("tp_cpu_mul", "({0} * {1})"),
    "add": ("tp_cpu_add", "({0} + {1})"),
    "sub": ("tp_cpu_sub", "({0} - {1})"),
    "div": ("tp_cpu_div", "({0} / {1})"),
    "relu": ("tp_cpu_relu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "relu")),
    "sigmoid": (
        "tp_cpu_sigmoid",
        _DISPATCH_UNARY_TEMPLATE.replace("{op}", "sigmoid"),
    ),
    "silu": ("tp_cpu_silu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "silu")),
    "gelu": ("tp_cpu_gelu", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "gelu")),
    "tanh": ("tp_cpu_tanh", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "tanh")),
    "exp": ("tp_cpu_exp", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "exp")),
    "sqrt": ("tp_cpu_sqrt", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "sqrt")),
    "abs": ("tp_cpu_abs", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "abs")),
    "neg": ("tp_cpu_neg", _DISPATCH_UNARY_TEMPLATE.replace("{op}", "neg")),
    # A nan-median over every element is one value: the middle element with
    # the NaNs left out, so there is no order to hand back beside it.
    "nanmedian": ("tp_cpu_nanmedian", "t0.nanmedian()"),
    # The median over every element is the same shape of answer, and the
    # order it came from is only handed back when an axis is named.
    "median": ("tp_cpu_median", "t0.median()"),
}

#: The arguments a convolution is given beyond its operands, in the order the
#: operation declares them.  A convolution's bias may or may not be there, and
#: which of the two forms the call takes is which of them it is.
_CONVOLUTION_LITERALS: tuple[tuple[str, str], ...] = (
    ("stride", "ints"),
    ("padding", "ints"),
    ("dilation", "ints"),
    ("transposed", "flag"),
    ("output_padding", "ints"),
    ("groups", "int"),
)

_CONVOLUTION_NO_BIAS_LITERALS: tuple[tuple[str, str], ...] = (
    (("bias", "optional_tensor"),) + _CONVOLUTION_LITERALS
)

#: Shims for an operation that takes an argument which may be absent, keyed by
#: the number of tensors the call hands over.  Such an operation has one form
#: per way it can be called -- a linear layer's bias may or may not be there --
#: and the forms are separate entries because the C++ functions behind them
#: take different arguments.  Each entry is the C++ function to call, the call
#: as it is spelled, and the arguments that are not buffers and how to write
#: each of them.
#: What a forward convolution of one rank takes after its operands.
_CONVOLUTION_TAIL: tuple[tuple[str, str], ...] = (
    (None, "ints"),
    (None, "ints"),
    (None, "ints"),
    (None, "int"),
)

#: What a transposed convolution takes after its operands: the same set with
#: the extra padding the transpose needs to size its result.
_CONVOLUTION_TRANSPOSE_TAIL: tuple[tuple[str, str], ...] = (
    (None, "ints"),
    (None, "ints"),
    (None, "ints"),
    (None, "int"),
    (None, "ints"),
)


def _convolution_forms(
    function: str, tail: tuple[tuple[str, str], ...]
) -> dict[int, tuple]:
    """The forms of a convolution, keyed by how many operands.

    The bias is an operand when the program gives one and an absent place
    otherwise, so there are two forms; ``tail`` is what follows the operands.
    """

    after_bias = ", ".join(f"{{s{index}}}" for index in range(1, len(tail) + 1))
    after_operands = ", ".join(f"{{s{index}}}" for index in range(len(tail)))
    return {
        2: (
            f"tp_cpu_{function}",
            f"tensorplay::Tensor::{function}(t0, t1, std::nullopt, {after_bias})",
            ((None, "tensor"), (None, "tensor"), ("bias", "optional_tensor"), *tail),
        ),
        3: (
            f"tp_cpu_{function}_bias",
            f"tensorplay::Tensor::{function}(t0, t1, t2, {after_operands})",
            ((None, "tensor"), (None, "tensor"), (None, "tensor"), *tail),
        ),
    }


_EXTERN_KERNEL_ARITY_SHIMS: dict[
    str, dict[int, tuple[str, str, tuple[tuple[str, str], ...]]]
] = {
    "linear": {
        2: (
            "tp_cpu_linear",
            _DISPATCH_LINEAR_NO_BIAS_TEMPLATE,
            (("bias", "optional_tensor"),),
        ),
        3: ("tp_cpu_linear_bias", _DISPATCH_LINEAR_TEMPLATE, ()),
    },
    "convolution": {
        2: (
            "tp_cpu_convolution",
            _DISPATCH_CONVOLUTION_NO_BIAS_TEMPLATE,
            _CONVOLUTION_NO_BIAS_LITERALS,
        ),
        3: (
            "tp_cpu_convolution_bias",
            _DISPATCH_CONVOLUTION_TEMPLATE,
            _CONVOLUTION_LITERALS,
        ),
    },
    # A group norm's scale and shift are optional, and the call is written with
    # every place it has whether or not a value stands in it, so the places are
    # what each form is described by.
    "group_norm": {
        1: (
            "tp_cpu_group_norm",
            "tensorplay::Tensor::group_norm(t0, {s0}, {s1}, {s2}, {s3})",
            (
                (None, "tensor"),
                ("num_groups", "int"),
                ("weight", "optional_tensor"),
                ("bias", "optional_tensor"),
                ("eps", "real"),
            ),
        ),
        3: (
            "tp_cpu_group_norm_bias",
            "tensorplay::Tensor::group_norm(t0, {s0}, t1, t2, {s1})",
            (
                (None, "tensor"),
                ("num_groups", "int"),
                (None, "tensor"),
                (None, "tensor"),
                ("eps", "real"),
            ),
        ),
        2: (
            "tp_cpu_group_norm_weight",
            "tensorplay::Tensor::group_norm(t0, {s0}, t1, std::nullopt, {s2})",
            (
                (None, "tensor"),
                ("num_groups", "int"),
                (None, "tensor"),
                ("bias", "optional_tensor"),
                ("eps", "real"),
            ),
        ),
    },
    # Attention reads three operands and, when the program supplies one, a
    # mask as a fourth.  The mask stands between them and the numbers, so the
    # two forms are separate: one writes the absent mask, the other hands the
    # mask over as the operand it is.
    "scaled_dot_product_attention": {
        3: (
            "tp_cpu_sdpa",
            "tensorplay::Tensor::scaled_dot_product_attention("
            "t0, t1, t2, std::nullopt, {s1}, {s2}, {s3}, {s4})",
            (
                (None, "tensor"),
                (None, "tensor"),
                (None, "tensor"),
                # The absent mask is still a place in the call, written as
                # nothing rather than left out.
                ("attn_mask", "optional_tensor"),
                ("dropout_p", "real"),
                ("is_causal", "flag"),
                ("scale", "optional_real"),
                ("enable_gqa", "flag"),
            ),
        ),
        4: (
            "tp_cpu_sdpa_mask",
            "tensorplay::Tensor::scaled_dot_product_attention("
            "t0, t1, t2, t3, {s0}, {s1}, {s2}, {s3})",
            (
                (None, "tensor"),
                (None, "tensor"),
                (None, "tensor"),
                (None, "tensor"),
                ("dropout_p", "real"),
                ("is_causal", "flag"),
                ("scale", "optional_real"),
                ("enable_gqa", "flag"),
            ),
        ),
    },
    # A transposed convolution spreads each input element out through the
    # filter rather than gathering a window in.  What it is given after the
    # operands is the same set as a forward convolution's, in the same order,
    # so the two forms differ only in whether the bias is there.
    "conv1d": _convolution_forms("conv1d", _CONVOLUTION_TAIL),
    "conv3d": _convolution_forms("conv3d", _CONVOLUTION_TAIL),
    "conv_transpose1d": _convolution_forms(
        "conv_transpose1d", _CONVOLUTION_TRANSPOSE_TAIL
    ),
    "conv_transpose2d": _convolution_forms(
        "conv_transpose2d", _CONVOLUTION_TRANSPOSE_TAIL
    ),
    "conv_transpose3d": _convolution_forms(
        "conv_transpose3d", _CONVOLUTION_TRANSPOSE_TAIL
    ),
}


#: Shims for an operation whose extra arguments do not change its shape, keyed
#: by the operation: the same template serves however many tensors it is
#: handed, and the literals are written where the template names them.
_EXTERN_KERNEL_LITERAL_SHIMS: dict[
    str, tuple[str, str, tuple[tuple[str, str], ...]]
] = {
    "addmm": (
        "tp_cpu_addmm",
        "tensorplay::DispatchStub<tensorplay::Tensor, const tensorplay::Tensor&, "
        "const tensorplay::Tensor&, const tensorplay::Tensor&, "
        "const tensorplay::Scalar&, const tensorplay::Scalar&>::call("
        'tensorplay::Dispatcher::singleton().findHandle("addmm"), '
        "tensorplay::DispatchKey::CPU, {0}, {1}, {2}, "
        "tensorplay::Scalar({s0}), tensorplay::Scalar({s1}))",
        (("beta", "real"), ("alpha", "real")),
    ),
    # A batch of products, all added onto one value: the two numbers that
    # weight the sum and the products are places of their own, exactly as they
    # are for the unbatched product.
    "addbmm": (
        "tp_cpu_addbmm",
        "tensorplay::Tensor::addbmm(t0, t1, t2, tensorplay::Scalar({s0}), "
        "tensorplay::Scalar({s1}))",
        (("beta", "real"), ("alpha", "real")),
    ),
    "baddbmm": (
        "tp_cpu_baddbmm",
        "tensorplay::Tensor::baddbmm(t0, t1, t2, tensorplay::Scalar({s0}), "
        "tensorplay::Scalar({s1}))",
        (("beta", "real"), ("alpha", "real")),
    ),
    # Adaptive pooling in each rank: the operand, then the extents the windows
    # are laid over.  The extents alternate with the operand, so the call is
    # read by the places its arguments stand in.
    "adaptive_avg_pool1d": (
        "tp_cpu_adaptive_avg_pool1d",
        "tensorplay::Tensor::adaptive_avg_pool1d(t0, {s0})",
        ((None, "tensor"), (None, "ints")),
    ),
    "adaptive_avg_pool2d": (
        "tp_cpu_adaptive_avg_pool2d",
        "tensorplay::Tensor::adaptive_avg_pool2d(t0, {s0})",
        ((None, "tensor"), (None, "ints")),
    ),
    "adaptive_avg_pool3d": (
        "tp_cpu_adaptive_avg_pool3d",
        "tensorplay::Tensor::adaptive_avg_pool3d(t0, {s0})",
        ((None, "tensor"), (None, "ints")),
    ),
    # A one-dimensional maximum reads a window, a step, a padding, a spacing
    # and whether a window may reach past the border.
    "max_pool1d": (
        "tp_cpu_max_pool1d",
        "tensorplay::Tensor::max_pool1d(t0, {s0}, {s1}, {s2}, {s3}, {s4})",
        (
            (None, "tensor"),
            (None, "ints"),
            (None, "ints"),
            (None, "ints"),
            (None, "ints"),
            (None, "flag"),
        ),
    ),
    "cumsum": (
        "tp_cpu_cumsum",
        "t0.cumsum({s0}, {s1})",
        (("dim", "int"), ("dtype", "optional_dtype")),
    ),
    # A running sum of exponentials: the axis it runs along and the type it
    # accumulates in are its two places.
    "logcumsumexp": (
        "tp_cpu_logcumsumexp",
        "t0.logcumsumexp({s0}, {s1})",
        (("dim", "int"), ("dtype", "optional_dtype")),
    ),
    # The product along one axis accumulates the same way the sum does, and it
    # takes the same pair of places: the axis, and the type to accumulate in.
    "cumprod": (
        "tp_cpu_cumprod",
        "t0.cumprod({s0}, {s1})",
        (("dim", "int"), ("dtype", "optional_dtype")),
    ),
    # A contraction takes its operands as a list rather than as separate
    # arguments, and the places it has beyond that list are the equation and
    # an optional path.  The list itself is the operands: each entry is a
    # buffer or a window, and the shim builds the vector from the slots it is
    # handed.  A call recorded from the variadic spelling writes the operands
    # one after another instead of in a list; both shapes are read.
    "einsum": (
        "tp_cpu_einsum",
        "tensorplay::Tensor::einsum({s0}, {tlist}, {s1})",
        ((None, "string"), (None, "tensor_list"), ("path", "optional_ints")),
    ),
    "div_Scalar_mode": (
        "tp_cpu_div_scalar_mode",
        "tensorplay::Tensor::div(t0, tensorplay::Scalar({s0}), {s1})",
        ((None, "real"), ("rounding_mode", "string")),
    ),
    # A draw has nothing to read: the extents it fills and the type and device
    # the program asked for are the whole of the call, and the shim fills the
    # buffer it is handed.
    "randn": (
        "tp_cpu_randn",
        "tensorplay::Tensor::randn({s0}, {s1}, {s2})",
        (
            (None, "ints"),
            ("dtype", "optional_dtype"),
            ("device", "optional_device"),
        ),
    ),
    "rand": (
        "tp_cpu_rand",
        "tensorplay::Tensor::rand({s0}, {s1}, {s2})",
        (
            (None, "ints"),
            ("dtype", "optional_dtype"),
            ("device", "optional_device"),
        ),
    ),
    # A product of a matrix and a vector, added to a value: the two numbers
    # that scale the value and the product are places of their own.
    "addmv": (
        "tp_cpu_addmv",
        "t0.addmv(t1, t2, {s0}, {s1})",
        (("beta", "real"), ("alpha", "real")),
    ),
    # The distance between two rows over every pair of rows: the order of the
    # norm and how the pairs are traversed are the call's two places.
    "cdist": (
        "tp_cpu_cdist",
        "tensorplay::Tensor::cdist(t0, t1, {s0}, {s1})",
        ((None, "real"), ("compute_mode", "optional_int")),
    ),
    "divide_Scalar_mode": (
        "tp_cpu_divide_scalar_mode",
        "tensorplay::Tensor::divide(t0, tensorplay::Scalar({s0}), {s1})",
        ((None, "real"), ("rounding_mode", "string")),
    ),
}


#: Shims for an operation that leaves more than one result behind, keyed by the
#: name the generated code calls it.  A sort answers with its values and the
#: order that produced them; each half is a buffer of its own, so the shim is
#: handed one slot per result and copies each into place.
#:
#: Each entry is the C++ function to call, the call as it is spelled -- the
#: operands as ``{0}``, ``{1}``, ... and the arguments that are not buffers as
#: ``{s0}``, ``{s1}``, ... -- and those arguments and how to write each.  A
#: name of ``None`` means the argument stands in the call without one, in which
#: case the values are taken in the order they were written.
_EXTERN_KERNEL_MULTI_SHIMS: dict[
    str, tuple[str, str, tuple[tuple[str | None, str], ...]]
] = {
    "tp_sort_stable": (
        "tp_cpu_sort_stable",
        "t0.sort({s0}, {s1}, {s2})",
        (("stable", "optional_flag"), ("dim", "int"), ("descending", "flag")),
    ),
    "tp_topk": (
        "tp_cpu_topk",
        "tensorplay::Tensor::topk(t0, {s0}, {s1}, {s2}, {s3})",
        (
            (None, "int"),
            (None, "int"),
            (None, "flag"),
            (None, "flag"),
        ),
    ),
    "tp_median_dim": (
        "tp_cpu_median_dim",
        "t0.median({s0}, {s1})",
        ((None, "int"), (None, "flag")),
    ),
    "tp_mode": (
        "tp_cpu_mode",
        "t0.mode({s0}, {s1})",
        ((None, "int"), (None, "flag")),
    ),
    "tp_kthvalue": (
        "tp_cpu_kthvalue",
        "t0.kthvalue({s0}, {s1}, {s2})",
        ((None, "int"), (None, "int"), (None, "flag")),
    ),
    # The running maximum along one axis answers with the maxima and the place
    # each came from, so it leaves two buffers behind like a sort does.
    "tp_cummax": (
        "tp_cpu_cummax",
        "t0.cummax({s0})",
        (("dim", "int"),),
    ),
    "tp_cummin": (
        "tp_cpu_cummin",
        "t0.cummin({s0})",
        (("dim", "int"),),
    ),
    # Adaptive pooling answers with the pooled values and where in each window
    # the chosen one sat.  Its output size alternates with its operand, so the
    # places are what describes the call.
    "tp_adaptive_max_pool2d_with_indices": (
        "tp_cpu_adaptive_max_pool2d_with_indices",
        "tensorplay::Tensor::adaptive_max_pool2d_with_indices(t0, {s0})",
        ((None, "tensor"), (None, "ints")),
    ),
    "tp_adaptive_max_pool3d": (
        "tp_cpu_adaptive_max_pool3d",
        "tensorplay::Tensor::adaptive_max_pool3d(t0, {s0})",
        ((None, "tensor"), (None, "ints")),
    ),
    # The one-dimensional maximum reads a window, a step, a padding, a spacing
    # and whether a window may reach past the border, and leaves the places it
    # chose behind.  The three-dimensional maximum asks for the same five
    # values as the one-dimensional one.
    "tp_max_pool1d_with_indices": (
        "tp_cpu_max_pool1d_with_indices",
        "tensorplay::Tensor::max_pool1d_with_indices(t0, {s0}, {s1}, {s2}, "
        "{s3}, {s4})",
        (
            (None, "tensor"),
            (None, "ints"),
            (None, "ints"),
            (None, "ints"),
            (None, "ints"),
            (None, "flag"),
        ),
    ),
    # A bag of embeddings answers with the sums, the bag each value landed in,
    # how many values each bag held and where the largest one sat; all four
    # are written before the caller picks the one it wants.
    "tp__embedding_bag": (
        "tp_cpu_embedding_bag",
        "tensorplay::Tensor::_embedding_bag(t0, t1, t2, {s0}, {s1}, {s2}, "
        "{s3}, {s4}, {s5})",
        (
            (None, "flag"),
            (None, "int"),
            (None, "flag"),
            (None, "optional_tensor"),
            (None, "flag"),
            (None, "int"),
        ),
    ),
}



#: How each kind of literal is written.  Each is given the text the program
#: used and returns the C++ that spells the same value.
_LITERAL_WRITERS: dict[str, Any] = {}


def _literal_writer(kind: str):
    """Name a way of writing one kind of literal, for the table above."""

    def register(fn):
        _LITERAL_WRITERS[kind] = fn
        return fn

    return register


@_literal_writer("int")
def _write_int(text: str) -> str:
    return _scalar_literal(text)


@_literal_writer("real")
def _write_real(text: str) -> str:
    return _scalar_literal(text)


@_literal_writer("optional_real")
def _write_optional_real(text: str) -> str:
    """A number that may not have been supplied at all."""

    if text.strip() == "None":
        return "std::nullopt"
    return _scalar_literal(text)


@_literal_writer("optional_int")
def _write_optional_int(text: str) -> str:
    """A whole number that may not have been supplied at all."""

    if text.strip() == "None":
        return "std::nullopt"
    return f"std::optional<int64_t>({_write_int(text)})"


@_literal_writer("flag")
def _write_flag(text: str) -> str:
    if text == "True":
        return "true"
    if text == "False":
        return "false"
    raise NotImplementedError(
        f"C++ wrapper cannot write the flag {text!r}"
    )


@_literal_writer("optional_flag")
def _write_optional_flag(text: str) -> str:
    """A flag that may not have been supplied at all.

    The absent case is what an optional argument is for, and it is written as
    nothing rather than as one of the two flags.
    """

    if text == "None":
        return "std::nullopt"
    return _write_flag(text)


@_literal_writer("string")
def _write_string(text: str) -> str:
    """A word the call takes, written where it stands.

    The value arrives as it was written in the program, quotes and all; what
    the call takes is the word itself.
    """

    body = text.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "'\"":
        body = body[1:-1]
    return f'std::string("{body}")'


@_literal_writer("optional_dtype")
def _write_optional_dtype(text: str) -> str:
    """An element type that may not have been asked for.

    The type is written as the name the program used -- ``tensorplay.float64``
    -- and what the call takes is the enumerator of that name, so the spelling
    is worked out from the name rather than kept in a table that would have to
    be extended for every type the framework gains.
    """

    if text == "None":
        return "std::nullopt"
    name = text.strip().rsplit(".", 1)[-1]
    words = re.findall(r"[A-Za-z]+|[0-9]+", name)
    spelled = "".join(
        word if word[0].isupper() else (word[:1].upper() + word[1:])
        for word in words
    )
    # The unsigned types are spelled with a capital I in the middle.
    if spelled.startswith("Uint"):
        spelled = "UInt" + spelled[len("Uint") :]
    return f"tensorplay::DType::{spelled}"


@_literal_writer("ints")
def _write_ints(text: str) -> str:
    """A list of extents, as the vector the call takes.

    An empty list is an empty vector, which is not the same as a list holding
    nothing: the distinction is the number of axes the operation walks.
    """

    body = text.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1]
    elif body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    elif body and all(
        item.strip().lstrip("-").isdigit()
        for item in body.split(",")
        if item.strip()
    ):
        # A place that takes a list may be given one number instead: an extent
        # written on its own is the list holding just it.
        return "std::vector<int64_t>{" + body + "}"
    else:
        raise NotImplementedError(
            f"C++ wrapper cannot write the list {text!r}"
        )
    items = [item.strip() for item in body.split(",") if item.strip()]
    return "std::vector<int64_t>{" + ", ".join(items) + "}"


@_literal_writer("optional_ints")
def _write_optional_ints(text: str) -> str:
    """A list of extents that may not have been supplied at all.

    An absent list and an empty one ask the operation for the same thing --
    every default the call provides -- so the absent case is the empty vector.
    """

    return _write_ints("[]" if text is None else text)


@_literal_writer("optional_tensor")
def _write_optional_tensor(text: str) -> str:
    """A value that was not supplied, where the call takes an optional one."""

    if text != "None":
        raise NotImplementedError(
            f"C++ wrapper cannot write the argument {text!r}"
        )
    return "std::nullopt"


@_literal_writer("optional_device")
def _write_optional_device(text: str) -> str:
    """The device a result is to live on, when one was named.

    The spelling is the one the program used -- ``'cpu'`` or ``'cuda:1'`` --
    and the device parses that spelling itself, so it is handed the word
    rather than a table of types that would have to grow with every device.
    """

    if text == "None":
        return "std::nullopt"
    body = text.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "'\"":
        body = body[1:-1]
    return f'tensorplay::Device("{body}")'


def _literal_value(kind: str, text: str) -> str:
    """One literal, spelled for C++ in the way its kind requires."""

    writer = _LITERAL_WRITERS.get(kind)
    if writer is None:
        raise NotImplementedError(
            f"C++ wrapper does not know how to write a {kind!r} argument"
        )
    return writer(text)


def _extern_literals(name: str) -> tuple[tuple[str, str], ...]:
    """Every argument an operation may pass that is not a buffer.

    Which of them a particular call has depends on how many tensors it hands
    over -- a call without a bias names an absent one -- so this is the union,
    and the form the call takes is what says which of them it is given.
    """

    found = _EXTERN_KERNEL_LITERAL_SHIMS.get(name)
    if found is not None:
        return found[2]
    by_arity = _EXTERN_KERNEL_ARITY_SHIMS.get(name)
    if by_arity is None:
        return ()
    names: list[tuple[str, str]] = []
    for _, _, literals in by_arity.values():
        for item in literals:
            if item not in names:
                names.append(item)
    return tuple(names)


def _extern_shim(
    name: str, arity: int
) -> tuple[str | None, str | None, tuple[tuple[str, str], ...]]:
    """The C++ function an extern call is written as, and how it is spelled.

    ``name`` is the operation, ``arity`` how many tensors the call hands over.
    The last item names the arguments that are not buffers and how to write
    each of them.
    """

    by_arity = _EXTERN_KERNEL_ARITY_SHIMS.get(name)
    if by_arity is not None:
        found = by_arity.get(arity)
        if found is None:
            return None, None, ()
        shim_name, template, literals = found
        return shim_name, template, literals
    found = _EXTERN_KERNEL_SHIMS.get(name)
    if found is not None:
        shim_name, template = found
        return shim_name, template, ()
    found = _EXTERN_KERNEL_LITERAL_SHIMS.get(name)
    if found is None:
        return None, None, ()
    shim_name, template, literals = found
    return shim_name, template, literals


def _is_tensor_operand(node: Any) -> bool:
    """Whether a call's argument is read as memory rather than as a value.

    A buffer is, and so is a window onto one: both are handed to the call as a
    place to read elements from, and both are written into the entry point's
    table by the same route.
    """

    return isinstance(node, (ir.Buffer, ir.MutableBox, ir.ReinterpretView))


def _is_list_text(text: Any) -> bool:
    """Whether an argument as written spells a list."""

    return (
        isinstance(text, str)
        and text.strip().startswith("[")
        and text.strip().endswith("]")
    )


def _split_named(arg: Any) -> tuple[str | None, str, str]:
    """A call argument that was passed by name, split into its parts.

    A value that is not text, or text with no name in front of it, is a
    positional argument: it has no name, and the empty name says so.
    """

    if isinstance(arg, str) and "=" in arg:
        name, _, value = arg.partition("=")
        return name.strip(), "=", value
    return None, "", arg


def _extern_literal_names(name: str) -> tuple[str, ...]:
    """The arguments the shim for an operation expects as literals."""

    return tuple(arg_name for arg_name, _ in _extern_literals(name))


def _scalar_literal(value: str) -> str:
    """A number as it was written in the program, spelled for C++.

    A name-value pair arrives as text, and the value's type is what the
    operation was given: an integer literal stays an integer, and a real stays
    a real.  Anything that is not a plain number is refused rather than
    guessed at, because a wrong value here is a wrong result rather than a
    build failure.
    """

    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        raise NotImplementedError(
            f"C++ wrapper cannot write the argument value {value!r}"
        ) from None
    if number.is_integer() and "." not in text and "e" not in text.lower():
        return str(int(number))
    return repr(number)


def _build_extern_shim_impl(
    shim_name: str,
    template: str,
    n_tensors: int,
    literal_values: tuple[str, ...] = (),
    n_outputs: int = 1,
) -> str:
    params = ", ".join(
        [f"TpTensorBuffer* arg{i}" for i in range(n_tensors)]
        + [f"TpTensorBuffer* out{i}" for i in range(n_outputs)]
    )
    lines = []
    for i in range(n_tensors):
        lines.append(
            f"tensorplay::Tensor t{i} = tp_buffer_to_tensor(*arg{i});"
        )
    substitutions = {str(i): f"t{i}" for i in range(n_tensors)}
    substitutions.update(
        {f"s{i}": value for i, value in enumerate(literal_values)}
    )
    # An operation that takes its operands as a list is handed the vector it
    # reads them from; the entries are the shim's own tensor parameters, in
    # the order the call wrote them.
    substitutions["tlist"] = (
        "std::vector<tensorplay::Tensor>{"
        + ", ".join(f"t{i}" for i in range(n_tensors))
        + "}"
    )
    expr = template.format(
        *[f"t{i}" for i in range(n_tensors)], **substitutions
    )
    if n_outputs == 1:
        lines.append(f"tensorplay::Tensor result = {expr};")
        # The call's result is written into the buffer it belongs in.  The
        # copy is by value rather than by bytes because neither side is
        # necessarily laid out the obvious way: a contraction answers in the
        # order that suits it, and the buffer it belongs in is described by
        # its own shape and stride.
        lines.append("tp_buffer_to_tensor(*out0).copy_(result);")
    else:
        # An operation that answers with several results answers with them
        # together, so what is copied is each part of what came back, in the
        # order the operation answers.
        lines.append(f"auto result = {expr};")
        for i in range(n_outputs):
            lines.append(
                f"tp_buffer_to_tensor(*out{i}).copy_(std::get<{i}>(result));"
            )
    body = "\n        ".join(lines)
    return f"""
extern "C" int64_t {shim_name}({params}) {{
    try {{
        {body}
        return 0;
    }} catch (const std::exception& e) {{
        fprintf(stderr, "%s failed: %s\\n", "{shim_name}", e.what());
        return 1;
    }} catch (...) {{
        fprintf(stderr, "%s failed: unknown error\\n", "{shim_name}");
        return 1;
    }}
}}
"""


class CppWrapperCodegen(PythonWrapperCodegen):
    """Generate a C++ wrapper around the compiled C++ kernels.

    Currently supports CPU graphs whose schedule consists of C++ kernels
    (the ``CppScheduling`` backend).  Triton kernels and extern-kernel-only
    schedules raise ``NotImplementedError`` until the corresponding C++
    wrappers are implemented.
    """

    def __init__(self) -> None:
        self._kernel_decls: list[str] = []
        self._kernel_sources: list[str] = []
        self._extern_shim_sources: list[str] = []
        #: Which shim each distinct call got, and which names are taken.  Two
        #: calls of one operation can differ in the arguments that are not
        #: buffers, and each is written into a shim of its own, so a name is
        #: only reused for the call it was written for.
        self._extern_shim_keys: dict[Any, str] = {}
        self._extern_shim_names: set[str] = set()
        super().__init__()
        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False

        # Buffers are addressed through a flat pointer array at runtime.
        self._tensor_index: dict[str, int] = {}
        self._sizevar_index: dict[Any, int] = {}
        self._num_tensors = 0
        self._num_sizevars = 0
        self._tensor_data_name = "tensor_data"
        self._sizevar_name = "sizevars"

        #: Metadata for buffers referenced by the generated wrapper, keyed by
        #: buffer name: (shape, stride, dtype, device).  Consumed by the
        #: Python side that builds ``tensor_data`` before calling ``call``.
        self.buffer_meta: dict[str, tuple[Any, Any, Any, Any]] = {}

        #: What each view the program reads or returns looks like: the buffer
        #: it sits on, its shape, stride, offset and type.  A view is named by
        #: the expression that spells it, so these are not buffer names.
        self.view_meta: dict[
            str, tuple[str, tuple[Any, ...], tuple[Any, ...], Any, Any]
        ] = {}

        #: Names of the graph outputs, in return order.
        self.output_names: list[str] = []

    # -- argument/state plumbing ----------------------------------------

    def _register_tensor(self, name: str) -> int:
        if name not in self._tensor_index:
            self._tensor_index[name] = self._num_tensors
            self._num_tensors += 1
        return self._tensor_index[name]

    def _register_sizevar(self, sym: Any) -> int:
        key = sym if isinstance(sym, sympy.Basic) else sympy.sympify(sym)
        if key not in self._sizevar_index:
            self._sizevar_index[key] = self._num_sizevars
            self._num_sizevars += 1
        return self._sizevar_index[key]

    def _tensor_ref(self, name: str, cpp_dtype: str) -> str:
        idx = self._register_tensor(name)
        return f"({cpp_dtype}){self._tensor_data_name}[{idx}].data"

    def _tensor_buf_ref(self, name: str) -> str:
        idx = self._register_tensor(name)
        return f"&{self._tensor_data_name}[{idx}]"

    def _sizevar_ref(self, sym: Any) -> str:
        idx = self._register_sizevar(sym)
        return f"{self._sizevar_name}[{idx}]"

    def _register_outputs(self, output_refs) -> None:
        """Record what each result of the region sits in.

        A result that is a constant has nothing behind it: the printed program
        would describe the slot the constant occupies but no call would ever
        fill it, so the caller would read whatever the allocation happened to
        hold.  Such a region is refused here, before anything is printed, so
        it is run the way it was written.

        A result that is a view is named by the expression that spells the
        window it selects; the buffers it reads are registered where the
        expression is written.
        """

        self.output_names = list(output_refs)
        for name in output_refs:
            if name in V.graph.constants:
                raise NotImplementedError(
                    f"C++ wrapper cannot write a region whose result is the "
                    f"constant {name!r}: the printed program has no call that "
                    f"fills its slot"
                )
            self._register_tensor(name)
            self._record_buffer_meta(name)

    def _record_buffer_meta(self, name: str) -> None:
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            return
        try:
            shape = tuple(int(s) for s in buf.get_size())
            stride = tuple(int(s) for s in buf.get_stride())
        except (TypeError, ValueError):
            shape = tuple(buf.get_size())
            stride = tuple(buf.get_stride())
        self.buffer_meta[name] = (
            shape,
            stride,
            buf.get_dtype(),
            buf.get_device(),
        )

    def _record_view(
        self,
        name: str,
        base: str,
        size: Any,
        stride: Any,
        offset: Any,
        dtype: Any,
    ) -> None:
        """Record what a view of another buffer looks like.

        The generated code names a view by the expression that spells it, so
        the name is not a buffer's own.  What the caller of the entry point has
        to know is which buffer the bytes live in and how they are read, which
        is the base buffer together with the view's own shape, stride, offset
        and type.
        """

        try:
            size = tuple(int(s) for s in size)
            stride = tuple(int(s) for s in stride)
            offset = int(offset)
        except (TypeError, ValueError):
            pass
        self.view_meta[name] = (base, size, stride, offset, dtype)
        self._register_tensor(name)

    # -- overrides -------------------------------------------------------

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: "PythonWrapperCodegen | None",
        partition_signatures: Any = None,
    ) -> "CppWrapperCodegen":
        if is_subgraph:
            raise NotImplementedError(
                "C++ wrapper does not support subgraphs yet"
            )
        return CppWrapperCodegen()

    def set_launcher_fn_name(self) -> None:
        self.launcher_fn_name = "call"

    def write_constant(self, name: str, hashed: str) -> None:
        # Constants are embedded in the generated C++ or passed from Python;
        # there is no separate constant namespace in the C++ wrapper.
        pass

    def write_header(self) -> None:
        self.header.splice('#include "tensorplay/GeneratedCode.h"')
        self.header.splice('#include "Tensor.h"')
        self.header.splice("#include <cstring>")
        self.header.splice("#include <cstdio>")
        self.header.splice("#include <vector>")
        self.header.writeline("")
        self.header.splice(
            """
            struct TpTensorBuffer {
                void* data;
                const int64_t* sizes;
                const int64_t* strides;
                int64_t dim;
                int32_t dtype;
                int32_t device_type;
            };

            static tensorplay::Tensor tp_buffer_to_tensor(const TpTensorBuffer& buf) {
                std::vector<int64_t> sizes(buf.sizes, buf.sizes + buf.dim);
                std::vector<int64_t> strides(buf.strides, buf.strides + buf.dim);
                int64_t numel = 1;
                for (int64_t s : sizes) numel *= s;
                tensorplay::Device dev(static_cast<tensorplay::DeviceType>(buf.device_type));
                tensorplay::Storage storage(
                    tensorplay::DataPtr(buf.data, tensorplay::deleteNothing, dev),
                    numel * tensorplay::elementSize(static_cast<tensorplay::ScalarType>(buf.dtype)),
                    nullptr);
                return tensorplay::Tensor(
                    storage, sizes, strides,
                    static_cast<tensorplay::ScalarType>(buf.dtype), 0);
            }
            """
        )
        self.header.writeline("")

    def write_prefix(self) -> None:
        # Input handling and buffer allocation happen on the Python side of
        # the C++ wrapper; nothing is written into the generated translation
        # unit before the kernels are called.
        pass

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = True,
        cpp_definition: str | None = None,
    ) -> None:
        if cpp_definition is None:
            raise NotImplementedError(
                "C++ wrapper requires cpp_definition for kernel declarations"
            )
        self._kernel_decls.append(cpp_definition)
        self._kernel_sources.append(kernel_body)

    def codegen_sizevar(self, x: sympy.Expr) -> str:
        x = sympy.sympify(x)
        if x.is_number:
            return str(int(x))
        return self._sizevar_ref(x)

    def write_triton_header_once(self) -> None:
        pass

    def write_get_raw_stream_header_once(self) -> None:
        pass

    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str:
        return "0"

    def generate_profiler_mark_wrapper_call(self, stack) -> None:
        pass

    def generate_start_graph(self) -> None:
        pass

    def generate_end_graph(self) -> None:
        pass

    def generate_proton_finalize(self) -> None:
        pass

    def can_reuse(self, input_buffer, output_buffer=None) -> bool:
        """Whether one buffer's storage can be handed to another value.

        A generated wrapper gives every buffer it names a slot of its own and
        allocates one buffer per slot, so there is no storage to hand over: a
        kernel that read its input and wrote its result in place would read the
        result's own uninitialized buffer instead.  In-place reuse is not
        offered here, and a region that would have reused a buffer gets one
        more buffer rather than a wrong answer.
        """

        return False

    def codegen_reinterpret_view(
        self,
        data,
        size,
        stride,
        offset,
        writeline,
        dtype=None,
    ) -> str:
        """The name a view is called by, and what has to be known about it.

        A view is named by the expression that spells it -- the buffer it sits
        on, the shape it reads, the stride it steps by and where it starts --
        because that expression is what the generated kernels were written
        against.  Nothing is written here: the entry point is handed the view's
        own address at call time, which is settled from what is recorded here.
        """

        name = super().codegen_reinterpret_view(
            data, size, stride, offset, writeline, dtype
        )
        if name == data.get_name():
            # The view asks for exactly the layout the buffer already has, so
            # the buffer itself is what is read.
            return name
        # What the expression describes is the view, not the buffer: its shape
        # and stride are what it was given, and its start is that offset.
        view_dtype = dtype if dtype is not None else data.dtype
        self._record_view(name, data.get_name(), size, stride, offset, view_dtype)
        return name

    def generate_debug_sync(self, buffer) -> None:
        pass

    @staticmethod
    def _op_name_from_kernel(kernel_name: str) -> str:
        """Derive the registered op name from a generated C++ kernel name.

        ``"tp_cpu_matmul"`` -> ``"matmul"``, ``"mm_out"`` -> ``"mm"``,
        ``"tp_mm_out"`` -> ``"mm"``, ``"matmul_default"`` -> ``"matmul"``.

        The spelling a kernel is called by carries the overload it was
        generated for; what is registered is the operation in front of it.
        """

        name = kernel_name
        if name.startswith("tp_cpu_"):
            name = name[len("tp_cpu_") :]
        for suffix in ("_out", "_default"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        name = name.removeprefix("tp_").removeprefix("_")
        return name

    @staticmethod
    def _tensor_arg_name(arg: Any, writer: Any) -> str:
        """The name a call argument is read through.

        A view is read as the window it selects rather than as the buffer
        behind it, so it is named by the expression that spells it; that
        expression is what the entry point is given, and what it describes.
        """

        if isinstance(arg, (ir.Buffer, ir.MutableBox, ir.ReinterpretView)):
            return arg.codegen_reference(writer)
        return arg.get_name()

    def _tensor_slot(self, name: str) -> str:
        """Register an argument's buffer, or refuse it if it has none."""

        if name not in V.graph.name_to_buffer and name not in self.view_meta:
            raise NotImplementedError(
                "C++ wrapper extern kernel calls take buffers and views of "
                f"them; got {name!r}"
            )
        self._register_tensor(name)
        return self._tensor_buf_ref(name)

    def _generate_extern_kernel_alloc_helper(
        self, extern_kernel, args
    ) -> None:
        if isinstance(extern_kernel.layout, ir.NoneLayout):
            return
        if getattr(extern_kernel, "outputs", None):
            self._generate_fallback_kernel_helper(extern_kernel, args)
            return
        output_name = extern_kernel.get_name()
        self._register_tensor(output_name)
        self._record_buffer_meta(output_name)

        tensor_args = list(getattr(extern_kernel, "tensor_args", None) or [])
        if not tensor_args:
            # A window onto a buffer is as much an operand as the buffer is:
            # the call reads the elements it selects, so it is counted and
            # handed over as one.
            tensor_args = [
                a for a in extern_kernel.inputs if _is_tensor_operand(a)
            ]

        # The shim table is keyed by the operation, and the name the generated
        # code calls this by is the one that carries it.  Where nothing settled
        # a name, the operation itself is asked: a plain function is named by
        # its own, and an operation reached through the table by its.
        kernel_name = getattr(extern_kernel, "cpp_kernel_name", None)
        if kernel_name:
            name = self._op_name_from_kernel(kernel_name)
        else:
            target = (
                getattr(extern_kernel, "kernel", None) or extern_kernel.op_overload
            )
            name = getattr(target, "__name__", None)
        shim_name, template, literals = _extern_shim(name, len(tensor_args))
        if shim_name is None:
            if not tensor_args:
                raise NotImplementedError(
                    "C++ wrapper does not support extern kernels without "
                    "tensor inputs"
                )
            raise NotImplementedError(
                f"C++ wrapper has no shim for extern kernel {name!r}"
            )
        shim_name = self._extern_shim_source(
            shim_name,
            template,
            len(tensor_args),
            self._extern_literals_of(name, literals, args),
        )

        arg_refs = [
            self._tensor_slot(self._tensor_arg_name(t, self.wrapper_call))
            for t in tensor_args
        ]
        arg_refs.append(self._tensor_buf_ref(output_name))
        self.writeline(
            f"TP_CHECK({shim_name}({', '.join(arg_refs)}) == 0, "
            f'"extern kernel {shim_name} failed");'
        )

    def _extern_shim_source(
        self,
        shim_name: str,
        template: str,
        n_tensors: int,
        literal_values: tuple[str, ...] = (),
        n_outputs: int = 1,
    ) -> str:
        """Write a shim for one call of an operation, and name it.

        The same operation can be called with different arguments that are not
        buffers -- a convolution with a padding of one and one with a padding
        of zero -- and each of those is a different call, so each needs a shim
        of its own.  Two calls that agree share one.
        """

        key = (template, n_tensors, tuple(literal_values), n_outputs)
        named = self._extern_shim_keys.get(key)
        if named is not None:
            return named
        name = shim_name
        index = 2
        while name in self._extern_shim_names:
            name = f"{shim_name}_{index}"
            index += 1
        self._extern_shim_names.add(name)
        self._extern_shim_keys[key] = name
        self._extern_shim_sources.append(
            _build_extern_shim_impl(
                name, template, n_tensors, literal_values, n_outputs
            )
        )
        return name

    def _extern_literals_of(
        self, op_name: str, literals, args
    ) -> tuple[str, ...]:
        """The arguments an extern call was given that are not buffers.

        Each arrives as text next to the name it was passed under; the
        operation's own argument names are what says which is which, and a form
        that expects one and was not given it is an error rather than a default
        that was never asked for.
        """

        if not literals:
            return ()
        if any(kind in ("tensor", "tensor_list") for _, kind in literals):
            # A call written with its operands in fixed positions says so: each
            # argument is matched with the place it stands in, and a place that
            # holds an operand -- or the list an operation takes its operands
            # in -- contributes nothing to what is written.
            return self._extern_literals_in_order(op_name, literals, args)
        by_name = {}
        positional = []
        for arg in args:
            name, _, value = _split_named(arg)
            if name is not None:
                by_name[name] = value
            elif not (
                isinstance(arg, str)
                and (arg in V.graph.name_to_buffer or arg in self.view_meta)
            ):
                # An operand is written as the buffer it names; anything else
                # without a name is a value the operation was given.
                positional.append(arg)
        values = []
        for name, kind in literals:
            if name is None:
                # The argument stands in the call without a name, so the values
                # are taken in the order they were written.
                if not positional:
                    if kind.startswith("optional_"):
                        # An argument the call stopped short of is the absent
                        # value, which is what an optional argument is for.
                        values.append("std::nullopt")
                        continue
                    raise NotImplementedError(
                        f"C++ wrapper extern kernel {op_name!r} was called "
                        f"without its {kind!r} argument"
                    )
                values.append(_literal_value(kind, positional.pop(0)))
                continue
            if name not in by_name:
                if kind.startswith("optional_"):
                    values.append("std::nullopt")
                    continue
                raise NotImplementedError(
                    f"C++ wrapper extern kernel {op_name!r} was called "
                    f"without its {name!r} argument"
                )
            values.append(_literal_value(kind, by_name[name]))
        return tuple(values)

    def _extern_literals_in_order(
        self, op_name: str, literals, args
    ) -> tuple[str, ...]:
        """The arguments of a call written in fixed positions.

        A call whose operands and values alternate is described by the places
        they stand in: each entry of the form says what is at that place, and an
        entry that names an operand contributes nothing, because the operand is
        handed over as a slot rather than written here.

        The operands of a call that takes them as a list may arrive as one list
        or as a run of arguments written one after another -- a call recorded
        from a variadic spelling has no list at all.  Both are the same call,
        so both are consumed here.
        """

        values = []
        index = 0
        for name, kind in literals:
            if kind == "tensor":
                if index >= len(args):
                    raise NotImplementedError(
                        f"C++ wrapper extern kernel {op_name!r} was called "
                        f"with fewer arguments than its {kind!r} argument"
                    )
                index += 1
                continue
            if kind == "tensor_list":
                if index < len(args) and _is_list_text(args[index]):
                    # The operands arrived as the list the operation declares.
                    index += 1
                    continue
                # The operands arrived one after another, each an argument of
                # its own; they are consumed for as long as they name memory.
                first = index
                while index < len(args) and self._is_operand_text(args[index]):
                    index += 1
                if index == first:
                    raise NotImplementedError(
                        f"C++ wrapper extern kernel {op_name!r} was called "
                        "without the operands it takes as a list"
                    )
                continue
            if index >= len(args):
                if kind == "optional_ints":
                    # An absent list asks for the operation's defaults, which
                    # is the empty vector.
                    values.append(_literal_value(kind, None))
                    continue
                raise NotImplementedError(
                    f"C++ wrapper extern kernel {op_name!r} was called with "
                    f"fewer arguments than its {name or kind!r} argument"
                )
            arg = args[index]
            index += 1
            _, _, value = _split_named(arg)
            values.append(_literal_value(kind, value))
        if index != len(args):
            raise NotImplementedError(
                f"C++ wrapper extern kernel {op_name!r} was called with "
                f"{len(args) - index} argument(s) nothing describes"
            )
        return tuple(values)

    def _is_operand_text(self, text: Any) -> bool:
        """Whether an argument as written names memory the call reads."""

        return (
            isinstance(text, str)
            and (text in V.graph.name_to_buffer or text in self.view_meta)
        )

    def codegen_multi_output(self, node: ir.MultiOutput) -> None:
        """One of several results of a single call.

        A call that answers with several results writes each of them into the
        buffer it belongs in, so the buffer is the result and there is nothing
        to take out of anything.  A result that is one part of an answer this
        wrapper does not write that way is refused rather than left as a name
        nothing ever fills.
        """

        producer = V.graph.name_to_buffer.get(node.input_name(0))
        if not isinstance(producer, ir.FallbackKernel):
            raise NotImplementedError(
                "C++ wrapper takes several results only from a call it writes "
                f"as a whole; got {type(producer).__name__}"
            )

    def _codegen_assert_size_stride(
        self,
        code,
        name: str,
        size: str,
        stride: str,
        op_name: str,
        dtype=None,
    ) -> None:
        """The checks that a result really has the shape it was given.

        Every buffer this wrapper hands a call is allocated by the entry point
        from the shape and stride recorded for it, so there is nothing here
        that could disagree with what the call wrote: the check would guard a
        value the entry point chose itself.
        """

        return

    def _generate_extern_kernel_out_helper(
        self,
        kernel_name,
        output_name,
        output_view_name,
        args,
        device,
        stack_traces,
    ) -> None:
        self._register_tensor(output_name)
        self._record_buffer_meta(output_name)
        op_name = self._op_name_from_kernel(kernel_name)
        tensor_refs = []
        for arg in args:
            if isinstance(arg, str) and (
                arg in V.graph.name_to_buffer or arg in self.view_meta
            ):
                tensor_refs.append(self._tensor_slot(arg))
                continue
            name, _, _ = _split_named(arg)
            if name in _extern_literal_names(op_name):
                # A number is written into the call where it stands, so it is
                # left to the shim's own template to spell it.
                continue
            raise NotImplementedError(
                "C++ wrapper extern kernel out calls support only tensor "
                f"arguments; got {arg!r} for {op_name!r}"
            )
        # The arity is the number of tensors the call hands over: the result
        # is passed separately, and an operation with an optional argument has
        # a form for each of the ways it can be called without it.
        shim_name, template, literals = _extern_shim(
            op_name, len(tensor_refs)
        )
        if shim_name is None:
            raise NotImplementedError(
                f"C++ wrapper has no shim for extern kernel {op_name!r}"
            )
        shim_name = self._extern_shim_source(
            shim_name,
            template,
            len(tensor_refs),
            self._extern_literals_of(op_name, literals, args),
        )
        out_ref = self._tensor_buf_ref(output_name)
        self.writeline(
            f"TP_CHECK({shim_name}({', '.join(tensor_refs + [out_ref])}) == 0, "
            f'"extern kernel {shim_name} failed");'
        )

    def _generate_extern_kernel_multi_out_helper(
        self, kernel_name, output_names, args, device, stack_traces
    ) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support extern kernel multi-out calls yet"
        )

    def _generate_fallback_kernel_helper(
        self,
        extern_kernel,
        args,
    ) -> None:
        """Write a call that leaves more than one result behind.

        Some operations answer with a pair -- a sort's values and the order
        that produced them -- and neither half is the call's result on its own:
        each is a buffer of its own, which is why they arrive here as several.
        What is written is a shim that calls the operation once and copies each
        half into the buffer it belongs in, in the order the operation answers.

        An operation whose several results are one of them a place an argument
        already had is not this: there the call writes back into memory it was
        handed rather than into memory of its own, and that is not a call this
        can describe.
        """

        if not isinstance(extern_kernel, ir.FallbackKernel):
            raise NotImplementedError(
                "C++ wrapper cannot write the several results of "
                f"{type(extern_kernel).__name__}"
            )
        output_names = [o.get_name() for o in extern_kernel.outputs]
        for name in output_names:
            self._register_tensor(name)
            self._record_buffer_meta(name)

        tensor_args = list(getattr(extern_kernel, "tensor_args", None) or [])
        if not tensor_args:
            # A window onto a buffer is as much an operand as the buffer is:
            # the call reads the elements it selects, so it is counted and
            # handed over as one.
            tensor_args = [
                a for a in extern_kernel.inputs if _is_tensor_operand(a)
            ]
        if not tensor_args:
            raise NotImplementedError(
                "C++ wrapper does not support extern kernels without "
                "tensor inputs"
            )

        kernel_name = getattr(extern_kernel, "cpp_kernel_name", None)
        name = kernel_name
        if name is None:
            target = getattr(extern_kernel, "kernel", None) or extern_kernel.op_overload
            name = getattr(target, "__name__", None)
        found = _EXTERN_KERNEL_MULTI_SHIMS.get(name)
        if found is None and kernel_name:
            found = _EXTERN_KERNEL_MULTI_SHIMS.get(
                self._op_name_from_kernel(kernel_name)
            )
        if found is None:
            raise NotImplementedError(
                "C++ wrapper has no shim for the several results of "
                f"{name!r}"
            )
        shim_name, template, literals = found
        shim_name = self._extern_shim_source(
            shim_name,
            template,
            len(tensor_args),
            self._extern_literals_of(name, literals, args),
            len(output_names),
        )

        arg_refs = [
            self._tensor_slot(self._tensor_arg_name(t, self.wrapper_call))
            for t in tensor_args
        ]
        arg_refs.extend(self._tensor_buf_ref(name) for name in output_names)
        self.writeline(
            f"TP_CHECK({shim_name}({', '.join(arg_refs)}) == 0, "
            f'"extern kernel {shim_name} failed");'
        )

    #: The lines these two fallbacks are written as, and what the host
    #: translation unit does with them.  Neither is a call the wrapper can
    #: spell -- one writes through an index and the other reduces through one
    #: -- so both are refused here, before anything is printed: a refusal
    #: leaves the region to the interpreter, where the call is written as it
    #: was captured, while printing one would leave a translation unit that
    #: cannot compile.
    def generate_index_put_fallback(self, node) -> None:
        self._generate_index_put_fallback_helper(node)

    def generate_scatter_fallback(self, node) -> None:
        self._generate_scatter_fallback_helper(node)

    def _generate_index_put_fallback_helper(self, node) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support index_put fallback yet"
        )

    def _generate_scatter_fallback_helper(self, node) -> None:
        raise NotImplementedError(
            "C++ wrapper does not support scatter fallback yet"
        )

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta=None,
        tp_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ) -> None:
        if triton:
            raise NotImplementedError(
                "C++ wrapper does not support triton kernels yet"
            )
        if device is not None and device.type != "cpu":
            raise NotImplementedError(
                "C++ wrapper currently supports CPU kernels only"
            )
        cpp_args = []
        for arg, arg_type in zip(call_args, arg_types):
            if arg_type.endswith("*"):
                cpp_args.append(self._tensor_ref(arg, arg_type))
                self._record_buffer_meta(arg)
            else:
                expr = sympy.sympify(arg)
                if expr.is_number:
                    cpp_args.append(str(int(expr)))
                else:
                    cpp_args.append(self._sizevar_ref(expr))
        self.writeline(f"{kernel_name}({', '.join(cpp_args)});")

    def generate_return(self, output_refs) -> None:
        # Outputs are written through tensor_data; there is no return value.
        pass

    def generate_before_suffix(self, result) -> None:
        result.writeline("")
        result.writeline(
            f'extern "C" void call(TpTensorBuffer* {self._tensor_data_name}, '
            f"long* {self._sizevar_name}) {{"
        )

    def generate_after_suffix(self, result) -> None:
        pass

    def generate_end(self, result) -> None:
        result.writeline("}")

    def add_benchmark_harness(self, output) -> None:
        pass

    def _generate(self, is_inference):
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)

            with self.set_writeline(self.wrapper_call, self.wrapper_call.writeline):
                for line in self.lines:
                    if isinstance(line, MemoryPlanningLine):
                        # Buffer allocation and reuse are handled by the
                        # Python runtime of the C++ wrapper; the generated
                        # translation unit only references buffers through
                        # the flat tensor_data array.
                        if isinstance(line, AllocateLine):
                            self._register_tensor(line.node.get_name())
                            self._record_buffer_meta(line.node.get_name())
                        continue
                    if isinstance(line, FreeLine):
                        continue
                    if isinstance(line, WrapperLine):
                        line.codegen(self.wrapper_call)
                    else:
                        self.wrapper_call.writeline(line)

            output_refs = self.get_output_refs()
            self._register_outputs(output_refs)
            self.generate_return(output_refs)

        result = IndentedBuffer()
        result.splice(self.imports)
        result.writeline("")
        result.splice(self.header)
        for decl in self._kernel_decls:
            result.writeline(decl)
        result.writeline("")
        for impl in self._extern_shim_sources:
            result.splice(impl)
            result.writeline("")
        result.splice(self.subgraph_definitions)
        self.finalize_prefix()
        result.splice(self.prefix)

        self.generate_before_suffix(result)

        with result.indent(self.get_wrapper_call_indent()):
            result.splice(self.wrapper_call)

        result.splice(self.suffix)
        self.generate_after_suffix(result)
        self.generate_end(result)

        full_source = result.getvaluewithlinemap()
        value = full_source.value
        if self._kernel_sources:
            value = value + "\n\n" + "\n\n".join(self._kernel_sources)

        for name in list(self._tensor_index):
            if name not in self.buffer_meta:
                self._record_buffer_meta(name)

        tensor_input_names = [
            name
            for name in V.graph.graph_input_names
            if name in V.graph.graph_inputs_original
        ]
        # Every value the caller hands over is described, whether the generated
        # code names it or only reads it through a window: the caller fills the
        # slot from the shape it was given, and a package has to be able to
        # check that against the value it was built for.
        for name in tensor_input_names:
            self._record_buffer_meta(name)
        code = CppWrapperCode(
            value=value,
            line_map=full_source.line_map,
            buffer_meta=self.buffer_meta,
            output_names=self.output_names,
            tensor_index=self._tensor_index,
            sizevar_index=self._sizevar_index,
            num_tensors=self._num_tensors,
            num_sizevars=self._num_sizevars,
            graph_input_names=tensor_input_names,
            view_meta=self.view_meta,
        )
        return (
            code,
            self.kernel_declarations.getvaluewithlinemap(),
        )
