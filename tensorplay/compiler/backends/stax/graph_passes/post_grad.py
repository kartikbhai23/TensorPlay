import operator
from typing import Any

import sympy
import tensorplay as tp
from tensorplay.graph import Graph, Node, map_arg
from tensorplay.graph.experimental.symbolic_shapes import statically_known_true, sym_eq
from tensorplay.graph.experimental.sympy_functions import OrderedSet
from tensorplay._decomp import register_decomposition
from tensorplay.primitives.common import (
    is_boolean_dtype,
    is_expandable_to,
    is_integer_dtype,
)

from .. import config
from ..fx_utils import get_fake_args_kwargs, get_node_storage
from ..utils import is_gpu, is_view
from ..pattern_matcher import (
    Arg,
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    get_arg_value,
    get_mutation_region_id,
    register_graph_pattern,
    register_lowering_pattern as _register_lowering_pattern,
)


tp_ops = tp.ops.tp
prims = tp.ops.prims

#: The pattern tables the passes draw from, applied in order: a match found by
#: an earlier table is taken, so a table written later has the first say on
#: which of two ways of writing the same computation is used.
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def register_lowering_pattern(
    pattern,
    extra_check=lambda match: True,
    pass_number=1,
    *,
    output_metadata_ignores_input_storage: bool = True,
    output_metadata_is_input: int | str | None = None,
    output_metadata_fn=None,
):
    """Register a replacement of one way of writing a computation by another."""

    return _register_lowering_pattern(
        pattern,
        extra_check,
        pass_dict=pass_patterns[pass_number],
        output_metadata_ignores_input_storage=output_metadata_ignores_input_storage,
        output_metadata_is_input=output_metadata_is_input,
        output_metadata_fn=output_metadata_fn,
    )


def same_meta(node1: Node, node2: Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and isinstance(val1, tp.Tensor)
        and isinstance(val2, tp.Tensor)
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != tp.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
        # Check conjugate and negative bits - a clone that resolves these is not a no-op
        and val1.is_conj() == val2.is_conj()
        and val1.is_neg() == val2.is_neg()
    )




noop_registry: dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry)(
            (cond, nop_arg)  # type: ignore[arg-type]
        )
        return cond

    return register_fun


def _needs_spmd_graph_preservation() -> bool:
    """Check if SPMD graph preservation is needed for distributed overlap."""
    return (
        config.tp_distributed_optimizations.enable_overlap_scheduling
        or config.reorder_for_compute_comm_overlap
    )


@register_noop_decomp(tp_ops.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if _needs_spmd_graph_preservation():
        # Keep no-op slices so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    if start is None or end is None:
        return False

    slice_dim_size = self.shape[dim]
    if (
        statically_known_true(sym_eq(start, 0))
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_dim_size)
        )
        and statically_known_true(sym_eq(step, 1))
    ):
        return True
    return False


@register_noop_decomp(tp_ops.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    slice_scatter_dim_size = self.shape[dim]
    if (
        self.shape == src.shape
        and start == 0
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_scatter_dim_size)
        )
        and step == 1
    ):
        return True
    return False


@register_noop_decomp(tp_ops.repeat)
def repeat_noop(self, repeats):
    return all(r == 1 for r in repeats)


@register_noop_decomp(tp_ops.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    if _needs_spmd_graph_preservation():
        # Keep no-op pads so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    return all(p == 0 for p in padding)


# A change of type and a change of device are not registered here.  Both are
# plain functions here rather than dispatched operations, so a node naming one
# is a call to that function and not a call that a decomposition stands in
# front of -- there is nothing wrapped around it to take off, and a
# registration would have nothing to attach to.


@register_noop_decomp([tp_ops.ceil, tp_ops.floor, tp_ops.round, tp_ops.trunc])
def int_noop(x):
    return is_integer_dtype(x.dtype)


@register_noop_decomp([tp_ops.pow])
def pow_noop(a, b):
    return isinstance(b, int) and b == 1


@register_noop_decomp([tp_ops.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    return len(inputs) == 1


@register_noop_decomp(tp_ops.view.default)
def view_default_noop(arg, size):
    return statically_known_true(sym_eq(arg.shape, tuple(size)))


@register_noop_decomp(tp_ops.view.dtype)
def view_dtype_noop(arg, dtype):
    return arg.dtype == dtype


# Note, we also always have a check for identical metadata, which is why these
# are safe
@register_noop_decomp([tp_ops.copy], nop_arg=1)
@register_noop_decomp([tp_ops.alias, tp_ops.clone])
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: Graph):
    """
    Removes both operations that are essentially tp_ops.clone and operations that are essentially tp_ops.alias from the graph.
    """
    inputs = OrderedSet[Node]()
    input_storages = OrderedSet[int | None]()
    output_storages = OrderedSet[int | None]()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    if output_node.op != "output":
        raise AssertionError(f"expected output node, got {output_node.op}")
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, Node):
                continue

            if node.target is tp_ops.copy.default:
                dst = node.args[0]
                if (
                    isinstance(dst, Node)
                    and dst.op == "call_function"
                    and dst.kwargs.get("pin_memory") is True
                ):
                    continue

            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)


@register_graph_pattern(
    CallFunction(
        tp_ops.reciprocal.default,
        CallFunction(tp_ops.sqrt.default, KeywordArg("value")),
    ),
    pass_dict=pass_patterns[1],
)
def reciprocal_sqrt_to_rsqrt(match: Match, value):
    """Write the reciprocal of a square root as the one operation it is.

    Dividing one by a square root is the reciprocal square root, which is one
    value per element rather than two.
    """

    def replacement(argument):
        return tp_ops.rsqrt.default(argument)

    match.replace_by_example(replacement, [value])


_cat_before_prefix = CallFunction(tp_ops.cat.default, Arg(), 1, _users=2)


def _prefix_fits_first_cat_input(match: Match) -> bool:
    size = match.kwargs["size"]
    first = match.args[0][0]
    value = first.meta.get("val")
    return (
        isinstance(size, int)
        and size >= 0
        and value is not None
        and statically_known_true(size <= value.size(1))
    )


@register_lowering_pattern(
    CallFunction(
        tp_ops.cat.default,
        [
            _cat_before_prefix,
            CallFunction(
                tp_ops.slice.Tensor,
                _cat_before_prefix,
                1,
                0,
                KeywordArg("size"),
                1,
            ),
        ],
        1,
    ),
    extra_check=_prefix_fits_first_cat_input,
)
def cat_slice_cat(match, cat_input, size):
    from ..op_lowerings import lower_cat, slice_
    from ..ir import ConcatKernel, TensorBox

    first, *rest = cat_input
    inputs = [first, *rest, slice_(first, 1, 0, size)]
    if (
        first.get_device().type == "cpu"
        and all(value.get_dtype() == first.get_dtype() for value in inputs)
        and all(sympy.sympify(value.get_size()[1]).is_positive is True for value in inputs)
    ):
        return TensorBox(ConcatKernel.create(inputs, 1))
    return lower_cat(inputs, 1)


cat_slice_cat._tp_cache_key = "cat_slice_cat_v2"


def _pointless_cumsum_non_scalar(match: Match) -> bool:
    # A cumsum over zero dimensions answers the filled value itself, which
    # the lowering already answers without a walk.
    return len(match.kwargs["shape"]) > 0


@register_graph_pattern(
    CallFunction(
        tp_ops.cumsum.default,
        CallFunction(
            tp_ops.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=Ignored(),
            device=KeywordArg("device"),
            pin_memory=Ignored(),
            _users=MULTIPLE,
        ),
        KeywordArg("dim"),
        # The dtype the scan records itself with rides along as a trailing
        # optional argument; the rewrite answers it through the output node.
        Ignored(),
        _users=MULTIPLE,
    ),
    extra_check=_pointless_cumsum_non_scalar,
    pass_dict=pass_patterns[1],
)
def pointless_cumsum(match: Match, shape, fill_value, device, dim):
    """A ramp of the fill value is what a filled run sums to.

    Reading the pattern from OPTForCausalLM: every row of the filled tensor
    holds one value repeated, so the running total along the summed axis is
    the value times 1..dim_size -- writable as a multiply on an arange, which
    is one kernel instead of a fill followed by a sequential scan.
    """

    filled = match.output_node().args[0]
    dtype = filled.kwargs.get("dtype")
    if dtype is None or not hasattr(dtype, "name"):
        dtype = fill_value.dtype if isinstance(fill_value, tp.Tensor) else tp.float32
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        fill_value = int(bool(fill_value) if is_boolean_dtype(dtype) else fill_value)
        # The running total promotes every integral input to int64.
        dtype = tp.int64

    out_dtype = match.output_node().kwargs.get("dtype") or dtype
    bool_out = out_dtype == tp.bool
    integral_out = bool_out or is_integer_dtype(out_dtype)
    if integral_out:
        fill_value = int(bool(fill_value) if bool_out else fill_value)
    acc_dtype = tp.int64 if integral_out else tp.float64

    # Every ingredient is a constant of the capture -- the shape, the fill,
    # the device -- but the ramp cannot be traced as a factory call (the
    # shape walks refuse symbolic sizes) nor handed over as a constant (the
    # replacement join wants nodes).  It is built as a chain of registered
    # operations inside the graph; the stand-in values the chain needs are
    # refreshed by the propagation at the end of the pass.
    node = match.output_node()
    graph = node.graph
    inter_shape = [1] * len(shape)
    inter_shape[dim] = shape[dim]
    ops = tp.ops.tp
    with graph.inserting_before(node):
        idx = graph.create_node(
            "call_function", ops.arange.start_step,
            (1, shape[dim] + 1),
            kwargs={"device": device, "dtype": acc_dtype},
        )
        mul = graph.create_node(
            "call_function", ops.mul.Tensor, (idx, fill_value)
        )
        view = graph.create_node(
            "call_function", ops.view.default, (mul, inter_shape)
        )
        expanded = graph.create_node(
            "call_function", ops.expand.default, (view, shape)
        )
        if bool_out or integral_out:
            replaced = graph.create_node(
                "call_function", ops.to.dtype, (expanded, out_dtype)
            )
        else:
            replaced = expanded
    node.replace_all_uses_with(replaced)
    graph.erase_node(node)


def view_to_reshape(graph: Graph) -> None:
    """Retarget ``view`` to ``reshape``.

    The lowering answers both spellings with the same kernel, but downstream
    readers -- the fusion decisions above, a serialized deployment form --
    would otherwise see two names for one operation.
    """

    for node in graph.find_nodes(op="call_function", target=tp_ops.view.default):
        node.target = tp_ops.reshape.default
    for node in graph.find_nodes(op="call_method", target="view"):
        node.target = tp_ops.reshape.default
        node.op = "call_function"


def _is_bias_like_addmm_input(inp: Node, output: Node) -> bool:
    # The wider upstream check also accepts computed inputs whose value is a
    # real tensor; the values recorded here are descriptions, so only the
    # shapes a capture read directly qualify.
    return inp.op in ("placeholder", "get_attr")


#: The device the region being rewritten runs on.  The values the patterns
#: see are stand-ins on the meta device, so a decision that depends on where
#: the kernels would run reads the region's device instead.
_region_input_device: str | None = None


def _match_device_type(value: Any) -> str:
    """The device type a match's value would run on.

    The stand-ins the patterns match against all report ``meta``; the region
    itself was opened with a real device, recorded when the pass ran.
    """

    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is None or device_type == "meta":
        if _region_input_device is not None:
            return _region_input_device
        return "cuda" if tp.cuda.is_available() else "cpu"
    return device_type


def is_pointwise_use(use: Node) -> bool:
    """Whether every consumer behind ``use`` is a pointwise computation.

    Reads through views and selections: reshaping a value does not change
    what is computed on it.
    """

    if use.op != "call_function":
        return False
    target = use.target
    if target is operator.getitem or is_view(target):
        return all(is_pointwise_use(user) for user in use.users)
    return "pointwise" in getattr(target, "tags", ())


def should_prefer_unfused_addmm(match: Match) -> bool:
    """Whether an addmm is worth reading as a pointwise chain over a matmul.

    Only the GPU fuses the add into the matmul's epilogue; elsewhere the
    fused form is already one kernel and unfusing it adds operations.
    """

    inp = match.kwargs["inp"]
    value = inp.meta.get("val")
    if value is None or not is_gpu(_match_device_type(value)):
        return False
    mat1, mat2 = match.args
    keys = {
        (getattr(v, "dtype", None), getattr(v, "device", None))
        for v in (value, mat1.meta.get("val"), mat2.meta.get("val"))
    }
    if len(keys) != 1:
        return False
    beta = match.kwargs.get("beta", 1)
    # A zero beta ignores the added input outright; expanding its shape to
    # the product's needs an analysis this check does not carry, so the
    # non-CUDA side keeps the fused form.
    if beta == 0 and value.device.type != "cuda":
        return False
    output = match.output_node()
    if not _is_bias_like_addmm_input(inp, output):
        return False
    return all(is_pointwise_use(use) for use in output.users)


@register_graph_pattern(
    CallFunction(
        tp_ops.addmm.default,
        KeywordArg("inp"),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    extra_check=should_prefer_unfused_addmm,
    pass_dict=pass_patterns[2],
)
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp, alpha, beta):
    def repl(inp, x1, x2, alpha, beta):
        mm_result = x1 @ x2
        if alpha != 1:
            mm_result = alpha * mm_result
        if beta != 1:
            inp = beta * inp
        return inp + mm_result

    match.replace_by_example(repl, [inp, mat1, mat2, alpha, beta])


def should_prefer_unfused_baddbmm(match: Match) -> bool:
    inp = match.kwargs["inp"]
    value = inp.meta.get("val")
    if value is None or not is_gpu(_match_device_type(value)):
        return False
    mat1, mat2 = match.args
    keys = {
        (getattr(v, "dtype", None), getattr(v, "device", None))
        for v in (value, mat1.meta.get("val"), mat2.meta.get("val"))
    }
    if len(keys) != 1:
        return False
    output = match.output_node()
    if not _is_bias_like_addmm_input(inp, output):
        return False
    return all(is_pointwise_use(use) for use in output.users)


@register_graph_pattern(
    CallFunction(
        tp_ops.baddbmm.default,
        KeywordArg("inp"),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    extra_check=should_prefer_unfused_baddbmm,
    pass_dict=pass_patterns[2],
)
def unfuse_bias_baddbmm_to_pointwise(match: Match, mat1, mat2, *, inp, alpha, beta):
    def repl(inp, x1, x2, alpha, beta):
        bmm_result = x1 @ x2
        if alpha != 1:
            bmm_result = alpha * bmm_result
        if beta != 1:
            inp = beta * inp
        return inp + bmm_result

    match.replace_by_example(repl, [inp, mat1, mat2, alpha, beta])


def _floating_value(value: Any) -> bool:
    """Whether a recorded value, or its description, holds real numbers."""

    flag = getattr(value, "is_floating_point", None)
    if flag is not None:
        return bool(flag() if callable(flag) else flag)
    dtype = getattr(value, "dtype", None)
    return bool(getattr(dtype, "is_floating_point", False))


def is_valid_addmm_fusion(match: Match) -> bool:
    """Whether a product and an added input are worth writing as one call.

    The added input is folded into the product's epilogue, which needs it to
    be broadcastable to the product's shape and to hold the same numbers as
    the operands; a different dtype would make the fused call compute
    something the addition did not.
    """

    inp = match.kwargs["inp"]
    inp_value = inp.meta.get("val")
    if inp_value is None or not hasattr(inp_value, "shape"):
        return False
    mat1, mat2 = match.args
    mat1_value = mat1.meta.get("val")
    mat2_value = mat2.meta.get("val")
    if mat1_value is None or mat2_value is None:
        return False
    mat1_shape = tuple(getattr(mat1_value, "shape", ()))
    mat2_shape = tuple(getattr(mat2_value, "shape", ()))
    if len(mat1_shape) != 2 or len(mat2_shape) != 2:
        return False
    if not is_expandable_to(
        tuple(inp_value.shape), (mat1_shape[0], mat2_shape[1])
    ):
        return False
    if inp_value.dtype != mat1_value.dtype or inp_value.dtype != mat2_value.dtype:
        return False
    # The addition may carry a scale of its own; the single call means the
    # unscaled addition.
    alpha = get_arg_value(match.output_node(), 2, "alpha")
    if alpha is not None and alpha != 1:
        return False
    # Where the fusion would spoil a chain the region could fold into the
    # product's epilogue anyway, the two calls are left as they are.
    return not should_prefer_unfused_addmm(match)


@register_graph_pattern(
    CallFunction(
        tp_ops.add.Tensor,
        CallFunction(tp_ops.mm.default, Arg(), Arg()),
        KeywordArg("inp"),
    ),
    extra_check=is_valid_addmm_fusion,
    pass_dict=pass_patterns[2],
)
@register_graph_pattern(
    CallFunction(
        tp_ops.add.Tensor,
        KeywordArg("inp"),
        CallFunction(tp_ops.mm.default, Arg(), Arg()),
    ),
    extra_check=is_valid_addmm_fusion,
    pass_dict=pass_patterns[2],
)
def fuse_addmm(match: Match, mat1, mat2, *, inp):
    """A product with an added input is the product-with-bias operation.

    Written as two calls, the addition is a pass over the product's result;
    written as one, the bias is folded into the product's own epilogue and
    the intermediate never exists.
    """

    node = match.output_node()
    graph = node.graph
    with graph.inserting_before(node):
        fused = graph.create_node(
            "call_function", tp_ops.addmm.default, (inp, mat1, mat2)
        )
    # The call computes what the addition computed, so the value the node
    # stood for is the value it now stands for.
    fused.meta.update(node.meta)
    node.replace_all_uses_with(fused)
    match.erase_nodes()


def _is_addcdiv_fma_eligible(match: Match) -> bool:
    """Whether a scaled quotient and an addition are worth re-fusing.

    The single-call form rounds the quotient once and takes the product and
    the sum in one step where the device has such a step; a chain of three
    calls rounds between each of them.  Only a device with the step gains
    anything, and the single-call form asks for real numbers throughout.
    """

    inp_value = match.kwargs["inp"].meta.get("val")
    if inp_value is None or not _floating_value(inp_value):
        return False
    out_value = match.output_node().meta.get("val")
    if out_value is None or _match_device_type(out_value) not in ("cuda", "xpu"):
        return False
    for key in ("t1", "t2"):
        node = match.kwargs.get(key)
        value = node.meta.get("val") if isinstance(node, Node) else node
        if value is None or not _floating_value(value):
            return False
    # A quotient rounded toward zero is not the quotient the single call
    # takes; only the unrounded one is.
    quotient = next(
        (n for n in match.nodes if n.target is tp_ops.div.Tensor), None
    )
    if quotient is not None and get_arg_value(
        quotient, 2, "rounding_mode"
    ) not in (None, "None"):
        return False
    # The scale is a number; a tensor there is a different operation.
    return not isinstance(match.kwargs.get("value"), Node)


@register_graph_pattern(
    CallFunction(
        tp_ops.add.Tensor,
        KeywordArg("inp"),
        CallFunction(
            tp_ops.mul.Scalar,
            CallFunction(tp_ops.div.Tensor, KeywordArg("t1"), KeywordArg("t2")),
            KeywordArg("value"),
        ),
    ),
    extra_check=_is_addcdiv_fma_eligible,
    pass_dict=pass_patterns[2],
)
@register_graph_pattern(
    CallFunction(
        tp_ops.add.Tensor,
        KeywordArg("inp"),
        CallFunction(
            tp_ops.mul.Tensor,
            CallFunction(tp_ops.div.Tensor, KeywordArg("t1"), KeywordArg("t2")),
            KeywordArg("value"),
        ),
    ),
    extra_check=_is_addcdiv_fma_eligible,
    pass_dict=pass_patterns[2],
)
def fuse_addcdiv(match: Match, inp, t1, t2, value):
    """This, plus a scaled quotient, is the single call that says so."""

    node = match.output_node()
    graph = node.graph
    with graph.inserting_before(node):
        fused = graph.create_node(
            "call_function", tp_ops.addcdiv.default, (inp, t1, t2),
            {"value": value},
        )
    fused.meta.update(node.meta)
    node.replace_all_uses_with(fused)
    match.erase_nodes()


#: The two ways a value is cut into pieces of given sizes: one takes the sizes
#: as a list, the other takes them by name.
_SPLIT_TARGETS = (tp_ops.split.sizes, tp_ops.split_with_sizes.default)


def _split_and_cat(match: Match) -> tuple[Node | None, Node | None]:
    """The one split and the one cat a match found, or ``(None, None)``."""

    splits = [node for node in match.nodes if node.target in _SPLIT_TARGETS]
    cats = [node for node in match.nodes if node.target is tp_ops.cat.default]
    if len(splits) != 1 or len(cats) != 1:
        return None, None
    return splits[0], cats[0]


def _is_split_rejoined_by_cat(match: Match) -> bool:
    """Whether a cat puts back every piece of a split, in the original order.

    Joining pieces that were just cut from one value gives the value back, so
    the pair is a copy that the program does not need.  Only a cat of every
    piece along the same dimension, in the order the pieces were cut, says that.
    """

    split_node, cat_node = _split_and_cat(match)
    if split_node is None or cat_node is None:
        return False
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    sizes = get_arg_value(split_node, 1, "split_sizes")
    if not isinstance(sizes, (list, tuple)) or not sizes:
        return False
    pieces = get_arg_value(cat_node, 0)
    if not isinstance(pieces, (list, tuple)) or len(pieces) != len(sizes):
        return False
    indices = [get_arg_value(piece, 1) for piece in pieces]
    if any(index is None for index in indices):
        return False
    return indices == list(range(len(sizes)))


@register_graph_pattern(
    CallFunction(
        tp_ops.cat.default,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    [tp_ops.split.sizes, tp_ops.split_with_sizes.default],
                    KeywordArg("source"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    extra_check=_is_split_rejoined_by_cat,
    pass_dict=pass_patterns[2],
)
def split_rejoined_by_cat(match: Match, source):
    """Drop a split whose every piece is joined back in the same order."""

    node = match.output_node()
    node.replace_all_uses_with(source)
    for matched in reversed(match.nodes):
        if not matched.users:
            match.graph.erase_node(matched)


def _is_cat_split_back_to_pieces(match: Match) -> bool:
    """Whether a split cuts a cat exactly at the joins between its pieces.

    Cutting a joined value at the sizes of the pieces it was joined from gives
    those pieces back, so the join and the cut can both be dropped.  The cut
    has to be along the joining dimension, and each piece has to be exactly the
    size of the cut that reads it.
    """

    split_node, cat_node = _split_and_cat(match)
    if split_node is None or cat_node is None:
        return False
    dim = get_arg_value(split_node, 2, "dim")
    if dim != get_arg_value(cat_node, 1, "dim"):
        return False
    sizes = get_arg_value(split_node, 1, "split_sizes")
    pieces = get_arg_value(cat_node, 0)
    if not isinstance(sizes, (list, tuple)) or not isinstance(pieces, (list, tuple)):
        return False
    if not sizes or len(sizes) != len(pieces):
        return False
    for piece, size in zip(pieces, sizes):
        if not isinstance(size, int) or not isinstance(dim, int):
            return False
        value = piece.meta.get("val")
        if value is None or not statically_known_true(sym_eq(value.size(dim), size)):
            return False
    # What reads the pieces of the cut has to be the cut itself, so that each
    # reader can be handed the piece it asked for.
    for user in split_node.users:
        if user.target is not operator.getitem:
            return False
        index = get_arg_value(user, 1)
        if not isinstance(index, int) or index < 0 or index >= len(pieces):
            return False
    return True


@register_graph_pattern(
    CallFunction(
        [tp_ops.split.sizes, tp_ops.split_with_sizes.default],
        CallFunction(
            tp_ops.cat.default,
            KeywordArg("cat_input"),
            KeywordArg("cat_dim"),
            _users=MULTIPLE,
        ),
        KeywordArg("sizes"),
        KeywordArg("dim"),
    ),
    extra_check=_is_cat_split_back_to_pieces,
    pass_dict=pass_patterns[2],
)
def cat_split_back_to_pieces(match: Match, cat_input, cat_dim, sizes, dim):
    """Hand each reader of a cut the piece of the join it asked for."""

    split_node = match.output_node()
    graph = match.graph
    for user in list(split_node.users):
        user.replace_all_uses_with(cat_input[get_arg_value(user, 1)])
        graph.erase_node(user)
    for matched in reversed(match.nodes):
        if not matched.users:
            graph.erase_node(matched)


def remove_profiler_ops(graph: Graph) -> None:
    """Drop the spans that only tell the profiler where work happened.

    A span marks a stretch of the program for the profiler and computes
    nothing.  A compiled region is one call from the profiler's point of view,
    so a span inside it labels work that is no longer kept apart; left in, it
    gives every value it touches a second reader to consider, which keeps a
    fusion from seeing that the value's producer could move.
    """

    from tensorplay.profiler.profiler import (
        _record_function_enter,
        _record_function_exit,
    )

    spans = {_record_function_enter, _record_function_exit}
    doomed = [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target in spans
    ]
    # The end of a span reads its beginning, so the ends go first.
    for node in reversed(doomed):
        graph.erase_node(node)


def remove_assert_ops(graph: Graph) -> None:
    """Drop the promises a graph makes about a value's metadata.

    An assertion about a value's shape, dtype or layout states what the caller
    already promised, and nothing the region computes comes from it.  Left in,
    it gives the value a second reader that is not a computation, which keeps
    a rewrite from moving the value's producer next to the reader that wants
    it.
    """

    packet = getattr(tp_ops, "_assert_tensor_metadata", None)
    if packet is None:
        return
    for node in list(graph.find_nodes(op="call_function", target=packet.default)):
        # The assertion produces no value, so nothing reads it; one that is
        # read is not a promise about a value and is left alone.
        if node.users:
            continue
        graph.erase_node(node)


def reorder_for_locality(graph: Graph) -> None:
    """Move each value's producer next to the read that consumes it.

    A value computed early and read late is carried through everything in
    between: a buffer held, written and read again.  Moving the computation to
    just before its reader keeps the result where the read finds it.

    A producer that draws random numbers stays where it is, because the order
    of the draws is part of what the program computes.  So does a producer on
    the other side of a write to an input: the write is the boundary between
    two orderings of the same values, and moving across it would change which
    values the write sees.
    """

    seen = OrderedSet[Node]()
    # Nothing moves across the write-back that functionalization leaves at the
    # end of a graph, so the walk backwards stops at the first such write.
    first_write = next(
        iter(graph.find_nodes(op="call_function", target=tp_ops.copy_.default)),
        None,
    )
    past_writes = first_write is None

    def move_if_settled(producer: Node, reader: Node) -> None:
        if (
            producer.op == "call_function"
            # A tuple unpacking names positions in a result rather than doing
            # work of its own; it stays beside the call that produced it.
            and producer.target is not operator.getitem
            and all(user in seen for user in producer.users)
            and get_mutation_region_id(graph, reader)
            == get_mutation_region_id(graph, producer)
            and "nondeterministic_seeded" not in getattr(producer.target, "tags", ())
        ):
            reader.prepend(producer)

    for node in reversed(graph.nodes):
        seen.add(node)
        if not past_writes:
            past_writes = node is first_write
            continue
        map_arg((node.args, node.kwargs), lambda producer: move_if_settled(producer, node))


def decompose_auto_functionalized(graph: Graph) -> bool:
    """Each functional stand-in for a mutating operator, as what it does.

    ``auto_functionalized(op, **kwargs)`` stands for ``op`` run on private
    copies of the arguments it writes, returning ``op``'s own fresh outputs
    followed by the copies.  Written out that way -- one copy per written
    argument, then ``op`` writing into the copies -- the copies are values this
    region made, so writing into them is a mutation of the region's own memory
    that the lowering orders like any other.  Left whole, the stand-in would be
    a call returning a tuple that nothing here knows how to take apart.

    Returns whether anything was rewritten.
    """

    from tensorplay._higher_order_ops.auto_functionalize import (
        auto_functionalized,
        get_mutable_args,
        returns_without_aliases,
    )

    clone = tp_ops.clone.default
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target is not auto_functionalized:
            continue
        changed = True
        op = node.args[0]
        schema = op._schema
        kwargs = dict(node.kwargs)
        names, _ = get_mutable_args(op)
        kept = returns_without_aliases(op)
        value = node.meta.get("val")
        values = list(value) if isinstance(value, (tuple, list)) else [value]
        new_value_meta = values[len(kept):]

        def copy_of(arg: Any, meta: Any) -> Node:
            made = graph.call_function(clone, (arg,))
            made.meta["val"] = meta if meta is not None else arg.meta.get("val")
            return made

        with graph.inserting_before(node):
            copies: list[Any] = []
            for position, name in enumerate(names):
                arg = kwargs.get(name)
                meta = new_value_meta[position] if position < len(new_value_meta) else None
                if arg is None:
                    copies.append(None)
                    continue
                if isinstance(arg, (list, tuple)):
                    metas = list(meta) if isinstance(meta, (list, tuple)) else [None] * len(arg)
                    made = [copy_of(a, m) for a, m in zip(arg, metas)]
                else:
                    made = copy_of(arg, meta)
                kwargs[name] = made
                copies.append(made)
            positional = tuple(kwargs[a.name] for a in schema.arguments if not a.kwarg_only)
            keywords = {a.name: kwargs[a.name] for a in schema.arguments if a.kwarg_only}
            call = graph.call_function(op, positional, keywords)

            # What the call returns: its fresh outputs as recorded, and for a
            # return that is one of the written arguments, that argument's copy.
            written = {}
            for name, made in zip(names, copies):
                for argument in schema.mutated_arguments():
                    if argument.name == name:
                        for alias in argument.alias_info.before_set:
                            written[alias] = made
            returns = []
            for index, ret in enumerate(schema.returns):
                if index in kept:
                    returns.append(values[kept.index(index)])
                    continue
                alias = next(iter(ret.alias_info.before_set & set(written)), None)
                target = written.get(alias)
                returns.append(target.meta.get("val") if isinstance(target, Node) else None)
            call.meta["val"] = returns[0] if len(schema.returns) == 1 else tuple(returns)

        for user in list(node.users):
            if user.target is not operator.getitem or user.args[0] is not node:
                raise RuntimeError(f"unexpected reader of {node}: {user}")
            index = user.args[1]
            if index < len(kept):
                if len(schema.returns) == 1:
                    replacement = call
                else:
                    with graph.inserting_after(call):
                        replacement = graph.call_function(
                            operator.getitem, (call, kept[index])
                        )
                    replacement.meta["val"] = values[index]
            else:
                replacement = copies[index - len(kept)]
            if isinstance(replacement, list):
                # A written list of tensors: each element is read on its own.
                for reader in list(user.users):
                    if reader.target is not operator.getitem:
                        raise RuntimeError(f"unexpected reader of {user}: {reader}")
                    reader.replace_all_uses_with(replacement[reader.args[1]])
                    graph.erase_node(reader)
            else:
                user.replace_all_uses_with(replacement)
            graph.erase_node(user)
        graph.erase_node(node)
    return changed


def post_grad_passes(gm, is_inference: bool, input_device=None) -> None:
    """Rewrite a graph whose operations are settled and functional.

    Run once on the forward graph and once on the backward graph.  Every step
    preserves what the program computes; only the way it is written changes,
    so the caller is handed the same results through fewer and cheaper
    operations.

    The rewritten graph is left ready to be read; regenerating the module's
    own code is the business of whoever runs it, and the artifact this region
    is lowered into reads the graph rather than calling it.
    """

    from ..custom_graph_pass import get_custom_graph_passes
    from .group_batch_fusion import group_batch_fusion_passes
    from .joint_graph import lazy_init

    # Registration is shared with the joint-graph rewrites: one table per
    # family, filled once however either side reaches it first.
    lazy_init(input_device)
    # The values the patterns match against are stand-ins on the meta device;
    # decisions that depend on where kernels would run read this instead.
    global _region_input_device
    if input_device is not None:
        _region_input_device = str(getattr(input_device, "type", input_device))
    elif _region_input_device is None:
        _region_input_device = "cuda" if tp.cuda.is_available() else "cpu"
    graph = gm.graph

    if config.dce:
        # What nothing reads costs a kernel and is not part of the result.
        graph.eliminate_dead_code()

    if config.reorder_for_locality and (
        is_inference or config.reorder_for_locality_in_training
    ):
        reorder_for_locality(graph)

    for custom_pass in get_custom_graph_passes(config.post_grad_custom_pre_pass):
        custom_pass(graph)

    # A span the profiler was told about marks work that this region no longer
    # keeps apart, so it is dropped before anything looks for a fusion.
    remove_profiler_ops(graph)

    if config.pattern_matcher:
        # Producers of the same kind next to each other are grouped so that
        # the batch they form is one call rather than several.
        group_batch_fusion_passes(graph, pre_grad=False)
        # Operations whose result is their input, or which cancel out, cost a
        # kernel each until they are gone.
        remove_noop_ops(graph)
        # A promise about a value says nothing about the computation, so it
        # must not stand between a producer and the rewrites that want to
        # move it.
        remove_assert_ops(graph)
        # The product's add is pointwise; the fusion wants to see it as one.
        for patterns in pass_patterns:
            patterns.apply(graph)
        # One name for one shape change: readers behind the fusions and the
        # serialized form both want reshape, not view.
        view_to_reshape(graph)
        # The rewrites a program asks for by name are registered on first use,
        # so a program that asks for none pays nothing for them.
        requested = config.post_grad_fusion_options
        if requested:
            from ..utils import OPTIMUS_EXCLUDE_POST_GRAD
            from .group_batch_fusion import POST_GRAD_FUSIONS
            from .split_cat import POST_GRAD_PATTERNS

            for pass_name in requested:
                # The batch fusions are not pattern tables and are applied above.
                if (
                    pass_name in POST_GRAD_FUSIONS
                    or pass_name in OPTIMUS_EXCLUDE_POST_GRAD
                ):
                    continue
                patterns = POST_GRAD_PATTERNS.get(pass_name)
                if patterns is None:
                    raise RuntimeError(
                        f"unknown post-grad pattern pass {pass_name!r}"
                    )
                patterns.apply(graph)

    for custom_pass in get_custom_graph_passes(config.post_grad_custom_post_pass):
        custom_pass(graph)

