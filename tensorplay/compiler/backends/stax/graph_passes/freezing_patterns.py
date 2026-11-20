"""Graph patterns that become available once a region's state is constant.

The rewrites here join matrix products that share an input and read constant
weights, and drop conversions that no longer change a value.  Both need the
weights to be constants before they can fire, so they run as part of freezing
rather than as part of the ordinary post-grad pass list.

Weight prepacking is deliberately not part of this list: the packed forms this
build declares have no CPU kernel, so a rewrite into one would produce a call
that cannot run.
"""

from __future__ import annotations

import functools

import tensorplay as tp

from .. import config
from ..op_lowerings import select_decomp_table
from ..pattern_matcher import (
    _return_true,
    CallFunction,
    fwd_only,
    Ignored,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)
from ..utils import GPU_TYPES


tp_ops = tp.ops.tp
prims = tp.ops.prims


def _pattern_decompositions() -> dict:
    """The decomposition table, keeping a matrix product spelled as written.

    The patterns below are written as ``inp @ weight``.  The trace that turns
    a pattern into something to match would otherwise break that product into
    a two-dimensional product, while the region being matched keeps the call
    the program wrote.  Both spellings lower to the same kernel, so the
    pattern is traced with the spelling the region uses.
    """

    table = select_decomp_table()
    table.pop(tp_ops.matmul.default, None)
    table.pop(tp_ops.matmul.out, None)
    return table

# Three rounds, applied in order.  A rewrite that creates a value another
# rewrite reads belongs in a later round; keeping one round per stage makes
# the order the rewrites see each other explicit.
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def freezing_passes(gm: tp.graph.GraphModule, aot_example_inputs) -> None:
    """Fold and rewrite a graph whose parameters are now constants."""

    lazy_init()

    constant_fold(gm)
    for pattern in pass_patterns:
        pattern.apply(gm.graph)

    # A rewrite can leave the graph in an order where a value is used before
    # the node that produces it; the lowering requires a topological order.
    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


def lazy_init(input_device: tp.device | None = None) -> None:
    addmm_patterns_init()


def register_freezing_graph_pattern(
    pattern, extra_check=_return_true, pass_number=0
):
    while pass_number > len(pass_patterns) - 1:
        pass_patterns.append(PatternMatcherPass())
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=pass_patterns[pass_number],
    )


def _example_matmul_device() -> str:
    for gpu in GPU_TYPES:
        module = getattr(tp, gpu, None)
        if module is not None and module.is_available():
            return gpu
    return "cpu"


@functools.cache
def addmm_patterns_init() -> None:
    """Join matrix products that share an input and constant weights.

    The joined product is one wider matrix product against the concatenated
    weights; the split back into the original results is a view, so the
    caller sees the same values and shapes.  Only the CPU spelling is gated:
    on an accelerator the wider product is a choice the pattern's own shape
    check makes worth taking, while on the CPU it is off unless asked for.
    """

    device = _example_matmul_device()
    val = functools.partial(tp.empty, (10, 10), device=device, requires_grad=False)
    scale = functools.partial(tp.empty, (10,), device=device, requires_grad=False)

    def check_int8_woq_concat_linear_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if not is_cpu or not config.cpp.enable_concat_linear:
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        if not all(
            match.kwargs[wgt].target is prims.convert_element_type
            for wgt in weight_inputs
        ):
            return False

        if not all(
            next(iter(match.kwargs[wgt]._input_nodes.keys())).meta["val"].dtype
            is tp.int8
            for wgt in weight_inputs
        ):
            return False

        return all(
            match.kwargs[wgt].meta["val"].dtype is tp.bfloat16
            for wgt in weight_inputs
        )

    def check_concat_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if is_cpu and not config.cpp.enable_concat_linear:
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]
        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")
            equal_shape_inputs.append(bias_inputs)

        for equal_shape_group in equal_shape_inputs:
            inps = [match.kwargs[name] for name in equal_shape_group]
            if not all(
                inp.op == "get_attr"
                and inp.meta["val"].shape[:-1] == inps[0].meta["val"].shape[:-1]
                for inp in inps
            ):
                return False
        return True

    def int8_woq_fusion_pattern(inp, w1, w2, w3, s1, s2, s3):
        return ((inp @ w1) * s1, (inp @ w2) * s2, (inp @ w3) * s3)

    def int8_woq_fusion_replacement(inp, w1, w2, w3, s1, s2, s3):
        cat_w = tp.cat((w1, w2, w3), dim=1)
        cat_s = tp.cat((s1, s2, s3), dim=0)
        mm = (inp @ cat_w).mul(cat_s)
        n1, n2 = w1.size(1), w2.size(1)
        return mm.tensor_split([n1, n1 + n2], dim=-1)

    register_replacement(
        int8_woq_fusion_pattern,
        int8_woq_fusion_replacement,
        [val(), val(), val(), val(), scale(), scale(), scale()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_int8_woq_concat_linear_weights,
        exclusive_arg_names=("w1", "w2", "w3", "s1", "s2", "s3"),
        get_decomp_fn=_pattern_decompositions,
    )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        weights = (w1, w2, w3)
        cat_t = tp.cat(weights, dim=1)
        mm = inp @ cat_t
        return mm.split([w.size(1) for w in weights], dim=1)

    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
        get_decomp_fn=_pattern_decompositions,
    )

    def matmul_fuse_pattern_two(inp, w1, w2):
        return (inp @ w1, inp @ w2)

    def matmul_replacement_two(inp, w1, w2):
        weights = (w1, w2)
        cat_t = tp.cat(weights, dim=1)
        mm = inp @ cat_t
        return mm.split([w.size(1) for w in weights], dim=1)

    register_replacement(
        matmul_fuse_pattern_two,
        matmul_replacement_two,
        [val(), val(), val()],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2"),
        get_decomp_fn=_pattern_decompositions,
    )

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            tp_ops.addmm(b1, inp, w1),
            tp_ops.addmm(b2, inp, w2),
            tp_ops.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        weights = (w1, w2, w3)
        cat_w = tp.cat(weights, dim=1)
        cat_b = tp.cat((b1, b2, b3))
        return tp_ops.addmm(cat_b, inp, cat_w).split(
            [w.size(1) for w in weights], dim=1
        )

    register_replacement(
        addmm_fuse_pattern_second,
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        fwd_only,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
        get_decomp_fn=_pattern_decompositions,
    )


def same_dtype(match: Match) -> bool:
    return match.output_node().args[0].meta["val"].dtype == match.kwargs["dtype"]


@register_graph_pattern(
    CallFunction(
        prims.convert_element_type,
        Ignored(),
        KeywordArg("dtype"),
    ),
    pass_dict=pass_patterns[0],
    extra_check=same_dtype,
)
def unnecessary_dtype_convert(match: Match, **kwargs) -> None:
    """Drop a conversion whose source already has the requested dtype.

    A folded normalization can leave such a conversion behind: the arithmetic
    that made it necessary was computed in a different dtype and is gone by
    the time this reads the graph.
    """

    graph = match.graph
    node = match.output_node()
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)


# Imported late: the folding module imports the graph passes, and the constant
# folding this file calls must be the same object the freezing pipeline sees.
from ..constant_folding import constant_fold  # noqa: E402
