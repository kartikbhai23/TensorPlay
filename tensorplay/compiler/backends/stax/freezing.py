"""Turn an inference region's read-only parameters into constants.

A weight that is only read is the same value on every call.  Leaving it as an
argument makes the region ask for it each time and hides everything that could
be computed from it alone -- a normalization folded into the convolution in
front of it, a transposed copy of a packed weight -- behind a load.  This
module finds the arguments that are such weights, writes their values into the
region as constants, and lets the ordinary folding and matching passes work on
them.

The region keeps every one of its arguments.  An input whose value is now a
constant is simply not read any more, so a caller's calling convention does
not change and a region that cannot be lowered can still fall back to its
unfrozen graph without rebuilding anything.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import tensorplay as tp
from tensorplay.graph import Graph, Node, map_arg

from . import config
from .constant_folding import constant_fold
from .freezing_utils import enter_freezing, record_has_frozen_params
from .graph_passes.freezing_patterns import freezing_passes
from .graph_passes.post_grad import view_to_reshape
from .pattern_matcher import (
    compute_mutation_region_ids,
    same_mutation_regions,
)


log = logging.getLogger(__name__)

#: The number of leading arguments of a region that are module state rather
#: than user inputs.  A caller that knows this sets it on the captured graph;
#: a graph without it has no state it can identify, and freezing does nothing.
STATE_INPUT_COUNT = "_tensorplay_state_input_count"


def state_input_count(gm: Any) -> int:
    """How many leading arguments of ``gm`` are module state."""

    count = getattr(gm, STATE_INPUT_COUNT, 0)
    if not isinstance(count, int) or count < 0:
        return 0
    return count


# Operations that can make a new value share storage with an input.  The list
# is deliberately wider than the set that always aliases: treating a copying
# operation as aliasing only means a weight stays an argument.
_ALIASING_TARGETS = frozenset(
    {
        "alias",
        "as_strided",
        "chunk",
        "conj",
        "contiguous",
        "detach",
        "diagonal",
        "expand",
        "expand_as",
        "flatten",
        "imag",
        "movedim",
        "narrow",
        "permute",
        "real",
        "reshape",
        "reshape_as",
        "select",
        "slice",
        "split",
        "split_with_sizes",
        "squeeze",
        "swapaxes",
        "swapdims",
        "t",
        "to",
        "transpose",
        "unbind",
        "unflatten",
        "unsqueeze",
        "view",
        "view_as",
    }
)

# A Python method spelling of the same operations.
_ALIASING_METHODS = _ALIASING_TARGETS

_INPLACE_NAME = re.compile(r"^(?P<name>.*?)_?(?:\.[^.]+)?$")


def _target_name(target: Any) -> str:
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return getattr(schema, "name", "") or ""
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name.split(".")[0]
    if isinstance(target, str):
        return target
    return ""


def _is_inplace(target: Any) -> bool:
    name = _target_name(target)
    if not name:
        return False
    # ``copy_`` updates in place although its operator name does not end in an
    # underscore in every spelling.
    return name.endswith("_") or name in {"set_", "resize_", "copy_"}


def _is_aliasing(node: Node) -> bool:
    if node.op == "call_method":
        return isinstance(node.target, str) and node.target in _ALIASING_METHODS
    if node.op != "call_function":
        return False
    return _target_name(node.target) in _ALIASING_TARGETS


def _flatten_arguments(value: Any) -> list[Any]:
    if isinstance(value, Node):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_arguments(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten_arguments(item))
        return out
    return []


def _aliased_nodes(graph: Graph, roots: set[Node]) -> set[Node]:
    """Every node that can stand for the same storage as one of ``roots``."""

    aliases = set(roots)
    for node in graph.nodes:
        if not _is_aliasing(node):
            continue
        inputs = _flatten_arguments(node.args) + _flatten_arguments(node.kwargs)
        if any(inp in aliases for inp in inputs):
            aliases.add(node)
    return aliases


def _must_remain_an_argument(graph: Graph, placeholder: Node) -> bool:
    """Whether a state argument may be written or handed back to the caller.

    A parameter that an operation updates in place is state the caller can
    observe, and one whose storage reaches the output is a value the caller
    may write through.  Neither is a constant, whatever its value is now.
    """

    aliases = _aliased_nodes(graph, {placeholder})
    output = graph.output_node
    if any(node in aliases for node in _flatten_arguments(output.args)):
        return True
    for node in graph.nodes:
        if node.op == "output" or not _is_inplace(node.target):
            continue
        inputs = _flatten_arguments(node.args) + _flatten_arguments(node.kwargs)
        if any(inp in aliases for inp in inputs):
            return True
    return False


def _constant_name(gm: Any) -> str:
    count = getattr(gm, "_frozen_param_count", 0)
    while True:
        name = f"_frozen_param{count}"
        count += 1
        if not hasattr(gm, name) and name not in getattr(gm, "_buffers", {}):
            gm._frozen_param_count = count
            return name


def _replace_uses_with_constant(gm: Any, node: Node, value: Any) -> None:
    """Make every use of ``node`` read ``value``; leave the argument in place."""

    name = _constant_name(gm)
    with gm.graph.inserting_before(node):
        constant = gm.graph.create_node("get_attr", name, (), {})
    constant.meta.update(node.meta)
    node.replace_all_uses_with(constant)
    gm.register_buffer(name, value)
    setattr(gm, name, value)


def replace_state_with_constants(
    gm: Any,
    example_inputs: list[Any],
    count: int,
) -> int:
    """Write the leading state arguments of ``gm`` into it as constants.

    Returns how many arguments became constants.  Every argument stays in the
    graph, so the caller's convention is unchanged; the returned count only
    says how much of the state the region no longer reads.
    """

    placeholders = gm.graph.placeholders
    frozen = 0
    for index, node in enumerate(placeholders[:count]):
        if index >= len(example_inputs):
            break
        value = example_inputs[index]
        if not isinstance(value, tp.Tensor):
            continue
        if _must_remain_an_argument(gm.graph, node):
            continue
        _replace_uses_with_constant(gm, node, value)
        frozen += 1
    if frozen:
        gm.recompile()
    return frozen


def _hash_key(value: Any, env: dict[Node, Node]) -> Any:
    if isinstance(value, Node):
        return env.get(value, value)
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_hash_key(item, env) for item in value))
    if isinstance(value, dict):
        return (
            "mapping",
            tuple(sorted((repr(key), _hash_key(item, env)) for key, item in value.items())),
        )
    try:
        hash(value)
    except TypeError:
        return ("object", id(value))
    return (type(value).__name__, value)


def fx_graph_cse(graph: Graph) -> Graph:
    """Copy ``graph`` with repeated pure expressions reduced to one node.

    Two identical calls are the same value unless an operation between them
    can change what the call reads.  The mutation regions recorded on the
    nodes answer that question; nodes in different regions are never merged.
    Placeholders, attribute reads, outputs, impure calls and random draws are
    copied as they stand.
    """

    compute_mutation_region_ids(graph)
    new_graph = Graph()
    env: dict[Node, Node] = {}
    seen: dict[Any, Node] = {}

    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output") or node.is_impure():
            copied = new_graph.node_copy(node, lambda value: env[value])
            env[node] = copied
            continue

        key = (
            node.target,
            _hash_key(node.args, env),
            _hash_key(node.kwargs, env),
        )
        previous = seen.get(key)
        if previous is not None and same_mutation_regions(node, previous):
            env[node] = previous
            continue
        copied = new_graph.node_copy(node, lambda value: env[value])
        env[node] = copied
        seen[key] = copied

    return new_graph


def freeze(
    gm: Any,
    example_inputs: list[Any],
    count: int | None = None,
) -> Any:
    """Bake the read-only leading state of ``gm`` into the graph.

    ``count`` is the number of leading arguments that are module state; by
    default the count recorded on the graph is used.  Returns the same graph
    module, changed in place.
    """

    if count is None:
        count = state_input_count(gm)
    if not count:
        return gm
    if config.freezing_discard_parameters:
        raise RuntimeError(
            "freezing_discard_parameters is not supported: this build has no "
            "way to discard an eager module's parameters while guaranteeing "
            "that a later read of one is reported rather than silently "
            "answered from a stale value"
        )
    if config.cpp.weight_prepack:
        raise RuntimeError(
            "cpp.weight_prepack is not supported: the packed weight forms "
            "this build declares have no CPU kernel, so a packed weight "
            "would reach a call that cannot run"
        )
    with enter_freezing():
        return _freeze(gm, example_inputs, count)


def _freeze(gm: Any, example_inputs: list[Any], count: int) -> Any:
    # A weight laid out channels last cannot always be viewed; retargeting
    # views to reshapes first keeps the folding below from seeing a view whose
    # source changed layout.
    view_to_reshape(gm.graph)

    frozen = replace_state_with_constants(gm, example_inputs, count)
    if not frozen:
        return gm

    gm.graph = fx_graph_cse(gm.graph)
    gm.recompile()

    freezing_passes(gm, example_inputs)
    constant_fold(gm)

    record_has_frozen_params(gm)
    log.debug("frozen %d of %d leading arguments", frozen, count)
    return gm
