"""Pattern rewriting, foreach grouping and split-scan tuning, as units.

The pattern matcher decides which rules run on a graph and how a handler is
bound to what it matched; the grouping decides which collectives may be asked
for together; the split-scan configs decide the smallest reduction block a
kernel may use.  Each has a contract that a full compile only exercises for
some shapes, so the contract is checked directly.
"""

import operator
import types

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.graph_passes import bucketing
from tensorplay.compiler.backends.stax.heuristics.registry import (
    get_codegen_heuristic,
)
from tensorplay.compiler.backends.stax.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    PatternMatcherPass,
    register_graph_pattern,
)
from tensorplay.graph import Tracer


def _trace(fn):
    return Tracer().trace(fn, sample_inputs={"x": tp.randn(3)})


def _targets(gm):
    return [
        node.target
        for node in gm.graph.nodes
        if node.op == "call_function"
    ]


# ---------------------------------------------------------------------------
# pattern matcher
# ---------------------------------------------------------------------------


def test_a_graph_pattern_rewrites_the_matched_node():
    gm = _trace(lambda x: (x + 1.0) * 2.0)
    pass_dict = PatternMatcherPass(pass_name="test_rewrite")

    @register_graph_pattern(CallFunction(operator.mul, Arg(), Arg()), pass_dict=pass_dict)
    def handler(match, left, right):
        node = match.nodes[-1]
        with match.graph.inserting_before(node):
            replacement = match.graph.call_function(
                operator.truediv, (left, right)
            )
        node.replace_all_uses_with(replacement)
        match.erase_nodes()

    assert pass_dict.apply(gm) == 1
    assert operator.truediv in _targets(gm)
    assert operator.mul not in _targets(gm)


def test_an_extra_check_can_decline_a_match():
    gm = _trace(lambda x: (x + 1.0) * 2.0)
    pass_dict = PatternMatcherPass(pass_name="test_declined")

    @register_graph_pattern(
        CallFunction(operator.mul, Arg(), Arg()),
        extra_check=lambda match: False,
        pass_dict=pass_dict,
    )
    def handler(match, left, right):  # pragma: no cover - must not run
        raise AssertionError("a declined match must not reach the handler")

    assert pass_dict.apply(gm) == 0
    assert operator.mul in _targets(gm)


def test_a_keyword_pattern_binds_by_argument_name():
    gm = _trace(lambda x: (x + 1.0) * 2.0)
    pass_dict = PatternMatcherPass(pass_name="test_keyword")
    seen = []

    @register_graph_pattern(
        CallFunction(operator.add, Arg(), KeywordArg("other")),
        pass_dict=pass_dict,
    )
    def handler(match, *args, **kwargs):
        seen.append((args, kwargs))

    assert pass_dict.apply(gm) == 1
    (args, kwargs) = seen[0]
    # The traced node holds its arguments positionally; the name still binds
    # to the position the operation's signature gives it.
    assert len(args) == 1
    assert kwargs == {"other": 1.0}


def test_a_pattern_that_matches_nothing_leaves_the_graph_alone():
    gm = _trace(lambda x: (x + 1.0) * 2.0)
    before = _targets(gm)
    pass_dict = PatternMatcherPass(pass_name="test_unmatched")

    @register_graph_pattern(
        CallFunction(operator.sub, Arg(), Arg()), pass_dict=pass_dict
    )
    def handler(match, left, right):  # pragma: no cover - must not run
        raise AssertionError("nothing matched, so nothing runs")

    assert pass_dict.apply(gm) == 0
    assert _targets(gm) == before


def test_an_empty_pass_and_a_bare_graph_are_accepted():
    gm = _trace(lambda x: x + 1.0)
    assert PatternMatcherPass(pass_name="test_empty").apply(gm) == 0
    # A Graph is accepted in place of its owner.
    assert PatternMatcherPass(pass_name="test_graph").apply(gm.graph) == 0


def test_a_non_graph_input_is_rejected():
    pass_dict = PatternMatcherPass(pass_name="test_bad")

    @register_graph_pattern(
        CallFunction(operator.add, Arg(), Arg()), pass_dict=pass_dict
    )
    def handler(match, left, right):  # pragma: no cover - must not run
        raise AssertionError("nothing matched, so nothing runs")

    with pytest.raises(RuntimeError, match="GraphModule or a Graph"):
        pass_dict.apply(object())


# ---------------------------------------------------------------------------
# foreach grouping
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Only what grouping reads: dtype and shape."""

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


def test_foreach_groups_are_split_by_dtype_and_shape():
    inputs = [
        _FakeTensor(tp.float32, (2, 3)),
        _FakeTensor(tp.float32, (2, 3)),
        _FakeTensor(tp.float64, (2, 3)),
    ]
    out_dtypes = [tp.float32, tp.float32, tp.float64]

    # One group per (source dtype, destination dtype, shape); -1 separates.
    assert bucketing._compute_foreach_groups(inputs, out_dtypes) == [0, 1, -1, 2]


def test_foreach_grouping_returns_nothing_for_a_single_group():
    inputs = [_FakeTensor(tp.float32, (4,)), _FakeTensor(tp.float32, (4,))]
    assert bucketing._compute_foreach_groups(
        inputs, [tp.float32, tp.float32]
    ) is None


def test_foreach_grouping_needs_hinted_extents():
    class _Symbolic:
        pass

    inputs = [
        _FakeTensor(tp.float32, (2,)),
        _FakeTensor(tp.float32, (_Symbolic(),)),
    ]
    with pytest.raises(AssertionError, match="int or SymInt"):
        bucketing._compute_foreach_groups(inputs, [tp.float32, tp.float32])
    assert bucketing._hint_int_or_raise(4, context="test") == 4
    with pytest.raises(AssertionError, match="int or SymInt"):
        bucketing._hint_int_or_raise(4.0, context="test")


# ---------------------------------------------------------------------------
# split scan
# ---------------------------------------------------------------------------


def _config(kwargs):
    return types.SimpleNamespace(kwargs=dict(kwargs), num_warps=4, num_stages=2)


def test_split_scan_raises_small_reduction_blocks_to_the_minimum(monkeypatch):
    heuristic = get_codegen_heuristic("reduction", "cuda")
    generated = [
        _config({"XBLOCK": 128, "R0_BLOCK": 64}),
        _config({"XBLOCK": 64, "R0_BLOCK": 1024}),
        _config({"XBLOCK": 32, "RSPLIT_SIZE": 1}),
    ]
    monkeypatch.setattr(heuristic, "get_configs", lambda **kwargs: generated)

    configs = heuristic.get_split_scan_configs(
        size_hints={"x": 128, "r0_": 4096}, tp_meta={}, triton_meta={}
    )
    # Every key that names a reduction block is floored at 256 by default,
    # while the program-extent key keeps its size.
    assert configs[0].kwargs["R0_BLOCK"] == 256
    assert configs[1].kwargs["R0_BLOCK"] == 1024
    assert configs[0].kwargs["XBLOCK"] == 128
    assert configs[2].kwargs == {"XBLOCK": 32, "RSPLIT_SIZE": 256}


def test_split_scan_uses_the_requested_minimum(monkeypatch):
    heuristic = get_codegen_heuristic("reduction", "cuda")
    monkeypatch.setattr(
        heuristic,
        "get_configs",
        lambda **kwargs: [_config({"XBLOCK": 16, "R0_BLOCK": 32})],
    )

    configs = heuristic.get_split_scan_configs(
        size_hints={"x": 16, "r0_": 32},
        tp_meta={"min_split_scan_rblock": 512},
        triton_meta={},
    )
    assert configs[0].kwargs["R0_BLOCK"] == 512
    assert configs[0].kwargs["XBLOCK"] == 16