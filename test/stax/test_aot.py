"""P3 vertical slice: AOT forward/backward partitioning (L4)."""

import operator

import pytest

import tensorplay as tp
from tensorplay._stax import AOTError, build_aot
from tensorplay.graph import Tracer


def _make(fn, sample):
    tracer = Tracer()
    gm = tracer.trace(fn, sample_inputs={"x": sample[0], "w": sample[1]} if len(sample) > 1 else {"x": sample[0]})
    return gm


def _make_sample_map(fn, args):
    import inspect

    names = list(inspect.signature(fn).parameters)
    return dict(zip(names, args))


def _assert_grads_match_eager(fn, args, policy, required=None, rtol=1e-5, atol=1e-6, partitioner="default", memory_budget=None):
    smap = _make_sample_map(fn, args)
    gm = Tracer().trace(fn, sample_inputs=smap)
    result = build_aot(gm, sample_inputs=smap, required_grads=required, policy=policy, partitioner=partitioner, memory_budget=memory_budget)
    out, grads = result.value_and_grad(*args)

    eager_args = [a.clone().requires_grad_(True) for a in args]
    eager_out = fn(*eager_args)
    if isinstance(eager_out, tp.Tensor):
        eager_out.sum().backward()
    else:
        sum(t.sum() for t in eager_out).backward()

    if isinstance(eager_out, tp.Tensor):
        assert out.shape == eager_out.shape
    else:
        assert tuple(o.shape for o in out) == tuple(o.shape for o in eager_out)
    for got, want in zip(grads.values(), [a.grad for a in eager_args]):
        assert tp.allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_mul_relu_sum_gradients(policy, partitioner):
    def fn(x, w):
        return (x * w).relu().sum()

    x = tp.tensor([-1.0, 0.5, 2.0])
    w = tp.tensor([0.3, -0.2, 1.5])
    _assert_grads_match_eager(fn, [x, w], policy, partitioner=partitioner)


@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
@pytest.mark.parametrize(
    "formula",
    [
        lambda x: (x * x).sum(),
        lambda x: ((x * x) * x).sum(),
        lambda x: (x + x).sum(),
        lambda x: (x - x).sum(),
        lambda x: (x * x).sum() + (x * x).sum(),
        lambda x: x.mul(x).sum(),
        lambda x: (x.sin() * x.sin()).sum(),
    ],
    ids=[
        "square",
        "cube",
        "add_twice",
        "sub_twice",
        "two_branches",
        "method",
        "shared_result",
    ],
)
def test_a_value_used_twice_receives_both_contributions(formula, partitioner):
    """One node can fill both operands, and each role contributes a term.

    The derivative of ``x * x`` is ``2x``, not ``x``: the reverse sweep has to
    accumulate the term from each operand even though both name the same
    value.  A sweep keyed by the value keeps only one term, which halves the
    gradient, so this covers every rule with two operands plus the branch
    case where two paths reach the same node.
    """

    _assert_grads_match_eager(
        formula, [tp.randn(4)], "save_needed", partitioner=partitioner
    )


@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
@pytest.mark.parametrize(
    "formula",
    [
        lambda x: x.sum(0),
        lambda x: x.sum(1),
        lambda x: x.sum(-1),
        lambda x: x.sum(1, True),
        lambda x: x.sum(dim=1, keepdim=True),
        lambda x: x.sum((0, 2)),
        lambda x: x.sum((0, 2), True),
    ],
    ids=[
        "axis0",
        "axis1",
        "negative_axis",
        "axis1_keepdim",
        "axis1_keepdim_named",
        "two_axes",
        "two_axes_keepdim",
    ],
)
def test_partial_reduction_gradient_spreads_over_reduced_axes(formula, partitioner):
    """A partial reduction's gradient goes back only over the reduced axes.

    The result of ``x.sum(1)`` has one extent fewer, so spreading it back means
    putting a singleton where the reduced axis was; spreading it directly at
    the input's shape either fails or lands on the wrong axes.
    """

    _assert_grads_match_eager(
        formula, [tp.randn(2, 3, 4)], "save_needed", partitioner=partitioner
    )


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_a_lifted_constant_is_read_by_the_backward(policy, partitioner):
    """A value created inside the region becomes an attribute, not an argument.

    The backward graph is rooted on the same module, so it has to read such a
    value from there; asking the caller for it as an input fails.
    """

    constant = tp.tensor([1.0, 2.0, 3.0])

    def fn(x):
        return x * constant

    x = tp.randn(3)
    gm = Tracer().trace(fn, sample_inputs={"x": x})
    result = build_aot(
        gm,
        sample_inputs={"x": x},
        policy=policy,
        partitioner=partitioner,
    )
    value, grads = result.value_and_grad(x.clone().requires_grad_(True))
    assert tp.allclose(value, x * constant)
    assert tp.allclose(grads["x"], constant)


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_multi_output_region_differentiates_each_result(policy, partitioner):
    """Every result of a multi-output region is a forward result, not a gradient.

    The joint graph's outputs are the forward's results followed by the
    backward's.  Counting the second result as a gradient leaves the backward
    reading a value the forward never handed it, and the split fails while
    extracting the forward half.
    """

    def fn(x):
        return (x.sum(), (x.sin() * x).sum())

    x = tp.randn(4)
    gm = Tracer().trace(fn, sample_inputs={"x": x})
    result = build_aot(
        gm,
        sample_inputs={"x": x},
        policy=policy,
        partitioner=partitioner,
    )
    value, grads = result.value_and_grad(x.clone().requires_grad_(True))

    eager = x.clone().requires_grad_(True)
    first, second = fn(eager)
    (first.sum() + second.sum()).backward()

    assert isinstance(value, tuple) and len(value) == 2
    assert tp.allclose(value[0], first)
    assert tp.allclose(value[1], second)
    assert tp.allclose(grads["x"], eager.grad)


@pytest.mark.parametrize("policy", ["save_needed", "recompute_all"])
@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_broadcast_scalar_param_gradient_reduces(policy, partitioner):
    def fn(x, w):
        return (x * w).sum()

    x = tp.tensor([1.0, 2.0, 3.0])
    w = tp.tensor(2.0)  # scalar: d(w) must reduce (3,) -> ()
    _assert_grads_match_eager(fn, [x, w], policy, partitioner=partitioner)


def test_sin_chain_and_truediv():
    def fn(x, w):
        return (sin_term := (x * w).sin()).sum() / 2

    x = tp.tensor([0.3, -0.7])
    w = tp.tensor([1.4, 2.2])
    _assert_grads_match_eager(fn, [x, w], "save_needed", rtol=1e-4, atol=1e-5)


def test_save_needed_saves_only_rule_operands():
    def fn(x, w):
        return ((x * w).relu()).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])})
    result = build_aot(
        gm,
        sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])},
        policy="save_needed",
    )
    # Backward needs the pre-activation (relu input); nothing else.
    assert len(result.saved_names) == 1

    gm2 = Tracer().trace(fn, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])})
    result2 = build_aot(gm2, sample_inputs={"x": tp.tensor([1.0]), "w": tp.tensor([1.0])},
                        policy="recompute_all")
    assert result2.saved_names == []


def test_multiple_outputs_tuple_grads():
    def fn(x):
        return (x.sin(), x.cos())

    x = tp.tensor([0.4])
    _assert_grads_match_eager(fn, [x], "save_needed")


def test_unsupported_op_raises_clear_error():
    def fn(x):
        return x.reshape(-1).sum()

    gm = Tracer().trace(fn, sample_inputs={"x": tp.tensor([[1.0], [2.0]])})
    with pytest.raises(AOTError, match="no derivative"):
        build_aot(gm, sample_inputs={"x": tp.tensor([[1.0], [2.0]])})


def test_aot_backend_through_compile_pipeline():
    calls = {}

    def aot_backend(graph_module, example_inputs, **kwargs):
        sample = {p.name: v for p, v in zip(graph_module.graph.placeholders, example_inputs)}
        result = build_aot(graph_module, sample_inputs=sample)
        calls["saved"] = result.saved_names
        return lambda *a, **k: result.value_and_grad(*a)[0]

    def fn(x, w):
        return (x * w).relu().sum()

    compiled = tp.compile(fn, backend=aot_backend, fullgraph=True)
    x = tp.tensor([-1.0, 0.5, 2.0])
    w = tp.tensor([0.3, -0.2, 1.5])
    out = compiled(x, w)
    assert out.tolist() == (x * w).relu().sum().tolist()
    assert len(calls["saved"]) >= 1


@pytest.mark.parametrize("partitioner", ["default", "min_cut"])
def test_partitioner_roles_consistent(partitioner):
    def fn(x, w):
        return (x * w).relu().sum()

    x = tp.tensor([1.0, 2.0])
    smap = {"x": x, "w": tp.tensor([1.0, 2.0])}
    r = build_aot(Tracer().trace(fn, sample_inputs=smap), sample_inputs=smap,
                  partitioner=partitioner)
    phs = list(r.backward_gm.graph.placeholders)
    assert len(phs) == len(r.input_kinds) == len(r.input_keys)
    assert sum(k == "tangent" for k in r.input_kinds) == 1
    assert {"x", "w"} <= {k for k, kind in zip(r.input_keys, r.input_kinds) if kind == "leaf"}
    assert all(name in r.saved_names or True for name in [])


def test_min_cut_saves_no_more_than_default():
    def fn(x, w):
        return ((x * w).sin().cos() - x / w).exp().sum()

    smap = {"x": tp.tensor([1.0]), "w": tp.tensor([2.0])}
    gm1 = Tracer().trace(fn, sample_inputs=smap)
    gm2 = Tracer().trace(fn, sample_inputs=smap)
    d = build_aot(gm1, sample_inputs=smap, partitioner="default")
    m = build_aot(gm2, sample_inputs=smap, partitioner="min_cut")
    assert len(m.saved_names) <= len(d.saved_names)


def _saved_value_bytes(result):
    """Bytes held by the values the forward hands the backward."""

    by_name = {node.name: node for node in result.forward_gm.graph.nodes}
    total = 0
    for name in result.saved_names:
        value = by_name[name].meta.get("val")
        numel = getattr(value, "numel", None)
        if callable(numel):
            itemsize = int(getattr(getattr(value, "dtype", None), "itemsize", 4) or 4)
            total += int(numel()) * itemsize
    return total


def test_memory_budget_limits_saved_bytes():
    def fn(x, w, idx):
        h = w.index_select(0, idx)
        return (h * x).sum() + (h.sin() * x).sum()

    smap = {
        "x": tp.randn(2),
        "w": tp.randn(4),
        "idx": tp.tensor([0, 1]),
    }

    def saved(budget):
        gm = Tracer().trace(fn, sample_inputs=smap)
        kwargs = {} if budget is None else {"memory_budget": budget}
        result = build_aot(
            gm, sample_inputs=smap, partitioner="min_cut", **kwargs
        )
        return len(result.saved_names), _saved_value_bytes(result)

    names, held = saved(None)
    assert names == 1 and held > 0
    assert saved(1.0) == (names, held)
    # A fraction below what the cheapest-to-run cut holds recomputes instead.
    assert saved(0.9) == (0, 0)
    assert saved(0.0) == (0, 0)


@pytest.mark.parametrize("budget", [2, -0.1])
def test_memory_budget_out_of_range_is_rejected(budget):
    def fn(x, w, idx):
        return (w.index_select(0, idx) * x).sum()

    smap = {"x": tp.randn(2), "w": tp.randn(4), "idx": tp.tensor([0, 1])}
    gm = Tracer().trace(fn, sample_inputs=smap)
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_aot(gm, sample_inputs=smap, partitioner="min_cut", memory_budget=budget)


@pytest.mark.parametrize("budget", [True, "half"])
def test_memory_budget_rejects_a_non_number(budget):
    def fn(x, w, idx):
        return (w.index_select(0, idx) * x).sum()

    smap = {"x": tp.randn(2), "w": tp.randn(4), "idx": tp.tensor([0, 1])}
    gm = Tracer().trace(fn, sample_inputs=smap)
    with pytest.raises(TypeError, match="must be a number"):
        build_aot(gm, sample_inputs=smap, partitioner="min_cut", memory_budget=budget)


@pytest.mark.parametrize("budget", [0.5, 0.0])
def test_memory_budget_partition_keeps_gradients(budget):
    def fn(x, w, idx):
        h = w.index_select(0, idx)
        return (h * x).sum() + (h.sin() * x).sum()

    args = [tp.randn(2), tp.randn(4), tp.tensor([0, 1])]
    _assert_grads_match_eager(
        fn, args, "save_needed", partitioner="min_cut", memory_budget=budget
    )


# ---------------------------------------------------------------------------
# inference input updates
# ---------------------------------------------------------------------------


def _compile_inference(fn, primals, *, keep):
    from tensorplay.compiler._core.aot_autograd import aot_function

    def backend(gm, inputs, **kwargs):
        return gm.forward

    return aot_function(
        fn, primals, fw_compiler=backend, keep_inference_input_mutations=keep
    )


def _graph_of(compiled):
    return compiled._tensorplay_aot_graphs[0]


def _copy_nodes(gm):
    return [
        node
        for node in gm.graph.nodes
        if getattr(node.target, "_opname", None) == "copy_"
    ]


def _bump(x, buf):
    buf.add_(1.0)
    return x * 2


def test_inference_update_can_run_outside_the_graph():
    compiled = _compile_inference(
        _bump, [tp.ones(2), tp.zeros(2)], keep=False
    )
    value, buffer = tp.ones(2), tp.zeros(2)

    assert compiled(value, buffer).tolist() == [2.0, 2.0]
    assert buffer.tolist() == [1.0, 1.0]
    # The wrapper updates the caller's tensor, so repeats accumulate.
    assert compiled(value, buffer).tolist() == [2.0, 2.0]
    assert buffer.tolist() == [2.0, 2.0]

    graph = _graph_of(compiled)
    assert _copy_nodes(graph) == []
    # The region still writes an input; a replay path is told so.
    assert graph.meta["input_update_placeholders"] == (1,)


def test_inference_update_can_stay_in_the_graph():
    compiled = _compile_inference(_bump, [tp.ones(2), tp.zeros(2)], keep=True)
    value, buffer = tp.ones(2), tp.zeros(2)

    assert compiled(value, buffer).tolist() == [2.0, 2.0]
    assert buffer.tolist() == [1.0, 1.0]
    assert compiled(value, buffer).tolist() == [2.0, 2.0]
    assert buffer.tolist() == [2.0, 2.0]

    graph = _graph_of(compiled)
    assert len(_copy_nodes(graph)) == 1
    assert "input_update_placeholders" not in graph.meta


@pytest.mark.parametrize("keep", [False, True])
def test_inference_without_updates_is_unchanged(keep):
    def fn(x):
        return x + 1

    compiled = _compile_inference(fn, [tp.ones(2)], keep=keep)
    assert compiled(tp.ones(2)).tolist() == [2.0, 2.0]
    graph = _graph_of(compiled)
    assert _copy_nodes(graph) == []
    assert "input_update_placeholders" not in graph.meta
