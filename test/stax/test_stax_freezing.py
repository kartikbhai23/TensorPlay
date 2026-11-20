"""Freezing an inference region's read-only state into its graph.

Freezing is observed where the compiler first sees the region: the arguments
that are module state stop being read once their values are written in, while
the arguments themselves stay in place so the callable keeps its convention.
"""

import pytest
import tensorplay as tp

from tensorplay.compiler.backends.stax import config


def _compile_capturing_graph(module, example, **settings):
    """Compile ``module`` and return its callable and the post-grad graph."""

    seen = []

    def record(graph):
        seen.append(graph)

    with config.patch(
        post_grad_custom_post_pass=record,
        cache_graph_artifacts=False,
        **settings,
    ):
        compiled = tp.compile(module, strict_native=True)
        result = compiled(example)
    assert seen
    return result, seen[-1]


def _placeholder_users(graph):
    return {
        node.name: len(node.users)
        for node in graph.nodes
        if node.op == "placeholder"
    }


def _frozen_attrs(graph):
    return [
        node.target
        for node in graph.nodes
        if node.op == "get_attr" and str(node.target).startswith("_frozen_param")
    ]


class _Linear(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = tp.nn.Parameter(tp.randn(4, 8))
        self.bias = tp.nn.Parameter(tp.randn(4))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


def test_an_inference_weight_becomes_a_constant():
    tp.manual_seed(0)
    module = _Linear().eval()
    x = tp.randn(3, 8)
    expected = module(x)

    with tp.no_grad():
        result, graph = _compile_capturing_graph(module, x, freezing=True)

    assert (result - expected).abs().max().item() < 1e-6
    # The two state arguments are present and unread; the user's input is the
    # only argument the region still reads.
    users = _placeholder_users(graph)
    assert sorted(users.values()) == [0, 0, 1]
    assert len(_frozen_attrs(graph)) == 2


def test_freezing_is_off_by_default():
    tp.manual_seed(0)
    module = _Linear().eval()
    x = tp.randn(3, 8)

    with tp.no_grad():
        result, graph = _compile_capturing_graph(module, x)

    assert result.shape == (3, 4)
    assert _frozen_attrs(graph) == []
    assert all(count > 0 for count in _placeholder_users(graph).values())


def test_a_written_state_argument_stays_an_argument():
    from tensorplay.compiler.backends.stax.freezing import freeze
    from tensorplay.graph import Graph, GraphModule

    graph = Graph()
    weight = graph.placeholder("weight")
    x = graph.placeholder("x")
    product = graph.call_function(tp.ops.tp.mm.default, (x, weight))
    # An update that writes back into the argument, as an input-mutation
    # region keeps it.
    graph.call_function(tp.ops.tp.add_.Scalar, (weight, 1))
    graph.output(product)
    module = GraphModule({}, graph)

    freeze(module, [tp.randn(4, 4), tp.randn(3, 4)], count=1)

    assert _frozen_attrs(graph) == []


def test_discarding_the_eager_parameters_refuses():
    module = _Linear().eval()
    x = tp.randn(3, 8)

    with tp.no_grad():
        with config.patch(freezing=True, freezing_discard_parameters=True):
            with pytest.raises(RuntimeError, match="freezing_discard_parameters"):
                tp.compile(module, strict_native=True)(x)


def test_weight_prepacking_refuses():
    module = _Linear().eval()
    x = tp.randn(3, 8)

    with tp.no_grad():
        with config.patch({"freezing": True, "cpp.weight_prepack": True}):
            with pytest.raises(RuntimeError, match="weight_prepack"):
                tp.compile(module, strict_native=True)(x)


class _ThreeBranches(tp.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = tp.nn.Parameter(tp.randn(6, 8))
        self.w2 = tp.nn.Parameter(tp.randn(5, 8))
        self.w3 = tp.nn.Parameter(tp.randn(7, 8))

    def forward(self, x):
        return x @ self.w1.t(), x @ self.w2.t(), x @ self.w3.t()


def test_concatenating_constant_weights_joins_three_products():
    tp.manual_seed(0)
    module = _ThreeBranches().eval()
    x = tp.randn(3, 8)
    expected = module(x)

    with tp.no_grad():
        result, graph = _compile_capturing_graph(
            module,
            x,
            **{"freezing": True, "cpp.enable_concat_linear": True},
        )

    for got, want in zip(result, expected):
        assert (got - want).abs().max().item() < 1e-6
    products = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is tp.ops.tp.matmul.default
    ]
    assert len(products) == 1
