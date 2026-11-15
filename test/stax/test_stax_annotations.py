"""What the callable annotations do, and do not, change about capture.

Two of them belong to the scripting path, which this build does not have: they
record a mark on the callable and leave capture alone.  The substitution
decorator does change capture, and its folding hint describes a shortcut that
capture already takes for calls whose arguments are all constants.
"""

import pytest

import tensorplay as tp


@pytest.fixture
def probe():
    """A backend that records the operations of each captured region."""

    seen: list[list[str]] = []

    def backend(graph_module, example_inputs, **kwargs):
        seen.append(
            [
                node.target
                if isinstance(node.target, str)
                else getattr(node.target, "_opname", None)
                or getattr(node.target, "__name__", None)
                or repr(node.target)
                for node in graph_module.graph.nodes
                if node.op in ("call_function", "call_method")
            ]
        )
        return graph_module.forward

    tp.compiler.register_backend(backend, name="annotation_probe")
    yield seen
    tp.compiler.unregister_backend("annotation_probe")


def test_a_helper_marked_unused_is_traced_like_any_other(probe):
    """The mark belongs to the scripting path; capture still traces the call."""

    @tp.compiler.unused
    def helper(a):
        return a.sin() * 2.0

    def fn(x):
        return helper(x) + 1.0

    compiled = tp.compile(fn, backend="annotation_probe")
    x = tp.tensor([0.5, -0.5])

    assert tp.allclose(compiled(x), fn(x))
    assert "sin" in probe[0]


def test_definition_marks_are_recorded_and_change_no_capture(probe):
    @tp.compiler.export
    def exported(a):
        return a + 1.0

    @tp.compiler.overload_method
    def overload(a):
        return a - 1.0

    assert exported.__tensorplay_compiler_export__ is True
    assert overload.__tensorplay_compiler_overload__ is True

    def fn(x):
        return exported(x) + overload(x)

    compiled = tp.compile(fn, backend="annotation_probe")
    x = tp.tensor([1.0, 2.0])
    assert tp.allclose(compiled(x), fn(x))
    # Both bodies were traced, so neither mark kept its call as one opaque op.
    assert probe[0].count("add") >= 1 and probe[0].count("sub") >= 1


def test_a_substituted_call_is_captured_as_the_replacement(probe):
    def original(a):
        return a * 3.0

    @tp.compiler.substitute_in_graph(original)
    def replacement(a):
        return a + a + a

    def fn(x):
        return original(x)

    compiled = tp.compile(fn, backend="annotation_probe")
    x = tp.tensor([1.0, 2.0])

    assert tp.allclose(compiled(x), tp.tensor([3.0, 6.0]))
    # The replacement's two additions are what was recorded, not one multiply.
    assert probe[0].count("add") == 2
    assert "mul" not in probe[0]


@pytest.mark.parametrize("fold", [True, False])
def test_a_constant_only_call_never_becomes_a_node(probe, fold):
    """Capture evaluates it while tracing, whatever the hint says.

    The hint asks for exactly that shortcut through the original function;
    with every argument a plain constant there is nothing to record either
    way, so both settings give the same graph.
    """

    def original(a):
        return a * 3.0

    @tp.compiler.substitute_in_graph(original, can_constant_fold_through=fold)
    def replacement(a):
        return a + a + a

    def fn(x):
        return x + original(4.0)

    compiled = tp.compile(fn, backend="annotation_probe")
    x = tp.tensor([1.0, 2.0])

    assert tp.allclose(compiled(x), tp.tensor([13.0, 14.0]))
    assert probe[0] == ["add"]


def test_skipping_the_signature_check_allows_a_different_arity(probe):
    def original(a, b):
        return a + b

    with pytest.raises(TypeError):
        tp.compiler.substitute_in_graph(original)(
            lambda a: a
        )

    @tp.compiler.substitute_in_graph(original, skip_signature_check=True)
    def replacement(a, *rest):
        return a + a

    def fn(x):
        return original(x, x)

    compiled = tp.compile(fn, backend="annotation_probe")
    x = tp.tensor([2.0, 3.0])
    assert tp.allclose(compiled(x), tp.tensor([4.0, 6.0]))