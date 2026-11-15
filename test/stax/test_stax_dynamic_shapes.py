"""Shape policies: declared dimensions, static marks and automatic generalization."""

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax.backend import stax
from tensorplay.export.dynamic_shapes import Dim
from tensorplay.graph._utils import GraphCaptureError


def _counting_backend(artifacts):
    def backend(gm, inputs, **kwargs):
        artifact = stax(gm, inputs, **kwargs)
        artifacts.append(artifact)
        return artifact

    backend.lowers_operator_graphs = True
    backend.supports_symbolic_shapes = True
    backend.trace_decompositions = stax.trace_decompositions
    return backend


def _plain_backend(gm, inputs, **kwargs):
    """A backend that has not declared symbolic capture."""

    return gm.forward


def test_declared_dimensions_share_one_artifact_across_their_sizes():
    artifacts = []
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: Dim("batch")}},
        strict_native=True,
    )
    for rows in (3, 5, 7):
        value = tp.randn(rows, 4)
        assert tp.allclose(compiled(value), value + 1)
    assert len(artifacts) == 1

    value = tp.randn(3, 9)
    assert tp.allclose(compiled(value), value + 1)
    assert len(artifacts) == 2


def test_positional_dynamic_shapes_name_the_arguments():
    artifacts = []
    compiled = tp.compile(
        lambda x, y: x + y,
        backend=_counting_backend(artifacts),
        dynamic_shapes=[{0: Dim("rows")}, {0: Dim("rows")}],
    )
    for rows in (2, 4, 6):
        x, y = tp.randn(rows, 3), tp.randn(rows, 3)
        assert tp.allclose(compiled(x, y), x + y)
    assert len(artifacts) == 1


def test_declared_dimension_range_rejects_examples_outside_it():
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend([]),
        dynamic_shapes={"x": {0: Dim("batch", min=4)}},
    )
    with pytest.raises(ValueError, match="batch"):
        compiled(tp.randn(2, 3))


def test_declared_dimension_static_size_mismatch_is_reported():
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend([]),
        dynamic_shapes={"x": {1: 4}},
    )
    with pytest.raises(ValueError, match="expected static size 4"):
        compiled(tp.randn(3, 5))


def test_unknown_argument_name_is_reported_at_compile_time():
    with pytest.raises(ValueError, match="does not match any argument"):
        tp.compile(
            lambda x: x + 1,
            backend=_counting_backend([]),
            dynamic_shapes={"missing": {0: Dim("batch")}},
        )


def test_bare_dimension_entry_is_rejected():
    with pytest.raises(TypeError, match="per-axis"):
        tp.compile(
            lambda x: x + 1,
            backend=_counting_backend([]),
            dynamic_shapes={"x": Dim("batch")},
        )


def test_declared_dimensions_need_a_symbolic_backend():
    with pytest.raises(RuntimeError, match="supports_symbolic_shapes"):
        tp.compile(
            lambda x: x + 1,
            backend=_plain_backend,
            dynamic_shapes={"x": {0: Dim("batch")}},
        )


def test_mark_static_before_capture_pins_the_axis():
    artifacts = []

    def fn(x):
        return x + 1

    example = tp.randn(3, 4)
    tp.compiler.mark_static(example, 0)
    compiled = tp.compile(
        fn,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: Dim("batch"), 1: Dim("cols")}},
        strict_native=True,
    )
    assert tp.allclose(compiled(example), example + 1)
    other = tp.randn(3, 9)
    assert tp.allclose(compiled(other), other + 1)
    assert len(artifacts) == 1

    changed = tp.randn(4, 4)
    assert tp.allclose(compiled(changed), changed + 1)
    assert len(artifacts) == 2


def test_mark_static_inside_capture_reads_the_example_size():
    artifacts = []

    def fn(x):
        tp.compiler.mark_static(x, 0)
        return x[: x.shape[0] - 1] + 1

    compiled = tp.compile(
        fn,
        backend=_counting_backend(artifacts),
        dynamic=True,
        strict_native=True,
    )
    for rows in (3, 3, 7):
        value = tp.randn(rows, 4)
        assert tp.allclose(compiled(value), value[: rows - 1] + 1)
    assert len(artifacts) == 2


@pytest.mark.parametrize(
    "spec",
    [
        {"x": {0: Dim("batch")}},
        [{0: Dim("batch")}],
    ],
)
def test_decorator_form_keeps_the_shape_specification(spec):
    artifacts = []

    @tp.compile(backend=_counting_backend(artifacts), dynamic_shapes=spec)
    def fn(x):
        return x + 1

    for rows in (2, 4, 6):
        value = tp.randn(rows, 3)
        assert tp.allclose(fn(value), value + 1)
    assert len(artifacts) == 1


def test_automatic_dynamic_generalizes_after_a_size_change():
    artifacts = []
    compiled = tp.compile(lambda x: x + 1, backend=_counting_backend(artifacts))
    for rows in (3, 5, 7):
        value = tp.randn(rows, 4)
        assert tp.allclose(compiled(value), value + 1)
    assert len(artifacts) == 2
    assert compiled._tensorplay_dynamic is True
    assert len(compiled._tensorplay_cache) == 1


def test_automatic_dynamic_leaves_other_guard_misses_specialized():
    artifacts = []

    def fn(x):
        return x + 1 if bool((x > 0).all()) else x - 1

    compiled = tp.compile(fn, backend=_counting_backend(artifacts), fullgraph=True)
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 3.0]
    assert compiled(tp.tensor([-3.0])).tolist() == [-4.0]
    assert compiled._tensorplay_dynamic is None


def test_automatic_dynamic_can_be_disabled():
    artifacts = []
    previous = tp.compiler.config.automatic_dynamic_shapes
    tp.compiler.config.automatic_dynamic_shapes = False
    try:
        compiled = tp.compile(lambda x: x + 1, backend=_counting_backend(artifacts))
        for rows in (3, 5):
            value = tp.randn(rows, 4)
            assert tp.allclose(compiled(value), value + 1)
    finally:
        tp.compiler.config.automatic_dynamic_shapes = previous
    assert len(artifacts) == 2
    assert compiled._tensorplay_dynamic is None


def test_automatic_dynamic_does_not_generalize_on_layout_change():
    artifacts = []
    compiled = tp.compile(lambda x: x + 1, backend=_counting_backend(artifacts))
    value = tp.randn(4, 4)
    assert tp.allclose(compiled(value), value + 1)
    transposed = value.t()
    assert tp.allclose(compiled(transposed), transposed + 1)
    assert len(artifacts) == 2
    assert compiled._tensorplay_dynamic is None


@pytest.mark.parametrize(
    "dynamic, dynamic_shapes", [(False, None), (None, False)]
)
def test_explicit_static_policy_keeps_sizes_specialized(dynamic, dynamic_shapes):
    artifacts = []
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend(artifacts),
        dynamic=dynamic,
        dynamic_shapes=dynamic_shapes,
    )
    for rows in (3, 5):
        value = tp.randn(rows, 4)
        assert tp.allclose(compiled(value), value + 1)
    assert len(artifacts) == 2
    assert compiled._tensorplay_dynamic is False


def test_disallow_in_graph_refuses_to_absorb_the_call():
    def boundary(x):
        return x + 1

    tp.compiler.disallow_in_graph(boundary)
    compiled = tp.compile(
        lambda x: boundary(x) * 2,
        backend=_counting_backend([]),
    )
    with pytest.raises(GraphCaptureError, match="disallowed"):
        compiled(tp.randn(3))