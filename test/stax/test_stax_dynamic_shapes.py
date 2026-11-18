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


def test_shared_dimension_must_hold_one_size():
    compiled = tp.compile(
        lambda x, y: x.sum(0) * y.sum(0),
        backend=_counting_backend([]),
        dynamic_shapes={"x": {0: Dim("batch")}, "y": {0: Dim("batch")}},
    )
    with pytest.raises(ValueError, match="conflicting example sizes"):
        compiled(tp.randn(3, 2), tp.randn(4, 2))


def test_mark_static_overrides_a_shared_dimension():
    artifacts = []

    def fn(x, y):
        return x.sum(0) * y.sum(0)

    fixed = tp.randn(4, 2)
    tp.compiler.mark_static(fixed, 0)
    compiled = tp.compile(
        fn,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: Dim("batch")}, "y": {0: Dim("batch")}},
    )
    value = tp.randn(3, 2)
    assert tp.allclose(compiled(value, fixed), value.sum(0) * fixed.sum(0))


def test_derived_dimension_rejects_a_size_it_cannot_produce():
    artifacts = []
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: 2 * Dim("half", min=2)}},
    )
    with pytest.raises(ValueError, match="not of the form 2\\*half"):
        compiled(tp.randn(5, 3))
    assert artifacts == []


def test_derived_dimension_rejects_an_unreachable_size_after_a_capture():
    artifacts = []
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: 2 * Dim("half", min=2)}},
    )
    value = tp.randn(4, 3)
    assert tp.allclose(compiled(value), value + 1)
    with pytest.raises(ValueError, match="not of the form 2\\*half"):
        compiled(tp.randn(5, 3))
    assert len(artifacts) == 1


def test_derived_dimension_stays_consistent_with_its_root():
    compiled = tp.compile(
        lambda x, y: x.sum(0) * y.sum(0),
        backend=_counting_backend([]),
        dynamic_shapes={"x": {0: Dim("half")}, "y": {0: 2 * Dim("half")}},
    )
    with pytest.raises(ValueError, match="example sizes give 6 for the root"):
        compiled(tp.randn(3, 2), tp.randn(8, 2))


def test_declared_maximum_is_enforced_on_later_calls():
    artifacts = []
    compiled = tp.compile(
        lambda x: x + 1,
        backend=_counting_backend(artifacts),
        dynamic_shapes={"x": {0: Dim("rows", min=2, max=5)}},
    )
    value = tp.randn(3, 4)
    assert tp.allclose(compiled(value), value + 1)
    with pytest.raises(ValueError, match="violates rows\\(min=2, max=5\\)"):
        compiled(tp.randn(6, 4))
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

def test_a_symbolic_region_can_hold_a_product():
    """A product under symbolic extents, served at more than one size.

    No candidate of the product templates is written for extents that are not
    known, so the lowering answers nothing there -- which used to leave the
    region's result as a name nothing ever filled.  A lowering that answers
    nothing has said the call is not one it covers, and the framework computes
    it: the region keeps the product as a call of its own and answers for every
    size the one artifact was captured for.
    """

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) / 7
    compiled = tp.compile(lambda a: tp.mm(a, weight).relu(), dynamic=True)
    for rows in (2, 3, 5):
        value = tp.randn(rows, 4)
        assert tp.allclose(compiled(value), tp.mm(value, weight).relu(), atol=1e-5)
