"""The rewrites that run once a region's operations are settled.

The pipeline that runs after the gradient graph is settled rewrites the graph
before it is lowered, so what it does is observed here through the hook that
receives the graph at the end of that pipeline.  Compilation happens on the
first call, which is therefore made while the hook is still installed.
"""

import pytest
import tensorplay as tp

from tensorplay.compiler.backends.stax import config

METADATA_ASSERT = tp.ops.tp._assert_tensor_metadata.default


def _compiled_order(fn, *args, **settings):
    """Compile and call ``fn``, reporting the operations its region kept."""

    seen = []

    def record(graph):
        seen.append([node.target for node in graph.nodes])

    with config.patch(post_grad_custom_post_pass=record, **settings):
        compiled = tp.compile(fn, strict_native=True)
        result = compiled(*args)
    return result, seen


def test_a_metadata_assertion_is_dropped_before_lowering():
    def fn(x):
        METADATA_ASSERT(x, (2, 3), None, None)
        return x.relu() + 1

    result, seen = _compiled_order(fn, tp.tensor([[-1.0, 2.0, 3.0], [-1.0, 2.0, 3.0]]))

    assert seen
    assert METADATA_ASSERT not in seen[-1]
    assert result.tolist() == [[1.0, 3.0, 4.0], [1.0, 3.0, 4.0]]


def test_a_producer_moves_next_to_the_read_that_wants_it():
    def fn(x):
        carried = x.relu()
        other = x.sin()
        late = x.cos()
        return (carried + other) * late

    result, seen = _compiled_order(fn, tp.tensor([[-1.0, 2.0]]))

    targets = seen[-1]
    total = targets.index(tp.ops.tp.add.Tensor)
    # The addition reads the first two branches and is followed by the third
    # one, which is only needed afterwards.
    assert targets[total - 2 : total] == [
        tp.ops.tp.relu.default,
        tp.ops.tp.sin.default,
    ]
    assert targets.index(tp.ops.tp.cos.default) > total
    assert result.shape == (1, 2)


def test_a_random_draw_keeps_the_place_it_was_written_in():
    def fn(x):
        draw = tp.randn(2, 2)
        carried = x.relu()
        other = x.sin()
        return (carried + other) * draw

    result, seen = _compiled_order(fn, tp.tensor([[-1.0, 2.0]]))

    targets = seen[-1]
    # Drawing the value earlier is part of the program: moving the draw would
    # change every value drawn after it.
    assert targets.index(tp.ops.tp.randn.default) < targets.index(
        tp.ops.tp.relu.default
    )
    assert result.shape == (2, 2)


def test_the_locality_order_can_be_turned_off():
    def fn(x):
        carried = x.relu()
        other = x.sin()
        late = x.cos()
        return (carried + other) * late

    result, seen = _compiled_order(
        fn, tp.tensor([[-1.0, 2.0]]), reorder_for_locality=False
    )

    targets = seen[-1]
    assert targets.index(tp.ops.tp.cos.default) < targets.index(tp.ops.tp.add.Tensor)
    assert result.shape == (1, 2)


def test_the_reciprocal_of_a_square_root_is_written_once():
    def fn(x):
        return 1.0 / x.sqrt()

    result, seen = _compiled_order(fn, tp.tensor([[4.0, 16.0]]))

    assert tp.ops.tp.rsqrt.default in seen[-1]
    assert tp.ops.tp.sqrt.default not in seen[-1]
    assert result.tolist() == [[0.5, 0.25]]


def test_a_split_rebuilt_by_a_cat_collapses_without_being_asked_for():
    def fn(x):
        left, right = x.split([2, 2], dim=1)
        return tp.cat([left, right], dim=1) + 0

    result, seen = _compiled_order(fn, tp.arange(8).reshape(2, 4))

    assert tp.ops.tp.cat.default not in seen[-1]
    assert tp.ops.tp.split.sizes not in seen[-1]
    assert result.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_split_given_sizes_rebuilt_by_a_cat_collapses():
    def fn(x):
        left, right = tp.split_with_sizes(x, [2, 2], 1)
        return tp.cat([left, right], dim=1)

    result, seen = _compiled_order(fn, tp.arange(8).reshape(2, 4))

    assert tp.ops.tp.split_with_sizes.default not in seen[-1]
    assert tp.ops.tp.cat.default not in seen[-1]
    assert result.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_a_cat_of_the_pieces_in_another_order_is_left_alone():
    def fn(x):
        left, right = x.split([2, 2], dim=1)
        return tp.cat([right, left], dim=1) + 0

    result, seen = _compiled_order(fn, tp.arange(8).reshape(2, 4))

    assert tp.ops.tp.cat.default in seen[-1]
    assert tp.ops.tp.split.sizes in seen[-1]
    assert result.tolist() == [[2, 3, 0, 1], [6, 7, 4, 5]]


def test_a_partial_rebuild_of_a_split_still_needs_the_threshold_pass():
    def fn(x):
        left, right = x.split([2, 2], dim=1)
        return tp.cat([left], dim=1) + right

    _, seen = _compiled_order(fn, tp.arange(8).reshape(2, 4))

    assert tp.ops.tp.split.sizes in seen[-1]


def test_a_split_of_a_cat_hands_back_the_pieces():
    def fn(x, y):
        joined = tp.cat([x, y], dim=1)
        left, right = joined.split([2, 2], dim=1)
        return left * 2 + right

    result, seen = _compiled_order(
        fn, tp.arange(4).reshape(2, 2), tp.ones(2, 2)
    )

    assert tp.ops.tp.cat.default not in seen[-1]
    assert tp.ops.tp.split.sizes not in seen[-1]
    assert result.tolist() == [[1.0, 3.0], [5.0, 7.0]]


def test_a_split_of_a_cat_that_cuts_elsewhere_is_left_alone():
    def fn(x, y):
        joined = tp.cat([x, y], dim=1)
        left, right = joined.split([1, 3], dim=1)
        return left * 2 + right

    _, seen = _compiled_order(fn, tp.arange(4).reshape(2, 2), tp.ones(2, 2))

    assert tp.ops.tp.cat.default in seen[-1]
    assert tp.ops.tp.split.sizes in seen[-1]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [0, 3, 5])
def test_cat_with_a_prefix_of_its_first_input_is_joined_once(device, width):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def fn(a, b):
        joined = tp.cat([a, b], 1)
        return tp.cat([joined, joined[:, :width]], 1)

    a = tp.arange(10, device=device, dtype=tp.float32).reshape(2, 5)
    b = tp.ones((2, 4), device=device)
    expected = fn(a, b)
    actual, seen = _compiled_order(fn, a, b)
    assert actual.tolist() == expected.tolist()
    assert tp.ops.tp.cat.default not in seen[-1]
    assert any(getattr(target, "_tp_lowering_function", False) for target in seen[-1])


@pytest.mark.parametrize("width", [6, -1])
def test_cat_prefix_outside_first_input_keeps_both_joins(width):
    def fn(a, b):
        joined = tp.cat([a, b], 1)
        return tp.cat([joined, joined[:, :width]], 1)

    a = tp.arange(10, dtype=tp.float32).reshape(2, 5)
    b = tp.ones((2, 4))
    expected = fn(a, b)
    actual, seen = _compiled_order(fn, a, b)
    assert actual.tolist() == expected.tolist()
    assert seen[-1].count(tp.ops.tp.cat.default) == 2


@pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.float64])
def test_cat_prefix_rewrite_preserves_values_for_noncontiguous_inputs(dtype):
    def fn(a, b):
        joined = tp.cat([a, b], 1)
        return tp.cat([joined, joined[:, :3]], 1) + 2

    a = tp.arange(30, dtype=dtype).reshape(5, 6).transpose(0, 1)
    b = tp.arange(18, dtype=dtype).reshape(3, 6).transpose(0, 1)
    actual, _ = _compiled_order(fn, a, b)
    assert actual.tolist() == fn(a, b).tolist()


def test_cat_prefix_rewrite_handles_mixed_types():
    def fn(a, b):
        joined = tp.cat([a, b], 1)
        return tp.cat([joined, joined[:, :2]], 1)

    a = tp.arange(6, dtype=tp.int32).reshape(2, 3)
    b = tp.ones((2, 4))
    actual, _ = _compiled_order(fn, a, b)
    expected = fn(a, b)
    assert actual.dtype == expected.dtype
    assert actual.tolist() == expected.tolist()


def test_a_view_is_renamed_to_a_reshape():
    def fn(x):
        return (x.view(6) + 1).view(2, 3)

    result, seen = _compiled_order(fn, tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    assert seen
    targets = seen[-1]
    assert tp.ops.tp.view.default not in targets
    assert tp.ops.tp.reshape.default in targets
    assert result.tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_an_addmm_with_a_bias_like_input_is_read_as_a_product_and_an_add(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("cuda not available")

    def fn(x, w, b):
        return (tp.addmm(b, x, w) * 2).relu()

    x = tp.randn(4, 5, device=device)
    w = tp.randn(5, 3, device=device)
    b = tp.randn(3, device=device)
    eager = fn(x, w, b)

    result, seen = _compiled_order(fn, x, w, b)
    if device == "cuda":
        assert tp.ops.tp.addmm.default not in seen[-1]
        assert tp.ops.tp.mm.default in seen[-1]
    assert tp.allclose(result, eager)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_an_addmm_outside_the_fusion_range_stays_fused(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("cuda not available")

    def fn(x, w):
        # 加进来的值是算出来的，不是捕获直接读到的形状：解融的判定不收，
        # 乘积保持融合形态。
        bias = x.sum(1)
        return tp.relu(tp.addmm(bias, x, w)) * 2

    x = tp.randn(4, 4, device=device)
    w = tp.randn(4, 4, device=device)
    eager = fn(x, w)

    result, seen = _compiled_order(fn, x, w)
    assert tp.ops.tp.addmm.default in seen[-1]
    assert tp.allclose(result, eager)
