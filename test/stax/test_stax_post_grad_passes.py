"""The rewrites that run once a region's operations are settled.

The pipeline that runs after the gradient graph is settled rewrites the graph
before it is lowered, so what it does is observed here through the hook that
receives the graph at the end of that pipeline.  Compilation happens on the
first call, which is therefore made while the hook is still installed.
"""

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