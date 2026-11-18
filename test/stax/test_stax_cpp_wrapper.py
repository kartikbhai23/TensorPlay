"""The C++ wrapper: a region written as one translation unit.

A region compiled for the host is printed as C++ rather than as Python, and
what it prints is one entry point that is called with the region's buffers.
The tests below compile through that route and check two things: that the
region was actually built (a region that fell back to the graph executor says
so with a warning, and would answer correctly anyway), and that the answer is
the one the region computes when it is run as it was written.

The pointwise cases matter most.  A kernel that reads one buffer and writes
another is given two pointers; one that is written to read and write the same
buffer is given one.  Which of the two a region gets is decided while the
kernel is printed, and a kernel that reads its own uninitialized result is the
mistake these cases are here to catch.
"""

import warnings

import pytest

import tensorplay as tp


def _check(fn, args):
    """Compile ``fn`` and compare what it returns with what it computes."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = tp.compile(fn, options={"stax.cuda_codegen": True})
        got = compiled(*args)
    fell_back = [
        str(item.message)
        for item in caught
        if "Stax lowering failed" in str(item.message)
    ]
    if fell_back:
        raise AssertionError(
            f"the region fell back instead of being written: {fell_back[0]}"
        )
    expected = fn(*args)
    for one, other in zip(
        got if isinstance(got, tuple) else (got,),
        expected if isinstance(expected, tuple) else (expected,),
    ):
        if not tp.allclose(one, other):
            raise AssertionError(
                f"got {one.tolist()}, expected {other.tolist()}"
            )


# --------------------------------------------------------------------------
# elementwise regions
# --------------------------------------------------------------------------


def test_elementwise_chain():
    _check(lambda a: (a * 2 + 1).relu(), [tp.tensor([-2.0, 0.0, 3.0])])


def test_elementwise_chain_over_a_matrix():
    _check(
        lambda a: ((a + 1) * 0.5).tanh(),
        [tp.tensor([[1.0, -1.0], [0.5, 0.25]])],
    )


def test_region_with_several_results():
    _check(lambda a: (a * 2, a + 1), [tp.tensor([1.0, 2.0])])


def test_region_reading_a_constant():
    weight = tp.arange(6, dtype=tp.float32).reshape(2, 3)
    _check(lambda a: (a * weight).relu().sum(dim=-1), [tp.ones(2, 3)])


# --------------------------------------------------------------------------
# regions that hand a call to the runtime
# --------------------------------------------------------------------------


def test_matrix_multiply():
    other = tp.arange(12, dtype=tp.float32).reshape(4, 3)
    _check(lambda a: tp.mm(a, other), [tp.ones(2, 4)])


def test_matrix_multiply_written_as_an_operator():
    other = tp.arange(12, dtype=tp.float32).reshape(4, 3)
    _check(lambda a: a @ other, [tp.ones(2, 4)])


def test_batched_matrix_multiply():
    other = tp.arange(12, dtype=tp.float32).reshape(2, 2, 3)
    _check(lambda a: tp.bmm(a, other), [tp.ones(2, 1, 2)])


def test_linear_with_a_bias():
    weight = tp.arange(8, dtype=tp.float32).reshape(4, 2)
    bias = tp.arange(4, dtype=tp.float32)
    _check(lambda a: tp.nn.functional.linear(a, weight, bias), [tp.ones(3, 2)])


def test_linear_without_a_bias():
    weight = tp.arange(8, dtype=tp.float32).reshape(4, 2)
    _check(lambda a: tp.nn.functional.linear(a, weight), [tp.ones(3, 2)])


def test_two_layer_network():
    first = tp.arange(8, dtype=tp.float32).reshape(4, 2)
    second = tp.arange(12, dtype=tp.float32).reshape(3, 4)
    bias = tp.arange(3, dtype=tp.float32)

    def network(a):
        hidden = tp.nn.functional.relu(tp.nn.functional.linear(a, first))
        return tp.nn.functional.linear(hidden, second, bias)

    _check(network, [tp.tensor([[1.0, 2.0], [-1.0, 0.5]])])


# --------------------------------------------------------------------------
# a pointwise kernel that reads what another kernel wrote
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pointwise",
    [
        lambda a: a.tanh(),
        lambda a: a.exp(),
        lambda a: -a,
        lambda a: a * 2.0,
        lambda a: a + 1.0,
        lambda a: a.abs(),
        lambda a: a.relu(),
    ],
    ids=["tanh", "exp", "neg", "mul", "add", "abs", "relu"],
)
def test_pointwise_after_a_matrix_multiply(pointwise):
    other = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(lambda a: pointwise(tp.mm(a, other)), [tp.tensor([[1.0, 2.0, 3.0, 4.0]])])


def test_pointwise_chain_after_a_matrix_multiply():
    other = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: (tp.mm(a, other) * 2.0).tanh().relu(),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]])],
    )


def test_sum_after_a_matrix_multiply():
    other = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, other).sum(dim=-1),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0]])],
    )


def test_linear_then_pointwise():
    weight = tp.arange(8, dtype=tp.float32).reshape(4, 2) - 3
    _check(
        lambda a: tp.nn.functional.linear(a, weight).tanh(),
        [tp.tensor([[1.0, 2.0], [0.0, -1.0]])],
    )

# --------------------------------------------------------------------------
# a matrix multiply over a bias: the call that takes a number as well
# --------------------------------------------------------------------------


def test_matrix_multiply_over_a_bias():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    bias = tp.arange(3, dtype=tp.float32) / 3
    _check(
        lambda a: tp.mm(a, weight) + bias,
        [tp.tensor([[1.0, 2.0, 3.0, 4.0]])],
    )


def test_addmm_called_directly():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    bias = tp.arange(3, dtype=tp.float32) / 3
    _check(
        lambda a: tp.addmm(bias, a, weight),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0]])],
    )


def test_matrix_multiply_over_a_bias_then_pointwise():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    bias = tp.arange(3, dtype=tp.float32) / 3
    _check(
        lambda a: (tp.mm(a, weight) + bias).relu(),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0]])],
    )


# --------------------------------------------------------------------------
# views: the same bytes read another way
# --------------------------------------------------------------------------


def test_transposed_result():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).transpose(0, 1),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]])],
    )


def test_reshaped_result():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).reshape(3, 2),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]])],
    )


def test_sliced_result_reads_from_its_offset():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight)[:, 1:3],
        [tp.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]])],
    )


def test_transposed_input_is_read_as_the_window_it_selects():
    weight = tp.arange(6, dtype=tp.float32).reshape(3, 2) - 2
    _check(
        lambda a: tp.mm(a.transpose(0, 1), weight),
        [tp.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
    )


def test_two_results_that_are_views_of_one_buffer():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: (tp.mm(a, weight).transpose(0, 1), tp.mm(a, weight).reshape(3, 2)),
        [tp.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.5, 2.0]])],
    )


def test_view_of_a_graph_input_is_returned():
    _check(lambda a: a.transpose(0, 1), [tp.tensor([[1.0, 2.0, 3.0]])])


# --------------------------------------------------------------------------
# a convolution: the call whose extents are lists rather than buffers
# --------------------------------------------------------------------------


def _conv_weight():
    return tp.arange(54, dtype=tp.float32).reshape(2, 3, 3, 3) / 7


def test_convolution():
    _check(lambda a: tp.conv2d(a, _conv_weight()), [tp.randn(1, 3, 8, 8)])


def test_convolution_with_a_bias():
    _check(
        lambda a: tp.conv2d(a, _conv_weight(), tp.arange(2, dtype=tp.float32) / 5),
        [tp.randn(1, 3, 8, 8)],
    )


def test_convolution_with_padding_and_stride():
    weight = _conv_weight()
    _check(
        lambda a: tp.conv2d(a, weight, padding=1),
        [tp.randn(1, 3, 8, 8)],
    )
    _check(
        lambda a: tp.conv2d(a, weight, stride=2),
        [tp.randn(1, 3, 8, 8)],
    )


def test_convolution_then_pointwise():
    weight = _conv_weight()
    _check(
        lambda a: tp.conv2d(a, weight, padding=1).relu(),
        [tp.randn(1, 3, 8, 8)],
    )


def test_two_convolutions():
    """A hand-written network: the second call reads what the first wrote."""

    first = _conv_weight()
    second = tp.arange(8, dtype=tp.float32).reshape(4, 2, 1, 1) / 7

    def network(a):
        hidden = tp.nn.functional.relu(tp.conv2d(a, first, padding=1))
        return tp.conv2d(hidden, second)

    _check(network, [tp.randn(1, 3, 8, 8)])


# --------------------------------------------------------------------------
# a call that answers with more than one result
# --------------------------------------------------------------------------


def _pair_input():
    return tp.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 0.0]])


def test_a_sort_answers_with_both_halves():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).sort(dim=-1),
        [_pair_input()],
    )


def test_a_sort_read_by_either_half():
    """The values and the order are two buffers, and either may be the result."""

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).sort(dim=-1)[1],
        [_pair_input()],
    )
    _check(
        lambda a: tp.mm(a, weight).sort(dim=-1)[0] * 2.0,
        [_pair_input()],
    )


def test_a_sort_that_keeps_its_axes():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).sort(dim=-1, descending=True),
        [_pair_input()],
    )


def test_topk():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(lambda a: tp.mm(a, weight).topk(2, dim=-1), [_pair_input()])


def test_a_median_and_a_mode():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.median(tp.mm(a, weight), dim=-1),
        [_pair_input()],
    )
    _check(
        lambda a: tp.mode(tp.mm(a, weight), dim=-1),
        [_pair_input()],
    )


def test_a_kth_value():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.kthvalue(tp.mm(a, weight), 2, dim=-1),
        [_pair_input()],
    )


def test_a_cumulative_sum():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).cumsum(dim=-1),
        [_pair_input()],
    )


def test_a_cumulative_sum_in_another_type():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).cumsum(dim=-1, dtype=tp.float64),
        [_pair_input()],
    )


def test_a_rounded_division():
    """A division that rounds keeps the rounding as part of the call.

    The mode is a word the call takes rather than a buffer, so it is written
    where it stands; an integral quotient is what the two spellings are for.
    """

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.div(tp.mm(a, weight) * 10, 3.0, rounding_mode="floor"),
        [_pair_input()],
    )
    _check(
        lambda a: tp.divide(tp.mm(a, weight) * 10, 3.0, rounding_mode="trunc"),
        [_pair_input()],
    )


def test_a_symbolic_region_serves_every_size():
    """One region, written once, answering for more than one number of rows.

    A product has no candidate written for extents that are not known, so the
    region keeps the product as a call of its own; the point of capturing
    symbolically is that the same region answers for every size afterwards.
    """

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) / 7
    fn = lambda a: tp.mm(a, weight).relu()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = tp.compile(
            fn, options={"stax.cuda_codegen": True}, dynamic=True
        )
        answers = [compiled(tp.randn(rows, 4)) for rows in (2, 3, 5)]
    fell_back = [
        str(item.message)
        for item in caught
        if "Stax lowering failed" in str(item.message)
    ]
    if fell_back:
        raise AssertionError(
            f"the region fell back instead of being written: {fell_back[0]}"
        )
    for rows, answer in zip((2, 3, 5), answers):
        assert tuple(answer.shape) == (rows, 3)
