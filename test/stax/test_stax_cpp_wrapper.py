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


def _check_seeded(fn, args):
    """Compile ``fn`` and compare, with the draws on both sides the same.

    A region that draws numbers has nothing to compare its answer against
    unless both sides start from the same state, so the generator is set to
    the same state before each of them runs.  The region still has to be
    written rather than fallen back from, which is checked first.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = tp.compile(fn, options={"stax.cuda_codegen": True})
        tp.manual_seed(1234)
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
    tp.manual_seed(1234)
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


def test_a_batch_of_products_added_onto_one_value():
    """Every product in the batch lands on the same value, scaled two ways.

    The batched forms are separate calls from the unbatched ones, and the two
    numbers that weight the value and the products are places of their own.
    """

    weight = tp.arange(20, dtype=tp.float32).reshape(4, 5) / 7
    addend = tp.ones(2, 5)
    rows = tp.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 0.0]])

    def batch(a):
        return a.reshape(1, 2, 4)

    def batched(a):
        return weight.reshape(1, 4, 5)

    _check(lambda a: tp.addbmm(addend, batch(a), batched(a)), [rows])
    _check(
        lambda a: tp.addbmm(addend, batch(a), batched(a), beta=0.5, alpha=2.0),
        [rows],
    )
    _check(
        lambda a: tp.baddbmm(addend.reshape(1, 2, 5), batch(a), batched(a)),
        [rows],
    )
    _check(
        lambda a: tp.baddbmm(
            addend.reshape(1, 2, 5), batch(a), batched(a), beta=0.5, alpha=2.0
        ),
        [rows],
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


def test_a_transposed_convolution():
    """The transposed call spreads each input element out through the filter.

    Its weight is read with the input's channels first, and the places after
    the operands are the ones a forward convolution has.
    """

    weight = _conv_weight()
    _check(
        lambda a: tp.conv_transpose2d(
            a, weight, stride=2, padding=1, output_padding=1
        ),
        [tp.randn(1, 2, 8, 8)],
    )
    _check(
        lambda a: tp.conv_transpose2d(
            a,
            weight,
            tp.arange(3, dtype=tp.float32) / 5,
            stride=2,
            padding=1,
            output_padding=1,
        ),
        [tp.randn(1, 2, 8, 8)],
    )


def test_convolutions_of_other_ranks():
    """One- and three-dimensional convolutions take the same places.

    They are separate calls from the two-dimensional one, and what they take
    after the operands is the same set in the same order.
    """

    one = tp.arange(36, dtype=tp.float32).reshape(4, 3, 3) / 7
    three = tp.arange(324, dtype=tp.float32).reshape(4, 3, 3, 3, 3) / 7
    _check(
        lambda a: tp.conv1d(a, one, padding=1),
        [tp.randn(1, 3, 8)],
    )
    _check(
        lambda a: tp.conv1d(a, one, tp.arange(4, dtype=tp.float32) / 5, stride=2),
        [tp.randn(1, 3, 8)],
    )
    _check(
        lambda a: tp.conv3d(a, three, padding=1),
        [tp.randn(1, 3, 4, 4, 4)],
    )


def test_transposed_convolutions_of_each_rank():
    one = tp.arange(18, dtype=tp.float32).reshape(2, 3, 3) / 7
    three = tp.arange(162, dtype=tp.float32).reshape(2, 3, 3, 3, 3) / 7
    _check(
        lambda a: tp.conv_transpose1d(
            a, one, stride=2, padding=1, output_padding=1
        ),
        [tp.randn(1, 2, 8)],
    )
    _check(
        lambda a: tp.conv_transpose3d(
            a, three, stride=2, padding=1, output_padding=1
        ),
        [tp.randn(1, 2, 4, 4, 4)],
    )


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


def test_a_cumulative_product():
    """The product along an axis accumulates like the sum, in either type."""

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).cumprod(dim=-1),
        [_pair_input()],
    )
    _check(
        lambda a: tp.mm(a, weight).cumprod(dim=1, dtype=tp.float64),
        [_pair_input()],
    )


def test_a_running_sum_of_exponentials():
    """A running sum of exponentials takes an axis and a type to accumulate.

    The type is optional: a call that does not name one leaves it absent, and
    the absent form is what the call is written with.
    """

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.logcumsumexp(tp.mm(a, weight), 1),
        [_pair_input()],
    )
    _check(
        lambda a: tp.logcumsumexp(tp.mm(a, weight), 1, dtype=tp.float64),
        [_pair_input()],
    )


def test_the_median_of_every_element():
    """Over every element the median is one value, with no order beside it."""

    _check(
        lambda a: tp.median(a),
        [tp.tensor([[4.0, 1.0, 3.0], [2.0, 5.0, 0.0]])],
    )


def test_a_running_maximum():
    """A running maximum keeps the place each maximum came from."""

    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).cummax(dim=-1),
        [_pair_input()],
    )


def test_a_running_minimum():
    weight = tp.arange(12, dtype=tp.float32).reshape(4, 3) - 5
    _check(
        lambda a: tp.mm(a, weight).cummin(dim=-1),
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


# --------------------------------------------------------------------------
# an operand that is a window onto a buffer
# --------------------------------------------------------------------------


def test_a_product_whose_operand_is_a_window():
    """A reshaped weight is read through the window, not through the buffer.

    The window is a place to read elements from as much as the buffer is, so
    it is one of the call's operands: the entry point is given the address its
    first element sits at, and the kernel reads the elements it selects.
    """

    weight = tp.arange(12, dtype=tp.float32).reshape(1, 4, 3) / 7
    _check(
        lambda a: tp.matmul(a, weight.reshape(4, 3)),
        [tp.arange(8, dtype=tp.float32).reshape(2, 4)],
    )


def test_a_product_of_two_windows():
    """Both operands are windows, and neither is the buffer it sits on."""

    weight = tp.arange(12, dtype=tp.float32).reshape(1, 4, 3) / 7
    _check(
        lambda a: tp.matmul(
            a.reshape(2, 1, 4), weight
        ),
        [tp.arange(8, dtype=tp.float32).reshape(2, 4)],
    )


def test_a_group_norm():
    """A normalization whose scale and shift are optional.

    The call is written with every place it has whether or not a value stands in
    each, so the two forms are told apart by how many operands are handed over.
    """

    weight = tp.arange(4, dtype=tp.float32) / 5 + 0.5
    bias = tp.arange(4, dtype=tp.float32) / 7
    _check(
        lambda a: tp.nn.functional.group_norm(a, 2, weight, bias),
        [tp.randn(2, 4, 8, 8)],
    )


def test_a_group_norm_without_a_scale_or_shift():
    weight = tp.arange(4, dtype=tp.float32) / 5 + 0.5
    _check(
        lambda a: tp.nn.functional.group_norm(a, 2, None, None, eps=1e-3),
        [tp.randn(2, 4, 8, 8)],
    )
    _check(
        lambda a: tp.nn.functional.group_norm(a, 2, weight, None).relu(),
        [tp.randn(2, 4, 8, 8)],
    )


# --------------------------------------------------------------------------
# attention: three operands and a handful of numbers
# --------------------------------------------------------------------------


def test_scaled_dot_product_attention():
    """The attention call, with its mask absent and its dropout off.

    Only three of the places it has hold operands; the rest are written as they
    stand, and the absent ones are written as nothing.
    """

    qkv = tp.randn(2, 4, 8)
    _check(
        lambda a: tp.nn.functional.scaled_dot_product_attention(a, a, a),
        [qkv],
    )


def test_scaled_dot_product_attention_is_causal():
    qkv = tp.randn(2, 4, 8)
    _check(
        lambda a: tp.nn.functional.scaled_dot_product_attention(
            a, a, a, is_causal=True
        ),
        [qkv],
    )


def test_scaled_dot_product_attention_with_a_scale():
    qkv = tp.randn(2, 4, 8)
    _check(
        lambda a: tp.nn.functional.scaled_dot_product_attention(
            a, a, a, scale=0.5
        ),
        [qkv],
    )


# --------------------------------------------------------------------------
# a contraction whose operands are a list
# --------------------------------------------------------------------------


def _einsum_operands():
    return [
        tp.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, 0.0]]),
        tp.arange(12, dtype=tp.float32).reshape(4, 3) / 7,
    ]


def test_einsum():
    """A contraction takes its operands in a list rather than as arguments.

    The list is the operands, and the places it has beyond the list are the
    equation and the path; the shim builds the vector from the slots.
    """

    a, weight = _einsum_operands()
    _check(lambda x: tp.einsum("ij,jk->ik", x, weight), [a])


def test_einsum_with_three_operands_and_a_window():
    """The third operand is a transposed weight, read through its window."""

    a, weight = _einsum_operands()
    _check(
        lambda x: tp.einsum("ij,jk,kl->il", x, weight, weight.transpose(0, 1)),
        [a],
    )


def test_einsum_over_a_batch():
    a, weight = _einsum_operands()
    _check(
        lambda x: tp.einsum(
            "bij,bjk->bik", x.reshape(2, 1, 4), weight.reshape(1, 4, 3)
        ),
        [a],
    )


def test_einsum_of_a_buffer_with_itself():
    a, _ = _einsum_operands()
    _check(lambda x: tp.einsum("ij,ij->i", x, x), [a])


def test_einsum_then_pointwise():
    a, weight = _einsum_operands()
    _check(lambda x: tp.einsum("ij,jk->ik", x, weight).relu(), [a])


# --------------------------------------------------------------------------
# a middle value with the NaNs left out, and adaptive pooling
# --------------------------------------------------------------------------


def test_nanmedian():
    """The middle element with the NaNs left out is a single value."""

    _check(
        lambda a: tp.nanmedian(a),
        [tp.tensor([[1.0, 2.0, float("nan"), 4.0], [-1.0, 0.5, 2.0, 0.0]])],
    )


def test_adaptive_max_pool_with_indices():
    """Adaptive pooling answers with the values and the places they sat.

    Its output size alternates with its operand, so the places are what
    describes the call.
    """

    image = tp.randn(1, 3, 8, 8)
    _check(
        lambda a: tp.nn.functional.adaptive_max_pool2d_with_indices(a, (4, 4)),
        [image],
    )
    _check(
        lambda a: tp.nn.functional.adaptive_max_pool2d_with_indices(
            a, (3, 5)
        )[1],
        [image],
    )


def test_adaptive_average_pool_of_each_rank():
    """Adaptive average pooling reads the extents after its operand.

    Each rank is a call of its own; the extents alternate with the operand, so
    the places of the call are what describes it.
    """

    line = tp.randn(1, 2, 8)
    grid = tp.randn(1, 2, 4, 4)
    cube = tp.randn(1, 2, 2, 2, 4)
    _check(lambda a: tp.nn.functional.adaptive_avg_pool1d(a, 4), [line])
    _check(lambda a: tp.nn.functional.adaptive_avg_pool2d(a, (2, 2)), [grid])
    _check(lambda a: tp.nn.functional.adaptive_avg_pool3d(a, (1, 1, 2)), [cube])


def test_the_three_dimensional_adaptive_maximum_pool():
    """The values and the places they came from, over three axes."""

    cube = tp.randn(1, 2, 2, 4, 8)
    _check(
        lambda a: tp.nn.functional.adaptive_max_pool3d_with_indices(
            a, (1, 2, 4)
        ),
        [cube],
    )


def test_a_one_dimensional_maximum_pool():
    """The window, step, padding, spacing and border rule are five places.

    They are written in the order the call writes them, and a window that
    cannot slide a whole extra step keeps the same five places, so the border
    rule is spelled rather than assumed.
    """

    line = tp.randn(1, 2, 8)
    _check(lambda a: tp.nn.functional.max_pool1d(a, 2), [line])
    _check(
        lambda a: tp.nn.functional.max_pool1d(a, 3, stride=2, padding=1), [line]
    )
    _check(
        lambda a: tp.nn.functional.max_pool1d(a, 3, stride=2, ceil_mode=True),
        [line],
    )
    _check(
        lambda a: tp.nn.functional.max_pool1d_with_indices(a, 3, stride=2),
        [line],
    )
    _check(
        lambda a: tp.nn.functional.max_pool1d_with_indices(a, 3, stride=2)[1],
        [line],
    )


def test_scaled_dot_product_attention_with_a_mask():
    """The mask is an operand, and it stands between the three and the numbers.

    When the program supplies a mask it is handed over as the operand it is,
    so the call has four operands rather than three and an absent one.
    """

    qkv = tp.randn(2, 4, 8)
    mask = tp.zeros(4, 4)
    _check(
        lambda a: tp.nn.functional.scaled_dot_product_attention(
            a, a, a, attn_mask=mask
        ),
        [qkv],
    )
    _check(
        lambda a: tp.nn.functional.scaled_dot_product_attention(
            a, a, a, attn_mask=mask, scale=0.3
        ),
        [qkv],
    )


def test_scaled_dot_product_attention_with_shared_heads():
    """Fewer key heads than query heads: the query heads are grouped."""

    query = tp.randn(1, 2, 4, 8)
    key = tp.randn(1, 1, 4, 8)
    _check(
        lambda q, k: tp.nn.functional.scaled_dot_product_attention(
            q, k, k, enable_gqa=True
        ),
        [query, key],
    )


# --------------------------------------------------------------------------
# a draw, which reads nothing
# --------------------------------------------------------------------------


def test_a_draw_beside_a_computed_value():
    """A draw has no operand: its extents, type and device are the whole call.

    What it produces stays random, so the two sides are compared from the same
    starting state of the generator rather than value by value.
    """

    x = tp.randn(2, 4)
    _check_seeded(lambda a: a + tp.randn(2, 4), [x])
    _check_seeded(lambda a: a * tp.rand(2, 4), [x])


def test_a_region_that_is_only_a_draw():
    _check_seeded(lambda: tp.randn(3, 4), [])


# --------------------------------------------------------------------------
# a product with a vector, distances between rows, and bags of embeddings
# --------------------------------------------------------------------------


def test_a_matrix_times_a_vector_added_to_a_value():
    """The two scalars that scale the value and the product are places."""

    weight = tp.arange(20, dtype=tp.float32).reshape(4, 5) / 7
    vec = tp.arange(5, dtype=tp.float32) / 7
    _check(lambda a: tp.addmv(a, weight, vec), [tp.randn(4)])
    _check(
        lambda a: tp.addmv(a, weight, vec, beta=0.5, alpha=2.0),
        [tp.randn(4)],
    )


def test_a_distance_between_every_pair_of_rows():
    """The order of the norm is a place; how pairs are traversed is another."""

    _check(lambda a: tp.cdist(a, a * 2), [tp.randn(2, 4)])
    _check(lambda a: tp.cdist(a, a * 2, p=1.5), [tp.randn(2, 4)])


def test_a_bag_of_embeddings():
    """Every bag is combined into one row, in each of the three ways."""

    weight = tp.arange(15, dtype=tp.float32).reshape(5, 3) / 7
    indices = tp.tensor([0, 3, 1], dtype=tp.int64)
    offsets = tp.tensor([0, 2], dtype=tp.int64)
    for mode in ("sum", "mean", "max"):
        _check(
            lambda a: tp.nn.functional.embedding_bag(
                indices, weight, offsets, mode=mode
            ),
            [tp.randn(2, 4)],
        )


# --------------------------------------------------------------------------
# what is refused rather than written
# --------------------------------------------------------------------------


def test_a_scattered_write_is_refused_before_it_is_printed():
    """A scattered write is not a call the host wrapper can spell.

    The refusal comes before anything is printed, so the region is run as it
    was written; printing first would leave a translation unit that cannot
    compile, and the region would fall back only after the compiler had
    rejected it.
    """

    def fn(a):
        index = tp.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=tp.int64)
        return a.scatter_add(0, index, tp.ones(2, 4))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = tp.compile(fn, options={"stax.cuda_codegen": True})
        got = compiled(tp.ones(2, 4))
    refused = [
        str(item.message)
        for item in caught
        if "does not support scatter fallback" in str(item.message)
    ]
    assert refused, "the region was not refused"
    assert tp.allclose(got, fn(tp.ones(2, 4)))


def test_a_result_that_is_a_constant_is_refused_before_it_is_printed():
    """A region whose whole work is a constant has no call to print.

    The slot is described like any other result, but nothing writes it, so the
    caller would read an uninitialized allocation.  Refusing the region keeps
    the answer correct: it is run as it was written, which hands back the
    constant itself.
    """

    def fn(a):
        return tp.nn.functional.one_hot(tp.tensor([0, 2, 1]), 4).to(tp.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compiled = tp.compile(fn, options={"stax.cuda_codegen": True})
        got = compiled(tp.ones(2, 4))
    refused = [
        str(item.message)
        for item in caught
        if "whose result is the constant" in str(item.message)
    ]
    assert refused, "the region was not refused"
    assert tp.allclose(got, fn(tp.ones(2, 4)))
