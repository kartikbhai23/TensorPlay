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
