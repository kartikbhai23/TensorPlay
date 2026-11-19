"""Writing a product that is narrow in one dimension as pointwise work.

A product whose result is written one row at a time spends its time reading
the large operand over and over; the same arithmetic written as a broadcast
multiply and a sum reads each operand once.  The rewrite runs only when asked
for by name and only where the shapes make it worthwhile, and what it
produces must equal what the product did.
"""

import pytest
import tensorplay as tp

from tensorplay.compiler.backends.stax import config

pytestmark = pytest.mark.skipif(
    not tp._C.has_mkldnn(),
    reason="the product table is filled by builds that carry oneDNN",
)


def _compiled_order(fn, *args, **settings):
    """Compile and call ``fn``, reporting the operations its region kept."""

    seen = []

    def record(graph):
        seen.append([node.target for node in graph.nodes])

    with config.patch(post_grad_custom_post_pass=record, **settings):
        compiled = tp.compile(fn, strict_native=True)
        result = compiled(*args)
    return result, seen[-1]


DECOMPOSE = {"post_grad_fusion_options": {"decompose_mm_pass": {}}}


def test_the_table_is_filled_when_a_region_is_lowered():
    # Importing the pass module is what fills its table; a region that is
    # compiled at all is enough to have asked for it.
    _compiled_order(lambda x: x + 1, tp.randn(2, 2))

    from tensorplay.compiler.backends.stax.graph_passes.split_cat import (
        POST_GRAD_PATTERNS,
    )

    assert POST_GRAD_PATTERNS["decompose_mm_pass"].patterns


def test_a_product_with_one_row_is_written_as_multiply_and_sum():
    def fn(a, b):
        return tp.mm(a, b)

    a = tp.randn(1, 8)
    b = tp.randn(8, 16)
    result, targets = _compiled_order(fn, a, b, **DECOMPOSE)

    assert tp.ops.tp.mm.default not in targets
    assert tp.ops.tp.mul.Tensor in targets
    assert tp.allclose(result, tp.mm(a, b), atol=1e-5)


def test_the_product_with_a_bias_is_written_the_same_way():
    def fn(a, b, bias):
        return tp.addmm(bias, a, b)

    a = tp.randn(1, 8)
    b = tp.randn(8, 16)
    bias = tp.randn(1, 16)
    result, targets = _compiled_order(fn, a, b, bias, **DECOMPOSE)

    assert tp.ops.tp.addmm.default not in targets
    assert tp.ops.tp.mul.Tensor in targets
    assert tp.ops.tp.add.Tensor in targets
    assert tp.allclose(result, tp.addmm(bias, a, b), atol=1e-5)


def test_a_batch_with_one_matrix_is_written_the_same_way():
    def fn(a, b):
        return tp.bmm(a, b)

    a = tp.randn(1, 2, 8)
    b = tp.randn(1, 8, 16)
    result, targets = _compiled_order(fn, a, b, **DECOMPOSE)

    assert tp.ops.tp.bmm.default not in targets
    assert tp.ops.tp.mul.Tensor in targets
    assert tp.allclose(result, tp.bmm(a, b), atol=1e-5)


def test_a_product_is_kept_whole_unless_the_rewrite_is_asked_for():
    def fn(a, b):
        return tp.mm(a, b)

    result, targets = _compiled_order(fn, tp.randn(1, 8), tp.randn(8, 16))

    assert tp.ops.tp.mm.default in targets
    assert result.shape == (1, 16)


def test_a_product_outside_the_shape_limits_is_kept_whole():
    # Two rows is already past the point where writing the product one row at
    # a time was judged to pay off on a host.
    def fn(a, b):
        return tp.mm(a, b)

    a = tp.randn(2, 8)
    b = tp.randn(8, 16)
    result, targets = _compiled_order(fn, a, b, **DECOMPOSE)

    assert tp.ops.tp.mm.default in targets
    assert result.shape == (2, 16)