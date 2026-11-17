"""The prepared matrix product on the host: what it is offered to, and the answer.

A product is the largest thing in most regions, so it is the one place where
the measurement matters most -- and also the one where a template that is
offered when it should not be, or an answer that is accepted when it is wrong,
would be least visible.  These tests pin both sides: the kernel is chosen for
the calls it was written for, and what it produces is the product the framework
produces.
"""

import os

import pytest

import tensorplay as tp
from tensorplay.compiler.backends.stax import config
from tensorplay.compiler.backends.stax import ir
from tensorplay.compiler.backends.stax import kernel_cache
from tensorplay.compiler.backends.stax.templates import select_algorithm
from tensorplay.compiler.backends.stax.utils import counters

#: The settings a measured product is chosen under, with only the prepared
#: product on offer.  Naming one backend is what makes the counter below mean
#: "this kernel and no other".
MEASURED_AS_CPP = {
    "max_autotune": True,
    "max_autotune_gemm": True,
    "max_autotune_gemm_backends": "CPP",
}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep this test's generated code and decisions to itself."""

    monkeypatch.setenv("TP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(kernel_cache, "_default_caches", {})
    # Which candidate was measured is remembered for the process, not for the
    # cache directory, so a test that ran before could hand this one a decision
    # -- and a kernel that was already built, which is a kernel this test did
    # not see being written.  Each test measures its own.
    monkeypatch.setattr(
        select_algorithm, "_ALGORITHM_SELECTOR_CACHE", None
    )
    return tmp_path


def _templated_kernels():
    return int(counters["tp"]["cpp_templated_kernel_counter"])


def _compiled(fn, *args, settings=None):
    """The program compiled for the host, and its answer."""

    with config.patch(settings or MEASURED_AS_CPP):
        compiled = tp.compile(fn, backend="stax")
        return compiled(*args)


def _matches_eager(fn, *args, atol=1e-4, settings=None):
    """Whether the compiled call answers what the same call answers uncompiled."""

    out = _compiled(fn, *args, settings=settings)
    expected = fn(*args)
    assert out.shape == expected.shape
    assert float((out - expected).abs().max().item()) <= atol
    return out


def test_a_product_is_written_as_one_prepared_kernel():
    """The measured product is emitted, not handed to the library."""

    def mm(x, weight):
        return tp.mm(x, weight)

    before = _templated_kernels()
    x = tp.randn(64, 128)
    weight = tp.randn(128, 96)
    _matches_eager(mm, x, weight)
    assert _templated_kernels() == before + 1


def test_a_product_with_a_bias_folds_the_bias_into_the_kernel():
    """The bias is added where the result is stored, not by a second pass."""

    def addmm(bias, x, weight):
        return tp.addmm(bias, x, weight)

    x = tp.randn(128, 256)
    weight = tp.randn(256, 129)
    before = _templated_kernels()
    _matches_eager(addmm, tp.randn(129), x, weight)
    _matches_eager(addmm, tp.randn(128, 129), x, weight)
    assert _templated_kernels() == before + 2


def test_a_shape_the_register_blocks_do_not_divide_still_matches():
    """A result whose column count is not a whole block is written correctly."""

    def mm(x, weight):
        return tp.mm(x, weight)

    before = _templated_kernels()
    # Every extent is deliberately awkward: the last block of each axis is a
    # partial one, which is where a kernel that assumed whole blocks would read
    # or write past what it was given.
    _matches_eager(mm, tp.randn(37, 23), tp.randn(23, 71))
    assert _templated_kernels() == before + 1


@pytest.mark.parametrize("dtype", [tp.bfloat16, tp.float16])
def test_a_narrow_product_is_computed_in_float(dtype):
    """A narrow product widens its operands and narrows what it stores."""

    def mm(x, weight):
        return tp.mm(x, weight)

    x = tp.randn(64, 96).to(dtype)
    weight = tp.randn(96, 48).to(dtype)
    before = _templated_kernels()
    _matches_eager(mm, x, weight, atol=2e-2)
    assert _templated_kernels() == before + 1


def test_the_positional_spelling_of_a_product_with_a_bias_is_lowered():
    """A call written through the function that declares the scalars is lowered.

    The function takes the addend's multiplier and then the product's, and
    records them in that order -- so a lowering that only accepted them by name
    would refuse every call written that way, which is what a program written
    against the function produces.  The second call is the one that matters: a
    call compiled once is captured again when a later call's shapes differ, and
    that second capture is the one written this way.
    """

    import warnings

    def addmm(bias, x, weight):
        return tp.addmm(bias, x, weight)

    x = tp.randn(64, 128)
    weight = tp.randn(128, 96)
    row_bias = tp.randn(96)
    matrix_bias = tp.randn(64, 96)
    with warnings.catch_warnings():
        # A call that could not be lowered is answered by the graph executor
        # with a warning, so a warning here is the failure this pins.
        warnings.simplefilter("error", UserWarning)
        with config.patch(MEASURED_AS_CPP):
            compiled = tp.compile(addmm, backend="stax")
            row = compiled(row_bias, x, weight)
            matrix = compiled(matrix_bias, x, weight)
    assert float((row - tp.addmm(row_bias, x, weight)).abs().max().item()) <= 1e-4
    assert float((matrix - tp.addmm(matrix_bias, x, weight)).abs().max().item()) <= 1e-4


def test_nothing_is_prepared_when_nothing_is_measured():
    """Without a measurement the library's own kernel answers, as before."""

    def mm(x, weight):
        return tp.mm(x, weight)

    before = _templated_kernels()
    _matches_eager(
        mm,
        tp.randn(64, 128),
        tp.randn(128, 96),
        settings={"max_autotune": False, "max_autotune_gemm": False},
    )
    assert _templated_kernels() == before


def test_a_type_the_kernel_does_not_compute_in_is_not_offered():
    """A product of another type is the library's, and is answered by it.

    Asked under the ordinary list of backends rather than the one that names
    only this kernel, so that the framework's own product is on the table and
    the answer is a real answer rather than the absence of any candidate.
    """

    def mm(x, weight):
        return tp.mm(x, weight)

    x = tp.randn(32, 48).to(tp.float64)
    weight = tp.randn(48, 16).to(tp.float64)
    before = _templated_kernels()
    _matches_eager(
        mm,
        x,
        weight,
        atol=1e-9,
        settings={"max_autotune": True, "max_autotune_gemm": True},
    )
    assert _templated_kernels() == before


def test_both_spellings_of_a_product_reach_the_same_lowering():
    """A call records its default form, and that form has to be the tuned one.

    A record of a product names the default overload of the operation, while
    the table is keyed by the name a call carries.  An entry registered under
    one spelling only is an entry a program never reaches, and the tuning it
    describes would then never happen for any call a program writes.
    """

    from tensorplay.compiler.backends.stax.op_lowerings import LOWERINGS
    from tensorplay.compiler.backends.stax.templates import mm as products

    assert LOWERINGS["mm"] is products.tuned_mm
    assert LOWERINGS["mm.default"].__module__ == products.__name__
    assert LOWERINGS["addmm"] is products.tuned_addmm
    assert LOWERINGS["addmm.default"] is products.lower_addmm_default


def test_a_registered_operation_answers_a_call_that_names_where_it_writes():
    """The name a wrapper calls is one that accepts an output buffer.

    A generated wrapper says where a result goes, and the operation it names has
    to be one that can be told: a plain function of the same name answers the
    call without the destination and rejects it.
    """

    from tensorplay.compiler.backends.stax.templates import mm as products
    from tensorplay.compiler.backends.stax.templates import select_algorithm

    x = tp.randn(4, 8)
    weight = tp.randn(8, 16)
    out = tp.empty(4, 16)
    written = getattr(select_algorithm.extern_kernels, products.framework_mm.name)
    written(x, weight, out=out)
    assert float((out - tp.mm(x, weight)).abs().max().item()) == 0.0


def test_a_size_hint_covers_a_symbol_nothing_pins_down():
    """A kernel's own loop symbol still gets a size to write code for.

    The hint is what a size is chosen from -- how much work one thread takes --
    and a symbol that only a loop body knows has no value to be read, so the
    fallback stands in rather than the question failing.
    """

    import sympy

    from tensorplay.graph.experimental._size_hinting import _optimization_hint_base
    from tensorplay.graph.experimental.symbolic_shapes import ShapeEnv

    start, end = sympy.Symbol("m_start"), sympy.Symbol("m_end")
    # Neither symbol is pinned to anything, so both stand in with the fallback
    # and the difference collapses: the hint is a number, which is the point --
    # what the number is has no meaning beyond being a size to write code for.
    hint = _optimization_hint_base(ShapeEnv(), (end - start) * (end - start) + 1, None, 8)
    assert hint == 1


def test_a_buffer_reports_what_it_reads_without_running_its_body():
    """Reading the names a body reads does not compute what it stores.

    A body is written against the operations a kernel offers, and one written as
    an expression of its reads rather than as calls of them is still a body
    whose reads are known without the expression being evaluated.
    """

    from tensorplay.compiler.backends.stax.loops import V

    body = ir.Pointwise(
        device=tp.device("cpu"),
        dtype=tp.float32,
        inner_fn=lambda index: V.ops.add(
            V.ops.load("left", index[0]), V.ops.load("right", index[0])
        ),
        ranges=[4],
    )

    buffer = ir.ComputedBuffer(
        name="sum_of_two",
        layout=ir.FixedLayout(tp.device("cpu"), tp.float32, [4]),
        data=body,
    )
    assert buffer.get_read_names() == {"left", "right"}
    assert buffer.num_reads() == 2


def test_the_cache_directories_are_the_ones_this_test_made(tmp_path):
    """A guard on the fixture above, so a test cannot quietly share artifacts."""

    assert os.environ["TP_CACHE_DIR"] == str(tmp_path)