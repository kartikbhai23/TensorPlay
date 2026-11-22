"""linspace / logspace across every element type they accept."""

import math

import pytest

import tensorplay as tp


def _close(got, want, tol=1e-5):
    assert len(got) == len(want), (got, want)
    for a, b in zip(got, want):
        assert abs(a - b) <= tol * max(1.0, abs(b)), (got, want)


# --------------------------------------------------------------- linspace


def test_linspace_float_is_unchanged():
    _close(tp.linspace(0.0, 1.0, 5).tolist(), [0.0, 0.25, 0.5, 0.75, 1.0])
    _close(tp.linspace(0.0, 1.0, 5, dtype=tp.float64).tolist(),
           [0.0, 0.25, 0.5, 0.75, 1.0])


def test_linspace_pins_both_endpoints():
    values = tp.linspace(-3.0, 7.0, 11).tolist()
    assert values[0] == -3.0
    assert values[-1] == 7.0


@pytest.mark.parametrize("dtype", [tp.int8, tp.int16, tp.int32, tp.int64,
                                   tp.uint8])
def test_linspace_truncates_toward_zero_for_integers(dtype):
    # step 2.5: 0, 2.5, 5, 7.5, 10 truncated to the element type
    assert tp.linspace(0, 10, 5, dtype=dtype).tolist() == [0, 2, 5, 7, 10]


def test_linspace_integer_exact_step():
    assert tp.linspace(0, 3, 4, dtype=tp.int64).tolist() == [0, 1, 2, 3]
    assert tp.linspace(10, 0, 6, dtype=tp.int32).tolist() == [10, 8, 6, 4, 2, 0]


def test_linspace_narrows_the_endpoints_to_the_element_type():
    # the endpoints land where int32 puts them, so the walk runs 0 -> 3
    assert tp.linspace(0.5, 3.5, 4, dtype=tp.int32).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16])
def test_linspace_reduced_precision(dtype):
    values = tp.linspace(0.0, 1.0, 5, dtype=dtype).to(tp.float32).tolist()
    _close(values, [0.0, 0.25, 0.5, 0.75, 1.0], tol=1e-2)


def test_linspace_complex():
    values = tp.linspace(0.0, 1.0, 3, dtype=tp.complex64)
    _close(values.real.tolist(), [0.0, 0.5, 1.0])
    _close(values.imag.tolist(), [0.0, 0.0, 0.0])


def test_linspace_edge_step_counts():
    assert tp.linspace(2.0, 9.0, 0).numel() == 0
    assert tp.linspace(2.0, 9.0, 1).tolist() == [2.0]
    assert tp.linspace(2, 9, 1, dtype=tp.int32).tolist() == [2]
    with pytest.raises(RuntimeError):
        tp.linspace(0.0, 1.0, -1)


# --------------------------------------------------------------- logspace


def test_logspace_float_is_unchanged():
    _close(tp.logspace(0.0, 3.0, 4).tolist(), [1.0, 10.0, 100.0, 1000.0])


@pytest.mark.parametrize("dtype", [tp.int16, tp.int32, tp.int64])
def test_logspace_integer(dtype):
    assert tp.logspace(0, 3, 4, dtype=dtype).tolist() == [1, 10, 100, 1000]


def test_logspace_honours_the_base():
    _close(tp.logspace(0.0, 4.0, 5, base=2.0).tolist(),
           [1.0, 2.0, 4.0, 8.0, 16.0])
    assert tp.logspace(0, 4, 5, base=2.0, dtype=tp.int32).tolist() == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("dtype", [tp.float16, tp.bfloat16])
def test_logspace_reduced_precision(dtype):
    values = tp.logspace(0.0, 2.0, 3, dtype=dtype).to(tp.float32).tolist()
    _close(values, [1.0, 10.0, 100.0], tol=1e-2)


def test_logspace_complex():
    values = tp.logspace(0.0, 2.0, 3, dtype=tp.complex64)
    _close(values.real.tolist(), [1.0, 10.0, 100.0], tol=1e-4)


def test_logspace_edge_step_counts():
    assert tp.logspace(0.0, 3.0, 0).numel() == 0
    _close(tp.logspace(2.0, 9.0, 1).tolist(), [100.0])
    assert tp.logspace(2, 9, 1, dtype=tp.int32).tolist() == [100]
    with pytest.raises(RuntimeError):
        tp.logspace(0.0, 1.0, -1)


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("factory", [tp.linspace, tp.logspace])
def test_cuda_range_accepts_scalar_views_and_can_write_an_empty_result(factory):
    bounds = tp.tensor([99.0, 0.0, 2.0], device="cuda")
    start, end = bounds[1], bounds[2]
    assert start.item() == 0.0 and end.item() == 2.0
    _close(factory(start, end, 3, device="cuda").tolist(), factory(0, 2, 3).tolist())
    out = tp.empty((3,), dtype=tp.float64, device="cuda")
    assert factory(start, end, 0, out=out).shape == out.shape == (0,)
    with pytest.raises(RuntimeError, match="does not match dtype"):
        factory(start, end, 0, dtype=tp.float32, out=out)
    graph = tp.cuda.CUDAGraph()
    with tp.cuda.graph(graph):
        result = factory(start, end, 0, dtype=tp.float64, device="cuda")
        marker = end + 1
    graph.replay()
    assert result.shape == (0,) and result.dtype == tp.float64
    assert marker.item() == 3.0


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_scalar_reads_complex_and_unsigned_values():
    values = tp.tensor([0j, 2 - 3j], dtype=tp.complex128, device="cuda")
    assert values[1].item() == 2 - 3j
    values = tp.tensor([0, 2**58], dtype=tp.uint64, device="cuda")
    assert values[1].item() == 2**58


@pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("factory", [tp.linspace, tp.logspace])
def test_cuda_range_reads_mixed_bounds_on_the_active_stream(factory):
    with tp.cuda.stream(tp.cuda.Stream()):
        first = tp.tensor([11, 0], dtype=tp.int64, device="cuda")[1]
        last = tp.tensor([13.0, 2.0], dtype=tp.float32, device="cuda")[1]
        out = tp.empty(3, dtype=tp.float64, device="cuda")
        factory(first, last, 3, out=out)
        _close(out.tolist(), factory(0, 2, 3).tolist())


def test_unsupported_dtype_reports_the_name():
    with pytest.raises(Exception):
        tp.linspace(0.0, 1.0, 4, dtype=tp.bool)
