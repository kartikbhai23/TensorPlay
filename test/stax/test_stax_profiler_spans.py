"""What a profiler span leaves in a captured region.

A span marks a stretch of a program for the profiler and computes nothing.
When a program is captured, the span is written into the graph as the calls
that open and close it, and the rewrites that settle a region drop the pair
again: the region is one call from the profiler's point of view, so a span
inside it would label work that is no longer kept apart.

The graph is observed through the hooks around those rewrites.  Compilation
happens on the first call, which is therefore made while the hooks are still
installed.
"""

import contextlib

import tensorplay as tp

from tensorplay.compiler import config as frontend_config
from tensorplay.compiler.backends.stax import config
from tensorplay.profiler.profiler import (
    _record_function_enter,
    _record_function_exit,
)

SPANS = {_record_function_enter, _record_function_exit}


def _span_calls(graph):
    """How many span markers the graph carries, in the order they appear."""

    return [
        node.target
        for node in graph.nodes
        if node.op == "call_function" and node.target in SPANS
    ]


@contextlib.contextmanager
def _spans_captured(as_spans):
    previous = frontend_config.capture_profiler_record_function
    frontend_config.capture_profiler_record_function = as_spans
    try:
        yield
    finally:
        frontend_config.capture_profiler_record_function = previous


def _observed(fn, *inputs, capture, backward=False):
    """Compile and call ``fn``, reporting the spans each region saw.

    Returns the call's results along with the spans found before and after the
    rewrites, one entry per region: the forward region, and the backward one
    when the call is a training call.
    """

    before, after = [], []

    def record_before(graph):
        before.append(_span_calls(graph))

    def record_after(graph):
        after.append(_span_calls(graph))

    with config.patch(
        post_grad_custom_pre_pass=record_before,
        post_grad_custom_post_pass=record_after,
    ), _spans_captured(capture):
        compiled = tp.compile(fn, strict_native=True)
        results = compiled(*inputs)
        if backward:
            results.backward()
    return results, before, after


def _spanning(fn):
    """Wrap ``fn`` in a span, so every capture of it carries one."""

    def spanned(argument):
        with tp.profiler.record_function("region work"):
            return fn(argument)

    return spanned


def test_a_span_reaches_the_rewrites_and_is_dropped_by_them():
    def fn(x):
        return x.relu() + 1

    results, before, after = _observed(_spanning(fn), tp.tensor([[-1.0, 2.0]]), capture=True)

    assert before[-1] == [_record_function_enter, _record_function_exit]
    assert after[-1] == []
    assert results.tolist() == [[1.0, 3.0]]


def test_a_span_is_not_captured_unless_asked_for():
    def fn(x):
        return x.relu() + 1

    results, before, after = _observed(_spanning(fn), tp.tensor([[-1.0, 2.0]]), capture=False)

    assert before[-1] == []
    assert after[-1] == []
    assert results.tolist() == [[1.0, 3.0]]


def test_a_training_region_captures_a_span_by_the_same_rule():
    def fn(x):
        return (x.relu() + 1).sum()

    x = tp.tensor([[-1.0, 2.0]], requires_grad=True)

    results, before, after = _observed(_spanning(fn), x, capture=True, backward=True)

    # The span was written around the forward computation, so the forward
    # region carries it and the backward one does not; neither keeps it past
    # the rewrites.
    assert len(before) == 2
    assert before[0] == [_record_function_enter, _record_function_exit]
    assert before[1] == []
    assert all(spans == [] for spans in after)
    assert results.tolist() == 4.0
    assert x.grad.tolist() == [[0.0, 1.0]]


def test_a_training_region_keeps_no_span_when_capture_is_off():
    def fn(x):
        return (x.relu() + 1).sum()

    results, before, after = _observed(
        _spanning(fn), tp.tensor([[-1.0, 2.0]], requires_grad=True), capture=False
    )

    assert all(spans == [] for spans in before)
    assert all(spans == [] for spans in after)
    assert results.tolist() == 4.0

def test_a_captured_span_is_not_reported_by_the_profiler():
    def fn(x):
        return x.relu() + 1

    with _spans_captured(True):
        compiled = tp.compile(_spanning(fn), strict_native=True)
        # The first call compiles the region; the span is only observed on the
        # calls that run what was compiled.
        compiled(tp.tensor([[-1.0, 2.0]]))
        with tp.profiler.profile() as prof:
            results = compiled(tp.tensor([[-1.0, 2.0]]))

    # The region is one call from the profiler's point of view, so the span
    # written inside it labels work that is no longer kept apart and is
    # dropped rather than reported around the compiled call.
    assert "region work" not in [name for name, *_ in prof.events]
    assert results.tolist() == [[1.0, 3.0]]


def test_a_span_outside_a_captured_region_is_still_reported():
    with _spans_captured(True):
        with tp.profiler.profile() as prof:
            with tp.profiler.record_function("plain work"):
                x = tp.ones([2])
                y = x.add(x)

    assert "plain work" in [name for name, *_ in prof.events]
    assert y.tolist() == [2.0, 2.0]
