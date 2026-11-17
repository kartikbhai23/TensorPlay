"""mode="reduce-overhead" replay shell around a stax-compiled artifact.

Logic tests run against a stand-in native surface and stand-in CUDA tensors;
the numerics of a real capture are covered by the CUDA-gated cases at the
bottom.
"""

from types import SimpleNamespace

import pytest

import tensorplay as tp
from tensorplay._stax.cudagraphs import (
    CudaGraphError,
    CudagraphCompiledCallable,
    cudagraph_wrap,
    wrap_skip_reason,
)


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------


class FakeDevice:
    def is_cuda(self):
        return True

    def __repr__(self):
        return "cuda:0"


class FakeTensor:
    """Duck-typed CUDA tensor delegating storage to a real CPU tensor."""

    def __init__(self, data):
        self.data = data
        self.device = FakeDevice()

    @property
    def shape(self):
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def requires_grad(self):
        return self.data.requires_grad

    def clone(self):
        return FakeTensor(self.data.clone())

    def copy_(self, other):
        self.data.copy_(other.data)
        return self

    def data_ptr(self):
        return self.data.data_ptr()

    def stride(self):
        return tuple(self.data.stride())

    def tolist(self):
        return self.data.tolist()


def _fake_is_tensor(value):
    return isinstance(value, (tp.Tensor, FakeTensor))


@pytest.fixture
def fake_cuda_probe(monkeypatch):
    """Pretend inputs are CUDA; drop traced metas (they stay CPU here)."""

    import tensorplay._stax.cudagraphs as cg

    monkeypatch.setattr(cg, "_is_tensor", _fake_is_tensor)
    monkeypatch.setattr(cg, "get_device_node_mapping", lambda gm: {})


class FakeGraph:
    def __init__(self):
        self.in_capture = False
        self.captured = False
        self.replays = 0
        self.resets = 0
        self.recompute = None

    def capture_begin(self, pool=0, capture_error_mode="global", stream=None):
        assert not self.in_capture
        self.in_capture = True

    def capture_end(self):
        assert self.in_capture
        self.in_capture = False
        self.captured = True

    def replay(self):
        assert self.captured
        # a real recorded graph re-executes its kernels against the captured
        # input storage; the stand-in recomputes its output the same way
        if self.recompute is not None:
            self.recompute()
        self.replays += 1

    def reset(self):
        self.resets += 1


class FakeNative:
    CUDAGraph = FakeGraph


class FakeNativeNoCapture(FakeNative):
    class CUDAGraph(FakeGraph):
        def capture_begin(self, *args, **kwargs):
            raise RuntimeError("capture rejected by the driver")


def _twice_artifact(graph):
    """Callable standing in for a compiled kernel: out = in * 2.

    During the capture window it registers a recompute hook so the stand-in
    replay refreshes the static output from the captured input storage, the
    way a real recorded graph re-executes its kernels against that storage.
    """

    def artifact(x):
        out = FakeTensor(x.data * 2)
        if graph.in_capture:
            def recompute():
                out.data = x.data * 2

            graph.recompute = recompute
        return out

    return artifact


def _trace_gm(fn, *args):
    """Trace ``fn`` and stamp tensor metadata (the default capture pipeline)."""

    from tensorplay.graph import Tracer
    from tensorplay.graph.passes import ShapeProp

    gm = Tracer().trace(
        fn, sample_inputs={name: value for name, value in zip(("a", "b", "c"), args)}
    )
    ShapeProp(list(args))(gm)
    return gm


def _clean_gm():
    return _trace_gm(lambda a: (a * 2).relu(), tp.tensor([1.0]))


def _capture_stub(entry):
    """Give wrap_skip_reason a truthy compiled artifact without behavior."""
    return entry


# --------------------------------------------------------------------------
# eligibility
# --------------------------------------------------------------------------


def test_skip_reason_dynamic():
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [tp.tensor([1.0])], dynamic=True)
    assert reason == "skipping cudagraphs due to dynamic shape specialization"


def test_skip_reason_cpu_input():
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [tp.tensor([1.0])])
    assert reason == "skipping cudagraphs due to input on a non-CUDA device"


def test_grad_demanding_input_is_capturable(fake_cuda_probe):
    # Parameters demand grad by construction, so refusing capture on that
    # would keep every inference call off the graph.  The wrapper guards the
    # replay against recording calls instead.
    gm = _clean_gm()
    value = FakeTensor(tp.tensor([1.0], requires_grad=True))
    assert wrap_skip_reason(_capture_stub(1), gm, [value]) is None


def test_skip_reason_non_tensor_input(fake_cuda_probe):
    gm = _clean_gm()
    reason = wrap_skip_reason(_capture_stub(1), gm, [FakeTensor(tp.tensor([1.0])), [1, 2]])
    assert "unsupported input type" in reason


def test_skip_reason_mutated_inputs(fake_cuda_probe):
    gm = _trace_gm(lambda a: a.add_(1.0), tp.tensor([1.0]))
    reason = wrap_skip_reason(
        _capture_stub(1), gm, [FakeTensor(tp.tensor([1.0]))]
    )
    assert reason == "skipping cudagraphs due to mutated inputs (a)"


def test_skip_reason_incompatible_op(fake_cuda_probe):
    gm = _trace_gm(lambda a: a.nonzero(), tp.tensor([1.0, 0.0]))
    reason = wrap_skip_reason(
        _capture_stub(1), gm, [FakeTensor(tp.tensor([1.0, 0.0]))]
    )
    assert reason == "skipping cudagraphs due to incompatible op (nonzero)"


def test_skip_reason_clean_region_is_none(fake_cuda_probe):
    gm = _clean_gm()
    assert (
        wrap_skip_reason(_capture_stub(1), gm, [FakeTensor(tp.tensor([1.0]))])
        is None
    )


# --------------------------------------------------------------------------
# wrapper behavior (stand-in native)
# --------------------------------------------------------------------------


def test_replay_reads_the_caller_storage(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    artifact = _twice_artifact(graph)
    wrapped, reason = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert reason is None and isinstance(wrapped, CudagraphCompiledCallable)
    assert not wrapped._trees

    held = FakeTensor(tp.tensor([3.0]))
    out1 = wrapped(held)
    assert out1.tolist() == [6.0]
    held.data.copy_(tp.tensor([-4.0]))  # fresh data, same storage
    out2 = wrapped(held)
    # the recording reads the caller's storage, so new values reach the graph
    assert out2.tolist() == [-8.0]
    # steady state is a replay, not an artifact call: one launch per call
    assert graph.replays == 2  # first call replays once after capture
    assert graph.captured


def test_drifted_input_is_staged_onto_the_recording(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        out = FakeTensor(x.data * 2)
        if graph.in_capture:

            def recompute(source=x, result=out):
                result.data = source.data * 2

            graph.recompute = recompute
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native, trees=True
    )
    held = FakeTensor(tp.tensor([3.0]))
    assert wrapped(held).tolist() == [6.0]
    assert graph.replays == 1  # capture + same-storage replay
    # A tensor at a different address is copied onto the recording's own
    # input; the graph never reads the caller's storage.
    other = FakeTensor(tp.tensor([-4.0]))
    assert wrapped(other).tolist() == [-8.0]
    assert graph.replays == 2
    # only the warmup and the capture window ran the artifact; the moved
    # input replayed like any other call
    assert calls == [[3.0], [3.0]]
    assert wrapped._state == "live"


def test_direct_binding_runs_the_artifact_for_drifted_inputs(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        out = FakeTensor(x.data * 2)
        if graph.in_capture:

            def recompute(source=x, result=out):
                result.data = source.data * 2

            graph.recompute = recompute
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native, trees=False
    )
    held = FakeTensor(tp.tensor([3.0]))
    assert wrapped(held).tolist() == [6.0]
    # With direct binding a moved tensor cannot be seen by the graph, so the
    # call runs the artifact uncaptured for that round.
    assert wrapped(FakeTensor(tp.tensor([-4.0]))).tolist() == [-8.0]
    assert graph.replays == 1
    assert calls == [[3.0], [3.0], [-4.0]]
    assert wrapped(held).tolist() == [6.0]
    assert graph.replays == 2


def test_returned_outputs_survive_later_replays(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    held = FakeTensor(tp.tensor([1.0]))
    first = wrapped(held)
    held.data.copy_(tp.tensor([5.0]))
    again = wrapped(held)
    assert first is not again
    # the earlier result is a buffer of its own, not the recording's static
    # output overwritten by the next replay
    assert first.tolist() == [2.0]
    assert again.tolist() == [10.0]


def test_recording_call_runs_the_artifact(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        return FakeTensor(x.data * 2)

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    with tp.enable_grad():
        out = wrapped(FakeTensor(tp.tensor([2.0], requires_grad=True)))
    assert out.tolist() == [4.0]
    # A replay writes no autograd history, so a call that is recording runs
    # the artifact and the graph is never recorded.
    assert calls == [[2.0]]
    assert not graph.captured

    with tp.no_grad():
        assert wrapped(FakeTensor(tp.tensor([3.0]))).tolist() == [6.0]
    assert graph.captured


def test_only_visible_outputs_are_copied(fake_cuda_probe):
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)

    def artifact(x):
        return (FakeTensor(x.data * 2), FakeTensor(x.data + 1))

    wrapped = CudagraphCompiledCallable(
        artifact, native=native, visible_outputs=(1,)
    )
    first = wrapped(FakeTensor(tp.tensor([2.0])))
    recorded = next(iter(wrapped._manager._entries.values())).static_outputs
    # the result the caller reads is its own buffer; the intermediate the
    # region also handed back is left where the graph wrote it
    assert first[1] is not recorded[1]
    assert first[1].tolist() == [3.0]
    assert first[0] is recorded[0]
    assert first[0].tolist() == [4.0]

    # the next replay may overwrite the intermediate, and the value the
    # caller kept is unaffected
    wrapped(FakeTensor(tp.tensor([5.0])))
    assert first[1].tolist() == [3.0]


def test_region_output_with_an_absent_part_is_replayed(fake_cuda_probe):
    """A module that leaves an auxiliary branch out returns None in its place.

    The result keeps that place: only the parts the program computed have a
    recorded buffer, and a call replaying the graph hands the rest back as the
    value the recording itself returned.
    """

    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    seen = []

    def artifact(x):
        out = FakeTensor(x.data * 2)
        seen.append(1)
        if graph.in_capture:
            def recompute():
                out.data = x.data * 2

            graph.recompute = recompute
        return (out, None)

    wrapped, _ = cudagraph_wrap(
        artifact, _clean_gm(), [FakeTensor(tp.tensor([1.0]))], native=native
    )
    held = FakeTensor(tp.tensor([1.0]))
    first = wrapped(held)
    assert graph.captured
    assert len(first) == 2
    assert first[0].tolist() == [2.0]
    assert first[1] is None

    held.data.copy_(tp.tensor([4.0]))
    again = wrapped(held)
    assert again[0].tolist() == [8.0]
    assert again[1] is None
    entry = next(iter(wrapped._manager._entries.values()))
    assert entry.replays == 2
    assert len(entry.static_outputs) == 2
    # the replay did not recompute the result through the artifact
    assert len(seen) == 2


def test_visible_output_indices_come_from_the_region_meta():
    from tensorplay._stax.cudagraphs import _visible_output_indices

    gm = _clean_gm()
    output = gm.graph.find_nodes(op="output")[0]
    assert _visible_output_indices(gm) is None
    output.meta["user_visible_output_idxs"] = [0]
    assert _visible_output_indices(gm) == (0,)


def test_repeated_drift_switches_to_staged_replay(fake_cuda_probe):
    gm = _clean_gm()
    graphs = []

    def new_graph():
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    native = SimpleNamespace(CUDAGraph=new_graph)

    def artifact(x):
        out = FakeTensor(x.data * 2)
        for graph in graphs:
            if graph.in_capture:
                def recompute(source=x, result=out):
                    result.data = source.data * 2

                graph.recompute = recompute
                break
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped(FakeTensor(tp.tensor([1.0]))).tolist() == [2.0]
    assert not wrapped._trees
    # a moved tensor cannot be read by a recording made against the caller's
    # storage; the first one runs the artifact, the repeat switches to staging
    assert wrapped(FakeTensor(tp.tensor([3.0]))).tolist() == [6.0]
    assert not wrapped._trees
    assert wrapped(FakeTensor(tp.tensor([4.0]))).tolist() == [8.0]
    assert wrapped._trees
    assert len(wrapped._manager._nodes) == 1
    # staged replay answers a moved input without running the artifact again
    assert wrapped(FakeTensor(tp.tensor([5.0]))).tolist() == [10.0]
    assert len(graphs) == 2


def test_each_layout_records_its_own_node(fake_cuda_probe):
    gm = _clean_gm()
    graphs = []

    def new_graph():
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    native = SimpleNamespace(CUDAGraph=new_graph)

    def artifact(x):
        out = FakeTensor(x.data * 2)
        for graph in graphs:
            if graph.in_capture:
                def recompute(source=x, result=out):
                    result.data = source.data * 2

                graph.recompute = recompute
                break
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native, trees=True
    )
    assert wrapped(FakeTensor(tp.tensor([1.0]))).tolist() == [2.0]
    # A different shape has no node yet: it is recorded below the path that
    # produced this call.
    assert wrapped(FakeTensor(tp.tensor([2.0, 3.0]))).tolist() == [4.0, 6.0]
    assert len(graphs) == 2
    paths = sorted(len(path) for path in wrapped._manager.tree())
    assert paths == [1, 2]
    # Coming back to the first layout reuses its recording instead of
    # recording a third node.
    assert wrapped(FakeTensor(tp.tensor([7.0]))).tolist() == [14.0]
    assert len(graphs) == 2


def test_capture_failure_falls_back_permanently(fake_cuda_probe):
    gm = _clean_gm()
    native = FakeNativeNoCapture()
    calls = []

    def artifact(x):
        calls.append(x.data.tolist())
        return FakeTensor(x.data * 2)

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped(FakeTensor(tp.tensor([2.0]))).tolist() == [4.0]
    assert wrapped(FakeTensor(tp.tensor([3.0]))).tolist() == [6.0]
    # capture warmup + the fallback call after the failed capture + the
    # second direct call
    assert len(calls) == 3


def test_kwargs_delegate_to_artifact(fake_cuda_probe):
    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    calls = []

    def artifact(x, scale=2.0):
        calls.append(scale)
        out = FakeTensor(x.data * scale)
        if graph.in_capture:
            def recompute():
                out.data = x.data * 2.0

            graph.recompute = recompute
        return out

    wrapped, _ = cudagraph_wrap(
        artifact, gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped(FakeTensor(tp.tensor([1.0]))).tolist() == [2.0]
    assert wrapped(FakeTensor(tp.tensor([1.0])), scale=3.0).tolist() == [3.0]
    # capture warmup + capture-window run used the default scale; the
    # keyword call skipped replay and ran the artifact directly
    assert calls == [2.0, 2.0, 3.0]


def test_wrapper_follows_the_cudagraph_trees_setting(fake_cuda_probe, monkeypatch):
    import tensorplay.compiler.backends.stax.config as stax_config

    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    monkeypatch.setattr(stax_config.triton, "cudagraph_trees", False)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert not wrapped._trees
    monkeypatch.setattr(stax_config.triton, "cudagraph_trees", True)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native
    )
    assert wrapped._trees


def test_reset_releases_wrapped_graphs(fake_cuda_probe):
    from tensorplay._stax.cudagraphs import CudagraphsBackend

    gm = _clean_gm()
    graph = FakeGraph()
    native = SimpleNamespace(CUDAGraph=lambda: graph)
    wrapped, _ = cudagraph_wrap(
        _twice_artifact(graph), gm, [FakeTensor(tp.tensor([1.0]))], native=native,
        trees=True,
    )
    wrapped(FakeTensor(tp.tensor([1.0])))
    manager = wrapped._manager
    assert manager in CudagraphsBackend._managers
    CudagraphsBackend.reset()
    assert not manager._nodes
    assert graph.resets == 1


# --------------------------------------------------------------------------
# mode wiring through tp.compile
# --------------------------------------------------------------------------


def test_reduce_overhead_mode_compiles_and_runs_on_cpu():
    compiled = tp.compile(lambda a: (a * 2).relu().sum(), mode="reduce-overhead")
    x = tp.tensor([1.0, -2.0, 3.0])
    assert compiled(x).item() == 8.0


def test_max_autotune_no_cudagraphs_mode_accepted():
    compiled = tp.compile(lambda a: (a * 2).relu(), mode="max-autotune-no-cudagraphs")
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]


def test_stax_cudagraphs_option_accepted():
    compiled = tp.compile(
        lambda a: (a * 2).relu(), options={"stax.cudagraphs": True}
    )
    assert compiled(tp.tensor([1.0, 2.0])).tolist() == [2.0, 4.0]


def test_stax_cudagraphs_option_validated():
    # the backend lowers lazily: the option contract bites on first call
    compiled = tp.compile(lambda a: a, options={"stax.cudagraphs": "yes"})
    with pytest.raises(RuntimeError, match="must be bool values"):
        compiled(tp.tensor([1.0]))


# --------------------------------------------------------------------------
# real CUDA capture (gated)
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="CUDA runtime is not available"
)


@requires_cuda
def test_reduce_overhead_replays_compiled_region():
    compiled = tp.compile(
        lambda a: (a * 2).relu().sum(dim=-1), mode="reduce-overhead"
    )
    with tp.no_grad():
        x = tp.randn(4, 8, device="cuda")
        first = compiled(x)
        expected1 = (x * 2).relu().sum(dim=-1)
        assert (first - expected1).abs().max().item() < 1e-5
        x.copy_(tp.randn(4, 8, device="cuda"))  # fresh data, same storage
        second = compiled(x)
        expected2 = (x * 2).relu().sum(dim=-1)
        assert (second - expected2).abs().max().item() < 1e-5
        # outputs are buffers of their own: the second replay does not
        # overwrite the first result
        assert first is not second
        assert (first - expected1).abs().max().item() < 1e-5
        # a tensor at a new address is staged onto the recording's own input
        # and replayed; it does not fall back to the uncaptured artifact
        y = tp.randn(4, 8, device="cuda")
        third = compiled(y)
        assert (third - (y * 2).relu().sum(dim=-1)).abs().max().item() < 1e-5


@requires_cuda
def test_reduce_overhead_skips_mutating_regions(caplog):
    compiled = tp.compile(lambda a: a.add_(1.0), mode="reduce-overhead")
    import logging

    with caplog.at_level(logging.WARNING, logger="tensorplay._stax.cudagraphs"):
        x = tp.zeros(4, device="cuda")
        compiled(x)
    assert "mutated inputs" in caplog.text


@requires_cuda
def test_reduce_overhead_falls_back_when_capture_fails(caplog, monkeypatch):
    import logging

    import tensorplay._stax.cudagraphs as cg

    class ExplodingGraph(FakeGraph):
        def capture_begin(self, *args, **kwargs):
            raise CudaGraphError("capture rejected by the driver")

    monkeypatch.setattr(
        cg, "_default_native", lambda: type("N", (), {"CUDAGraph": ExplodingGraph})
    )
    compiled = tp.compile(lambda a: (a * 2).relu().sum(), mode="reduce-overhead")
    with caplog.at_level(logging.WARNING, logger="tensorplay._stax.cudagraphs"):
        x = tp.randn(4, device="cuda")
        assert compiled(x).item() == (x * 2).relu().sum().item()
    assert "capture failed" in caplog.text


# --------------------------------------------------------------------------
# mode-to-options mapping
# --------------------------------------------------------------------------

def test_list_mode_options_mapping():
    from tensorplay.compiler import list_mode_options

    assert list_mode_options("default") == {}
    assert list_mode_options("reduce-overhead") == {"stax.cudagraphs": True}
    assert list_mode_options("max-autotune-no-cudagraphs") == {
        "stax.max_autotune": True,
        "stax.coordinate_descent_tuning": True,
    }
    assert list_mode_options("max-autotune") == {
        "stax.max_autotune": True,
        "stax.cudagraphs": True,
        "stax.coordinate_descent_tuning": True,
    }
    full = list_mode_options()
    assert set(full) == {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }
    with pytest.raises(RuntimeError, match="Unrecognized mode"):
        list_mode_options("fastest")


def test_stax_mode_patch_overlaid_by_explicit_options(monkeypatch):
    from tensorplay.compiler.backends.stax import backend as stax_backend
    from tensorplay.compiler.backends import cudagraphs as cg

    import contextlib

    patches: list[dict] = []

    @contextlib.contextmanager
    def fake_patch(settings):
        patches.append(dict(settings))
        yield

    def fake_lower(graph_module, example_inputs, **kwargs):
        return lambda *args: "lowered"

    monkeypatch.setattr(stax_backend, "_lower_stax_region", fake_lower)
    monkeypatch.setattr(stax_backend.config, "patch", fake_patch)

    wrapped: list = []

    def fake_wrap(compiled, graph_module, example_inputs, dynamic=None):
        wrapped.append(compiled)
        return compiled, None

    monkeypatch.setattr(cg, "cudagraph_wrap", fake_wrap)
    graph_like = SimpleNamespace()

    # the mode's patch lands as defaults and drives the replay wrap...
    stax_backend.stax(graph_like, [], mode="max-autotune")
    assert patches[0]["max_autotune"] is True
    assert patches[0]["coordinate_descent_tuning"] is True
    assert len(wrapped) == 1

    # ...and an explicit option wins over the mode patch per key
    stax_backend.stax(
        graph_like, [], mode="max-autotune", options={"stax.cudagraphs": False}
    )
    assert patches[1]["max_autotune"] is True
    assert len(wrapped) == 1  # no further wrap: explicit opt-out

    # a mode without the autotune knobs leaves them off
    stax_backend.stax(
        graph_like, [], mode="reduce-overhead", options={"stax.max_autotune": False}
    )
    assert "max_autotune" not in patches[2]
    assert "coordinate_descent_tuning" not in patches[2]
    assert len(wrapped) == 2


def test_max_autotune_mode_compiles_and_matches_eager():
    def fn(a):
        return (a * 3 + 1).relu().sum()

    compiled = tp.compile(fn, mode="max-autotune")
    x = tp.randn(16, 64)
    assert tp.allclose(compiled(x), fn(x))


def test_max_autotune_no_cudagraphs_mode_compiles_and_matches_eager():
    def fn(a):
        return (a.sin() * a).sum()

    compiled = tp.compile(fn, mode="max-autotune-no-cudagraphs")
    x = tp.randn(32, 16)
    assert tp.allclose(compiled(x), fn(x))


class _ScaledSteady(tp.nn.Module):
    """A module whose output depends on a plain attribute and a buffer."""

    def __init__(self):
        super().__init__()
        self.bias = tp.zeros(3, device="cuda")
        self.scale = 2.0

    def forward(self, x):
        return (x + self.bias) * self.scale


@requires_cuda
def test_steady_state_repeats_the_recorded_answer():
    module = _ScaledSteady()
    compiled = tp.compile(module, fullgraph=True, mode="reduce-overhead")
    x = tp.ones(3, device="cuda")
    with tp.no_grad():
        first = compiled(x)
        assert first.tolist() == [2.0, 2.0, 2.0]
        second = compiled(x)
        assert second.tolist() == [2.0, 2.0, 2.0]
        # Each replay answers with a tensor of its own, not the buffer the
        # previous call returned.
        assert first is not second
        assert first.tolist() == [2.0, 2.0, 2.0]


@requires_cuda
def test_steady_state_returns_to_the_slow_path_when_an_attribute_moves():
    module = _ScaledSteady()
    compiled = tp.compile(module, fullgraph=True, mode="reduce-overhead")
    x = tp.ones(3, device="cuda")
    with tp.no_grad():
        assert compiled(x).tolist() == [2.0, 2.0, 2.0]
        assert compiled(x).tolist() == [2.0, 2.0, 2.0]
        module.scale = 3.0
        assert compiled(x).tolist() == [3.0, 3.0, 3.0]
        assert compiled(x).tolist() == [3.0, 3.0, 3.0]


@requires_cuda
def test_steady_state_notices_a_value_written_in_place():
    module = _ScaledSteady()
    compiled = tp.compile(module, fullgraph=True, mode="reduce-overhead")
    x = tp.ones(3, device="cuda")
    with tp.no_grad():
        assert compiled(x).tolist() == [2.0, 2.0, 2.0]
        # The buffer keeps its identity and moves its contents: the version it
        # carries is what says the recording no longer describes the call.
        module.bias.add_(1.0)
        assert compiled(x).tolist() == [4.0, 4.0, 4.0]


@requires_cuda
def test_steady_state_follows_a_training_flag_flip():
    module = _ScaledSteady()
    compiled = tp.compile(module, fullgraph=True, mode="reduce-overhead")
    x = tp.ones(3, device="cuda")
    module.eval()
    with tp.no_grad():
        assert compiled(x).tolist() == [2.0, 2.0, 2.0]
        module.train()
        # The flag is part of what the module tree certifies; the call has to
        # come back through the wrapper rather than replay a stale tree.
        assert compiled(x).tolist() == [2.0, 2.0, 2.0]
